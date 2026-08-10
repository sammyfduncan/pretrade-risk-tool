"""Walk forward training harness.

Uses sklearn's TimeSeriesSplit: each fold trains on an expanding window 
and tests on the block immediately after it, so every test prediction comes
from a model that has never seen that bar or anything after it.

The gap parameter is load bearing: labels look horizon bars into the
future, so the last horizon training bars share future price action with
the first test bars. Skipping gap >= horizon bars between train and test
removes that overlap. AssetConfig enforces gap >= horizon at load time.

Concatenated per fold test predictions form the out of fold series; 
honest simulation of model predictions; which is only thing backtest
is allowed to consume.

Two corrections layer on top, both fitted strictly inside each fold:

  * sample weighting by label uniqueness (sample_weights.py), because
    overlapping forward windows make consecutive rows near-duplicates;
  * probability calibration (calibration.py), because the raw scores rank
    better than they measure.

Both are per-fold on purpose. Computing either across the whole dataset would
let test period rows influence how training period rows are weighted/scored — 
mild leak, but exactly the kind this project exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from commodity_ml.modeling.calibration import (
    CalibratedModel,
    IdentityCalibrator,
    calibration_split,
    fit_calibrator,
)
from commodity_ml.modeling.models import make_model
from commodity_ml.modeling.sample_weights import uniqueness_weights


@dataclass
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    model: object
    calibrator: object = field(default_factory=IdentityCalibrator)
    # Effective sample size after uniqueness weighting: how many genuinely
    # independent observations the fold actually trained on, versus n_train.
    effective_n_train: float | None = None


@dataclass
class WalkForwardResult:
    """Two out of fold series, both kept apart.

    oof_proba is raw model score. It is what the backtest ranks on,
    because the entry rule is a *quantile* of the prediction distribution and
    quantiles are a statement about ordering.

    oof_calibrated is same score mapped onto honest probabilities. It is
    what Brier is measured on and what position sizing would use later.
    """

    oof_proba: pd.Series          # raw out-of-fold score — ranking, entries
    oof_calibrated: pd.Series     # calibrated P(label=1) — Brier, sizing
    folds: list[FoldResult] = field(default_factory=list)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    gap: int,
    model_kind: str,
    exit_time: pd.Series | None = None,
    sample_weighting: str = "none",
    calibration: str = "none",
    purge_by_exit_time: bool = False,
) -> WalkForwardResult:
    """Run expanding window walk forward training and collect OOF predictions."""
    if not X.index.is_monotonic_increasing:
        raise ValueError("X must be sorted by time before walk-forward splitting.")
    if sample_weighting == "uniqueness" and exit_time is None:
        raise ValueError(
            "sample_weighting='uniqueness' needs each label's exit_time "
            "(the 'exit_time' column produced by labels.make_labels)."
        )
    if purge_by_exit_time and exit_time is None:
        raise ValueError("purge_by_exit_time=True requires exit_time")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    result = WalkForwardResult(
        oof_proba=pd.Series(dtype=float, name="proba"),
        oof_calibrated=pd.Series(dtype=float, name="proba_calibrated"),
    )
    raw_chunks: list[pd.Series] = []
    calibrated_chunks: list[pd.Series] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if purge_by_exit_time:
            test_start = X.index[test_idx[0]]
            train_exit = pd.to_datetime(exit_time.iloc[train_idx], utc=True)
            train_idx = train_idx[(train_exit < test_start).to_numpy()]
            if not len(train_idx):
                raise ValueError(f"Fold {fold} has no rows after exit-time purging")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        weights = _fold_weights(sample_weighting, X_train.index, exit_time)

        calibrator = _fit_fold_calibrator(
            X_train,
            y_train,
            weights,
            gap,
            model_kind,
            calibration,
            exit_time=exit_time,
            purge_by_exit_time=purge_by_exit_time,
        )

        model = _fit(make_model(model_kind), X_train, y_train, weights)
        raw = model.predict_proba(X.iloc[test_idx])[:, 1]
        test_index = X.index[test_idx]

        raw_chunks.append(pd.Series(raw, index=test_index, name="proba"))
        calibrated_chunks.append(
            pd.Series(calibrator.transform(raw), index=test_index, name="proba_calibrated")
        )
        result.folds.append(
            FoldResult(
                fold=fold,
                train_start=X.index[train_idx[0]],
                train_end=X.index[train_idx[-1]],
                test_start=X.index[test_idx[0]],
                test_end=X.index[test_idx[-1]],
                n_train=len(train_idx),
                n_test=len(test_idx),
                model=model,
                calibrator=calibrator,
                effective_n_train=_effective_n(weights),
            )
        )

    result.oof_proba = pd.concat(raw_chunks).sort_index()
    result.oof_calibrated = pd.concat(calibrated_chunks).sort_index()
    return result

def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_kind: str,
    exit_time: pd.Series | None = None,
    sample_weighting: str = "none",
    calibration: str = "none",
    gap: int = 0,
    purge_by_exit_time: bool = False,
) -> CalibratedModel:
    """Fits onto all history — the artifact for future/live prediction.

    Its quality estimate comes from the walk-forward folds, never from
    predictions on its own training data. The returned object carries the
    calibrator and the feature order with it, so the live path cannot apply
    one without the other.
    Quality estimate comes from walk forward folds, never from preds on own
    train data. Returned obj. carries calibrator and feature order with it.
    """
    weights = _fold_weights(sample_weighting, X.index, exit_time)
    calibrator = _fit_fold_calibrator(
        X,
        y,
        weights,
        gap,
        model_kind,
        calibration,
        exit_time=exit_time,
        purge_by_exit_time=purge_by_exit_time,
    )
    model = _fit(make_model(model_kind), X, y, weights)
    return CalibratedModel(
        model=model, calibrator=calibrator, feature_names=list(X.columns)
    )


def _fold_weights(
    sample_weighting: str,
    index: pd.DatetimeIndex,
    exit_time: pd.Series | None,
) -> np.ndarray | None:
    if sample_weighting == "none":
        return None
    if sample_weighting != "uniqueness":
        raise ValueError(f"Unknown sample_weighting: {sample_weighting!r}")
    return uniqueness_weights(index, exit_time.loc[index]).to_numpy()


def _fit_fold_calibrator(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: np.ndarray | None,
    gap: int,
    model_kind: str,
    calibration: str,
    *,
    exit_time: pd.Series | None = None,
    purge_by_exit_time: bool = False,
):
    """Fit the calibrator on a gapped tail of the training block only."""
    if calibration == "none":
        return IdentityCalibrator()

    split = calibration_split(len(X_train), gap)
    if split is None:
        return IdentityCalibrator()
    fit_rows, cal_rows = split
    if purge_by_exit_time:
        if exit_time is None:
            raise ValueError("Calibration purging requires exit_time")
        calibration_start = X_train.index[cal_rows[0]]
        fit_exit = pd.to_datetime(
            exit_time.loc[X_train.index[fit_rows]],
            utc=True,
        )
        fit_rows = fit_rows[(fit_exit < calibration_start).to_numpy()]
        if not len(fit_rows):
            return IdentityCalibrator()

    inner = _fit(
        make_model(model_kind),
        X_train.iloc[fit_rows],
        y_train.iloc[fit_rows],
        None if weights is None else weights[fit_rows],
    )
    raw = inner.predict_proba(X_train.iloc[cal_rows])[:, 1]
    return fit_calibrator(calibration, raw, y_train.iloc[cal_rows].to_numpy())


def _fit(model, X: pd.DataFrame, y: pd.Series, weights: np.ndarray | None):
    """Fit routing sample weights through a pipeline to final step

    Bare estimator takes sample_weight; pipeline needs it addressed to
    the step that consumes it (clf__sample_weight)
    """
    if weights is None:
        model.fit(X, y)
        return model
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0]
        model.fit(X, y, **{f"{final_step}__sample_weight": weights})
    else:
        model.fit(X, y, sample_weight=weights)
    return model


def _effective_n(weights: np.ndarray | None) -> float | None:
    """Kish effective sample size: (sum w)^2 / sum(w^2).

    With uniform weights this equals n. The gap between it and n_train is the
    honest measure of how much the overlapping labels were inflating the
    apparent sample.
    """
    if weights is None:
        return None
    return float(weights.sum() ** 2 / np.square(weights).sum())
