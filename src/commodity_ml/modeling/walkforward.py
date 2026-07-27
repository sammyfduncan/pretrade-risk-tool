"""Walk-forward training harness.

Uses sklearn's TimeSeriesSplit (never hand-rolled, never shuffled): each fold
trains on an expanding window of the past and tests on the block immediately
after it, so every test prediction comes from a model that has never seen
that bar or anything after it.

The `gap` parameter is load-bearing: labels look `horizon` bars into the
future, so the last `horizon` training bars share future price action with
the first test bars. Skipping `gap >= horizon` bars between train and test
removes that overlap. AssetConfig enforces gap >= horizon at load time.

The concatenated per-fold test predictions form the *out-of-fold* (OOF)
series — an honest simulation of "what would the model have predicted at the
time" — which is the only thing the backtest is allowed to consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from commodity_ml.modeling.models import make_model


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


@dataclass
class WalkForwardResult:
    oof_proba: pd.Series          # out-of-fold P(label=1), indexed by bar time
    folds: list[FoldResult] = field(default_factory=list)


def walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    gap: int,
    model_kind: str,
) -> WalkForwardResult:
    """Run expanding-window walk-forward training and collect OOF predictions."""
    if not X.index.is_monotonic_increasing:
        raise ValueError("X must be sorted by time before walk-forward splitting.")

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    result = WalkForwardResult(oof_proba=pd.Series(dtype=float, name="proba"))
    oof_chunks: list[pd.Series] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        model = make_model(model_kind)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof_chunks.append(pd.Series(proba, index=X.index[test_idx], name="proba"))
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
            )
        )

    result.oof_proba = pd.concat(oof_chunks).sort_index()
    return result


def fit_final_model(X: pd.DataFrame, y: pd.Series, model_kind: str):
    """Fit on all available history — the artifact for future/live prediction.

    Its quality estimate comes from the walk-forward folds, never from
    predictions on its own training data.
    """
    model = make_model(model_kind)
    model.fit(X, y)
    return model
