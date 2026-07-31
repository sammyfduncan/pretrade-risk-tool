"""Causal two-hour intraday volatility forecasting.

The registered design lives in ``docs/intraday_risk_design.md``.  This module
contains no trading direction and no strategy P&L: observed future closes form
the audit target, while every feature is available at the M30 decision time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from commodity_ml.data.schema import validate_candles
from commodity_ml.features.vol_layer import build_iv_features
from commodity_ml.intraday.event_blackouts import scheduled_release_times
from commodity_ml.intraday.execution import EntryBlackout, SessionCutoffOverride

BAR_DURATION = pd.Timedelta(minutes=30)
FORECAST_HORIZON_BARS = 4
FORECAST_HORIZON = FORECAST_HORIZON_BARS * BAR_DURATION
SESSION_TIMEZONE = "Europe/London"
SESSION_START = time(7, 0)
LAST_ENTRY = time(19, 30)
FLAT_CUTOFF = time(21, 30)
TRAILING_WINDOWS = (4, 16, 48)
TARGET_COLUMN = "fwd_log_rv_2h"

BASELINE_FEATURES = (
    "log_rv_trailing_4",
    "log_rv_trailing_16",
    "log_rv_trailing_48",
    "london_clock_sin",
    "london_clock_cos",
    "weekday_tue",
    "weekday_wed",
    "weekday_thu",
    "weekday_fri",
)
IV_FEATURES = ("gvz_log", "gvz_chg_5d")
CALENDAR_FEATURES = ("bls_release_next_120m", "fomc_release_next_120m")
ALL_FEATURES = BASELINE_FEATURES + IV_FEATURES + CALENDAR_FEATURES


@dataclass(frozen=True)
class VolatilityDataset:
    frame: pd.DataFrame
    eligible_price_rows: int
    matched_rows: int
    matched_coverage: float
    exclusions: dict[str, int]


@dataclass(frozen=True)
class LinearVolatilityModel:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray

    def predict(self, features: pd.DataFrame) -> pd.Series:
        _require_feature_contract(features, self.feature_names)
        values = features.loc[:, self.feature_names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Volatility forecast features must be finite")
        standardized = (values - self.mean) / self.scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        prediction = design @ self.coefficients
        return pd.Series(prediction, index=features.index, name="forecast_log_rv_2h")


@dataclass(frozen=True)
class VolatilityFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    r2_oos: float


@dataclass(frozen=True)
class WalkForwardVolatilityForecast:
    feature_names: tuple[str, ...]
    oof_prediction: pd.Series
    pooled_r2_oos: float
    folds: tuple[VolatilityFold, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class VolatilityAblation:
    baseline: WalkForwardVolatilityForecast
    augmented: WalkForwardVolatilityForecast
    delta_r2_oos: float
    per_fold_delta: tuple[float, ...]
    folds_improved: int


@dataclass(frozen=True)
class JointVolatilityComparison:
    """Registered J1 comparison against the already-known B0 plus IV model."""

    reference: WalkForwardVolatilityForecast
    joint: WalkForwardVolatilityForecast
    delta_r2_oos: float
    per_fold_delta: tuple[float, ...]
    folds_improved: int


def clean_gvz_observations(gvz: pd.Series) -> pd.Series:
    """Remove FRED non-observation rows without altering observed closes.

    FRED represents US-market holidays as dated NaNs.  They are not GVZ
    observations and therefore do not belong in a five-*observation* change.
    Removing them before any circular shift also keeps the observed and
    placebo availability masks identical.
    """

    if not isinstance(gvz, pd.Series):
        raise ValueError("GVZ must be a pandas Series")
    if not isinstance(gvz.index, pd.DatetimeIndex):
        raise ValueError("GVZ requires a DatetimeIndex")
    if not gvz.index.is_unique:
        raise ValueError("GVZ observation dates must be unique")
    clean = gvz.sort_index().dropna().astype(float)
    if clean.empty:
        raise ValueError("GVZ has no observed closes")
    if not np.isfinite(clean.to_numpy()).all() or (clean <= 0).any():
        raise ValueError("Observed GVZ closes must be finite and positive")
    clean.name = "gvz"
    return clean


def build_volatility_dataset(
    candles: pd.DataFrame,
    gvz: pd.Series,
    *,
    development_start: pd.Timestamp = pd.Timestamp("2021-01-01", tz="UTC"),
    development_end: pd.Timestamp = pd.Timestamp("2025-07-01", tz="UTC"),
    entry_blackouts: Sequence[EntryBlackout] = (),
    cutoff_overrides: Sequence[SessionCutoffOverride] = (),
    releases: pd.DataFrame | None = None,
) -> VolatilityDataset:
    """Build the one matched-row frame used by every registered ablation.

    ``development_end`` is exclusive.  Supplying any candle at or beyond it is
    refused so callers cannot load confirmation data and merely promise not to
    use it.
    """

    start = _utc(development_start, "development_start")
    end = _utc(development_end, "development_end")
    if start >= end:
        raise ValueError("development_start must precede development_end")

    frame = validate_candles(candles, "intraday-volatility")
    if len(frame) == 0:
        raise ValueError("Volatility dataset requires candles")
    if frame.index.max() >= end:
        raise ValueError(
            "confirmation_candles_loaded: pass a frame ending before the "
            "exclusive development cutoff"
        )

    close = frame["close"].astype(float)
    returns = _valid_close_returns(close)
    features = _baseline_features(returns, frame.index)
    target, target_contiguous = _forward_target(returns, frame.index)
    decision_time = frame.index + BAR_DURATION
    features.index = decision_time
    target.index = decision_time
    target_contiguous.index = decision_time

    gvz_features = build_iv_features(gvz, decision_time)
    release_frame = (
        _validate_release_frame(releases)
        if releases is not None
        else scheduled_release_times(start, end)
    )
    calendar = _calendar_features(decision_time, release_frame)
    combined = pd.concat([features, gvz_features, calendar, target], axis=1)

    in_development = (decision_time >= start) & (decision_time < end)
    in_session, before_cutoff = _session_masks(decision_time, cutoff_overrides)
    not_blackout = _outside_blackouts(decision_time, entry_blackouts)
    price_eligible = in_development & in_session & before_cutoff & not_blackout
    eligible_price_rows = int(price_eligible.sum())

    finite_target = np.isfinite(target.to_numpy(dtype=float))
    finite_baseline = np.isfinite(
        features.loc[:, BASELINE_FEATURES].to_numpy(dtype=float)
    ).all(axis=1)
    finite_iv = np.isfinite(gvz_features.loc[:, IV_FEATURES].to_numpy(dtype=float)).all(
        axis=1
    )
    finite_calendar = np.isfinite(
        calendar.loc[:, CALENDAR_FEATURES].to_numpy(dtype=float)
    ).all(axis=1)
    matched = (
        price_eligible
        & target_contiguous.to_numpy(dtype=bool)
        & finite_target
        & finite_baseline
        & finite_iv
        & finite_calendar
    )
    result = combined.loc[matched, list(ALL_FEATURES) + [TARGET_COLUMN]].copy()
    if not result.index.is_monotonic_increasing or not result.index.is_unique:
        raise ValueError("Volatility dataset index must be sorted and unique")

    matched_rows = len(result)
    coverage = matched_rows / eligible_price_rows if eligible_price_rows else 0.0
    exclusions = {
        "outside_development": int((~in_development).sum()),
        "outside_entry_session": int((in_development & ~in_session).sum()),
        "target_crosses_cutoff": int(
            (in_development & in_session & ~before_cutoff).sum()
        ),
        "entry_blackout": int(
            (in_development & in_session & before_cutoff & ~not_blackout).sum()
        ),
        "non_contiguous_target": int((price_eligible & ~target_contiguous).sum()),
        "invalid_target": int((price_eligible & ~finite_target).sum()),
        "incomplete_baseline": int((price_eligible & ~finite_baseline).sum()),
        "incomplete_gvz": int((price_eligible & ~finite_iv).sum()),
        "incomplete_calendar": int((price_eligible & ~finite_calendar).sum()),
    }
    return VolatilityDataset(
        frame=result,
        eligible_price_rows=eligible_price_rows,
        matched_rows=matched_rows,
        matched_coverage=coverage,
        exclusions=exclusions,
    )


def walk_forward_volatility(
    dataset: pd.DataFrame,
    feature_names: Sequence[str],
    *,
    n_splits: int = 5,
    gap: int = FORECAST_HORIZON_BARS,
) -> WalkForwardVolatilityForecast:
    """Produce honest expanding-window OOF forecasts and train-mean R-squared."""

    names = tuple(feature_names)
    _validate_forecast_frame(dataset, names, gap, n_splits)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    predictions: list[pd.Series] = []
    folds: list[VolatilityFold] = []
    sse_total = 0.0
    sst_total = 0.0

    X = dataset.loc[:, names]
    y = dataset[TARGET_COLUMN]
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X)):
        model = fit_linear_volatility_model(X.iloc[train_idx], y.iloc[train_idx])
        predicted = model.predict(X.iloc[test_idx])
        benchmark = float(y.iloc[train_idx].mean())
        actual = y.iloc[test_idx].to_numpy(dtype=float)
        values = predicted.to_numpy(dtype=float)
        sse = float(np.square(actual - values).sum())
        sst = float(np.square(actual - benchmark).sum())
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        sse_total += sse
        sst_total += sst
        predictions.append(predicted)
        folds.append(
            VolatilityFold(
                fold=fold,
                train_start=X.index[train_idx[0]],
                train_end=X.index[train_idx[-1]],
                test_start=X.index[test_idx[0]],
                test_end=X.index[test_idx[-1]],
                n_train=len(train_idx),
                n_test=len(test_idx),
                r2_oos=float(r2),
            )
        )
    pooled = 1.0 - sse_total / sst_total if sst_total > 0 else np.nan
    return WalkForwardVolatilityForecast(
        feature_names=names,
        oof_prediction=pd.concat(predictions).sort_index(),
        pooled_r2_oos=float(pooled),
        folds=tuple(folds),
    )


def evaluate_volatility_ablation(
    dataset: pd.DataFrame,
    family_features: Sequence[str],
) -> VolatilityAblation:
    """Compare one registered family with B0 on identical rows and folds."""

    family = tuple(family_features)
    if family not in (IV_FEATURES, CALENDAR_FEATURES):
        raise ValueError("Only the registered IV or calendar family may be evaluated")
    baseline = walk_forward_volatility(dataset, BASELINE_FEATURES)
    augmented = walk_forward_volatility(dataset, BASELINE_FEATURES + family)
    per_fold = tuple(
        augmented_fold.r2_oos - baseline_fold.r2_oos
        for baseline_fold, augmented_fold in zip(
            baseline.folds,
            augmented.folds,
            strict=True,
        )
    )
    return VolatilityAblation(
        baseline=baseline,
        augmented=augmented,
        delta_r2_oos=augmented.pooled_r2_oos - baseline.pooled_r2_oos,
        per_fold_delta=per_fold,
        folds_improved=sum(delta > 0 for delta in per_fold),
    )


def evaluate_joint_volatility(dataset: pd.DataFrame) -> JointVolatilityComparison:
    """Compare registered J1 against J0 on identical folds and rows."""

    reference_names = BASELINE_FEATURES + IV_FEATURES
    joint_names = ALL_FEATURES
    reference = walk_forward_volatility(dataset, reference_names)
    joint = walk_forward_volatility(dataset, joint_names)
    if not reference.oof_prediction.index.equals(joint.oof_prediction.index):
        raise RuntimeError("Joint volatility comparison changed OOF rows")
    per_fold = tuple(
        joint_fold.r2_oos - reference_fold.r2_oos
        for reference_fold, joint_fold in zip(
            reference.folds,
            joint.folds,
            strict=True,
        )
    )
    return JointVolatilityComparison(
        reference=reference,
        joint=joint,
        delta_r2_oos=joint.pooled_r2_oos - reference.pooled_r2_oos,
        per_fold_delta=per_fold,
        folds_improved=sum(delta > 0 for delta in per_fold),
    )


def fit_linear_volatility_model(
    features: pd.DataFrame,
    target: pd.Series,
) -> LinearVolatilityModel:
    names = tuple(features.columns)
    _require_feature_contract(features, names)
    values = features.to_numpy(dtype=float)
    y = pd.Series(target).reindex(features.index).to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(y).all():
        raise ValueError("Volatility model inputs must be finite")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (values - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return LinearVolatilityModel(names, mean, scale, coefficients)


def _valid_close_returns(close: pd.Series) -> pd.Series:
    logged = np.log(close.astype(float))
    returns = logged.diff()
    contiguous = close.index.to_series().diff().eq(BAR_DURATION).to_numpy()
    returns.iloc[~contiguous] = np.nan
    return returns.rename("m30_log_return")


def _baseline_features(
    returns: pd.Series,
    candle_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    valid = returns.dropna()
    out = pd.DataFrame(index=candle_index)
    for window in TRAILING_WINDOWS:
        rss = np.sqrt(valid.pow(2).rolling(window, min_periods=window).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            logged = np.log(rss)
        out[f"log_rv_trailing_{window}"] = logged.reindex(candle_index).ffill()

    decision = candle_index + BAR_DURATION
    local = decision.tz_convert(SESSION_TIMEZONE)
    minute = local.hour * 60 + local.minute
    angle = 2 * np.pi * minute / (24 * 60)
    out["london_clock_sin"] = np.sin(angle)
    out["london_clock_cos"] = np.cos(angle)
    weekday = local.dayofweek
    for day, name in ((1, "tue"), (2, "wed"), (3, "thu"), (4, "fri")):
        out[f"weekday_{name}"] = (weekday == day).astype(float)
    return out


def _forward_target(
    returns: pd.Series,
    candle_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series]:
    values = returns.to_numpy(dtype=float)
    n = len(values)
    target = np.full(n, np.nan)
    contiguous = np.zeros(n, dtype=bool)
    for position in range(n - FORECAST_HORIZON_BARS):
        future_index = candle_index[position + 1 : position + 1 + FORECAST_HORIZON_BARS]
        expected = pd.date_range(
            candle_index[position] + BAR_DURATION,
            periods=FORECAST_HORIZON_BARS,
            freq=BAR_DURATION,
            tz="UTC",
        )
        if not future_index.equals(expected):
            continue
        future = values[position + 1 : position + 1 + FORECAST_HORIZON_BARS]
        if not np.isfinite(future).all():
            continue
        realized = float(np.sqrt(np.square(future).sum()))
        contiguous[position] = True
        if realized > 0:
            target[position] = np.log(realized)
    return (
        pd.Series(target, index=candle_index, name=TARGET_COLUMN),
        pd.Series(contiguous, index=candle_index, name="target_contiguous"),
    )


def _calendar_features(
    decision_time: pd.DatetimeIndex,
    releases: pd.DataFrame,
) -> pd.DataFrame:
    times = releases.index
    kinds = releases["kind"].astype(str).to_numpy()
    result = pd.DataFrame(0.0, index=decision_time, columns=CALENDAR_FEATURES)
    values = times.asi8
    for position, decision in enumerate(decision_time):
        left = np.searchsorted(values, decision.value, side="right")
        right = np.searchsorted(
            values,
            (decision + FORECAST_HORIZON).value,
            side="right",
        )
        upcoming = set(kinds[left:right])
        result.iloc[position, 0] = float(bool(upcoming.intersection({"cpi", "nfp"})))
        result.iloc[position, 1] = float("fomc" in upcoming)
    return result


def _session_masks(
    decision_time: pd.DatetimeIndex,
    overrides: Sequence[SessionCutoffOverride],
) -> tuple[np.ndarray, np.ndarray]:
    local = decision_time.tz_convert(SESSION_TIMEZONE)
    clocks = np.array([stamp.time().replace(tzinfo=None) for stamp in local])
    in_session = np.array(
        [SESSION_START <= clock < LAST_ENTRY for clock in clocks],
        dtype=bool,
    )
    override_map = {item.session_date: item.flat_cutoff for item in overrides}
    if len(override_map) != len(overrides):
        raise ValueError("Duplicate cutoff override date")
    before_cutoff = np.zeros(len(local), dtype=bool)
    for position, stamp in enumerate(local):
        cutoff_clock = override_map.get(stamp.date(), FLAT_CUTOFF)
        cutoff = pd.Timestamp.combine(stamp.date(), cutoff_clock).tz_localize(
            SESSION_TIMEZONE
        )
        before_cutoff[position] = stamp + FORECAST_HORIZON <= cutoff
    return in_session, before_cutoff


def _outside_blackouts(
    decision_time: pd.DatetimeIndex,
    blackouts: Sequence[EntryBlackout],
) -> np.ndarray:
    allowed = np.ones(len(decision_time), dtype=bool)
    for blackout in blackouts:
        start = _utc(blackout.start, "blackout.start")
        end = _utc(blackout.end, "blackout.end")
        allowed &= ~((decision_time >= start) & (decision_time < end))
    return allowed


def _validate_release_frame(releases: pd.DataFrame) -> pd.DataFrame:
    if "kind" not in releases.columns:
        raise ValueError("Release frame requires a kind column")
    if not isinstance(releases.index, pd.DatetimeIndex) or releases.index.tz is None:
        raise ValueError("Release frame requires a timezone-aware DatetimeIndex")
    frame = releases[["kind"]].copy().sort_index()
    frame.index = frame.index.tz_convert("UTC")
    if not frame.index.is_unique:
        raise ValueError("Release timestamps must be unique")
    allowed = {"cpi", "nfp", "fomc"}
    unknown = set(frame["kind"].astype(str)) - allowed
    if unknown:
        raise ValueError(f"Unknown scheduled release kinds: {sorted(unknown)}")
    return frame


def _validate_forecast_frame(
    dataset: pd.DataFrame,
    names: tuple[str, ...],
    gap: int,
    n_splits: int,
) -> None:
    if not names or len(names) != len(set(names)):
        raise ValueError("feature_names must be non-empty and unique")
    missing = [name for name in names + (TARGET_COLUMN,) if name not in dataset]
    if missing:
        raise ValueError(f"Volatility dataset missing columns: {missing}")
    if not isinstance(dataset.index, pd.DatetimeIndex) or dataset.index.tz is None:
        raise ValueError("Volatility dataset requires a timezone-aware index")
    if not dataset.index.is_monotonic_increasing or not dataset.index.is_unique:
        raise ValueError("Volatility dataset must be sorted and unique")
    if gap < FORECAST_HORIZON_BARS:
        raise ValueError("gap must cover the four-bar volatility target")
    if n_splits != 5:
        raise ValueError("The registered experiment requires exactly five folds")
    values = dataset.loc[:, names + (TARGET_COLUMN,)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Volatility dataset contains non-finite model values")


def _require_feature_contract(
    features: pd.DataFrame,
    names: tuple[str, ...],
) -> None:
    if tuple(features.columns) != names:
        raise ValueError(
            f"Volatility feature contract mismatch: expected {list(names)}, "
            f"got {list(features.columns)}"
        )


def _utc(value: pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
