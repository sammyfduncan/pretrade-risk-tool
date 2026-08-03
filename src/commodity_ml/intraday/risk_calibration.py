"""Causal historical calibration for pre-trade range and excursion scenarios.

The module measures risk paths for both hypothetical sides at every eligible
timestamp. It contains no strategy signal, directional label, or trade P&L.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from commodity_ml.intraday.execution import EntryBlackout


BAR_DURATION = pd.Timedelta(minutes=30)
DEVELOPMENT_START = pd.Timestamp("2020-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-07-01", tz="UTC")
HORIZONS_MINUTES = (60, 120, 180, 240)
QUANTILES = (0.50, 0.80, 0.95, 0.99)
ROLLING_WINDOW = 252
MINIMUM_HISTORY = 126
LONDON = ZoneInfo("Europe/London")
OUTCOME_COLUMNS = (
    "range_bps",
    "long_adverse_bps",
    "short_adverse_bps",
    "spread_points",
    "spread_bps",
)
INSTRUMENT_TO_PREFIX = {"XAUUSD": "gold", "XAGUSD": "silver"}


@dataclass(frozen=True)
class CalibrationOutcomes:
    frame: pd.DataFrame
    exclusions: dict[str, int]
    eligible_rows: int


@dataclass(frozen=True)
class CalibrationResult:
    oof: pd.DataFrame
    profile: pd.DataFrame
    diagnostics: dict[str, Any]
    passed_data_quality: bool


@dataclass(frozen=True)
class CalibratedRiskLookup:
    instrument: str
    side: str
    horizon_minutes: int
    london_slot_minute: int
    profile_as_of: pd.Timestamp
    sample_count: int
    range_points: dict[str, float]
    adverse_points: dict[str, float]
    reference_spread_points: dict[str, float]
    broker_spread_eligible: bool = False
    volatility_sizing_approved: bool = False


def build_calibration_outcomes(
    synchronized_quotes: pd.DataFrame,
    *,
    instrument: str,
    horizon_minutes: int,
    development_start: pd.Timestamp = DEVELOPMENT_START,
    development_end: pd.Timestamp = DEVELOPMENT_END,
    entry_blackouts: Sequence[EntryBlackout] = (),
) -> CalibrationOutcomes:
    """Build exact future-path risk outcomes for one instrument and horizon."""

    if instrument not in INSTRUMENT_TO_PREFIX:
        raise ValueError(f"Unsupported calibration instrument: {instrument}")
    if horizon_minutes not in HORIZONS_MINUTES:
        raise ValueError(f"Unsupported calibration horizon: {horizon_minutes}")
    start = _utc(development_start, "development_start")
    end = _utc(development_end, "development_end")
    if start >= end:
        raise ValueError("development_start must precede development_end")
    frame = _validated_quote_frame(synchronized_quotes, instrument)
    if len(frame) == 0:
        raise ValueError("Calibration source is empty")
    if frame.index.min() < start or frame.index.max() >= end:
        raise ValueError(
            "calibration_source_outside_development: slice the source to the "
            "registered half-open development range before calibration"
        )

    bars = horizon_minutes // 30
    mid_open = (frame["bid_open"] + frame["ask_open"]) / 2
    mid_high = (frame["bid_high"] + frame["ask_high"]) / 2
    mid_low = (frame["bid_low"] + frame["ask_low"]) / 2
    future_high = mid_high.rolling(bars, min_periods=bars).max().shift(-(bars - 1))
    future_low = mid_low.rolling(bars, min_periods=bars).min().shift(-(bars - 1))

    endpoint = frame.index.to_series().shift(-(bars - 1))
    contiguous = endpoint.sub(frame.index) == (bars - 1) * BAR_DURATION
    local = frame.index.tz_convert(LONDON)
    slot = local.hour * 60 + local.minute
    in_entry_session = (slot >= 7 * 60) & (slot < 19 * 60 + 30)
    path_close = local + pd.Timedelta(minutes=horizon_minutes)
    cutoff = local.normalize() + pd.Timedelta(hours=21, minutes=30)
    before_cutoff = path_close <= cutoff
    outside_blackout = _outside_blackouts(frame.index, entry_blackouts)

    scale = 10_000 / mid_open
    outcomes = pd.DataFrame(
        {
            "entry_price": mid_open,
            "range_bps": (future_high - future_low) * scale,
            "long_adverse_bps": (mid_open - future_low).clip(lower=0) * scale,
            "short_adverse_bps": (future_high - mid_open).clip(lower=0) * scale,
            "spread_points": frame["ask_open"] - frame["bid_open"],
            "spread_bps": (frame["ask_open"] - frame["bid_open"]) * scale,
            "london_slot_minute": slot,
        },
        index=frame.index,
    )
    outcomes.index.name = "entry_time"
    finite = np.isfinite(outcomes.to_numpy(dtype=float)).all(axis=1)
    eligible = in_entry_session & before_cutoff & outside_blackout & contiguous & finite
    selected = outcomes.loc[eligible].copy()
    selected.insert(0, "instrument", instrument)
    selected.insert(1, "horizon_minutes", horizon_minutes)
    if (selected[list(OUTCOME_COLUMNS)] < 0).any().any():
        raise ValueError("Calibration outcomes must be non-negative")

    prior = in_entry_session
    exclusions = {
        "source_rows": len(frame),
        "outside_entry_session": int((~in_entry_session).sum()),
        "path_crosses_cutoff": int((prior & ~before_cutoff).sum()),
        "entry_blackout": int((prior & before_cutoff & ~outside_blackout).sum()),
        "non_contiguous_path": int(
            (prior & before_cutoff & outside_blackout & ~contiguous).sum()
        ),
        "invalid_outcome": int(
            (prior & before_cutoff & outside_blackout & contiguous & ~finite).sum()
        ),
        "eligible": len(selected),
    }
    return CalibrationOutcomes(
        frame=selected,
        exclusions=exclusions,
        eligible_rows=len(selected),
    )


def calibrate_outcomes(
    outcomes: pd.DataFrame,
    *,
    rolling_window: int = ROLLING_WINDOW,
    minimum_history: int = MINIMUM_HISTORY,
    quantiles: Sequence[float] = QUANTILES,
) -> CalibrationResult:
    """Create strictly-prior same-clock OOF estimates and a final profile."""

    _validate_calibration_parameters(rolling_window, minimum_history, quantiles)
    required = {
        "instrument",
        "horizon_minutes",
        "london_slot_minute",
        "entry_price",
        *OUTCOME_COLUMNS,
    }
    missing = sorted(required - set(outcomes.columns))
    if missing:
        raise ValueError(f"Calibration outcomes missing columns: {missing}")
    if not isinstance(outcomes.index, pd.DatetimeIndex) or outcomes.index.tz is None:
        raise ValueError("Calibration outcomes require a timezone-aware DatetimeIndex")

    frame = outcomes.sort_index(kind="mergesort").sort_values(
        ["instrument", "horizon_minutes", "london_slot_minute"],
        kind="mergesort",
    ).copy()
    group_columns = ["instrument", "horizon_minutes", "london_slot_minute"]
    groups = frame.groupby(group_columns, sort=False, observed=True)
    frame["prior_history_count"] = groups.cumcount()
    for metric in OUTCOME_COLUMNS:
        for quantile in quantiles:
            name = _quantile_column(metric, quantile)
            frame[name] = groups[metric].transform(
                lambda values, q=quantile: values.shift(1)
                .rolling(rolling_window, min_periods=minimum_history)
                .quantile(q, interpolation="higher")
            )

    estimate_columns = [
        _quantile_column(metric, quantile)
        for metric in OUTCOME_COLUMNS
        for quantile in quantiles
    ]
    calibrated = frame.loc[frame[estimate_columns].notna().all(axis=1)].copy()
    calibrated.sort_index(inplace=True)
    profile = _final_profile(
        frame,
        group_columns=group_columns,
        rolling_window=rolling_window,
        minimum_history=minimum_history,
        quantiles=quantiles,
    )
    diagnostics = _coverage_diagnostics(frame, calibrated, quantiles)
    diagnostics["eligible_rows"] = len(frame)
    diagnostics["calibrated_rows"] = len(calibrated)
    diagnostics["calibrated_fraction"] = len(calibrated) / len(frame) if len(frame) else 0
    diagnostics["profile_rows"] = len(profile)
    diagnostics["profile_min_sample_count"] = (
        int(profile["sample_count"].min()) if len(profile) else 0
    )
    passed = bool(
        len(calibrated) >= 1_000
        and diagnostics["calibrated_fraction"] >= 0.70
        and len(profile) > 0
        and diagnostics["profile_min_sample_count"] >= minimum_history
    )
    return CalibrationResult(
        oof=calibrated,
        profile=profile,
        diagnostics=diagnostics,
        passed_data_quality=passed,
    )


def lookup_calibrated_risk(
    profile: pd.DataFrame,
    *,
    instrument: str,
    decision_time: pd.Timestamp,
    horizon_minutes: int,
    entry_price: float,
    side: str,
    max_age_days: int = 90,
) -> CalibratedRiskLookup:
    """Convert one exact profile row from basis points to current price points."""

    if instrument not in INSTRUMENT_TO_PREFIX:
        raise ValueError(f"Unsupported calibration instrument: {instrument}")
    if horizon_minutes not in HORIZONS_MINUTES:
        raise ValueError(f"Unsupported calibration horizon: {horizon_minutes}")
    if side not in ("long", "short"):
        raise ValueError("side must be long or short")
    if isinstance(max_age_days, bool) or not isinstance(max_age_days, int) or max_age_days < 1:
        raise ValueError("max_age_days must be a positive integer")
    if not np.isfinite(entry_price) or entry_price <= 0:
        raise ValueError("entry_price must be finite and positive")
    now = _utc(decision_time, "decision_time")
    local = now.tz_convert(LONDON)
    slot = local.hour * 60 + local.minute
    matched = profile.loc[
        (profile["instrument"] == instrument)
        & (profile["horizon_minutes"] == horizon_minutes)
        & (profile["london_slot_minute"] == slot)
    ]
    if len(matched) != 1:
        raise ValueError("No unique calibration profile for instrument/horizon/clock")
    row = matched.iloc[0]
    sample_count = int(row["sample_count"])
    if sample_count < MINIMUM_HISTORY:
        raise ValueError("Calibration profile has insufficient history")
    profile_as_of = _utc(pd.Timestamp(row["sample_end"]), "profile.sample_end")
    if profile_as_of >= now:
        raise ValueError("Calibration profile must be strictly prior to decision_time")
    if now - profile_as_of > pd.Timedelta(days=max_age_days):
        raise ValueError("Calibration profile is stale")

    adverse_metric = f"{side}_adverse_bps"
    range_points = {
        _quantile_label(q): float(row[_quantile_column("range_bps", q)])
        * entry_price
        / 10_000
        for q in QUANTILES
    }
    adverse_points = {
        _quantile_label(q): float(row[_quantile_column(adverse_metric, q)])
        * entry_price
        / 10_000
        for q in QUANTILES
    }
    reference_spread = {
        _quantile_label(q): float(row[_quantile_column("spread_points", q)])
        for q in QUANTILES
    }
    return CalibratedRiskLookup(
        instrument=instrument,
        side=side,
        horizon_minutes=horizon_minutes,
        london_slot_minute=slot,
        profile_as_of=profile_as_of,
        sample_count=sample_count,
        range_points=range_points,
        adverse_points=adverse_points,
        reference_spread_points=reference_spread,
    )


def volatility_mapping_diagnostics(
    calibration_oof: pd.DataFrame,
    volatility_oof: pd.DataFrame,
) -> dict[str, Any]:
    """Measure the retained gold 2h forecast's OOF mapping to path risk."""

    subset = calibration_oof.loc[
        (calibration_oof["instrument"] == "XAUUSD")
        & (calibration_oof["horizon_minutes"] == 120)
    ].copy()
    required = {"forecast_log_rv_2h", "observed_log_rv_2h"}
    if not required.issubset(volatility_oof.columns):
        raise ValueError("Volatility OOF artifact has an unexpected schema")
    matched = subset.join(volatility_oof.loc[:, sorted(required)], how="inner")
    if len(matched) == 0:
        raise ValueError("No exact timestamp matches for volatility mapping")
    rv_scale_bps = np.exp(matched["forecast_log_rv_2h"].astype(float)) * 10_000
    report: dict[str, Any] = {
        "matched_rows": len(matched),
        "first": matched.index.min().isoformat(),
        "last": matched.index.max().isoformat(),
        "risk_information_only": True,
        "volatility_sizing_approved": False,
    }
    for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps"):
        actual = matched[metric].astype(float)
        ratio = actual / rv_scale_bps
        finite = ratio[np.isfinite(ratio)]
        report[metric] = {
            "forecast_log_correlation": _correlation(
                matched["forecast_log_rv_2h"].to_numpy(dtype=float),
                np.log(actual.clip(lower=1e-12)).to_numpy(dtype=float),
            ),
            "scale_ratio_p50": float(finite.quantile(0.50, interpolation="higher")),
            "scale_ratio_p80": float(finite.quantile(0.80, interpolation="higher")),
            "scale_ratio_p95": float(finite.quantile(0.95, interpolation="higher")),
        }
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_quote_frame(source: pd.DataFrame, instrument: str) -> pd.DataFrame:
    if not isinstance(source.index, pd.DatetimeIndex) or source.index.tz is None:
        raise ValueError("Calibration source requires a timezone-aware DatetimeIndex")
    if not source.index.is_monotonic_increasing or not source.index.is_unique:
        raise ValueError("Calibration timestamps must be sorted and unique")
    index = source.index.tz_convert("UTC")
    if (
        (index.second != 0).any()
        or (index.microsecond != 0).any()
        or (~np.isin(index.minute, (0, 30))).any()
    ):
        raise ValueError("Calibration timestamps must align to the M30 grid")
    prefix = INSTRUMENT_TO_PREFIX[instrument]
    names = [
        f"{prefix}_{side}_{field}"
        for side in ("bid", "ask")
        for field in ("open", "high", "low", "close")
    ]
    missing = sorted(set(names) - set(source.columns))
    if missing:
        raise ValueError(f"Calibration source missing quote columns: {missing}")
    frame = source.loc[:, names].copy()
    frame.columns = [name.removeprefix(f"{prefix}_") for name in names]
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Calibration quotes must be finite and positive")
    for field in ("open", "high", "low", "close"):
        if (frame[f"ask_{field}"] < frame[f"bid_{field}"]).any():
            raise ValueError(f"Calibration source contains crossed {field} quotes")
    for side in ("bid", "ask"):
        if (frame[f"{side}_high"] < frame[[f"{side}_open", f"{side}_close"]].max(axis=1)).any():
            raise ValueError(f"Calibration source has invalid {side} highs")
        if (frame[f"{side}_low"] > frame[[f"{side}_open", f"{side}_close"]].min(axis=1)).any():
            raise ValueError(f"Calibration source has invalid {side} lows")
    frame.index = index
    frame.index.name = "entry_time"
    return frame


def _outside_blackouts(
    index: pd.DatetimeIndex,
    blackouts: Sequence[EntryBlackout],
) -> np.ndarray:
    allowed = np.ones(len(index), dtype=bool)
    for blackout in blackouts:
        start = _utc(blackout.start, "blackout.start")
        end = _utc(blackout.end, "blackout.end")
        allowed &= ~((index >= start) & (index < end))
    return allowed


def _final_profile(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    rolling_window: int,
    minimum_history: int,
    quantiles: Sequence[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, sort=True, observed=True):
        recent = group.sort_index().tail(rolling_window)
        if len(recent) < minimum_history:
            continue
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "sample_count": len(recent),
                "sample_start": recent.index.min(),
                "sample_end": recent.index.max(),
            }
        )
        for metric in OUTCOME_COLUMNS:
            for quantile in quantiles:
                row[_quantile_column(metric, quantile)] = float(
                    recent[metric].quantile(quantile, interpolation="higher")
                )
        rows.append(row)
    profile = pd.DataFrame(rows)
    if len(profile):
        profile.sort_values(group_columns, inplace=True, ignore_index=True)
    return profile


def _coverage_diagnostics(
    eligible: pd.DataFrame,
    calibrated: pd.DataFrame,
    quantiles: Sequence[float],
) -> dict[str, Any]:
    coverage: dict[str, Any] = {}
    for metric in OUTCOME_COLUMNS:
        metric_report: dict[str, Any] = {}
        actual = calibrated[metric].astype(float)
        for quantile in quantiles:
            estimate = calibrated[_quantile_column(metric, quantile)].astype(float)
            nonzero = actual > 0
            ratio = estimate.loc[nonzero] / actual.loc[nonzero]
            metric_report[_quantile_label(quantile)] = {
                "nominal": quantile,
                "actual_coverage": float((actual <= estimate).mean()),
                "mean_forecast_to_outcome": float(ratio.mean()) if len(ratio) else None,
                "median_forecast_to_outcome": float(ratio.median()) if len(ratio) else None,
            }
        coverage[metric] = metric_report
    return {"coverage": coverage}


def _validate_calibration_parameters(
    rolling_window: int,
    minimum_history: int,
    quantiles: Sequence[float],
) -> None:
    for name, value in (
        ("rolling_window", rolling_window),
        ("minimum_history", minimum_history),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if minimum_history > rolling_window:
        raise ValueError("minimum_history cannot exceed rolling_window")
    if not quantiles or any(not 0 < float(value) < 1 for value in quantiles):
        raise ValueError("quantiles must lie strictly between zero and one")
    if tuple(sorted(set(quantiles))) != tuple(quantiles):
        raise ValueError("quantiles must be unique and increasing")


def _quantile_column(metric: str, quantile: float) -> str:
    return f"{metric}_{_quantile_label(quantile)}"


def _quantile_label(quantile: float) -> str:
    return f"p{int(round(100 * quantile)):02d}"


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def _utc(value: pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
