"""Fail-closed quote freshness, completeness, and spread diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_ml.data.schema import QUOTE_COLUMNS
from commodity_ml.intraday.cost_measurement import quote_round_trip_costs


@dataclass(frozen=True)
class SpreadAssessment:
    allowed: bool
    current_cost_r: float
    median_cost_r: float
    p95_cost_r: float
    observations: int
    reasons: tuple[str, ...]


def closed_quote_window(
    candles: pd.DataFrame,
    *,
    bar_duration: pd.Timedelta,
    now: pd.Timestamp,
    min_bars: int,
    max_staleness: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Return only complete, contiguous quote bars available at ``now``."""

    if bar_duration <= pd.Timedelta(0):
        raise ValueError("bar_duration must be positive")
    if isinstance(min_bars, bool) or not isinstance(min_bars, int) or min_bars < 1:
        raise ValueError("min_bars must be a positive integer")
    missing = [column for column in QUOTE_COLUMNS if column not in candles.columns]
    if missing:
        raise RuntimeError(f"quote_history_incomplete:{missing}")
    if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.tz is None:
        raise ValueError("candles require a timezone-aware DatetimeIndex")
    if not candles.index.is_monotonic_increasing or not candles.index.is_unique:
        raise ValueError("candles must be sorted and unique")

    decision_time = _utc(now)
    frame = candles.copy()
    frame.index = frame.index.tz_convert("UTC")
    closed = frame.loc[frame.index + bar_duration <= decision_time]
    if len(closed) < min_bars:
        raise RuntimeError(
            f"insufficient_closed_quote_bars:{len(closed)}<{min_bars}"
        )
    tail = closed.iloc[-min_bars:]
    expected = pd.date_range(
        tail.index[0],
        periods=min_bars,
        freq=bar_duration,
        tz="UTC",
    )
    if not tail.index.equals(expected):
        raise RuntimeError("non_contiguous_quote_history")

    age = decision_time - (tail.index[-1] + bar_duration)
    allowed_age = max_staleness if max_staleness is not None else bar_duration
    if age < pd.Timedelta(0):
        raise RuntimeError("latest_closed_bar_is_in_future")
    if age > allowed_age:
        raise RuntimeError(f"stale_closed_quote_bar:{age}")
    return tail


def assess_spread_history(
    candles: pd.DataFrame,
    *,
    stop_distance_pct: float,
    slippage_pct_per_side: float = 0.0,
    median_limit_r: float = 0.10,
    p95_limit_r: float = 0.25,
) -> SpreadAssessment:
    """Apply the frozen broker admission gate to observed round-trip friction."""

    for name, value in (
        ("stop_distance_pct", stop_distance_pct),
        ("median_limit_r", median_limit_r),
        ("p95_limit_r", p95_limit_r),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(slippage_pct_per_side) or slippage_pct_per_side < 0:
        raise ValueError("slippage_pct_per_side must be finite and non-negative")
    missing = [column for column in QUOTE_COLUMNS if column not in candles.columns]
    if missing:
        raise RuntimeError(f"quote_history_incomplete:{missing}")
    if candles.empty:
        raise RuntimeError("empty_quote_history")

    round_trip_cost = quote_round_trip_costs(
        candles,
        slippage_pct_per_side=slippage_pct_per_side,
    )
    cost_r = round_trip_cost / stop_distance_pct
    current = float(cost_r.iloc[-1])
    median = float(cost_r.median())
    p95 = float(cost_r.quantile(0.95))
    reasons: list[str] = []
    if median > median_limit_r:
        reasons.append("median_cost_above_0.10R")
    if p95 > p95_limit_r:
        reasons.append("p95_cost_above_0.25R")
    if current > p95_limit_r:
        reasons.append("current_cost_above_0.25R")

    return SpreadAssessment(
        allowed=not reasons,
        current_cost_r=current,
        median_cost_r=median,
        p95_cost_r=p95,
        observations=len(cost_r),
        reasons=tuple(reasons),
    )


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return timestamp.tz_convert("UTC")
