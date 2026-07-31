"""Causal H1-context/M30-decision dataset construction.

Candle timestamps are bar-open times. H1 values become available at the H1
close and are joined to an M30 signal only when that close is no later than the
M30 decision time. This is the structural guard against using a half-formed H1
bar in an intraday decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_ml.data.schema import QUOTE_COLUMNS, validate_candles
from commodity_ml.features.technical import atr

LONDON = "Europe/London"
H1_DURATION = pd.Timedelta(hours=1)
M30_DURATION = pd.Timedelta(minutes=30)
M15_DURATION = pd.Timedelta(minutes=15)


def build_intraday_frame(
    h1_candles: pd.DataFrame,
    m30_candles: pd.DataFrame,
    *,
    source: str,
    m15_candles: pd.DataFrame | None = None,
    max_h1_age: pd.Timedelta = pd.Timedelta(hours=2),
    max_m15_age: pd.Timedelta = pd.Timedelta(minutes=30),
) -> pd.DataFrame:
    """Build the shared causal frame used by rules, labels, and later ML."""

    h1 = validate_candles(h1_candles, f"{source}:H1")
    m30 = validate_candles(m30_candles, f"{source}:M30")
    _require_grid(h1.index, H1_DURATION, "H1")
    _require_grid(m30.index, M30_DURATION, "M30")
    if max_h1_age < H1_DURATION:
        raise ValueError("max_h1_age must be at least one H1 bar")

    h1_features = _h1_features(h1)
    m30_features = _m30_features(m30)
    left = m30_features.reset_index().rename(columns={"time": "signal_time"})
    left["decision_time"] = left["signal_time"] + M30_DURATION
    right = h1_features.reset_index().rename(columns={"time": "h1_source_time"})
    right["h1_available_time"] = right["h1_source_time"] + H1_DURATION

    merged = pd.merge_asof(
        left.sort_values("decision_time"),
        right.sort_values("h1_available_time"),
        left_on="decision_time",
        right_on="h1_available_time",
        direction="backward",
        tolerance=max_h1_age,
        allow_exact_matches=True,
    )
    merged = merged.set_index("signal_time").sort_index()
    merged.index = pd.DatetimeIndex(merged.index, name="time")
    merged["h1_context_age_minutes"] = (
        merged["decision_time"] - merged["h1_available_time"]
    ) / pd.Timedelta(minutes=1)

    invalid_context = (
        merged["h1_available_time"].notna()
        & (merged["h1_available_time"] > merged["decision_time"])
    )
    if invalid_context.any():  # pragma: no cover - merge_asof contract guard
        raise AssertionError("H1 context was joined before it became available")

    session = _session_features(m30, merged["decision_time"])
    for column in session:
        merged[column] = session[column]
    if m15_candles is not None:
        merged = attach_m15_context(
            merged,
            m15_candles,
            source=source,
            max_m15_age=max_m15_age,
        )
    return merged


def attach_m15_context(
    frame: pd.DataFrame,
    m15_candles: pd.DataFrame,
    *,
    source: str,
    max_m15_age: pd.Timedelta = pd.Timedelta(minutes=30),
) -> pd.DataFrame:
    """Attach optional completed-M15 execution context to M30 decisions.

    These columns are diagnostic challenger inputs only. The frozen P1/P2/P3
    rules and initial candidate-model feature lists do not consume them.
    """

    if "decision_time" not in frame:
        raise ValueError("M30 intraday frame is missing decision_time")
    if max_m15_age < M15_DURATION:
        raise ValueError("max_m15_age must be at least one M15 bar")
    m15 = validate_candles(m15_candles, f"{source}:M15")
    _require_grid(m15.index, M15_DURATION, "M15")
    right = _m15_features(m15).reset_index().rename(
        columns={"time": "m15_source_time"}
    )
    right["m15_available_time"] = right["m15_source_time"] + M15_DURATION
    left = frame.reset_index().rename(columns={"time": "signal_time"})
    merged = pd.merge_asof(
        left.sort_values("decision_time"),
        right.sort_values("m15_available_time"),
        left_on="decision_time",
        right_on="m15_available_time",
        direction="backward",
        tolerance=max_m15_age,
        allow_exact_matches=True,
    )
    merged = merged.set_index("signal_time").sort_index()
    merged.index = pd.DatetimeIndex(merged.index, name="time")
    merged["m15_context_age_minutes"] = (
        merged["decision_time"] - merged["m15_available_time"]
    ) / pd.Timedelta(minutes=1)
    invalid = (
        merged["m15_available_time"].notna()
        & (merged["m15_available_time"] > merged["decision_time"])
    )
    if invalid.any():  # pragma: no cover - merge_asof contract guard
        raise AssertionError("M15 context was joined before it became available")
    return merged


def _h1_features(candles: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=candles.index)
    ema20 = candles["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = candles["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    atr14 = atr(candles, length=14)
    out["h1_close"] = candles["close"]
    out["h1_ema20"] = ema20
    out["h1_ema50"] = ema50
    out["h1_ema50_lag3"] = ema50.shift(3)
    out["h1_atr14"] = atr14
    out["h1_ema20_dist"] = candles["close"] / ema20 - 1
    out["h1_ema50_dist"] = candles["close"] / ema50 - 1
    out["h1_ema50_slope3_atr"] = (ema50 - ema50.shift(3)) / atr14
    out["h1_atr_pct"] = atr14 / candles["close"]
    return out


def _m30_features(candles: pd.DataFrame) -> pd.DataFrame:
    out = candles.copy()
    ema20 = candles["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    atr14 = atr(candles, length=14)
    rolling = candles["close"].rolling(20)
    mean = rolling.mean()
    std = rolling.std(ddof=0)
    lower, upper = mean - 2 * std, mean + 2 * std
    width = upper - lower

    out["m30_ema20"] = ema20
    out["m30_atr14"] = atr14
    out["m30_bb_lower"] = lower
    out["m30_bb_upper"] = upper
    out["m30_ema20_dist"] = candles["close"] / ema20 - 1
    out["m30_atr_pct"] = atr14 / candles["close"]
    out["m30_bb_position"] = (candles["close"] - lower) / width
    out["m30_bb_width_pct"] = width / candles["close"]
    out["m30_body_pct"] = (candles["close"] - candles["open"]) / candles["close"]
    out["m30_range_pct"] = (candles["high"] - candles["low"]) / candles["close"]
    return out


def _m15_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Small execution-state set kept outside the initial M30 model contract."""

    out = pd.DataFrame(index=candles.index)
    ema20 = candles["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    atr14 = atr(candles, length=14)
    returns = candles["close"].pct_change()
    out["m15_close"] = candles["close"]
    out["m15_ema20_dist"] = candles["close"] / ema20 - 1
    out["m15_atr_pct"] = atr14 / candles["close"]
    out["m15_body_pct"] = (
        candles["close"] - candles["open"]
    ) / candles["close"]
    out["m15_range_pct"] = (
        candles["high"] - candles["low"]
    ) / candles["close"]
    out["m15_return1"] = returns
    out["m15_return4"] = candles["close"].pct_change(4)
    out["m15_realized_vol4"] = returns.rolling(4).std(ddof=0)
    return out


def _session_features(
    m30: pd.DataFrame,
    decision_time: pd.Series,
) -> pd.DataFrame:
    local_open = m30.index.tz_convert(LONDON)
    local_decision = pd.DatetimeIndex(decision_time).tz_convert(LONDON)
    session_date = pd.Index(local_open.date, name="london_session_date")
    minute_open = local_open.hour * 60 + local_open.minute
    minute_decision = local_decision.hour * 60 + local_decision.minute

    result = pd.DataFrame(index=m30.index)
    result["london_session_ordinal"] = pd.factorize(session_date)[0]
    result["london_open_minute"] = minute_open
    result["minutes_since_entry_start"] = minute_decision - 7 * 60
    result["minutes_until_last_entry"] = 19 * 60 + 30 - minute_decision

    session_mask = (minute_open >= 7 * 60) & (minute_open < 21 * 60 + 30)
    session_bars = m30.loc[session_mask].copy()
    session_bars["_session_date"] = np.asarray(session_date)[session_mask]
    aggregates = session_bars.groupby("_session_date").agg(
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_open=("open", "first"),
        session_close=("close", "last"),
        session_count=("close", "size"),
    )
    aggregates["previous_session_range"] = (
        aggregates["session_high"] - aggregates["session_low"]
    ).shift(1)
    aggregates["previous_session_return"] = (
        aggregates["session_close"] / aggregates["session_open"] - 1
    ).shift(1)
    date_values = pd.Series(np.asarray(session_date), index=m30.index)
    result["previous_session_range"] = date_values.map(
        aggregates["previous_session_range"]
    )
    result["previous_session_return"] = date_values.map(
        aggregates["previous_session_return"]
    )

    opening_mask = (minute_open >= 7 * 60) & (minute_open < 9 * 60)
    opening = m30.loc[opening_mask].copy()
    opening["_session_date"] = np.asarray(session_date)[opening_mask]
    opening_stats = opening.groupby("_session_date").agg(
        opening_range_high=("high", "max"),
        opening_range_low=("low", "min"),
        opening_bar_count=("close", "size"),
    )
    opening_stats["opening_range"] = (
        opening_stats["opening_range_high"] - opening_stats["opening_range_low"]
    )
    valid_range = opening_stats["opening_range"].where(
        opening_stats["opening_bar_count"] == 4
    )
    valid_medians = valid_range.dropna().shift(1).rolling(
        20,
        min_periods=20,
    ).median()
    opening_stats["opening_range_median20"] = valid_medians.reindex(
        opening_stats.index
    )
    opening_stats["opening_range_compression"] = (
        valid_range / opening_stats["opening_range_median20"]
    )

    opening_known = minute_decision >= 9 * 60
    for column in (
        "opening_range_high",
        "opening_range_low",
        "opening_range",
        "opening_range_median20",
        "opening_range_compression",
    ):
        values = date_values.map(opening_stats[column])
        result[column] = values.where(opening_known)

    # Normalize the prior session by volatility known through this M30 close.
    atr14 = atr(m30, length=14)
    result["previous_session_range_atr"] = result["previous_session_range"] / atr14
    return result


def _require_grid(
    index: pd.DatetimeIndex,
    duration: pd.Timedelta,
    label: str,
) -> None:
    if index.empty:
        raise ValueError(f"{label} candles are empty")
    midnight = index.normalize()
    offsets = index - midnight
    if ((offsets % duration) != pd.Timedelta(0)).any():
        raise ValueError(f"{label} candle timestamps are not aligned to {duration}")
