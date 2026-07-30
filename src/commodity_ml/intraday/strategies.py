"""The three frozen version-1 intraday primary strategy hypotheses.

These rules choose direction and create complete executable candidates. They do
not use ML. ML may later filter their candidates, but it cannot invent a side
or alter the registered stop, target, or timeout.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from commodity_ml.intraday.data import LONDON, M30_DURATION
from commodity_ml.intraday.execution import TradeCandidate

StrategyName = str


def generate_candidates(
    frame: pd.DataFrame,
    strategy: StrategyName,
) -> list[TradeCandidate]:
    """Generate candidates for one named frozen primary family."""

    generators: dict[str, Callable[[pd.DataFrame], list[TradeCandidate]]] = {
        "P1": opening_range_breakout,
        "P2": trend_pullback,
        "P3": regime_mean_reversion,
    }
    if strategy not in generators:
        raise ValueError(f"Unknown intraday strategy {strategy!r}; expected P1, P2, or P3")
    _validate_frame(frame)
    return generators[strategy](frame)


def generate_all_candidates(
    frame: pd.DataFrame,
) -> dict[str, list[TradeCandidate]]:
    _validate_frame(frame)
    return {
        "P1": opening_range_breakout(frame),
        "P2": trend_pullback(frame),
        "P3": regime_mean_reversion(frame),
    }


def opening_range_breakout(frame: pd.DataFrame) -> list[TradeCandidate]:
    """P1: compressed London opening range followed by H1-aligned expansion."""

    eligible = _entry_eligible(frame)
    local = frame.index.tz_convert(LONDON)
    session_dates = np.asarray(local.date)
    candidates: list[TradeCandidate] = []

    for session_date in pd.unique(session_dates):
        day = frame.loc[session_dates == session_date]
        if day.empty:
            continue
        compression = day["opening_range_compression"].dropna()
        if compression.empty or float(compression.iloc[-1]) > 1.25:
            continue
        opening_high = float(day["opening_range_high"].dropna().iloc[-1])
        opening_low = float(day["opening_range_low"].dropna().iloc[-1])

        day_local = day.index.tz_convert(LONDON)
        after_opening = day_local.hour * 60 + day_local.minute >= 9 * 60
        rows = day.loc[after_opening & eligible.loc[day.index]]
        for signal_time, row in rows.iterrows():
            atr_value = row["m30_atr14"]
            if not np.isfinite(atr_value) or atr_value <= 0:
                continue
            long_context = (
                row["h1_close"] > row["h1_ema50"]
                and row["h1_ema50"] > row["h1_ema50_lag3"]
            )
            short_context = (
                row["h1_close"] < row["h1_ema50"]
                and row["h1_ema50"] < row["h1_ema50_lag3"]
            )
            long_signal = long_context and row["close"] >= opening_high + 0.10 * atr_value
            short_signal = short_context and row["close"] <= opening_low - 0.10 * atr_value
            if long_signal or short_signal:
                candidates.append(
                    _candidate(
                        signal_time,
                        "long" if long_signal else "short",
                        atr_value,
                        target_r=1.5,
                        timeout_bars=6,
                        strategy="P1",
                    )
                )
                break
    return candidates


def trend_pullback(frame: pd.DataFrame) -> list[TradeCandidate]:
    """P2: H1 trend with a bounded M30 EMA20 touch and continuation close."""

    eligible = _entry_eligible(frame)
    atr_value = frame["m30_atr14"]
    long_regime = (
        (frame["h1_ema20"] > frame["h1_ema50"])
        & (frame["h1_close"] > frame["h1_ema20"])
        & (frame["h1_ema50"] > frame["h1_ema50_lag3"])
    )
    short_regime = (
        (frame["h1_ema20"] < frame["h1_ema50"])
        & (frame["h1_close"] < frame["h1_ema20"])
        & (frame["h1_ema50"] < frame["h1_ema50_lag3"])
    )
    long_signal = (
        long_regime
        & (frame["low"] <= frame["m30_ema20"])
        & (frame["close"] > frame["m30_ema20"])
        & (frame["close"] > frame["open"])
        & ((frame["close"] - frame["m30_ema20"]) <= 0.50 * atr_value)
    )
    short_signal = (
        short_regime
        & (frame["high"] >= frame["m30_ema20"])
        & (frame["close"] < frame["m30_ema20"])
        & (frame["close"] < frame["open"])
        & ((frame["m30_ema20"] - frame["close"]) <= 0.50 * atr_value)
    )
    return _daily_candidates(
        frame,
        eligible & (long_signal | short_signal),
        long_signal=long_signal,
        atr_value=atr_value,
        target_r=1.5,
        timeout_bars=8,
        strategy="P2",
    )


def regime_mean_reversion(frame: pd.DataFrame) -> list[TradeCandidate]:
    """P3: Bollinger re-entry only while completed H1 context is non-trending."""

    eligible = _entry_eligible(frame)
    atr_value = frame["m30_atr14"]
    non_trending = (
        (frame["h1_ema20"] - frame["h1_ema50"]).abs()
        <= 0.25 * frame["h1_atr14"]
    ) & (
        (frame["h1_ema50"] - frame["h1_ema50_lag3"]).abs()
        <= 0.25 * frame["h1_atr14"]
    )
    previous_below = frame["close"].shift(1) < frame["m30_bb_lower"].shift(1)
    previous_above = frame["close"].shift(1) > frame["m30_bb_upper"].shift(1)
    long_signal = (
        non_trending
        & previous_below
        & (frame["close"] > frame["m30_bb_lower"])
    )
    short_signal = (
        non_trending
        & previous_above
        & (frame["close"] < frame["m30_bb_upper"])
    )
    return _daily_candidates(
        frame,
        eligible & (long_signal | short_signal),
        long_signal=long_signal,
        atr_value=atr_value,
        target_r=1.0,
        timeout_bars=4,
        strategy="P3",
    )


def _daily_candidates(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    long_signal: pd.Series,
    atr_value: pd.Series,
    target_r: float,
    timeout_bars: int,
    strategy: str,
) -> list[TradeCandidate]:
    local_entry = (frame.index + M30_DURATION).tz_convert(LONDON)
    session_dates = np.asarray(local_entry.date)
    candidates: list[TradeCandidate] = []
    for session_date in pd.unique(session_dates):
        day_signal = signal & (session_dates == session_date)
        times = frame.index[day_signal.fillna(False)]
        for signal_time in times:
            distance = float(atr_value.loc[signal_time])
            if not np.isfinite(distance) or distance <= 0:
                continue
            candidates.append(
                _candidate(
                    signal_time,
                    "long" if bool(long_signal.loc[signal_time]) else "short",
                    distance,
                    target_r=target_r,
                    timeout_bars=timeout_bars,
                    strategy=strategy,
                )
            )
    return candidates


def _entry_eligible(frame: pd.DataFrame) -> pd.Series:
    entry = (frame.index + M30_DURATION).tz_convert(LONDON)
    minute = entry.hour * 60 + entry.minute
    return pd.Series(
        (minute >= 7 * 60) & (minute < 19 * 60 + 30),
        index=frame.index,
    )


def _candidate(
    signal_time: pd.Timestamp,
    side: str,
    atr_value: float,
    *,
    target_r: float,
    timeout_bars: int,
    strategy: str,
) -> TradeCandidate:
    return TradeCandidate(
        signal_time=signal_time,
        side=side,
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=timeout_bars,
        strategy=strategy,
        stop_distance_abs=float(atr_value),
        target_r_multiple=target_r,
    )


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {
        "open",
        "high",
        "low",
        "close",
        "m30_atr14",
        "m30_ema20",
        "m30_bb_lower",
        "m30_bb_upper",
        "h1_close",
        "h1_ema20",
        "h1_ema50",
        "h1_ema50_lag3",
        "h1_atr14",
        "opening_range_high",
        "opening_range_low",
        "opening_range_compression",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Intraday frame missing strategy columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Intraday frame requires a timezone-aware DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Intraday frame index must be sorted and unique")
