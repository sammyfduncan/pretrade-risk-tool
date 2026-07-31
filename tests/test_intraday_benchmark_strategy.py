"""Causal candidate construction for the frozen LBMA-PM hypothesis."""

from datetime import time

import numpy as np
import pandas as pd

from commodity_ml.intraday.benchmark_strategy import benchmark_continuation_candidates


def candles() -> pd.DataFrame:
    index = pd.date_range("2026-01-05", periods=3 * 96, freq="15min", tz="UTC")
    close = np.full(len(index), 100.0)
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 0.0,
        },
        index=index,
    )
    local = index.tz_convert("Europe/London")
    target_day = pd.Timestamp("2026-01-07").date()
    window = (local.date == target_day) & (local.hour == 14)
    positions = np.flatnonzero(window)
    for offset, position in enumerate(positions):
        frame.iloc[position, frame.columns.get_loc("open")] = 100.0 + offset * 0.5
        frame.iloc[position, frame.columns.get_loc("close")] = 100.5 + offset * 0.5
        frame.iloc[position, frame.columns.get_loc("high")] = 100.6 + offset * 0.5
        frame.iloc[position, frame.columns.get_loc("low")] = 99.9 + offset * 0.5
    return frame


def test_b1_uses_completed_pre_auction_hour_and_enters_next_bar():
    frame = candles()

    candidates = benchmark_continuation_candidates(frame)

    candidate = next(item for item in candidates if item.signal_time.day == 7)
    assert candidate.signal_time.hour == 14
    assert candidate.signal_time.minute == 45
    assert candidate.side == "long"
    assert candidate.max_holding_bars == 4
    assert candidate.target_r_multiple == 1.5


def test_benchmark_clock_is_explicit_not_hardcoded_to_pm():
    frame = candles()
    local = frame.index.tz_convert("Europe/London")
    window = (local.date == pd.Timestamp("2026-01-07").date()) & (local.hour == 13)
    positions = np.flatnonzero(window)
    for offset, position in enumerate(positions):
        frame.iloc[position, frame.columns.get_loc("open")] = 100.0 - offset * 0.5
        frame.iloc[position, frame.columns.get_loc("close")] = 99.5 - offset * 0.5
        frame.iloc[position, frame.columns.get_loc("high")] = 100.1 - offset * 0.5
        frame.iloc[position, frame.columns.get_loc("low")] = 99.4 - offset * 0.5

    candidates = benchmark_continuation_candidates(
        frame,
        benchmark_time=time(14, 0),
        strategy="B1_PLACEBO_1400",
    )

    candidate = next(item for item in candidates if item.signal_time.day == 7)
    assert candidate.signal_time.hour == 13
    assert candidate.signal_time.minute == 45
    assert candidate.side == "short"
    assert candidate.strategy == "B1_PLACEBO_1400"
