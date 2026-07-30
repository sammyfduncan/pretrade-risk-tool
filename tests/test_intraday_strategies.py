"""Exact-condition tests for the three frozen primary candidate families."""

import numpy as np
import pandas as pd

from commodity_ml.intraday.strategies import generate_candidates


def base_frame() -> pd.DataFrame:
    index = pd.date_range("2026-01-05 06:00", periods=28, freq="30min", tz="UTC")
    frame = pd.DataFrame(index=index)
    frame["open"] = 100.0
    frame["high"] = 100.2
    frame["low"] = 99.8
    frame["close"] = 100.0
    frame["volume"] = 0.0
    frame["m30_atr14"] = 1.0
    frame["m30_ema20"] = 100.0
    frame["m30_bb_lower"] = 98.0
    frame["m30_bb_upper"] = 102.0
    frame["h1_close"] = 101.0
    frame["h1_ema20"] = 100.5
    frame["h1_ema50"] = 100.0
    frame["h1_ema50_lag3"] = 99.5
    frame["h1_atr14"] = 2.0
    frame["opening_range_high"] = np.nan
    frame["opening_range_low"] = np.nan
    frame["opening_range_compression"] = np.nan
    return frame


def test_p1_emits_first_post_opening_h1_aligned_breakout_only():
    frame = base_frame()
    day = frame.index.tz_convert("Europe/London").date
    mask = np.asarray(day) == day[0]
    post_nine = frame.index.hour >= 9
    frame.loc[mask & post_nine, "opening_range_high"] = 101.0
    frame.loc[mask & post_nine, "opening_range_low"] = 99.0
    frame.loc[mask & post_nine, "opening_range_compression"] = 1.0
    frame.loc[pd.Timestamp("2026-01-05 09:30", tz="UTC"), "close"] = 101.2
    frame.loc[pd.Timestamp("2026-01-05 10:00", tz="UTC"), "close"] = 102.0

    candidates = generate_candidates(frame, "P1")

    assert len(candidates) == 1
    assert candidates[0].signal_time == pd.Timestamp("2026-01-05 09:30", tz="UTC")
    assert candidates[0].side == "long"
    assert candidates[0].stop_distance_abs == 1.0
    assert candidates[0].target_r_multiple == 1.5
    assert candidates[0].max_holding_bars == 6


def test_p2_requires_touch_reclaim_body_and_bounded_pullback():
    frame = base_frame()
    signal_time = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    frame.loc[signal_time, ["open", "low", "close"]] = [99.9, 99.7, 100.4]

    candidates = generate_candidates(frame, "P2")

    match = [candidate for candidate in candidates if candidate.signal_time == signal_time]
    assert len(match) == 1 and match[0].side == "long"
    assert match[0].target_r_multiple == 1.5
    assert match[0].max_holding_bars == 8

    too_far = frame.copy()
    too_far.loc[signal_time, "close"] = 100.6
    assert not any(
        candidate.signal_time == signal_time
        for candidate in generate_candidates(too_far, "P2")
    )


def test_p3_requires_nontrend_and_previous_band_excursion_reentry():
    frame = base_frame()
    frame["h1_ema20"] = 100.1
    frame["h1_ema50"] = 100.0
    frame["h1_ema50_lag3"] = 99.9
    previous = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    signal = pd.Timestamp("2026-01-05 10:30", tz="UTC")
    frame.loc[previous, "close"] = 97.5
    frame.loc[signal, "close"] = 98.2

    candidates = generate_candidates(frame, "P3")

    match = [candidate for candidate in candidates if candidate.signal_time == signal]
    assert len(match) == 1 and match[0].side == "long"
    assert match[0].target_r_multiple == 1.0
    assert match[0].max_holding_bars == 4
