"""Causality and session tests for the H1/M30 rebuild frame."""

import numpy as np
import pandas as pd

from commodity_ml.intraday.data import build_intraday_frame
from commodity_ml.intraday.ml_dataset import COMMON_FEATURES, STRATEGY_FEATURES


def candles(
    *,
    start: str,
    periods: int,
    freq: str,
    drift: float = 0.02,
) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    close = 100 + np.arange(periods) * drift
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": 0.0,
        },
        index=index,
    )


def test_m30_decision_uses_only_last_completed_h1_bar():
    h1 = candles(start="2026-01-05", periods=100, freq="1h")
    m30 = candles(start="2026-01-05", periods=200, freq="30min")

    frame = build_intraday_frame(h1, m30, source="test")
    row = frame.loc[pd.Timestamp("2026-01-06 09:00", tz="UTC")]

    assert row["decision_time"] == pd.Timestamp("2026-01-06 09:30", tz="UTC")
    assert row["h1_source_time"] == pd.Timestamp("2026-01-06 08:00", tz="UTC")
    assert row["h1_available_time"] == pd.Timestamp("2026-01-06 09:00", tz="UTC")
    assert row["h1_available_time"] <= row["decision_time"]


def test_future_h1_and_m30_poison_cannot_change_past_rows():
    h1 = candles(start="2026-01-05", periods=200, freq="1h")
    m30 = candles(start="2026-01-05", periods=400, freq="30min")
    cutoff = pd.Timestamp("2026-01-09 12:00", tz="UTC")

    original = build_intraday_frame(h1, m30, source="test").loc[:cutoff]
    poisoned_h1, poisoned_m30 = h1.copy(), m30.copy()
    poisoned_h1.loc[poisoned_h1.index > cutoff, ["open", "high", "low", "close"]] *= 50
    poisoned_m30.loc[
        poisoned_m30.index > cutoff,
        ["open", "high", "low", "close"],
    ] *= 50
    poisoned = build_intraday_frame(
        poisoned_h1,
        poisoned_m30,
        source="test",
    ).loc[:cutoff]

    pd.testing.assert_frame_equal(original, poisoned)


def test_opening_range_is_hidden_until_complete_and_uses_prior_20_sessions():
    h1 = candles(start="2025-12-01", periods=24 * 45, freq="1h")
    m30 = candles(start="2025-12-01", periods=48 * 45, freq="30min")
    frame = build_intraday_frame(h1, m30, source="test")
    local = frame.index.tz_convert("Europe/London")
    late = frame.loc[(local.hour == 10) & (local.minute == 0)]
    ready = late["opening_range_median20"].dropna()

    assert len(ready) > 0
    session_date = ready.index[0].tz_convert("Europe/London").date()
    day = frame.loc[np.asarray(local.date) == session_date]
    day_local = day.index.tz_convert("Europe/London")
    before = day.loc[(day_local.hour == 8) & (day_local.minute == 0)]
    after = day.loc[(day_local.hour == 9) & (day_local.minute == 0)]
    assert before["opening_range_high"].isna().all()
    assert after["opening_range_high"].notna().all()
    assert before["opening_impulse_atr"].isna().all()
    assert after["opening_impulse_atr"].notna().all()


def test_optional_m15_context_uses_only_completed_bars_and_is_not_a_model_input():
    h1 = candles(start="2026-01-05", periods=120, freq="1h")
    m30 = candles(start="2026-01-05", periods=240, freq="30min")
    m15 = candles(start="2026-01-05", periods=480, freq="15min")
    frame = build_intraday_frame(
        h1,
        m30,
        source="test",
        m15_candles=m15,
    )
    signal_time = pd.Timestamp("2026-01-06 09:00", tz="UTC")
    row = frame.loc[signal_time]

    assert row["decision_time"] == pd.Timestamp("2026-01-06 09:30", tz="UTC")
    assert row["m15_source_time"] == pd.Timestamp("2026-01-06 09:15", tz="UTC")
    assert row["m15_available_time"] == row["decision_time"]
    assert row["m15_context_age_minutes"] == 0

    poisoned = m15.copy()
    poisoned.loc[
        poisoned.index >= pd.Timestamp("2026-01-06 09:30", tz="UTC"),
        ["open", "high", "low", "close"],
    ] *= 100
    rebuilt = build_intraday_frame(
        h1,
        m30,
        source="test",
        m15_candles=poisoned,
    )
    m15_columns = [column for column in frame if column.startswith("m15_")]
    assert not set(m15_columns) & (
        set(COMMON_FEATURES)
        | {column for values in STRATEGY_FEATURES.values() for column in values}
    )
    pd.testing.assert_series_equal(
        frame.loc[signal_time, m15_columns],
        rebuilt.loc[signal_time, m15_columns],
    )
