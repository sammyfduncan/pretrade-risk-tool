"""Causality and exact-rule tests for the frozen P4 driver layer."""

import numpy as np
import pandas as pd

from commodity_ml.intraday.execution import EntryBlackout
from commodity_ml.intraday.macro_impulse import (
    REFERENCE_OBSERVATIONS,
    _prior_observed_reference,
    build_driver_state,
    generate_macro_impulse_candidates,
)


def _definitions(symbol: str = "ZNM0") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_symbol": [symbol],
            "instrument_class": ["F"],
            "expiration": pd.to_datetime(["2030-06-01T00:00:00Z"]),
        }
    )


def _daily_ohlcv(days: int = 75, symbol: str = "ZNM0") -> pd.DataFrame:
    records: list[dict] = []
    index: list[pd.Timestamp] = []
    for day, start in enumerate(pd.date_range("2020-01-01T10:00Z", periods=days, freq="D")):
        move = 0.0002 * (1 + (day % 7))
        for minute in range(30):
            open_price = 100.0 + day * 0.01 + minute * move
            close = open_price + move
            index.append(start + pd.Timedelta(minutes=minute))
            records.append(
                {
                    "raw_symbol": symbol,
                    "open": open_price,
                    "high": close + move,
                    "low": open_price - move,
                    "close": close,
                    "volume": 10 + minute,
                }
            )
    return pd.DataFrame(
        records,
        index=pd.DatetimeIndex(index, name="ts_event"),
    )


def test_same_clock_reference_excludes_current_observation() -> None:
    index = pd.date_range("2020-01-01", periods=REFERENCE_OBSERVATIONS + 1, freq="D", tz="UTC")
    values = pd.Series(np.arange(len(index), dtype=float), index=index)
    slot = pd.Series(600, index=index)

    median, mad, count = _prior_observed_reference(values, slot)

    last = index[-1]
    assert median.loc[last] == np.median(np.arange(REFERENCE_OBSERVATIONS))
    assert mad.loc[last] == np.median(
        np.abs(np.arange(REFERENCE_OBSERVATIONS) - np.median(np.arange(REFERENCE_OBSERVATIONS)))
    )
    assert count.loc[last] == REFERENCE_OBSERVATIONS


def test_future_price_poisoning_cannot_change_earlier_driver_state() -> None:
    ohlcv = _daily_ohlcv()
    original = build_driver_state(ohlcv, _definitions(), prefix="zn").frame
    cutoff = original.index[-3]
    poisoned = ohlcv.copy()
    poisoned.loc[poisoned.index >= cutoff, ["open", "high", "low", "close"]] *= 3
    changed = build_driver_state(poisoned, _definitions(), prefix="zn").frame

    pd.testing.assert_frame_equal(
        original.loc[original.index < cutoff],
        changed.loc[changed.index < cutoff],
    )


def test_bucket_with_only_24_traded_minutes_is_ineligible() -> None:
    ohlcv = _daily_ohlcv()
    target_start = pd.Timestamp("2020-03-10T10:00Z")
    incomplete = ohlcv.drop(
        ohlcv.loc[
            (ohlcv.index >= target_start + pd.Timedelta(minutes=24))
            & (ohlcv.index < target_start + pd.Timedelta(minutes=30))
        ].index
    )
    state = build_driver_state(incomplete, _definitions(), prefix="zn").frame
    decision = target_start + pd.Timedelta(minutes=30)

    assert state.at[decision, "zn_trade_minutes"] == 24
    assert not bool(state.at[decision, "zn_bar_eligible"])
    assert np.isnan(state.at[decision, "zn_z"])


def test_p4_takes_first_confirming_signal_per_london_day() -> None:
    index = pd.date_range("2024-01-08T08:00Z", periods=6, freq="30min")
    gold = pd.DataFrame(
        {
            "open": [2000.0] * 6,
            "close": [2001.0, 2002.0, 1999.0, 1998.0, 2001.0, 2002.0],
            "m30_atr14": [5.0] * 6,
            "decision_time": index + pd.Timedelta(minutes=30),
        },
        index=index,
    )
    state = pd.DataFrame(
        {"zn_z": [1.2] * 6, "euro_z": [1.4] * 6, "eligible": [True] * 6},
        index=index + pd.Timedelta(minutes=30),
    )

    candidates = generate_macro_impulse_candidates(gold, state)

    assert len(candidates) == 1
    assert candidates[0].signal_time == index[0]
    assert candidates[0].side == "long"
    assert candidates[0].strategy == "P4"
    assert candidates[0].stop_distance_abs == 5.0
    assert candidates[0].target_r_multiple == 1.5
    assert candidates[0].max_holding_bars == 6


def test_p4_rejects_gold_body_larger_than_one_atr() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-08T08:00Z")])
    gold = pd.DataFrame(
        {
            "open": [2000.0],
            "close": [2011.0],
            "m30_atr14": [10.0],
            "decision_time": index + pd.Timedelta(minutes=30),
        },
        index=index,
    )
    state = pd.DataFrame(
        {"zn_z": [2.0], "euro_z": [2.0], "eligible": [True]},
        index=index + pd.Timedelta(minutes=30),
    )

    assert generate_macro_impulse_candidates(gold, state) == []


def test_p4_selects_first_signal_after_a_registered_blackout() -> None:
    index = pd.date_range("2024-01-08T08:00Z", periods=2, freq="30min")
    gold = pd.DataFrame(
        {
            "open": [2000.0, 2000.0],
            "close": [2001.0, 2001.0],
            "m30_atr14": [5.0, 5.0],
            "decision_time": index + pd.Timedelta(minutes=30),
        },
        index=index,
    )
    state = pd.DataFrame(
        {"zn_z": [1.2, 1.2], "euro_z": [1.2, 1.2], "eligible": [True, True]},
        index=index + pd.Timedelta(minutes=30),
    )
    blackout = EntryBlackout(
        start=pd.Timestamp("2024-01-08T08:30Z"),
        end=pd.Timestamp("2024-01-08T09:00Z"),
    )

    candidates = generate_macro_impulse_candidates(
        gold,
        state,
        entry_blackouts=(blackout,),
    )

    assert len(candidates) == 1
    assert candidates[0].signal_time == index[1]
