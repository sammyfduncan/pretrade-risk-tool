"""Causality tests for the registered COMEX GC market-state layer."""

import pandas as pd

from commodity_ml.intraday.futures_state import (
    build_active_gc_m30,
    completed_session_selections,
)


def _definitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_symbol": ["GCF0", "GCG0"],
            "instrument_class": ["F", "F"],
            "expiration": pd.to_datetime(
                ["2020-06-01T00:00:00Z", "2020-07-01T00:00:00Z"]
            ),
        }
    )


def _ohlcv() -> pd.DataFrame:
    # Session labelled Jan 2: GCF0 wins 20 to 10. Session labelled Jan 3:
    # GCG0 later wins overwhelmingly, but that final volume cannot select the
    # contract during the Jan 3 session itself.
    times = pd.to_datetime(
        [
            "2020-01-01T23:00:00Z",
            "2020-01-01T23:00:00Z",
            "2020-01-02T23:00:00Z",
            "2020-01-02T23:00:00Z",
            "2020-01-02T23:01:00Z",
            "2020-01-02T23:01:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "raw_symbol": ["GCF0", "GCG0"] * 3,
            "open": [1500.0, 1501.0, 1510.0, 1511.0, 1510.5, 1512.0],
            "high": [1501.0, 1502.0, 1511.0, 1513.0, 1512.0, 1514.0],
            "low": [1499.0, 1500.0, 1509.0, 1510.0, 1510.0, 1511.0],
            "close": [1500.5, 1501.5, 1510.5, 1512.0, 1511.5, 1513.0],
            "volume": [20, 10, 5, 500, 5, 500],
        },
        index=pd.DatetimeIndex(times, name="ts_event"),
    ).sort_index(kind="stable")


def _statistics() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2020-01-01T20:00:00Z",
            "2020-01-02T20:00:00Z",
            "2020-01-01T20:00:01Z",
            "2020-01-02T20:00:01Z",
        ]
    )
    return pd.DataFrame(
        {
            "raw_symbol": ["GCF0", "GCF0", "GCG0", "GCG0"],
            "stat_type": [9, 9, 9, 9],
            "ts_ref": pd.to_datetime(
                [
                    "2019-12-31T00:00:00Z",
                    "2020-01-01T00:00:00Z",
                    "2019-12-31T00:00:00Z",
                    "2020-01-01T00:00:00Z",
                ]
            ),
            "quantity": [100, 110, 80, 90],
        },
        index=pd.DatetimeIndex(index, name="ts_recv"),
    ).sort_index()


def test_completed_session_volume_is_available_only_after_session_end():
    selections = completed_session_selections(_ohlcv(), _definitions())
    first = selections.iloc[0]
    assert first["active_symbol"] == "GCF0"
    assert first["available_time"] == pd.Timestamp("2020-01-02T22:00:00Z")


def test_current_session_final_volume_cannot_change_current_active_contract():
    active = build_active_gc_m30(_ohlcv(), _statistics(), _definitions())
    current = active.loc[pd.Timestamp("2020-01-02T23:00:00Z")]
    assert current["raw_symbol"] == "GCF0"
    assert current["gc_volume"] == 10


def test_poisoning_later_session_volume_does_not_change_earlier_bar():
    original = build_active_gc_m30(_ohlcv(), _statistics(), _definitions())
    poisoned = _ohlcv().copy()
    poisoned.loc[
        (poisoned.index == pd.Timestamp("2020-01-02T23:01:00Z"))
        & (poisoned["raw_symbol"] == "GCG0"),
        "volume",
    ] = 1_000_000
    changed = build_active_gc_m30(poisoned, _statistics(), _definitions())
    timestamp = pd.Timestamp("2020-01-02T23:00:00Z")
    pd.testing.assert_series_equal(original.loc[timestamp], changed.loc[timestamp])
