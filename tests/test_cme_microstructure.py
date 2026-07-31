"""Integrity and causality tests for the GC TBBO research boundary."""

import pandas as pd
import pytest

from commodity_ml.data.cme_microstructure import normalize_gc_tbbo
from commodity_ml.intraday.order_flow import build_active_order_flow


def _definitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "definition_time": pd.to_datetime(
                ["2020-01-01T00:00:00Z", "2020-01-01T00:00:00Z"]
            ),
            "instrument_id": [1, 2],
            "raw_symbol": ["GCF0", "GCG0"],
            "instrument_class": ["F", "F"],
            "expiration": pd.to_datetime(
                ["2020-06-01T00:00:00Z", "2020-07-01T00:00:00Z"]
            ),
        }
    )


def _ohlcv() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2020-01-01T23:00:00Z",
            "2020-01-01T23:00:00Z",
            "2020-01-02T23:00:00Z",
            "2020-01-02T23:00:00Z",
        ]
    )
    return pd.DataFrame(
        {
            "raw_symbol": ["GCF0", "GCG0", "GCF0", "GCG0"],
            "volume": [20, 10, 5, 100],
        },
        index=pd.DatetimeIndex(index, name="ts_event"),
    ).sort_index(kind="stable")


def _raw_tbbo() -> pd.DataFrame:
    index = pd.date_range("2020-01-02T23:00:10Z", periods=36, freq="1min")
    return pd.DataFrame(
        {
            "instrument_id": 1,
            "ts_event": index - pd.Timedelta(milliseconds=1),
            "action": "T",
            "side": "B",
            "price": 1510.1,
            "size": 5,
            "bid_px_00": 1510.0,
            "ask_px_00": 1510.1,
            "bid_sz_00": 10,
            "ask_sz_00": 5,
            "bid_ct_00": 2,
            "ask_ct_00": 1,
            "sequence": range(36),
        },
        index=pd.DatetimeIndex(index, name="ts_recv"),
    )


def test_tbbo_normalization_maps_aggressor_and_pretrade_book():
    result = normalize_gc_tbbo(_raw_tbbo(), _definitions())
    assert set(result["raw_symbol"]) == {"GCF0"}
    assert set(result["side"]) == {"B"}
    assert (result["ask_px_00"] > result["bid_px_00"]).all()


def test_tbbo_normalization_rejects_crossed_book():
    broken = _raw_tbbo()
    broken["ask_px_00"] = broken["bid_px_00"]
    try:
        normalize_gc_tbbo(broken, _definitions())
    except ValueError as exc:
        assert "crossed or locked" in str(exc)
    else:
        raise AssertionError("crossed TBBO book was accepted")


def test_order_flow_windows_end_strictly_before_decision():
    tbbo = normalize_gc_tbbo(_raw_tbbo(), _definitions())
    flow = build_active_order_flow(tbbo, _ohlcv(), _definitions())
    decision = pd.Timestamp("2020-01-02T23:30:00Z")
    assert flow.loc[decision, "gc_aggressive_imbalance_5m"] == 1.0
    assert flow.loc[decision, "gc_aggressive_imbalance_30m"] == 1.0
    assert flow.loc[decision, "gc_spread_ticks_5m"] == pytest.approx(1.0)


def test_poisoning_post_decision_trade_does_not_change_prior_flow():
    original_tbbo = normalize_gc_tbbo(_raw_tbbo(), _definitions())
    poisoned_tbbo = original_tbbo.copy()
    timestamp = pd.Timestamp("2020-01-02T23:31:10Z")
    poisoned_tbbo.loc[timestamp, "side"] = "A"
    poisoned_tbbo.loc[timestamp, "size"] = 1_000_000
    original = build_active_order_flow(original_tbbo, _ohlcv(), _definitions())
    poisoned = build_active_order_flow(poisoned_tbbo, _ohlcv(), _definitions())
    decision = pd.Timestamp("2020-01-02T23:30:00Z")
    pd.testing.assert_series_equal(original.loc[decision], poisoned.loc[decision])
