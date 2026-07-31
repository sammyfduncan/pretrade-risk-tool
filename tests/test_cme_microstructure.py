"""Integrity and causality tests for the GC TBBO research boundary."""

import pandas as pd
import pytest

from commodity_ml.data.cme_microstructure import normalize_gc_tbbo
from commodity_ml.intraday.order_flow import build_active_order_flow
from commodity_ml.intraday.order_flow import build_active_order_flow_partitions


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


def test_tbbo_normalization_reason_codes_crossed_book():
    broken = _raw_tbbo()
    broken["ask_px_00"] = broken["bid_px_00"]
    result = normalize_gc_tbbo(broken, _definitions())
    assert not result["book_valid"].any()


def test_order_flow_windows_end_strictly_before_decision():
    tbbo = normalize_gc_tbbo(_raw_tbbo(), _definitions())
    flow = build_active_order_flow(tbbo, _ohlcv(), _definitions())
    decision = pd.Timestamp("2020-01-02T23:30:00Z")
    assert flow.loc[decision, "gc_aggressive_imbalance_5m"] == 1.0
    assert flow.loc[decision, "gc_aggressive_imbalance_30m"] == 1.0
    assert flow.loc[decision, "gc_depth_imbalance_5m"] == pytest.approx(1 / 3)
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


def test_trade_intensity_reference_uses_prior_observed_sessions():
    session_starts = pd.date_range("2020-01-01T23:00:00Z", periods=22, freq="1D")
    ohlcv = pd.DataFrame(
        {"raw_symbol": "GCF0", "volume": 100},
        index=pd.DatetimeIndex(session_starts, name="ts_event"),
    )
    records = []
    for day, start in enumerate(session_starts[1:]):
        for minute in range(30):
            records.append(
                {
                    "ts_recv": start + pd.Timedelta(minutes=minute, seconds=10),
                    "raw_symbol": "GCF0",
                    "side": "B",
                    "size": 1 + day,
                    "bid_px_00": 1510.0,
                    "ask_px_00": 1510.1,
                    "bid_sz_00": 10,
                    "ask_sz_00": 5,
                    "book_valid": True,
                }
            )
    tbbo = pd.DataFrame(records).set_index("ts_recv").sort_index()
    tbbo.index = pd.DatetimeIndex(tbbo.index, name="ts_recv")
    flow = build_active_order_flow(tbbo, ohlcv, _definitions())
    final_decision = session_starts[-1] + pd.Timedelta(minutes=30)
    assert pd.notna(flow.loc[final_decision, "gc_trade_intensity_surprise_5m"])


def test_partition_streaming_matches_single_frame(tmp_path):
    tbbo = normalize_gc_tbbo(_raw_tbbo(), _definitions())
    midpoint = len(tbbo) // 2
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    tbbo.iloc[:midpoint].to_parquet(first)
    tbbo.iloc[midpoint:].to_parquet(second)
    expected = build_active_order_flow(tbbo, _ohlcv(), _definitions())
    actual = build_active_order_flow_partitions(
        [first, second], _ohlcv(), _definitions()
    )
    pd.testing.assert_frame_equal(expected, actual)
