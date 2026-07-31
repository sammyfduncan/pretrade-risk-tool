"""Matched M15 replay keeps M30 signal timing and candidate identity."""

import pandas as pd
import pytest

from commodity_ml.intraday.execution import (
    IntradayExecutionConfig,
    TradeCandidate,
    resolve_trade,
)
from commodity_ml.intraday.m15_execution import replay_m30_candidates_on_m15


def test_m15_replay_enters_when_the_m30_signal_candle_has_closed():
    index = pd.date_range("2026-01-05 09:00", periods=5, freq="15min", tz="UTC")
    candles = pd.DataFrame(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 101.2, 99.5, 101.0),
            (101.0, 101.1, 100.5, 100.8),
            (100.8, 100.9, 100.0, 100.2),
        ],
        columns=["open", "high", "low", "close"],
        index=index,
    )
    candidate = TradeCandidate(
        signal_time=index[0],
        side="long",
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=1,
        strategy="P1V2",
        stop_distance_abs=1.0,
        target_r_multiple=1.0,
    )

    ledger = replay_m30_candidates_on_m15(
        candles,
        [candidate],
        IntradayExecutionConfig(cost_pct_per_side=0.0),
    )

    row = ledger.iloc[0]
    assert row["signal_time"] == index[0]
    assert row["m15_anchor_time"] == index[1]
    assert row["entry_time"] == index[2]
    assert row["exit_reason"] == "target"
    assert row["bars_held"] == 1


def test_m15_path_can_resolve_a_coarse_both_bar_more_precisely():
    m30_index = pd.date_range("2026-01-05 09:00", periods=2, freq="30min", tz="UTC")
    m30 = pd.DataFrame(
        [(100.0, 100.2, 99.8, 100.0), (100.0, 101.2, 98.8, 100.0)],
        columns=["open", "high", "low", "close"],
        index=m30_index,
    )
    m15_index = pd.date_range("2026-01-05 09:00", periods=4, freq="15min", tz="UTC")
    m15 = pd.DataFrame(
        [
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 100.2, 99.8, 100.0),
            (100.0, 101.2, 99.5, 101.0),
            (101.0, 101.1, 98.8, 100.0),
        ],
        columns=["open", "high", "low", "close"],
        index=m15_index,
    )
    candidate = TradeCandidate(
        signal_time=m30_index[0],
        side="long",
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=1,
        strategy="P3V2",
        stop_distance_abs=1.0,
        target_r_multiple=1.0,
    )
    cfg = IntradayExecutionConfig(cost_pct_per_side=0.0)

    coarse = resolve_trade(m30, candidate, cfg)
    fine = replay_m30_candidates_on_m15(m15, [candidate], cfg).iloc[0]

    assert coarse.exit_reason == "stop"
    assert fine["exit_reason"] == "target"
    assert fine["entry_time"] == coarse.entry_time


def test_m15_replay_requires_an_m30_source_contract():
    empty = pd.DataFrame(
        columns=["open", "high", "low", "close"],
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    with pytest.raises(ValueError, match="M30 source"):
        replay_m30_candidates_on_m15(
            empty,
            [],
            IntradayExecutionConfig(bar_duration=pd.Timedelta(minutes=15)),
        )
