"""Causality, synchronization, and execution tests for frozen RV1."""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.intraday.relative_value import (
    RelativeValueCandidate,
    _prior_same_slot_reference,
    build_relative_value_state,
    generate_relative_value_candidates,
    remeasure_outcome,
    simulate_relative_value,
    synchronize_quotes,
)


def _candles(index: pd.DatetimeIndex, price: float) -> pd.DataFrame:
    move = np.sin(np.arange(len(index)) / 17) * price * 0.0001
    open_price = price + move
    close = open_price + np.cos(np.arange(len(index)) / 11) * price * 0.00005
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + price * 0.0001,
            "low": np.minimum(open_price, close) - price * 0.0001,
            "close": close,
            "volume": 1.0,
        },
        index=index,
    )


def _quoted_candles(index: pd.DatetimeIndex):
    gold_bid = _candles(index, 2000.0)
    gold_ask = gold_bid.copy()
    gold_ask[["open", "high", "low", "close"]] += 0.20
    silver_bid = _candles(index, 25.0)
    silver_ask = silver_bid.copy()
    silver_ask[["open", "high", "low", "close"]] += 0.02
    return gold_bid, gold_ask, silver_bid, silver_ask


def _execution_quotes() -> pd.DataFrame:
    index = pd.date_range("2024-01-08T08:00Z", periods=4, freq="30min")
    values = {
        "gold_bid_open": [100.0, 100.0, 101.0, 101.0],
        "gold_ask_open": [100.1, 100.1, 101.1, 101.1],
        "gold_bid_close": [100.0, 101.0, 101.0, 101.0],
        "gold_ask_close": [100.1, 101.1, 101.1, 101.1],
        "silver_bid_open": [50.0, 50.0, 49.5, 49.5],
        "silver_ask_open": [50.1, 50.1, 49.6, 49.6],
        "silver_bid_close": [50.0, 49.4, 49.4, 49.4],
        "silver_ask_close": [50.1, 49.5, 49.5, 49.5],
    }
    frame = pd.DataFrame(values, index=index)
    for leg in ("gold", "silver"):
        for field in ("open", "close"):
            frame[f"{leg}_mid_{field}"] = (
                frame[f"{leg}_bid_{field}"] + frame[f"{leg}_ask_{field}"]
            ) / 2
    return frame


def test_synchronization_is_exact_and_never_fills_a_missing_timestamp() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    streams = list(_quoted_candles(index))
    streams[-1] = streams[-1].drop(index[1])

    result = synchronize_quotes(*streams)

    assert index[1] not in result.frame.index
    assert len(result.frame) == 3
    assert result.report["shared_rows"] == 3
    assert result.report["union_rows"] == 4
    assert result.report["filled_rows"] == 0


def test_synchronization_rejects_crossed_open_quote() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    streams = list(_quoted_candles(index))
    streams[1].loc[index[0], ["open", "high", "low", "close"]] = (
        streams[0].loc[index[0], ["open", "high", "low", "close"]] - 1
    )

    with pytest.raises(ValueError, match="crossed open/close"):
        synchronize_quotes(*streams)


def test_same_slot_reference_excludes_current_residual() -> None:
    index = pd.date_range("2020-01-01", periods=61, freq="D", tz="UTC")
    values = pd.Series(np.arange(61, dtype=float), index=index)
    slot = pd.Series(600, index=index)

    median, _, count = _prior_same_slot_reference(values, slot)

    assert median.iloc[-1] == np.median(np.arange(60))
    assert count.iloc[-1] == 60


def test_future_quote_poisoning_cannot_change_earlier_state() -> None:
    index = pd.date_range("2020-01-01", periods=4_000, freq="30min", tz="UTC")
    quotes = synchronize_quotes(*_quoted_candles(index)).frame
    original = build_relative_value_state(quotes).frame
    cutoff = index[-100]
    poisoned = quotes.copy()
    poisoned.loc[poisoned.index >= cutoff, ["gold_mid_open", "gold_mid_close"]] *= 3
    changed = build_relative_value_state(poisoned).frame

    pd.testing.assert_frame_equal(
        original.loc[original.index < cutoff],
        changed.loc[changed.index < cutoff],
    )


def test_candidate_weights_are_gross_normalized_and_first_per_day() -> None:
    index = pd.date_range("2024-01-08T08:00Z", periods=4, freq="30min")
    state = pd.DataFrame(
        {
            "eligible": True,
            "z": [-2.5, -3.0, 2.5, 3.0],
            "beta": 0.5,
            "residual_scale": 0.003,
        },
        index=index,
    )

    candidates = generate_relative_value_candidates(state)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.side == "long_residual"
    assert candidate.gold_weight == pytest.approx(2 / 3)
    assert candidate.silver_weight == pytest.approx(-1 / 3)
    assert abs(candidate.gold_weight) + abs(candidate.silver_weight) == pytest.approx(1)
    assert candidate.initial_r == pytest.approx(0.002)


def test_two_leg_execution_uses_ask_to_buy_and_bid_to_sell() -> None:
    quotes = _execution_quotes()
    candidate = RelativeValueCandidate(
        signal_time=quotes.index[0],
        side="long_residual",
        beta=1.0,
        gold_weight=0.5,
        silver_weight=-0.5,
        initial_r=0.005,
    )

    ledger = simulate_relative_value(quotes, [candidate])

    row = ledger.iloc[0]
    expected_gold = 101.0 / 100.1 - 1
    expected_silver = 1 - 49.5 / 50.0
    assert row["status"] == "executed"
    assert row["exit_reason"] == "target"
    assert row["gross_return"] == pytest.approx(
        0.5 * expected_gold + 0.5 * expected_silver
    )
    assert row["net_return"] == pytest.approx(row["gross_return"] - 0.0001)


def test_spread_stress_remeasures_same_clocks_and_worsens_return() -> None:
    quotes = _execution_quotes()
    candidate = RelativeValueCandidate(
        signal_time=quotes.index[0],
        side="long_residual",
        beta=1.0,
        gold_weight=0.5,
        silver_weight=-0.5,
        initial_r=0.005,
    )
    row = simulate_relative_value(quotes, [candidate]).iloc[0]

    _, base = remeasure_outcome(
        quotes,
        row,
        spread_multiplier=1.0,
        additional_cost_per_side=0.00005,
    )
    _, stressed = remeasure_outcome(
        quotes,
        row,
        spread_multiplier=2.0,
        additional_cost_per_side=0.0001,
    )

    assert stressed < base
    assert row["entry_time"] == quotes.index[1]
    assert row["exit_bar_time"] == quotes.index[1]
