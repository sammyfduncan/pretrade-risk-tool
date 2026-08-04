from __future__ import annotations

import pandas as pd
import pytest

from commodity_ml.intraday.quote_path import (
    TradeExecutionRecord,
    outcome_from_executable_quotes,
)


ENTRY = pd.Timestamp("2026-08-03 10:00:00", tz="UTC")
EXIT = pd.Timestamp("2026-08-03 10:00:04", tz="UTC")


def quotes() -> pd.DataFrame:
    index = pd.date_range(ENTRY + pd.Timedelta(seconds=1), periods=5, freq="1s")
    return pd.DataFrame(
        {
            "bid": [100.0, 98.0, 103.0, 99.0, 500.0],
            "ask": [100.2, 98.2, 103.2, 99.2, 500.2],
        },
        index=index,
    )


def execution(side: str = "long") -> TradeExecutionRecord:
    if side == "long":
        expected_entry, actual_entry, actual_exit, pnl = 100.2, 100.3, 98.9, -3.1
    else:
        expected_entry, actual_entry, actual_exit, pnl = 100.0, 99.9, 99.3, 0.9
    return TradeExecutionRecord(
        intent_id=f"paper-{side}",
        side=side,
        entry_time=ENTRY,
        exit_time=EXIT,
        filled_size=2,
        expected_entry_price=expected_entry,
        actual_entry_price=actual_entry,
        actual_exit_price=actual_exit,
        value_per_price_unit=1,
        fees=0.2,
        peak_margin=50,
        realized_pnl=pnl,
        source="OANDA_PRACTICE",
        source_reference=f"DEAL-{side}",
    )


def test_long_excursion_uses_bid_and_excludes_quotes_after_exit():
    outcome, audit = outcome_from_executable_quotes(execution("long"), quotes())

    assert outcome.max_adverse_excursion_points == pytest.approx(2.3)
    assert outcome.max_favorable_excursion_points == pytest.approx(2.7)
    assert outcome.entry_spread_points == pytest.approx(0.2)
    assert outcome.exit_spread_points == pytest.approx(0.2)
    assert outcome.entry_adverse_slippage_points == pytest.approx(0.1)
    assert outcome.exit_adverse_slippage_points == pytest.approx(0.1)
    assert audit.executable_side == "bid"
    assert audit.quote_count == 4
    assert audit.price_path_pnl_after_fees == pytest.approx(-3.0)
    assert audit.broker_pnl_difference == pytest.approx(-0.1)


def test_short_excursion_uses_ask():
    outcome, audit = outcome_from_executable_quotes(execution("short"), quotes())

    assert outcome.max_adverse_excursion_points == pytest.approx(3.3)
    assert outcome.max_favorable_excursion_points == pytest.approx(1.7)
    assert outcome.entry_adverse_slippage_points == pytest.approx(0.1)
    assert outcome.exit_adverse_slippage_points == pytest.approx(0.1)
    assert audit.executable_side == "ask"
    assert audit.price_path_pnl_after_fees == pytest.approx(1.0)


def test_sparse_internal_path_is_refused():
    frame = quotes().iloc[[0, 3]]

    with pytest.raises(ValueError, match="internal_quote_gap"):
        outcome_from_executable_quotes(
            execution(), frame, max_internal_gap_seconds=2
        )


def test_missing_boundary_quote_is_refused():
    frame = quotes().iloc[2:]

    with pytest.raises(ValueError, match="entry_quote_boundary_gap"):
        outcome_from_executable_quotes(
            execution(), frame, max_boundary_gap_seconds=1
        )


@pytest.mark.parametrize("mutation, message", [
    ("crossed", "crossed"),
    ("duplicate", "unique"),
    ("naive", "timezone-aware"),
])
def test_invalid_quote_paths_are_refused(mutation: str, message: str):
    frame = quotes()
    if mutation == "crossed":
        frame.loc[frame.index[0], "ask"] = 99
    elif mutation == "duplicate":
        frame.index = pd.DatetimeIndex([frame.index[0], *frame.index[:-1]])
    else:
        frame.index = frame.index.tz_localize(None)

    with pytest.raises(ValueError, match=message):
        outcome_from_executable_quotes(execution(), frame)


def test_ohlc_bars_are_not_accepted_as_streaming_quotes():
    frame = pd.DataFrame(
        {"bid_close": [100, 99], "ask_close": [100.2, 99.2]},
        index=pd.date_range(ENTRY, periods=2, freq="1s"),
    )

    with pytest.raises(ValueError, match="requires bid and ask"):
        outcome_from_executable_quotes(execution(), frame)
