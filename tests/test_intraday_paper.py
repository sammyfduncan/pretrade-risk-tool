"""Dry paper-intent tests: no broker order is ever submitted."""

from dataclasses import replace

import pandas as pd

from commodity_ml.data.ig import (
    IGAccountSnapshot,
    IGMarketMetadata,
    IGMarketQuote,
)
from commodity_ml.intraday.paper import (
    PaperCandidate,
    PaperRuntimeState,
    build_paper_intent,
)


NOW = pd.Timestamp("2026-01-07 12:00", tz="UTC")


def account(equity: float = 100_000) -> IGAccountSnapshot:
    return IGAccountSnapshot(
        account_id="ABC123",
        currency="GBP",
        account_type="CFD",
        status="ENABLED",
        balance=equity,
        open_pnl=0,
        equity=equity,
        available=equity,
    )


def market(min_size: float = 0.1) -> IGMarketMetadata:
    return IGMarketMetadata(
        epic="CS.D.TEST.GOLD.IP",
        name="Spot Gold",
        instrument_type="COMMODITIES",
        unit="CONTRACTS",
        expiry="DFB",
        account_currency="GBP",
        min_size=min_size,
        size_increment=0.1,
        value_per_price_unit=1.0,
        contract_size=1.0,
        lot_size=1.0,
        streaming_prices_available=True,
    )


def quote(
    *,
    timestamp: pd.Timestamp = NOW - pd.Timedelta(seconds=10),
    spread: float = 0.02,
) -> IGMarketQuote:
    return IGMarketQuote(
        epic="CS.D.TEST.GOLD.IP",
        bid=100 - spread / 2,
        ask=100 + spread / 2,
        timestamp=timestamp,
        market_status="TRADEABLE",
        delay_seconds=0,
    )


def quote_history(spread: float = 0.02) -> pd.DataFrame:
    index = pd.date_range(end=NOW - pd.Timedelta(minutes=30), periods=100, freq="30min")
    frame = pd.DataFrame(index=index)
    for field, base in (
        ("open", 100.0),
        ("high", 100.2),
        ("low", 99.8),
        ("close", 100.0),
    ):
        frame[f"bid_{field}"] = base - spread / 2
        frame[f"ask_{field}"] = base + spread / 2
    return frame


def runtime(**overrides) -> PaperRuntimeState:
    values = {
        "day_start_equity": 100_000,
        "week_start_equity": 100_000,
        "entries_today": 0,
        "expected_open_positions": 0,
    }
    values.update(overrides)
    return PaperRuntimeState(**values)


def candidate(intent_id: str = "P1:2026-01-07T11:30Z") -> PaperCandidate:
    return PaperCandidate(
        intent_id=intent_id,
        strategy="P1",
        side="long",
        decision_time=NOW,
        stop_distance_pct=0.01,
        target_distance_pct=0.015,
    )


def test_valid_inputs_build_a_dry_sized_order_intent():
    decision = build_paper_intent(
        candidate(),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(),
        broker_open_position_ids=(),
    )

    assert decision.allowed
    assert decision.intent is not None and decision.intent.dry_run
    assert decision.intent.direction == "BUY"
    assert decision.intent.size > 0
    assert decision.intent.stop_level < decision.intent.reference_price
    assert decision.sizing.planned_stop_loss <= decision.sizing.risk_budget


def test_duplicate_or_reconciliation_mismatch_fails_before_sizing():
    duplicate = build_paper_intent(
        candidate(),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(submitted_intent_ids=(candidate().intent_id,)),
        broker_open_position_ids=(),
    )
    assert not duplicate.allowed
    assert "duplicate_intent_id" in duplicate.reasons
    assert duplicate.sizing is None

    mismatch = build_paper_intent(
        candidate("different"),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(expected_open_positions=0),
        broker_open_position_ids=("D1",),
    )
    assert not mismatch.allowed
    assert "position_reconciliation_mismatch" in mismatch.reasons


def test_stale_quote_and_wide_spread_fail_closed():
    stale = build_paper_intent(
        candidate(),
        account=account(),
        market=market(),
        quote=quote(timestamp=NOW - pd.Timedelta(minutes=2)),
        quote_candles=quote_history(),
        runtime=runtime(),
        broker_open_position_ids=(),
    )
    assert not stale.allowed
    assert any(reason.startswith("stale_quote") for reason in stale.reasons)

    wide = build_paper_intent(
        candidate("wide"),
        account=account(),
        market=market(),
        quote=quote(spread=0.5),
        quote_candles=quote_history(spread=0.5),
        runtime=runtime(),
        broker_open_position_ids=(),
    )
    assert not wide.allowed
    assert wide.spread is not None
    assert "median_cost_above_0.10R" in wide.reasons


def test_daily_loss_gate_requests_flatten():
    losing_account = replace(account(), equity=99_250, balance=99_250)
    decision = build_paper_intent(
        candidate(),
        account=losing_account,
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(expected_open_positions=1),
        broker_open_position_ids=("D1",),
    )

    assert not decision.allowed
    assert decision.flatten_required
    assert "daily_loss_limit" in decision.reasons


def test_unsafe_broker_minimum_size_is_rejected():
    decision = build_paper_intent(
        candidate(),
        account=account(equity=1_000),
        market=market(min_size=20),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(day_start_equity=1_000, week_start_equity=1_000),
        broker_open_position_ids=(),
    )

    assert not decision.allowed
    assert decision.reasons == ("minimum_size_exceeds_caps",)
