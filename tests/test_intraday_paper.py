"""Dry paper-intent tests: no broker order is ever submitted."""

from dataclasses import replace

import pandas as pd

from commodity_ml.data.broker import (
    BrokerAccountSnapshot,
    BrokerInstrumentMetadata,
    BrokerMarketQuote,
)
from commodity_ml.intraday.paper import (
    PaperCandidate,
    PaperRuntimeState,
    build_paper_intent,
)
from commodity_ml.intraday.risk import VolatilityRiskDecision


NOW = pd.Timestamp("2026-01-07 12:00", tz="UTC")


def account(equity: float = 100_000) -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id="ABC123",
        currency="GBP",
        account_type="SPREADBET",
        status="ENABLED",
        balance=equity,
        open_pnl=0,
        equity=equity,
        available=equity,
    )


def market(min_size: float = 0.1) -> BrokerInstrumentMetadata:
    return BrokerInstrumentMetadata(
        instrument="XAU_USD",
        name="Spot Gold",
        instrument_type="METAL",
        unit="UNITS",
        expiry="-",
        account_currency="GBP",
        min_size=min_size,
        size_increment=0.1,
        value_per_price_unit=1.0,
        notional_value_per_price_unit=1.0,
        margin_rate=0.05,
        maximum_order_size=20_000,
        streaming_prices_available=True,
    )


def quote(
    *,
    timestamp: pd.Timestamp = NOW - pd.Timedelta(seconds=10),
    spread: float = 0.02,
) -> BrokerMarketQuote:
    return BrokerMarketQuote(
        instrument="XAU_USD",
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

    wrong_identity = build_paper_intent(
        candidate("wrong-identity"),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(
            expected_open_positions=1,
            expected_open_position_ids=("EXPECTED",),
        ),
        broker_open_position_ids=("DIFFERENT",),
    )
    assert not wrong_identity.allowed
    assert "position_id_reconciliation_mismatch" in wrong_identity.reasons


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


def volatility(multiplier: float, reason: str) -> VolatilityRiskDecision:
    return VolatilityRiskDecision(
        allowed=multiplier > 0,
        multiplier=multiplier,
        forecast=-4.0,
        threshold_80=-3.5,
        threshold_95=-3.0,
        reference_count=500,
        reason=reason,
    )


def test_volatility_policy_blocks_or_reduces_paper_risk_without_increasing_it():
    blocked = build_paper_intent(
        candidate("vol-block"),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(),
        broker_open_position_ids=(),
        volatility_risk=volatility(0.0, "forecast_at_or_above_p95"),
    )
    reduced = build_paper_intent(
        candidate("vol-reduce"),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(),
        broker_open_position_ids=(),
        volatility_risk=volatility(0.5, "forecast_at_or_above_p80"),
    )
    full = build_paper_intent(
        candidate("vol-full"),
        account=account(),
        market=market(),
        quote=quote(),
        quote_candles=quote_history(),
        runtime=runtime(),
        broker_open_position_ids=(),
        volatility_risk=volatility(1.0, "forecast_below_p80"),
    )

    assert not blocked.allowed
    assert blocked.reasons == ("forecast_at_or_above_p95",)
    assert reduced.allowed and full.allowed
    assert reduced.sizing.risk_budget == full.sizing.risk_budget / 2
    assert reduced.intent.size <= full.intent.size
