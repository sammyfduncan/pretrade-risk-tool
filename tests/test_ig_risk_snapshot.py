from __future__ import annotations

import pandas as pd
import pytest

from commodity_ml.data.ig import IGAccountSnapshot, IGMarketQuote
from commodity_ml.intraday.ig_risk_snapshot import (
    build_ig_read_only_snapshot,
    margin_rate_from_details,
    positions_from_ig,
    recent_same_session_spreads,
)
from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload


NOW = pd.Timestamp("2026-08-03 11:30", tz="UTC")
EPIC = "CS.D.TEST.GOLD.IP"


def account(*, available: float = 80_000) -> IGAccountSnapshot:
    return IGAccountSnapshot(
        account_id="ABC12345",
        currency="GBP",
        account_type="SPREADBET",
        status="ENABLED",
        balance=100_000,
        open_pnl=0,
        equity=100_000,
        available=available,
    )


def details() -> dict:
    return {
        "instrument": {
            "epic": EPIC,
            "name": "Spot Gold",
            "type": "COMMODITIES",
            "expiry": "DFB",
            "marginDepositBands": [
                {"currency": "GBP", "margin": 5, "min": 0, "max": 50},
                {"currency": "GBP", "margin": 10, "min": 50, "max": None},
            ],
        },
        "dealingRules": {"minDealSize": {"value": 0.1}},
    }


def quote() -> IGMarketQuote:
    return IGMarketQuote(
        epic=EPIC,
        bid=2399.9,
        ask=2400.1,
        timestamp=NOW - pd.Timedelta(seconds=10),
        market_status="TRADEABLE",
        delay_seconds=0,
    )


def candles() -> pd.DataFrame:
    index = pd.date_range("2026-08-03 07:00", periods=9, freq="30min", tz="UTC")
    bid = pd.Series(2400.0, index=index)
    spread = pd.Series([0.2] * 8 + [0.4], index=index)
    return pd.DataFrame({"bid_close": bid, "ask_close": bid + spread})


def intent() -> dict:
    return {
        "intent_id": "gold-live-read-1",
        "instrument": "XAUUSD",
        "side": "long",
        "order_type": "market",
        "decision_time": NOW.isoformat(),
        "entry_price": 2390.0,
        "stop_price": 2380.0,
        "target_price": 2430.0,
        "requested_size": 10.0,
        "holding_period_minutes": 180,
        "allow_overnight": False,
        "max_slippage_points_per_side": 0.05,
    }


def test_first_margin_band_is_scoped_instead_of_flattening_tiers():
    margin, issues = margin_rate_from_details(details(), account_currency="GBP")

    assert issues == ()
    assert margin.rate == pytest.approx(0.05)
    assert margin.max_size == pytest.approx(50)
    assert margin.source == "instrument.marginDepositBands"


def test_margin_fallback_is_explicit_and_auditable():
    margin, issues = margin_rate_from_details(
        {"instrument": {}},
        account_currency="GBP",
        fallback_rate=0.08,
        fallback_max_size=5,
    )

    assert margin.rate == pytest.approx(0.08)
    assert margin.max_size == 5
    assert issues == ("using_configured_margin_fallback",)


def test_spreads_use_only_closed_rows_in_current_london_session():
    frame = candles()
    # A forming row at 11:30 must not contaminate the statistic.
    frame.loc[NOW, ["bid_close", "ask_close"]] = [2400, 2410]

    median, p95, rows = recent_same_session_spreads(
        frame,
        decision_time=NOW,
    )

    assert rows == 9
    assert median == pytest.approx(0.2)
    assert p95 == pytest.approx(0.32)


def test_read_only_snapshot_can_feed_complete_engine_assessment():
    snapshot = build_ig_read_only_snapshot(
        instrument_symbol="XAUUSD",
        trade_intent=intent(),
        account=account(),
        market_details=details(),
        quote=quote(),
        recent_candles=candles(),
        open_positions={"positions": []},
        size_increment=0.1,
        value_per_price_unit=1.0,
        day_start_equity=100_000,
        week_start_equity=100_000,
        entries_today=0,
        scheduled_event_blackout_active=False,
        now=NOW,
    )
    report = evaluate_pretrade_payload(snapshot.payload)

    assert snapshot.order_authority is False
    assert snapshot.provenance["margin_source"] == "instrument.marginDepositBands"
    assert "market_entry_reference_replaced_with_current_mid" in snapshot.issues
    assert report.decision == "ALLOW"
    assert report.trade["entry_price"] == pytest.approx(2400.0)


def test_unknown_runtime_anchors_remain_incomplete():
    snapshot = build_ig_read_only_snapshot(
        instrument_symbol="XAUUSD",
        trade_intent=intent(),
        account=account(),
        market_details=details(),
        quote=quote(),
        recent_candles=candles(),
        open_positions={"positions": []},
        size_increment=0.1,
        value_per_price_unit=1.0,
        now=NOW,
    )
    report = evaluate_pretrade_payload(snapshot.payload)

    assert report.decision == "INCOMPLETE"
    assert "missing:account.day_start_equity" in report.reasons
    assert "missing:market.scheduled_event_blackout_active" in report.reasons


def test_single_position_uses_conservative_total_used_margin():
    response = {
        "positions": [
            {
                "market": {"epic": EPIC, "bid": 2399.9, "offer": 2400.1},
                "position": {
                    "direction": "BUY",
                    "size": 2,
                    "stopLevel": 2380,
                },
            }
        ]
    }
    positions, issues = positions_from_ig(
        response,
        account=account(available=98_000),
        epic_to_instrument={EPIC: "XAUUSD"},
        value_per_price_unit_by_instrument={"XAUUSD": 1.0},
    )

    assert issues == ()
    assert positions[0]["side"] == "long"
    assert positions[0]["current_price"] == pytest.approx(2399.9)
    assert positions[0]["margin_required"] == pytest.approx(2_000)


def test_multiple_positions_refuse_fictional_margin_allocation():
    row = {
        "market.epic": EPIC,
        "market.bid": 2399.9,
        "market.offer": 2400.1,
        "position.direction": "SELL",
        "position.size": 1,
        "position.stopLevel": 2420,
    }
    positions, issues = positions_from_ig(
        pd.DataFrame([row, row]),
        account=account(),
        epic_to_instrument={EPIC: "XAUUSD"},
        value_per_price_unit_by_instrument={"XAUUSD": 1.0},
    )

    assert len(positions) == 2
    assert all(position["margin_required"] is None for position in positions)
    assert "multiple_positions_prevent_margin_allocation" in issues
