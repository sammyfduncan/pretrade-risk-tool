from __future__ import annotations

import pandas as pd
import pytest

from commodity_ml.data.broker import (
    BrokerAccountSnapshot,
    BrokerInstrumentMetadata,
    BrokerMarketQuote,
)
from commodity_ml.intraday.oanda_risk_snapshot import (
    build_oanda_read_only_snapshot,
    positions_from_oanda,
    recent_same_session_spreads,
)
from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload

NOW = pd.Timestamp("2026-08-03 11:30", tz="UTC")


def account() -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id="101-001-12345678-002",
        currency="GBP",
        account_type="SPREADBET",
        status="ENABLED",
        balance=100_000,
        open_pnl=0,
        equity=100_000,
        available=100_000,
        hedging_enabled=False,
    )


def metadata() -> BrokerInstrumentMetadata:
    return BrokerInstrumentMetadata(
        instrument="XAU_USD",
        name="Gold",
        instrument_type="METAL",
        unit="UNITS",
        expiry="-",
        account_currency="GBP",
        min_size=0.1,
        size_increment=0.1,
        value_per_price_unit=0.81,
        notional_value_per_price_unit=0.8,
        margin_rate=0.05,
        maximum_order_size=20_000,
        streaming_prices_available=True,
    )


def quote() -> BrokerMarketQuote:
    return BrokerMarketQuote(
        instrument="XAU_USD",
        bid=2399.9,
        ask=2400.1,
        timestamp=NOW - pd.Timedelta(seconds=10),
        market_status="TRADEABLE",
    )


def candles() -> pd.DataFrame:
    index = pd.date_range("2026-08-03 07:00", periods=9, freq="30min", tz="UTC")
    bid = pd.Series(2400.0, index=index)
    return pd.DataFrame({"bid_close": bid, "ask_close": bid + 0.2})


def intent() -> dict:
    return {
        "intent_id": "gold-read-1",
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


def test_spreads_use_only_closed_current_london_session_rows():
    frame = candles()
    frame.loc[NOW, ["bid_close", "ask_close"]] = [2400, 2410]
    median, p95, rows = recent_same_session_spreads(frame, decision_time=NOW)
    assert rows == 9
    assert median == pytest.approx(0.2)
    assert p95 == pytest.approx(0.2)


def test_read_only_snapshot_feeds_complete_engine_assessment():
    snapshot = build_oanda_read_only_snapshot(
        instrument_symbol="XAUUSD",
        trade_intent=intent(),
        account=account(),
        metadata=metadata(),
        quote=quote(),
        recent_candles=candles(),
        open_trades=[],
        day_start_equity=100_000,
        week_start_equity=100_000,
        entries_today=0,
        kill_switch_active=False,
        scheduled_event_blackout_active=False,
        now=NOW,
    )
    report = evaluate_pretrade_payload(snapshot.payload)
    assert snapshot.order_authority is False
    assert snapshot.provenance["margin_source"] == "account.instrument.marginRate"
    assert "broker_reports_hedging_disabled" in snapshot.issues
    assert report.decision == "ALLOW"
    assert report.trade["broker_instrument"] == "XAU_USD"


def test_unknown_runtime_anchors_remain_incomplete():
    snapshot = build_oanda_read_only_snapshot(
        instrument_symbol="XAUUSD",
        trade_intent=intent(),
        account=account(),
        metadata=metadata(),
        quote=quote(),
        recent_candles=candles(),
        open_trades=[],
        now=NOW,
    )
    report = evaluate_pretrade_payload(snapshot.payload)
    assert report.decision == "INCOMPLETE"
    assert "missing:account.day_start_equity" in report.reasons


def test_position_uses_sided_quote_exact_stop_and_computed_margin():
    trades = [
        {
            "id": "10",
            "instrument": "XAU_USD",
            "currentUnits": "2.0",
            "stopLossOrder": {"price": "2380.0"},
        }
    ]
    positions, issues = positions_from_oanda(
        trades,
        metadata_by_instrument={"XAU_USD": metadata()},
        quotes_by_instrument={"XAU_USD": quote()},
        symbol_by_instrument={"XAU_USD": "XAUUSD"},
    )
    assert issues == ()
    assert positions[0]["side"] == "long"
    assert positions[0]["current_price"] == pytest.approx(2399.9)
    assert positions[0]["margin_required"] == pytest.approx(2 * 2399.9 * 0.8 * 0.05)


def test_position_without_stop_remains_incomplete_not_invented():
    positions, issues = positions_from_oanda(
        [{"id": "10", "instrument": "XAU_USD", "currentUnits": "-1"}],
        metadata_by_instrument={"XAU_USD": metadata()},
        quotes_by_instrument={"XAU_USD": quote()},
        symbol_by_instrument={"XAU_USD": "XAUUSD"},
    )
    assert positions[0]["stop_price"] is None
    assert "position_0_missing_stop" in issues
