"""Offline tests for OANDA account, quote, candle and instrument adapters."""

import pandas as pd
import pytest

from commodity_ml.config import DataConfig, OANDAConfig
from commodity_ml.data.oanda import (
    _to_canonical,
    fetch_account_snapshot,
    fetch_broker_snapshot,
    fetch_recent_candles,
    instrument_metadata_from_responses,
    open_trade_ids_from_response,
    quotes_from_pricing,
)


def pricing(now: pd.Timestamp) -> dict:
    return {
        "prices": [
            {
                "instrument": "XAU_USD",
                "time": now.isoformat(),
                "status": "tradeable",
                "bids": [{"price": "2499.500"}],
                "asks": [{"price": "2500.000"}],
            }
        ],
        "homeConversions": [
            {
                "currency": "USD",
                "accountGain": "0.79",
                "accountLoss": "0.81",
                "positionValue": "0.80",
            }
        ],
    }


def instrument() -> dict:
    return {
        "name": "XAU_USD",
        "type": "METAL",
        "displayName": "Gold",
        "tradeUnitsPrecision": 1,
        "minimumTradeSize": "0.1",
        "maximumOrderUnits": "20000",
        "marginRate": "0.05",
    }


def candles() -> dict:
    return {
        "candles": [
            {
                "time": "2026-01-05T07:00:00Z",
                "complete": True,
                "volume": 12,
                "mid": {"o": "100", "h": "101", "l": "99", "c": "100.2"},
                "bid": {"o": "99.9", "h": "100.9", "l": "98.9", "c": "100.1"},
                "ask": {"o": "100.1", "h": "101.1", "l": "99.1", "c": "100.3"},
            }
        ]
    }


class FakeClient:
    account_id = "101-001-12345678-002"

    def __init__(self, now: pd.Timestamp):
        self.now = now
        self.last_count = None

    def account_summary(self):
        return {
            "id": self.account_id,
            "currency": "GBP",
            "balance": "100000",
            "unrealizedPL": "-125",
            "NAV": "99875",
            "marginAvailable": "90000",
            "hedgingEnabled": False,
        }

    def account_instruments(self, names):
        assert names == ["XAU_USD"]
        return [instrument()]

    def pricing(self, names):
        assert names == ["XAU_USD"]
        return pricing(self.now)

    def candles(self, name, granularity, *, count):
        self.last_count = count
        assert name == "XAU_USD"
        assert granularity == "M30"
        return candles()


def test_oanda_canonical_frame_retains_bid_ask_and_utc():
    out = _to_canonical(candles())
    assert str(out.index.tz) == "UTC"
    assert out.index.name == "time"
    assert out.iloc[0]["open"] == pytest.approx(100)
    assert out.iloc[0]["bid_close"] == pytest.approx(100.1)
    assert out.iloc[0]["ask_close"] == pytest.approx(100.3)


def test_recent_fetch_is_bounded_and_uses_configured_instrument():
    client = FakeClient(pd.Timestamp("2026-01-05 10:00", tz="UTC"))
    cfg = DataConfig(oanda=OANDAConfig(instrument="XAU_USD"))
    out = fetch_recent_candles(cfg, "M30", num_points=250, client=client, verbose=False)
    assert len(out) == 1
    assert client.last_count == 250


def test_account_equity_uses_authoritative_nav_and_reports_hedging():
    account = fetch_account_snapshot(
        FakeClient(pd.Timestamp("2026-01-05 10:00", tz="UTC")),
        account_type="SPREADBET",
    )
    assert account.currency == "GBP"
    assert account.equity == 99_875
    assert account.open_pnl == -125
    assert account.hedging_enabled is False


def test_account_snapshot_refuses_non_boolean_hedging_flag():
    client = FakeClient(pd.Timestamp("2026-01-05 10:00", tz="UTC"))
    client.account_summary = lambda: {
        **FakeClient.account_summary(client),
        "hedgingEnabled": "false",
    }
    with pytest.raises(RuntimeError, match="hedgingEnabled"):
        fetch_account_snapshot(client, account_type="SPREADBET")


def test_quote_freshness_and_market_status_fail_closed():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    quote = quotes_from_pricing(pricing(now - pd.Timedelta(seconds=10)))["XAU_USD"]
    quote.validate_for_entry(now=now)
    payload = pricing(now)
    payload["prices"][0]["status"] = "non-tradeable"
    closed = quotes_from_pricing(payload)["XAU_USD"]
    with pytest.raises(RuntimeError, match="market_not_tradeable"):
        closed.validate_for_entry(now=now)


def test_metadata_derives_precision_margin_and_conservative_conversion():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    metadata = instrument_metadata_from_responses(
        instrument(), pricing(now), account_currency="GBP"
    )
    assert metadata.min_size == pytest.approx(0.1)
    assert metadata.size_increment == pytest.approx(0.1)
    assert metadata.value_per_price_unit == pytest.approx(0.81)
    assert metadata.notional_value_per_price_unit == pytest.approx(0.80)
    assert metadata.margin_rate == pytest.approx(0.05)
    assert metadata.notional_per_size_unit(2500) == pytest.approx(2000)


def test_metadata_refuses_fractional_trade_unit_precision():
    malformed = {**instrument(), "tradeUnitsPrecision": 1.5}
    with pytest.raises(RuntimeError, match="precision must be an integer"):
        instrument_metadata_from_responses(
            malformed,
            pricing(pd.Timestamp("2026-01-05 10:00", tz="UTC")),
            account_currency="GBP",
        )


def test_combined_snapshot_and_trade_id_reconciliation():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    cfg = DataConfig(oanda=OANDAConfig(instrument="XAU_USD"))
    account, metadata, quote = fetch_broker_snapshot(cfg, client=FakeClient(now))
    assert account.account_id.endswith("-002")
    assert metadata.instrument == quote.instrument == "XAU_USD"
    assert open_trade_ids_from_response([{"id": "10"}, {"id": "11"}]) == ("10", "11")


def test_missing_home_conversion_refuses_account_currency_sizing():
    with pytest.raises(RuntimeError, match="home conversion"):
        instrument_metadata_from_responses(
            instrument(), {"prices": [], "homeConversions": []}, account_currency="GBP"
        )
