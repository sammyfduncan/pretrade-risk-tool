"""Offline tests for IG quote, account, and instrument adapters."""

import pandas as pd
import pytest

from commodity_ml.config import DataConfig, IGConfig
from commodity_ml.data.ig import (
    _to_canonical,
    fetch_account_snapshot,
    fetch_broker_snapshot,
    fetch_open_position_ids,
    fetch_recent_candles,
    market_metadata_from_details,
    market_quote_from_details,
    open_position_ids_from_response,
)


def flat_ig_prices() -> pd.DataFrame:
    index = pd.date_range("2026-01-05 07:00", periods=3, freq="30min")
    return pd.DataFrame(
        {
            "open.bid": [99.9, 100.9, 101.9],
            "high.bid": [100.4, 101.4, 102.4],
            "low.bid": [99.4, 100.4, 101.4],
            "close.bid": [100.1, 101.1, 102.1],
            "open.ask": [100.1, 101.1, 102.1],
            "high.ask": [100.6, 101.6, 102.6],
            "low.ask": [99.6, 100.6, 101.6],
            "close.ask": [100.3, 101.3, 102.3],
            "volume": [0, 0, 0],
        },
        index=index,
    )


def market_details(now: pd.Timestamp) -> dict:
    return {
        "instrument": {
            "epic": "CS.D.TEST.GOLD.IP",
            "name": "Spot Gold",
            "type": "COMMODITIES",
            "unit": "CONTRACTS",
            "expiry": "DFB",
            "contractSize": "1",
            "lotSize": 1,
            "streamingPricesAvailable": True,
        },
        "dealingRules": {"minDealSize": {"value": 0.1, "unit": "POINTS"}},
        "snapshot": {
            "bid": 2499.5,
            "ask": 2500.0,
            "updateTimestampUTC": int(now.timestamp()),
            "marketStatus": "TRADEABLE",
            "delayTime": 0,
        },
    }


class FakeService:
    flat_prices = object()

    def __init__(self, now: pd.Timestamp):
        self.now = now
        self.last_request = None

    def fetch_historical_prices_by_epic(self, **kwargs):
        self.last_request = kwargs
        return {"prices": flat_ig_prices()}

    def fetch_accounts(self):
        return pd.DataFrame(
            [
                {
                    "accountId": "ABC123",
                    "currency": "GBP",
                    "accountType": "CFD",
                    "status": "ENABLED",
                    "balance": 10_000,
                    "profitLoss": -125,
                    "available": 8_000,
                    "preferred": True,
                }
            ]
        )

    def fetch_market_by_epic(self, epic):
        assert epic == "CS.D.TEST.GOLD.IP"
        return market_details(self.now)

    def fetch_open_positions(self):
        return pd.DataFrame({"dealId": ["D1", "D2"]})


def test_ig_canonical_frame_retains_quotes_and_uses_utc():
    out = _to_canonical(flat_ig_prices())

    assert str(out.index.tz) == "UTC"
    assert out.index.name == "time"
    assert out.loc[out.index[0], "open"] == pytest.approx(100.0)
    assert out.loc[out.index[0], "bid_open"] == pytest.approx(99.9)
    assert out.loc[out.index[0], "ask_open"] == pytest.approx(100.1)


def test_legacy_hierarchical_trading_ig_frame_is_still_supported():
    flat = flat_ig_prices()
    bid = flat[["open.bid", "high.bid", "low.bid", "close.bid"]].copy()
    ask = flat[["open.ask", "high.ask", "low.ask", "close.ask"]].copy()
    bid.columns = ["Open", "High", "Low", "Close"]
    ask.columns = ["Open", "High", "Low", "Close"]
    last = pd.DataFrame({"Volume": [0, 0, 0]}, index=flat.index)
    hierarchical = pd.concat(
        [bid, ask, last],
        axis=1,
        keys=["bid", "ask", "last"],
    )

    out = _to_canonical(hierarchical)

    assert out["open"].iloc[0] == pytest.approx(100.0)
    assert out["volume"].eq(0).all()


def test_recent_fetch_is_bounded_and_requests_utc_flat_formatter():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    service = FakeService(now)
    cfg = DataConfig(ig=IGConfig(epic="CS.D.TEST.GOLD.IP"))

    out = fetch_recent_candles(
        cfg,
        "M30",
        num_points=250,
        service=service,
        verbose=False,
    )

    assert len(out) == 3
    assert service.last_request["numpoints"] == 250
    assert service.last_request["format"] is service.flat_prices


def test_account_equity_includes_open_pnl():
    account = fetch_account_snapshot(
        FakeService(pd.Timestamp("2026-01-05 10:00", tz="UTC"))
    )

    assert account.currency == "GBP"
    assert account.balance == 10_000
    assert account.open_pnl == -125
    assert account.equity == 9_875


def test_market_quote_freshness_and_delay_fail_closed():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    details = market_details(now - pd.Timedelta(seconds=10))
    quote = market_quote_from_details(details, now=now)
    quote.validate_for_entry(now=now)

    details["snapshot"]["delayTime"] = 1
    delayed = market_quote_from_details(details, now=now)
    with pytest.raises(RuntimeError, match="delayed_quote"):
        delayed.validate_for_entry(now=now)

    details["snapshot"]["delayTime"] = 0
    details["snapshot"]["updateTimestampUTC"] = int(
        (now - pd.Timedelta(minutes=2)).timestamp()
    )
    stale = market_quote_from_details(details, now=now)
    with pytest.raises(RuntimeError, match="stale_quote"):
        stale.validate_for_entry(now=now)


def test_sizing_metadata_requires_ticket_confirmed_fields():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    details = market_details(now)

    with pytest.raises(RuntimeError, match="missing_size_increment"):
        market_metadata_from_details(
            details,
            account_currency="GBP",
            size_increment=None,
            value_per_price_unit=1.0,
        )
    with pytest.raises(RuntimeError, match="missing_value_per_price_unit"):
        market_metadata_from_details(
            details,
            account_currency="GBP",
            size_increment=0.1,
            value_per_price_unit=None,
        )

    metadata = market_metadata_from_details(
        details,
        account_currency="GBP",
        size_increment=0.1,
        value_per_price_unit=0.8,
    )
    assert metadata.min_size == pytest.approx(0.1)
    assert metadata.notional_per_size_unit(2500) == pytest.approx(2000)


def test_combined_broker_snapshot_and_position_reconciliation_adapter():
    now = pd.Timestamp("2026-01-05 10:00", tz="UTC")
    service = FakeService(now)
    cfg = DataConfig(
        ig=IGConfig(
            epic="CS.D.TEST.GOLD.IP",
            size_increment=0.1,
            value_per_price_unit=0.8,
        )
    )

    account, metadata, quote = fetch_broker_snapshot(
        cfg,
        service=service,
        now=now,
    )

    assert account.account_id == "ABC123"
    assert metadata.epic == quote.epic == cfg.ig.epic
    assert fetch_open_position_ids(service) == ("D1", "D2")


def test_position_ids_parse_without_a_second_broker_read():
    nested = {
        "positions": [
            {"position": {"dealId": "D1"}},
            {"position": {"dealId": "D2"}},
        ]
    }

    assert open_position_ids_from_response(nested) == ("D1", "D2")
    assert open_position_ids_from_response(pd.DataFrame({"dealId": []})) == ()
