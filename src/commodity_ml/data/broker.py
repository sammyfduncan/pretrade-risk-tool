"""Broker-neutral account, instrument and executable-quote contracts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrokerAccountSnapshot:
    account_id: str
    currency: str
    account_type: str
    status: str
    balance: float
    open_pnl: float
    equity: float
    available: float
    hedging_enabled: bool | None = None

    def __post_init__(self) -> None:
        if not self.account_id or not self.currency:
            raise ValueError("broker account id and currency must be non-empty")
        for name in ("balance", "open_pnl", "equity", "available"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"broker account {name} must be finite")
        if not np.isclose(self.equity, self.balance + self.open_pnl, atol=1e-6):
            raise ValueError("broker account equity must equal balance plus open P&L")
        if self.hedging_enabled is not None and not isinstance(self.hedging_enabled, bool):
            raise ValueError("broker account hedging_enabled must be boolean or None")


@dataclass(frozen=True)
class BrokerMarketQuote:
    instrument: str
    bid: float
    ask: float
    timestamp: pd.Timestamp
    market_status: str
    delay_seconds: int = 0

    def __post_init__(self) -> None:
        if not self.instrument:
            raise ValueError("broker quote instrument must be non-empty")
        if not np.isfinite(self.bid) or not np.isfinite(self.ask):
            raise ValueError("broker bid/ask must be finite")
        if self.bid <= 0 or self.ask < self.bid:
            raise ValueError("broker bid/ask must be positive and non-crossed")
        if self.timestamp.tzinfo is None:
            raise ValueError("broker quote timestamp must be timezone-aware")
        if self.delay_seconds < 0:
            raise ValueError("broker quote delay cannot be negative")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_fraction(self) -> float:
        return self.spread / self.mid

    def age(self, now: pd.Timestamp | None = None) -> pd.Timedelta:
        return _utc_now(now) - self.timestamp

    def validate_for_entry(
        self,
        *,
        now: pd.Timestamp | None = None,
        max_age: pd.Timedelta = pd.Timedelta(seconds=60),
    ) -> None:
        if self.market_status != "TRADEABLE":
            raise RuntimeError(f"market_not_tradeable:{self.market_status}")
        if self.delay_seconds != 0:
            raise RuntimeError(f"delayed_quote:{self.delay_seconds}s")
        age = self.age(now)
        # Broker and local clocks can differ slightly. More than five seconds
        # ahead is suspicious; sub-second/low-second skew is operational noise.
        if age < -pd.Timedelta(seconds=5):
            raise RuntimeError("quote_timestamp_in_future")
        if age > max_age:
            raise RuntimeError(f"stale_quote:{age}")


@dataclass(frozen=True)
class BrokerInstrumentMetadata:
    instrument: str
    name: str
    instrument_type: str
    unit: str
    expiry: str
    account_currency: str
    min_size: float
    size_increment: float
    value_per_price_unit: float
    notional_value_per_price_unit: float
    margin_rate: float
    maximum_order_size: float | None
    streaming_prices_available: bool

    def __post_init__(self) -> None:
        if not self.instrument or not self.account_currency:
            raise ValueError("broker instrument and account currency must be non-empty")
        for name in (
            "min_size",
            "size_increment",
            "value_per_price_unit",
            "notional_value_per_price_unit",
            "margin_rate",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"broker metadata {name} must be finite and positive")
        if self.margin_rate > 1:
            raise ValueError("broker margin rate must be a decimal fraction <= 1")
        if self.maximum_order_size is not None and (
            not np.isfinite(self.maximum_order_size) or self.maximum_order_size <= 0
        ):
            raise ValueError("broker maximum order size must be finite and positive")
        if not isinstance(self.streaming_prices_available, bool):
            raise ValueError("broker streaming-prices flag must be boolean")

    def notional_per_size_unit(self, price: float) -> float:
        if not np.isfinite(price) or price <= 0:
            raise ValueError("price must be finite and positive")
        return price * self.notional_value_per_price_unit


def _utc_now(value: pd.Timestamp | None) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current.tz_convert("UTC")
