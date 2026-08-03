"""IG Markets API access — epic lookup, executable quotes, and account metadata.

IG is the *execution* broker for this project, and its historical price
endpoint is deliberately not used for bulk training data: the REST API allows
only **10,000 historical price points per week**, which resets weekly. Step 1
needs roughly 40,000 hourly bars, so a full training pull would take about a
month. Training history therefore comes from Dukascopy (see dukascopy.py),
and IG is used for:

  * finding the epic (IG's market identifier — not a ticker symbol),
  * pulling a recent window to verify IG's prices agree with Dukascopy's,
  * fetching the latest bars at prediction time,
  * supplying account/market facts for fail-closed paper sizing, and
  * placing paper/live orders in a later step.

`fetch_candles` therefore *estimates the cost before spending it* and refuses
to silently burn the weekly allowance. Every response's remaining allowance is
printed so the budget is always visible.

Credentials come from .env: IG_API_KEY, IG_USERNAME, IG_PASSWORD, IG_ACC_TYPE.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

import numpy as np
import pandas as pd

from commodity_ml.config import DataConfig
from commodity_ml.data.schema import GRANULARITY_TO_IG, validate_candles

SOURCE = "ig"

# IG's published REST limit; see labs.ig.com FAQ. Used only to warn before spending.
WEEKLY_HISTORICAL_ALLOWANCE = 10_000


def create_service():
    """Construct an unauthenticated IG service after validating credentials."""
    from trading_ig import IGService

    missing = [
        var
        for var in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD")
        if not os.environ.get(var)
    ]
    if missing:
        raise RuntimeError(
            f"Missing IG credentials in environment: {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in — create the API key from "
            "IG's web platform under My Account -> Settings -> API keys "
            "(use a DEMO account key while developing)."
        )

    return IGService(
        username=os.environ["IG_USERNAME"],
        password=os.environ["IG_PASSWORD"],
        api_key=os.environ["IG_API_KEY"],
        acc_type=os.environ.get("IG_ACC_TYPE", "demo"),
        acc_number=os.environ.get("IG_ACC_NUMBER") or None,
        use_rate_limiter=True,  # trading-ig throttles us below IG's per-minute cap
    )


def get_service():
    """Authenticate against IG and return a live IGService."""
    service = create_service()
    service.create_session()
    return service


def search_epics(service, search_term: str) -> pd.DataFrame:
    """Find candidate epics by name. IG epics look like 'CS.D.USCGC.TODAY.IP'
    and vary by account region, so they must be looked up, never guessed."""
    results = service.search_markets(search_term)
    columns = [
        column
        for column in (
            "epic",
            "instrumentName",
            "instrumentType",
            "expiry",
            "marketStatus",
            "bid",
            "offer",
            "delayTime",
            "streamingPricesAvailable",
        )
        if column in results
    ]
    return results[columns]


def estimate_points(granularity: str, start: str, end: str | None = None) -> int:
    """Roughly how many price points a date range will cost against the allowance."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    hours = (end_ts - start_ts).total_seconds() / 3600
    per_hour = {"M5": 12, "M15": 4, "M30": 2, "H1": 1, "H4": 0.25, "D": 1 / 24}[granularity]
    return int(hours * per_hour * (5 / 7))  # markets closed at weekends


def fetch_candles(
    cfg: DataConfig,
    granularity: str,
    service=None,
    allow_large_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch mid plus bid/ask candles, guarding the weekly allowance.

    Raises before making the request if the range would exceed the weekly
    allowance, unless allow_large_download=True.
    """
    if not cfg.ig.epic:
        raise ValueError(
            "No IG epic configured. Find it with:\n"
            '    python scripts/find_epic.py --search "spot gold"\n'
            "then set data.ig.epic in your config yaml."
        )
    if granularity not in GRANULARITY_TO_IG:
        raise ValueError(f"Unsupported granularity for IG: {granularity}")

    estimated = estimate_points(granularity, cfg.start, cfg.end)
    if estimated > WEEKLY_HISTORICAL_ALLOWANCE and not allow_large_download:
        raise RuntimeError(
            f"This request needs roughly {estimated:,} price points, but IG allows only "
            f"{WEEKLY_HISTORICAL_ALLOWANCE:,} per week.\n"
            "Use Dukascopy for training history (data.source: dukascopy) and keep IG for "
            "execution and recent bars. Pass allow_large_download=True to override."
        )

    service = service or get_service()
    if verbose:
        print(f"Fetching {cfg.ig.epic} {granularity} from IG "
              f"(~{estimated:,} points of the {WEEKLY_HISTORICAL_ALLOWANCE:,}/week allowance)")

    request = {
        "epic": cfg.ig.epic,
        "resolution": GRANULARITY_TO_IG[granularity],
        "start_date": pd.Timestamp(cfg.start).strftime("%Y-%m-%dT%H:%M:%S"),
        "end_date": (
            pd.Timestamp(cfg.end) if cfg.end else pd.Timestamp.utcnow()
        ).strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # trading-ig's default formatter indexes the local ``snapshotTime``. Its
    # flat formatter uses ``snapshotTimeUTC``, eliminating a DST ambiguity.
    if hasattr(service, "flat_prices"):
        request["format"] = service.flat_prices
    response = service.fetch_historical_prices_by_epic(**request)

    if verbose and isinstance(response, dict) and "allowance" in response:
        allowance = response["allowance"]
        print(f"  allowance remaining: {allowance.get('remainingAllowance'):,} "
              f"of {allowance.get('totalAllowance'):,} "
              f"(resets in {allowance.get('allowanceExpiry')}s)")

    return _to_canonical(response["prices"])


def fetch_recent_candles(
    cfg: DataConfig,
    granularity: str,
    *,
    num_points: int = 500,
    service=None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch a bounded recent quote window for parity/paper diagnostics."""

    if not cfg.ig.epic:
        raise ValueError("No data.ig.epic configured")
    if granularity not in GRANULARITY_TO_IG:
        raise ValueError(f"Unsupported granularity for IG: {granularity}")
    if isinstance(num_points, bool) or not isinstance(num_points, int) or num_points < 2:
        raise ValueError("num_points must be an integer >= 2")
    if num_points > WEEKLY_HISTORICAL_ALLOWANCE:
        raise ValueError(
            f"num_points cannot exceed the {WEEKLY_HISTORICAL_ALLOWANCE:,} weekly allowance"
        )

    service = service or get_service()
    if verbose:
        print(
            f"Fetching latest {num_points:,} {granularity} quote candles for "
            f"{cfg.ig.epic} from IG"
        )
    request = {
        "epic": cfg.ig.epic,
        "resolution": GRANULARITY_TO_IG[granularity],
        "numpoints": num_points,
    }
    if hasattr(service, "flat_prices"):
        request["format"] = service.flat_prices
    response = service.fetch_historical_prices_by_epic(**request)
    return _to_canonical(response["prices"])


def _to_canonical(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert trading-ig's flat or hierarchical frame without losing quotes."""

    if isinstance(prices.columns, pd.MultiIndex):
        bid, ask = prices["bid"], prices["ask"]
        quote_values = {
            f"bid_{field.lower()}": bid[field]
            for field in ("Open", "High", "Low", "Close")
        } | {
            f"ask_{field.lower()}": ask[field]
            for field in ("Open", "High", "Low", "Close")
        }
    else:
        candidates = {
            f"{side}_{field}": (f"{field}.{side}", f"{side}_{field}")
            for side in ("bid", "ask")
            for field in ("open", "high", "low", "close")
        }
        quote_values = {}
        for canonical, options in candidates.items():
            column = next((name for name in options if name in prices.columns), None)
            if column is None:
                raise ValueError(
                    "IG prices missing executable quote column for "
                    f"{canonical}; got {list(prices.columns)}"
                )
            quote_values[canonical] = prices[column]

    frame = pd.DataFrame(quote_values, index=prices.index)
    for field in ("open", "high", "low", "close"):
        frame[field] = (frame[f"bid_{field}"] + frame[f"ask_{field}"]) / 2

    # Spot gold is a CFD with no real traded volume; IG reports it when available.
    if "last" in prices.columns.get_level_values(0) and "Volume" in prices["last"].columns:
        frame["volume"] = prices["last"]["Volume"]
    elif "volume" in prices.columns:
        frame["volume"] = prices["volume"]
    elif "Volume" in prices.columns:
        frame["volume"] = prices["Volume"]
    else:
        frame["volume"] = 0.0

    ordered = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
    ]
    return validate_candles(frame[ordered], SOURCE)


@dataclass(frozen=True)
class IGAccountSnapshot:
    account_id: str
    currency: str
    account_type: str
    status: str
    balance: float
    open_pnl: float
    equity: float
    available: float

    def __post_init__(self) -> None:
        if not self.account_id or not self.currency:
            raise ValueError("IG account id and currency must be non-empty")
        for name in ("balance", "open_pnl", "equity", "available"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"IG account {name} must be finite")
        if not np.isclose(self.equity, self.balance + self.open_pnl):
            raise ValueError("IG account equity must equal balance plus open P&L")


@dataclass(frozen=True)
class IGMarketQuote:
    epic: str
    bid: float
    ask: float
    timestamp: pd.Timestamp
    market_status: str
    delay_seconds: int

    def __post_init__(self) -> None:
        if not self.epic:
            raise ValueError("IG quote epic must be non-empty")
        if not np.isfinite(self.bid) or not np.isfinite(self.ask):
            raise ValueError("IG bid/ask must be finite")
        if self.bid <= 0 or self.ask < self.bid:
            raise ValueError("IG bid/ask must be positive and non-crossed")
        if self.timestamp.tzinfo is None:
            raise ValueError("IG quote timestamp must be timezone-aware")
        if self.delay_seconds < 0:
            raise ValueError("IG quote delay cannot be negative")

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
        now = _utc_now(now)
        return now - self.timestamp

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
        if age < pd.Timedelta(0):
            raise RuntimeError("quote_timestamp_in_future")
        if age > max_age:
            raise RuntimeError(f"stale_quote:{age}")


@dataclass(frozen=True)
class IGMarketMetadata:
    epic: str
    name: str
    instrument_type: str
    unit: str
    expiry: str
    account_currency: str
    min_size: float
    size_increment: float
    value_per_price_unit: float
    contract_size: float | None
    lot_size: float | None
    streaming_prices_available: bool

    def __post_init__(self) -> None:
        if not self.epic or not self.account_currency:
            raise ValueError("IG metadata epic and account currency must be non-empty")
        for name in ("min_size", "size_increment", "value_per_price_unit"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"IG metadata {name} must be finite and positive")

    def notional_per_size_unit(self, price: float) -> float:
        """Account-currency notional for one broker size unit."""

        if not np.isfinite(price) or price <= 0:
            raise ValueError("price must be finite and positive")
        return price * self.value_per_price_unit


def fetch_account_snapshot(
    service,
    account_id: str | None = None,
) -> IGAccountSnapshot:
    """Read active-account equity, including open P&L, without guessing."""

    accounts = service.fetch_accounts()
    frame = _accounts_frame(accounts)
    requested = account_id or os.environ.get("IG_ACC_NUMBER")
    if requested:
        selected = frame.loc[frame["accountId"].astype(str) == str(requested)]
        if len(selected) != 1:
            raise RuntimeError(f"IG account {requested!r} was not found exactly once")
    else:
        preferred = (
            frame.loc[frame["preferred"].astype(bool)]
            if "preferred" in frame.columns
            else frame.iloc[0:0]
        )
        if len(preferred) == 1:
            selected = preferred
        elif len(frame) == 1:
            selected = frame
        else:
            raise RuntimeError(
                "Multiple IG accounts are available; set IG_ACC_NUMBER explicitly"
            )

    row = selected.iloc[0]
    balance = _finite_float(row.get("balance"), "account balance")
    open_pnl = _finite_float(row.get("profitLoss", 0.0), "account open P&L")
    available = _finite_float(row.get("available"), "account available funds")
    currency = str(row.get("currency") or row.get("currencyIsoCode") or "").upper()
    if not currency:
        raise RuntimeError("IG account response has no currency")

    return IGAccountSnapshot(
        account_id=str(row["accountId"]),
        currency=currency,
        account_type=str(row.get("accountType", "")),
        status=str(row.get("status", "")),
        balance=balance,
        open_pnl=open_pnl,
        equity=balance + open_pnl,
        available=available,
    )


def fetch_market_details(service, epic: str) -> Mapping[str, Any]:
    """Fetch one account-specific market description from IG."""

    if not epic:
        raise ValueError("epic must be non-empty")
    return _mapping(service.fetch_market_by_epic(epic))


def market_quote_from_details(
    details: Mapping[str, Any],
    *,
    now: pd.Timestamp | None = None,
) -> IGMarketQuote:
    """Parse the executable top-of-book quote from a market response."""

    instrument = _mapping(details.get("instrument", {}))
    snapshot = _mapping(details.get("snapshot", {}))
    epic = str(instrument.get("epic") or details.get("epic") or "")
    if not epic:
        raise RuntimeError("IG market response has no epic")
    bid = _finite_float(snapshot.get("bid"), "market bid")
    ask = _finite_float(snapshot.get("ask", snapshot.get("offer")), "market ask")
    if ask < bid:
        raise RuntimeError("crossed_market_quote")

    timestamp = _quote_timestamp(snapshot, now=now)
    delay = int(_finite_float(snapshot.get("delayTime", 0), "quote delay"))
    return IGMarketQuote(
        epic=epic,
        bid=bid,
        ask=ask,
        timestamp=timestamp,
        market_status=str(snapshot.get("marketStatus", "")).upper(),
        delay_seconds=delay,
    )


def market_metadata_from_details(
    details: Mapping[str, Any],
    *,
    account_currency: str,
    size_increment: float | None,
    value_per_price_unit: float | None,
) -> IGMarketMetadata:
    """Parse unambiguous broker facts and require confirmed sizing semantics."""

    instrument = _mapping(details.get("instrument", {}))
    rules = _mapping(details.get("dealingRules", {}))
    snapshot = _mapping(details.get("snapshot", {}))
    min_deal = _mapping(rules.get("minDealSize", {}))

    epic = str(instrument.get("epic") or details.get("epic") or "")
    if not epic:
        raise RuntimeError("IG market response has no epic")
    if size_increment is None:
        raise RuntimeError(
            "missing_size_increment: confirm the demo order-ticket size step and "
            "set data.ig.size_increment"
        )
    if value_per_price_unit is None:
        raise RuntimeError(
            "missing_value_per_price_unit: confirm account-currency P&L for one "
            "price-point move at size 1 and set data.ig.value_per_price_unit"
        )

    return IGMarketMetadata(
        epic=epic,
        name=str(instrument.get("name", "")),
        instrument_type=str(instrument.get("type", "")).upper(),
        unit=str(instrument.get("unit", "")).upper(),
        expiry=str(instrument.get("expiry", "-")),
        account_currency=account_currency.upper(),
        min_size=_finite_float(min_deal.get("value"), "minimum deal size"),
        size_increment=_positive_float(size_increment, "size increment"),
        value_per_price_unit=_positive_float(
            value_per_price_unit,
            "value per price unit",
        ),
        contract_size=_optional_float(instrument.get("contractSize")),
        lot_size=_optional_float(instrument.get("lotSize")),
        streaming_prices_available=bool(
            instrument.get("streamingPricesAvailable", False)
        ),
    )


def fetch_broker_snapshot(
    cfg: DataConfig,
    *,
    service=None,
    now: pd.Timestamp | None = None,
) -> tuple[IGAccountSnapshot, IGMarketMetadata, IGMarketQuote]:
    """Fetch account, market, and quote facts as one paper-decision snapshot."""

    if not cfg.ig.epic:
        raise ValueError("No data.ig.epic configured")
    service = service or get_service()
    account = fetch_account_snapshot(service)
    details = fetch_market_details(service, cfg.ig.epic)
    quote = market_quote_from_details(details, now=now)
    metadata = market_metadata_from_details(
        details,
        account_currency=account.currency,
        size_increment=cfg.ig.size_increment,
        value_per_price_unit=cfg.ig.value_per_price_unit,
    )
    if metadata.epic != cfg.ig.epic or quote.epic != cfg.ig.epic:
        raise RuntimeError("IG epic mismatch across configured market and response")
    return account, metadata, quote


def fetch_open_position_ids(service) -> tuple[str, ...]:
    """Return broker deal IDs for restart/reconciliation checks."""

    positions = service.fetch_open_positions()
    if isinstance(positions, pd.DataFrame):
        if positions.empty:
            return ()
        if "dealId" not in positions.columns:
            raise RuntimeError("IG open-position response has no dealId column")
        ids = positions["dealId"].astype(str).tolist()
    else:
        raw = _mapping(positions)
        ids = [
            str(_mapping(_mapping(item).get("position", {})).get("dealId", ""))
            for item in raw.get("positions", [])
        ]
    if any(not deal_id for deal_id in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("IG returned missing or duplicate open-position deal IDs")
    return tuple(ids)


def _accounts_frame(accounts: Any) -> pd.DataFrame:
    if isinstance(accounts, pd.DataFrame):
        frame = accounts.copy()
    else:
        raw = _mapping(accounts)
        frame = pd.DataFrame(raw.get("accounts", []))
        if "balance" in frame.columns and not frame.empty:
            expanded = pd.json_normalize(frame.pop("balance"))
            for column in expanded.columns:
                frame[column] = expanded[column].to_numpy()
    if frame.empty or "accountId" not in frame.columns:
        raise RuntimeError("IG returned no selectable accounts")
    return frame


def _quote_timestamp(
    snapshot: Mapping[str, Any],
    *,
    now: pd.Timestamp | None,
) -> pd.Timestamp:
    current = _utc_now(now)
    epoch = snapshot.get("updateTimestampUTC")
    if epoch is not None:
        value = _finite_float(epoch, "quote timestamp")
        unit = "ms" if value > 10_000_000_000 else "s"
        return pd.Timestamp(value, unit=unit, tz="UTC")

    raw = snapshot.get("updateTimeUTC")
    if raw is None:
        raise RuntimeError("market quote has no unambiguous UTC timestamp")
    text = str(raw)
    if "T" in text or "-" in text:
        parsed = pd.Timestamp(text)
        return parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")

    parsed_time = pd.Timestamp(text).time()
    candidate = pd.Timestamp.combine(current.date(), parsed_time).tz_localize("UTC")
    if candidate > current + pd.Timedelta(minutes=1):
        candidate -= pd.Timedelta(days=1)
    return candidate


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    try:
        return dict(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Expected mapping-like IG response, got {type(value)}") from exc


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"IG {name} is missing or non-numeric: {value!r}") from exc
    if not np.isfinite(number):
        raise RuntimeError(f"IG {name} is non-finite")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0:
        raise RuntimeError(f"IG {name} must be positive")
    return number


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return _finite_float(value, "optional market field")


def _utc_now(value: pd.Timestamp | None) -> pd.Timestamp:
    current = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    if current.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return current.tz_convert("UTC")
