"""OANDA v20 practice/live API access for executable metals data.

Dukascopy remains the deep research source. OANDA supplies account-specific
instrument contracts, recent bid/ask candles, top-of-book prices, positions,
and eventually paper/live execution. No function in this module submits an
order.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import requests

from commodity_ml.config import DataConfig
from commodity_ml.data.broker import (
    BrokerAccountSnapshot,
    BrokerInstrumentMetadata,
    BrokerMarketQuote,
)
from commodity_ml.data.schema import GRANULARITY_TO_OANDA, validate_candles

SOURCE = "oanda"
MAX_CANDLES_PER_REQUEST = 5_000
REST_BASES = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}
STREAM_BASES = {
    "practice": "https://stream-fxpractice.oanda.com",
    "live": "https://stream-fxtrade.oanda.com",
}


class OANDAClient:
    """Small explicit v20 client with bounded timeouts and sanitized errors."""

    def __init__(
        self,
        *,
        api_key: str,
        account_id: str,
        environment: str = "practice",
        session: requests.Session | None = None,
        timeout_seconds: float = 20.0,
    ) -> None:
        environment = environment.lower().strip()
        if environment not in REST_BASES:
            raise ValueError("OANDA environment must be practice or live")
        if not api_key or not account_id:
            raise ValueError("OANDA API key and account id must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("OANDA timeout must be positive")
        self.account_id = account_id
        self.environment = environment
        self.rest_base = REST_BASES[environment]
        self.stream_base = STREAM_BASES[environment]
        self.timeout_seconds = float(timeout_seconds)
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Accept-Datetime-Format": "RFC3339",
                "Content-Type": "application/json",
                "User-Agent": "commodity-ml/0.1",
            }
        )

    def account_summary(self) -> Mapping[str, Any]:
        return _mapping(self._get(f"/v3/accounts/{self.account_id}/summary").get("account"))

    def account_instruments(self, instruments: Iterable[str]) -> list[Mapping[str, Any]]:
        names = _instrument_list(instruments)
        payload = self._get(
            f"/v3/accounts/{self.account_id}/instruments",
            params={"instruments": ",".join(names)},
        )
        return [_mapping(item) for item in payload.get("instruments", [])]

    def pricing(self, instruments: Iterable[str]) -> Mapping[str, Any]:
        names = _instrument_list(instruments)
        return self._get(
            f"/v3/accounts/{self.account_id}/pricing",
            params={
                "instruments": ",".join(names),
                "includeHomeConversions": "true",
            },
        )

    def candles(
        self,
        instrument: str,
        granularity: str,
        *,
        count: int,
    ) -> Mapping[str, Any]:
        if count < 2 or count > MAX_CANDLES_PER_REQUEST:
            raise ValueError(f"OANDA candle count must be between 2 and {MAX_CANDLES_PER_REQUEST}")
        return self._get(
            f"/v3/accounts/{self.account_id}/instruments/{instrument}/candles",
            params={"price": "MBA", "granularity": granularity, "count": count},
        )

    def candles_range(
        self,
        instrument: str,
        granularity: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Mapping[str, Any]:
        return self._get(
            f"/v3/accounts/{self.account_id}/instruments/{instrument}/candles",
            params={
                "price": "MBA",
                "granularity": granularity,
                "from": _rfc3339(start),
                "to": _rfc3339(end),
            },
        )

    def open_trades(self) -> list[Mapping[str, Any]]:
        payload = self._get(f"/v3/accounts/{self.account_id}/openTrades")
        return [_mapping(item) for item in payload.get("trades", [])]

    def open_pricing_stream(self, instruments: Iterable[str]):
        names = _instrument_list(instruments)
        try:
            response = self.session.get(
                f"{self.stream_base}/v3/accounts/{self.account_id}/pricing/stream",
                params={"instruments": ",".join(names)},
                timeout=(self.timeout_seconds, self.timeout_seconds + 70),
                stream=True,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OANDA pricing stream unavailable: {type(exc).__name__}") from exc
        _raise_for_status(response)
        return response

    def _get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        try:
            response = self.session.get(
                f"{self.rest_base}{path}",
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"OANDA REST request unavailable: {type(exc).__name__}") from exc
        _raise_for_status(response)
        try:
            return _mapping(response.json())
        except (json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError("OANDA returned invalid JSON") from exc


def get_client() -> OANDAClient:
    api_key = os.environ.get("OANDA_API_KEY") or os.environ.get("OANDA_API_TOKEN")
    account_id = os.environ.get("OANDA_ACCOUNT_ID")
    missing = []
    if not api_key:
        missing.append("OANDA_API_KEY")
    if not account_id:
        missing.append("OANDA_ACCOUNT_ID")
    if missing:
        raise RuntimeError(
            "Missing OANDA credentials in environment: " + ", ".join(missing)
        )
    return OANDAClient(
        api_key=api_key,
        account_id=account_id,
        environment=os.environ.get("OANDA_ENV", "practice"),
    )


def fetch_account_snapshot(
    client: OANDAClient,
    *,
    account_type: str,
) -> BrokerAccountSnapshot:
    raw = client.account_summary()
    account_id = str(raw.get("id") or "")
    if account_id != client.account_id:
        raise RuntimeError("OANDA account response id does not match configured account")
    balance = _finite_float(raw.get("balance"), "account balance")
    open_pnl = _finite_float(raw.get("unrealizedPL", 0), "account unrealized P&L")
    nav = _finite_float(raw.get("NAV"), "account NAV")
    if not np.isclose(nav, balance + open_pnl, atol=1e-4):
        raise RuntimeError("OANDA NAV does not equal balance plus unrealized P&L")
    hedging = raw.get("hedgingEnabled")
    if hedging is not None and not isinstance(hedging, bool):
        raise RuntimeError("OANDA hedgingEnabled must be a boolean when supplied")
    return BrokerAccountSnapshot(
        account_id=account_id,
        currency=str(raw.get("currency") or "").upper(),
        account_type=account_type.upper(),
        status="ENABLED",
        balance=balance,
        open_pnl=open_pnl,
        equity=nav,
        available=_finite_float(raw.get("marginAvailable"), "available margin"),
        hedging_enabled=hedging,
    )


def quotes_from_pricing(payload: Mapping[str, Any]) -> dict[str, BrokerMarketQuote]:
    output: dict[str, BrokerMarketQuote] = {}
    for raw in payload.get("prices", []):
        item = _mapping(raw)
        instrument = str(item.get("instrument") or "")
        bids = item.get("bids") or []
        asks = item.get("asks") or []
        if not bids or not asks:
            raise RuntimeError(f"OANDA price for {instrument or 'unknown'} has no top of book")
        if instrument in output:
            raise RuntimeError(f"OANDA returned duplicate price for {instrument}")
        output[instrument] = BrokerMarketQuote(
            instrument=instrument,
            bid=_finite_float(_mapping(bids[0]).get("price"), "top bid"),
            ask=_finite_float(_mapping(asks[0]).get("price"), "top ask"),
            timestamp=_timestamp(item.get("time"), "price time"),
            market_status=str(item.get("status") or "").upper(),
            delay_seconds=0,
        )
    return output


def instrument_metadata_from_responses(
    instrument: Mapping[str, Any],
    pricing: Mapping[str, Any],
    *,
    account_currency: str,
) -> BrokerInstrumentMetadata:
    name = str(instrument.get("name") or "")
    if not name:
        raise RuntimeError("OANDA instrument response has no name")
    quote_currency = name.rsplit("_", 1)[-1].upper()
    conversion = _home_conversion(
        pricing,
        quote_currency=quote_currency,
        account_currency=account_currency,
        label=str(instrument.get("displayName") or name),
    )
    raw_precision = _finite_float(instrument.get("tradeUnitsPrecision"), "trade precision")
    if not raw_precision.is_integer():
        raise RuntimeError("OANDA trade unit precision must be an integer")
    precision = int(raw_precision)
    if precision < 0 or precision > 12:
        raise RuntimeError("OANDA trade unit precision is outside supported range")
    minimum = _positive_float(instrument.get("minimumTradeSize"), "minimum trade size")
    increment = 10.0 ** (-precision)
    if minimum < increment and not np.isclose(minimum, increment):
        raise RuntimeError("OANDA minimum trade size is below its declared unit precision")
    maximum = _optional_positive(instrument.get("maximumOrderUnits"))
    return BrokerInstrumentMetadata(
        instrument=name,
        name=str(instrument.get("displayName") or name),
        instrument_type=str(instrument.get("type") or "").upper(),
        unit="UNITS",
        expiry="-",
        account_currency=account_currency.upper(),
        min_size=minimum,
        size_increment=increment,
        # accountLoss is deliberately used for stop-risk: it is the adverse,
        # conservative conversion from quote currency into account currency.
        value_per_price_unit=_positive_float(conversion.get("accountLoss"), "account loss conversion"),
        notional_value_per_price_unit=_positive_float(
            conversion.get("positionValue"), "position value conversion"
        ),
        margin_rate=_positive_float(instrument.get("marginRate"), "instrument margin rate"),
        maximum_order_size=maximum,
        streaming_prices_available=True,
    )


def fetch_broker_snapshot(
    cfg: DataConfig,
    *,
    client: OANDAClient | None = None,
) -> tuple[BrokerAccountSnapshot, BrokerInstrumentMetadata, BrokerMarketQuote]:
    if not cfg.oanda.instrument:
        raise ValueError("No data.oanda.instrument configured")
    client = client or get_client()
    account = fetch_account_snapshot(client, account_type=cfg.oanda.account_type)
    instruments = client.account_instruments([cfg.oanda.instrument])
    if len(instruments) != 1:
        raise RuntimeError("OANDA did not return exactly one configured instrument")
    pricing = client.pricing([cfg.oanda.instrument])
    quotes = quotes_from_pricing(pricing)
    quote = quotes.get(cfg.oanda.instrument)
    if quote is None:
        raise RuntimeError("OANDA pricing omitted the configured instrument")
    metadata = instrument_metadata_from_responses(
        instruments[0], pricing, account_currency=account.currency
    )
    if metadata.instrument != quote.instrument:
        raise RuntimeError("OANDA instrument mismatch across metadata and price")
    return account, metadata, quote


def fetch_recent_candles(
    cfg: DataConfig,
    granularity: str,
    *,
    num_points: int = 500,
    client: OANDAClient | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    instrument = _configured_instrument(cfg)
    code = _granularity(granularity)
    client = client or get_client()
    if verbose:
        print(f"Fetching latest {num_points:,} {granularity} quote candles for {instrument} from OANDA")
    return _to_canonical(client.candles(instrument, code, count=num_points))


def fetch_candles(
    cfg: DataConfig,
    granularity: str,
    *,
    client: OANDAClient | None = None,
    allow_large_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    instrument = _configured_instrument(cfg)
    code = _granularity(granularity)
    start = _timestamp(cfg.start, "start")
    end = _timestamp(cfg.end, "end") if cfg.end else pd.Timestamp.now(tz="UTC")
    estimate = _estimated_candles(start, end, granularity)
    if estimate > MAX_CANDLES_PER_REQUEST and not allow_large_download:
        raise RuntimeError(
            f"OANDA range needs roughly {estimate:,} candles; bounded broker pulls "
            f"are capped at {MAX_CANDLES_PER_REQUEST:,}. Use Dukascopy for research "
            "history or pass allow_large_download=True and implement an explicit plan."
        )
    if estimate > MAX_CANDLES_PER_REQUEST:
        raise RuntimeError("multi-page OANDA history is intentionally not implemented")
    client = client or get_client()
    if verbose:
        print(f"Fetching {instrument} {granularity} from OANDA ({start} -> {end})")
    return _to_canonical(client.candles_range(instrument, code, start=start, end=end))


def fetch_open_position_ids(client: OANDAClient) -> tuple[str, ...]:
    return open_trade_ids_from_response(client.open_trades())


def open_trade_ids_from_response(trades: Any) -> tuple[str, ...]:
    rows = trades if isinstance(trades, list) else _mapping(trades).get("trades", [])
    ids = [str(_mapping(item).get("id") or "") for item in rows]
    if any(not trade_id for trade_id in ids) or len(ids) != len(set(ids)):
        raise RuntimeError("OANDA returned missing or duplicate open-trade ids")
    return tuple(ids)


def _to_canonical(payload: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    index: list[pd.Timestamp] = []
    for raw in payload.get("candles", []):
        candle = _mapping(raw)
        mid, bid, ask = (_mapping(candle.get(side)) for side in ("mid", "bid", "ask"))
        if not mid or not bid or not ask:
            raise RuntimeError("OANDA candle omitted midpoint or bid/ask OHLC")
        row: dict[str, Any] = {"volume": candle.get("volume", 0)}
        for field, short in (("open", "o"), ("high", "h"), ("low", "l"), ("close", "c")):
            row[field] = mid.get(short)
            row[f"bid_{field}"] = bid.get(short)
            row[f"ask_{field}"] = ask.get(short)
        rows.append(row)
        index.append(_timestamp(candle.get("time"), "candle time"))
    if not rows:
        raise RuntimeError("OANDA returned no candles")
    return validate_candles(pd.DataFrame(rows, index=pd.DatetimeIndex(index)), SOURCE)


def _home_conversion(
    payload: Mapping[str, Any],
    *,
    quote_currency: str,
    account_currency: str,
    label: str,
) -> Mapping[str, Any]:
    if quote_currency == account_currency.upper():
        return {"accountGain": 1.0, "accountLoss": 1.0, "positionValue": 1.0}
    matches = [
        _mapping(item)
        for item in payload.get("homeConversions", [])
        if str(_mapping(item).get("currency") or "").upper() == quote_currency
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"OANDA pricing lacks one {quote_currency} home conversion for {label}"
        )
    return matches[0]


def _configured_instrument(cfg: DataConfig) -> str:
    value = cfg.oanda.instrument
    if not value:
        raise ValueError("No data.oanda.instrument configured")
    return value


def _granularity(value: str) -> str:
    if value not in GRANULARITY_TO_OANDA:
        raise ValueError(f"Unsupported granularity for OANDA: {value}")
    return GRANULARITY_TO_OANDA[value]


def _instrument_list(values: Iterable[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(value).strip() for value in values))
    if not names or any(not value for value in names):
        raise ValueError("OANDA instruments must be non-empty")
    return names


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        payload = response.json()
        message = payload.get("errorMessage") or payload.get("errorCode")
    except (ValueError, json.JSONDecodeError):
        message = None
    raise RuntimeError(f"OANDA request failed ({response.status_code}): {message or 'no safe detail'}")


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    raise RuntimeError(f"Expected mapping-like OANDA response, got {type(value)}")


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"OANDA {name} is missing or non-numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OANDA {name} is missing or non-numeric") from exc
    if not np.isfinite(number):
        raise RuntimeError(f"OANDA {name} is non-finite")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite_float(value, name)
    if number <= 0:
        raise RuntimeError(f"OANDA {name} must be positive")
    return number


def _optional_positive(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return _positive_float(value, "optional positive field")


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"OANDA {name} is missing or invalid") from exc
    if pd.isna(timestamp):
        raise RuntimeError(f"OANDA {name} is missing or invalid")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _rfc3339(value: pd.Timestamp) -> str:
    return _timestamp(value, "request timestamp").isoformat().replace("+00:00", "Z")


def _estimated_candles(start: pd.Timestamp, end: pd.Timestamp, granularity: str) -> int:
    seconds = {"M5": 300, "M15": 900, "M30": 1800, "H1": 3600, "H4": 14400, "D": 86400}[granularity]
    return max(0, int((end - start).total_seconds() / seconds * 5 / 7))
