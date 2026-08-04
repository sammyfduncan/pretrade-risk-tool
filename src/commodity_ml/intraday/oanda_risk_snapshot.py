"""Build a fail-closed pre-trade payload from read-only OANDA v20 facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from commodity_ml.data.broker import (
    BrokerAccountSnapshot,
    BrokerInstrumentMetadata,
    BrokerMarketQuote,
)


@dataclass(frozen=True)
class OANDAReadOnlyRiskSnapshot:
    payload: dict[str, Any]
    provenance: dict[str, Any]
    issues: tuple[str, ...]
    order_authority: bool = False


def recent_same_session_spreads(
    candles: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    minimum_rows: int = 8,
) -> tuple[float | None, float | None, int]:
    """Measure closed M30 bid/ask spreads in the current London entry session."""

    required = {"bid_close", "ask_close"}
    if candles.empty or not required.issubset(candles.columns):
        return None, None, 0
    current = _utc(decision_time)
    frame = candles.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        return None, None, 0
    frame.index = index.tz_convert("UTC")
    closed_before = current.floor("30min")
    local_index = frame.index.tz_convert("Europe/London")
    local_now = current.tz_convert("Europe/London")
    minutes = local_index.hour * 60 + local_index.minute
    mask = (
        (frame.index < closed_before)
        & (local_index.date == local_now.date())
        & (minutes >= 7 * 60)
        & (minutes < 19 * 60 + 30)
    )
    spreads = (
        pd.to_numeric(frame.loc[mask, "ask_close"], errors="coerce")
        - pd.to_numeric(frame.loc[mask, "bid_close"], errors="coerce")
    )
    spreads = spreads[np.isfinite(spreads) & (spreads >= 0)]
    if len(spreads) < minimum_rows:
        return None, None, int(len(spreads))
    return float(spreads.median()), float(spreads.quantile(0.95)), int(len(spreads))


def build_oanda_read_only_snapshot(
    *,
    instrument_symbol: str,
    trade_intent: Mapping[str, Any],
    account: BrokerAccountSnapshot,
    metadata: BrokerInstrumentMetadata,
    quote: BrokerMarketQuote,
    recent_candles: pd.DataFrame,
    open_trades: Any,
    metadata_by_instrument: Mapping[str, BrokerInstrumentMetadata] | None = None,
    quotes_by_instrument: Mapping[str, BrokerMarketQuote] | None = None,
    symbol_by_instrument: Mapping[str, str] | None = None,
    day_start_equity: float | None = None,
    week_start_equity: float | None = None,
    entries_today: int | None = None,
    kill_switch_active: bool | None = None,
    scheduled_event_blackout_active: bool | None = None,
    now: pd.Timestamp | None = None,
) -> OANDAReadOnlyRiskSnapshot:
    """Translate authenticated GET responses into the generic risk contract."""

    current = _utc(now if now is not None else quote.timestamp)
    issues: list[str] = ["account_type_confirmed_in_config_not_exposed_by_oanda_v20"]
    if metadata.instrument != quote.instrument:
        issues.append("instrument_mismatch_between_metadata_and_quote")
    if account.hedging_enabled is False:
        issues.append("broker_reports_hedging_disabled")

    median_spread, p95_spread, spread_rows = recent_same_session_spreads(
        recent_candles,
        decision_time=current,
    )
    if median_spread is None:
        issues.append("insufficient_closed_same_session_spread_rows")

    all_metadata = dict(metadata_by_instrument or {})
    all_metadata.setdefault(metadata.instrument, metadata)
    all_quotes = dict(quotes_by_instrument or {})
    all_quotes.setdefault(quote.instrument, quote)
    symbols = dict(symbol_by_instrument or {})
    symbols.setdefault(metadata.instrument, instrument_symbol)
    positions, position_issues = positions_from_oanda(
        open_trades,
        metadata_by_instrument=all_metadata,
        quotes_by_instrument=all_quotes,
        symbol_by_instrument=symbols,
    )
    issues.extend(position_issues)

    intent = dict(trade_intent)
    if intent.get("instrument") in (None, ""):
        intent["instrument"] = instrument_symbol
    if intent.get("decision_time") in (None, ""):
        intent["decision_time"] = current.isoformat()
    if str(intent.get("order_type") or "").lower() == "market":
        supplied = intent.get("entry_price")
        intent["entry_price"] = quote.mid
        if supplied is not None and not np.isclose(float(supplied), quote.mid):
            issues.append("market_entry_reference_replaced_with_current_mid")

    for value, issue in (
        (day_start_equity, "missing_day_start_equity"),
        (week_start_equity, "missing_week_start_equity"),
        (entries_today, "missing_entries_today"),
        (kill_switch_active, "missing_kill_switch_state"),
        (scheduled_event_blackout_active, "missing_current_scheduled_event_state"),
    ):
        if value is None:
            issues.append(issue)

    payload = {
        "trade_intent": intent,
        "account": {
            "account_type": account.account_type,
            "status": account.status,
            "currency": account.currency,
            "equity": account.equity,
            "available_funds": account.available,
            "day_start_equity": day_start_equity,
            "week_start_equity": week_start_equity,
            "entries_today": entries_today,
            "kill_switch_active": kill_switch_active,
        },
        "instrument": {
            "instrument": instrument_symbol,
            "broker_instrument": metadata.instrument,
            "instrument_type": metadata.instrument_type,
            "expiry": metadata.expiry,
            "account_currency": metadata.account_currency,
            "min_size": metadata.min_size,
            "size_increment": metadata.size_increment,
            "value_per_price_unit": metadata.value_per_price_unit,
            "notional_value_per_price_unit": metadata.notional_value_per_price_unit,
            "margin_rate": metadata.margin_rate,
            "margin_rate_max_size": metadata.maximum_order_size,
            "commission_per_size_per_side": 0.0,
        },
        "market": {
            "broker_instrument": quote.instrument,
            "as_of": quote.timestamp.isoformat(),
            "bid": quote.bid,
            "ask": quote.ask,
            "market_open": quote.market_status == "TRADEABLE",
            "quote_delay_seconds": quote.delay_seconds,
            "scheduled_event_blackout_active": scheduled_event_blackout_active,
            "median_spread_points": median_spread,
            "p95_spread_points": p95_spread,
            "forecast_range_points": None,
            "forecast_horizon_minutes": None,
            "volatility_percentile": None,
            "volatility_artifact_approved": False,
        },
        "existing_positions": positions,
    }
    provenance = {
        "mode": "oanda_v20_read_only_snapshot",
        "order_authority": False,
        "account_id_suffix": account.account_id[-4:],
        "broker_instrument": metadata.instrument,
        "quote_timestamp": quote.timestamp.isoformat(),
        "quote_delay_seconds": quote.delay_seconds,
        "spread_sample_rows": spread_rows,
        "margin_source": "account.instrument.marginRate",
        "home_conversion_risk_source": "pricing.homeConversions.accountLoss",
        "home_conversion_notional_source": "pricing.homeConversions.positionValue",
        "open_trade_rows": len(positions),
        "hedging_enabled": account.hedging_enabled,
    }
    return OANDAReadOnlyRiskSnapshot(
        payload=payload,
        provenance=provenance,
        issues=tuple(dict.fromkeys(issues)),
    )


def positions_from_oanda(
    response: Any,
    *,
    metadata_by_instrument: Mapping[str, BrokerInstrumentMetadata],
    quotes_by_instrument: Mapping[str, BrokerMarketQuote],
    symbol_by_instrument: Mapping[str, str],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Parse OANDA open trades with exact sided liquidation prices and margin."""

    rows = response if isinstance(response, list) else _mapping(response).get("trades", [])
    issues: list[str] = []
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        trade = _mapping(raw)
        instrument = _text(trade.get("instrument"))
        units = _optional_number(trade.get("currentUnits"))
        metadata = metadata_by_instrument.get(instrument or "")
        quote = quotes_by_instrument.get(instrument or "")
        symbol = symbol_by_instrument.get(instrument or "")
        side = "long" if units is not None and units > 0 else "short" if units is not None and units < 0 else None
        size = abs(units) if units is not None else None
        current_price = (
            quote.bid if quote is not None and side == "long"
            else quote.ask if quote is not None and side == "short"
            else None
        )
        stop_order = _mapping(trade.get("stopLossOrder"))
        stop = _optional_number(stop_order.get("price"))
        margin = (
            size * current_price * metadata.notional_value_per_price_unit * metadata.margin_rate
            if size is not None and current_price is not None and metadata is not None
            else None
        )
        for missing, issue in (
            (symbol, f"position_{index}_unmapped_instrument"),
            (side, f"position_{index}_invalid_units"),
            (current_price, f"position_{index}_missing_liquidation_quote"),
            (stop, f"position_{index}_missing_stop"),
            (metadata, f"position_{index}_missing_instrument_metadata"),
            (margin, f"position_{index}_missing_margin"),
        ):
            if missing is None:
                issues.append(issue)
        output.append(
            {
                "instrument": symbol,
                "side": side,
                "size": size,
                "current_price": current_price,
                "stop_price": stop,
                "value_per_price_unit": (
                    metadata.value_per_price_unit if metadata is not None else None
                ),
                "margin_required": margin,
            }
        )
    return output, tuple(dict.fromkeys(issues))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _optional_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("snapshot timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
