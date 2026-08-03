"""Build a fail-closed pre-trade risk payload from read-only IG responses.

The functions in this module are deliberately pure: they accept already-read
broker responses and never authenticate, submit, amend, or close an order.
Missing account-specific sizing facts remain ``None`` so the risk engine emits
``INCOMPLETE`` instead of manufacturing a usable-looking assessment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from commodity_ml.data.ig import IGAccountSnapshot, IGMarketQuote


@dataclass(frozen=True)
class IGMarginRate:
    """One percentage margin rate and the size range for which it is valid."""

    rate: float | None
    max_size: float | None
    source: str | None


@dataclass(frozen=True)
class IGReadOnlyRiskSnapshot:
    """Risk-engine input plus an audit trail of its broker provenance."""

    payload: dict[str, Any]
    provenance: dict[str, Any]
    issues: tuple[str, ...]
    order_authority: bool = False


def margin_rate_from_details(
    details: Mapping[str, Any],
    *,
    account_currency: str,
    fallback_rate: float | None = None,
    fallback_max_size: float | None = None,
) -> tuple[IGMarginRate, tuple[str, ...]]:
    """Extract only an unambiguous percentage rate valid from size zero.

    Tiered margin is not flattened into one optimistic number.  The first band
    is usable only when it starts at zero; its upper bound is returned as a
    hard sizing scope.  A configured fallback remains visibly distinct.
    """

    instrument = _mapping(details.get("instrument"))
    bands = instrument.get("marginDepositBands")
    candidates: list[Mapping[str, Any]] = []
    if isinstance(bands, list):
        for raw in bands:
            band = _mapping(raw)
            currency = str(band.get("currency") or "").upper()
            if not currency or currency == account_currency.upper():
                candidates.append(band)

    parsed: list[tuple[float, float | None, float]] = []
    for band in candidates:
        minimum = _optional_number(band.get("min"))
        rate_percent = _optional_number(band.get("margin"))
        maximum = _optional_number(band.get("max"))
        if minimum is None or rate_percent is None:
            continue
        if minimum < 0 or not (0 < rate_percent <= 100):
            continue
        if maximum is not None and maximum <= minimum:
            continue
        parsed.append((minimum, maximum, rate_percent / 100.0))

    parsed.sort(key=lambda item: item[0])
    if parsed and np.isclose(parsed[0][0], 0.0):
        _, maximum, rate = parsed[0]
        return IGMarginRate(rate, maximum, "instrument.marginDepositBands"), ()

    unit = str(instrument.get("marginFactorUnit") or "").upper()
    factor = _optional_number(instrument.get("marginFactor"))
    if unit == "PERCENTAGE" and factor is not None and 0 < factor <= 100:
        return IGMarginRate(
            factor / 100.0,
            None,
            "instrument.marginFactor",
        ), ()

    if fallback_rate is not None:
        if not (0 < fallback_rate <= 1):
            raise ValueError("fallback margin rate must be in (0, 1]")
        if fallback_max_size is not None and fallback_max_size <= 0:
            raise ValueError("fallback margin maximum size must be positive")
        return IGMarginRate(
            float(fallback_rate),
            float(fallback_max_size) if fallback_max_size is not None else None,
            "config.data.ig.margin_rate",
        ), ("using_configured_margin_fallback",)

    return IGMarginRate(None, None, None), ("missing_usable_percentage_margin_rate",)


def recent_same_session_spreads(
    candles: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
    minimum_rows: int = 8,
) -> tuple[float | None, float | None, int]:
    """Measure closed-bar IG spreads from the current London entry session."""

    required = {"bid_close", "ask_close"}
    if candles.empty or not required.issubset(candles.columns):
        return None, None, 0
    current = _utc(decision_time)
    frame = candles.copy()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        return None, None, 0
    frame.index = index.tz_convert("UTC")
    # At M30, the row stamped at the current half-hour can still be forming.
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
    return (
        float(spreads.median()),
        float(spreads.quantile(0.95)),
        int(len(spreads)),
    )


def build_ig_read_only_snapshot(
    *,
    instrument_symbol: str,
    trade_intent: Mapping[str, Any],
    account: IGAccountSnapshot,
    market_details: Mapping[str, Any],
    quote: IGMarketQuote,
    recent_candles: pd.DataFrame,
    open_positions: Any,
    size_increment: float | None,
    value_per_price_unit: float | None,
    fallback_margin_rate: float | None = None,
    fallback_margin_rate_max_size: float | None = None,
    day_start_equity: float | None = None,
    week_start_equity: float | None = None,
    entries_today: int | None = None,
    scheduled_event_blackout_active: bool | None = None,
    epic_to_instrument: Mapping[str, str] | None = None,
    value_per_price_unit_by_instrument: Mapping[str, float] | None = None,
    now: pd.Timestamp | None = None,
) -> IGReadOnlyRiskSnapshot:
    """Translate point-in-time GET responses into the engine's JSON contract."""

    current = _utc(now if now is not None else quote.timestamp)
    instrument = _mapping(market_details.get("instrument"))
    rules = _mapping(market_details.get("dealingRules"))
    min_deal = _mapping(rules.get("minDealSize"))
    issues: list[str] = []

    epic = str(instrument.get("epic") or market_details.get("epic") or "") or None
    if epic != quote.epic:
        issues.append("epic_mismatch_between_market_details_and_quote")
    minimum_size = _optional_number(min_deal.get("value"))
    if minimum_size is None:
        issues.append("missing_minimum_deal_size")
    if size_increment is None:
        issues.append("missing_confirmed_size_increment")
    if value_per_price_unit is None:
        issues.append("missing_confirmed_value_per_price_unit")

    margin, margin_issues = margin_rate_from_details(
        market_details,
        account_currency=account.currency,
        fallback_rate=fallback_margin_rate,
        fallback_max_size=fallback_margin_rate_max_size,
    )
    issues.extend(margin_issues)

    median_spread, p95_spread, spread_rows = recent_same_session_spreads(
        recent_candles,
        decision_time=current,
    )
    if median_spread is None:
        issues.append("insufficient_closed_same_session_spread_rows")

    mappings = dict(epic_to_instrument or {})
    if epic:
        mappings.setdefault(epic, instrument_symbol)
    point_values = dict(value_per_price_unit_by_instrument or {})
    if value_per_price_unit is not None:
        point_values.setdefault(instrument_symbol, float(value_per_price_unit))
    positions, position_issues = positions_from_ig(
        open_positions,
        account=account,
        epic_to_instrument=mappings,
        value_per_price_unit_by_instrument=point_values,
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

    if day_start_equity is None:
        issues.append("missing_day_start_equity")
    if week_start_equity is None:
        issues.append("missing_week_start_equity")
    if entries_today is None:
        issues.append("missing_entries_today")
    if scheduled_event_blackout_active is None:
        issues.append("missing_current_scheduled_event_state")

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
        },
        "instrument": {
            "instrument": instrument_symbol,
            "epic": epic,
            "instrument_type": str(instrument.get("type") or "").upper() or None,
            "expiry": str(instrument.get("expiry") or "") or None,
            "account_currency": account.currency,
            "min_size": minimum_size,
            "size_increment": size_increment,
            "value_per_price_unit": value_per_price_unit,
            "margin_rate": margin.rate,
            "margin_rate_max_size": margin.max_size,
            "commission_per_size_per_side": 0.0,
        },
        "market": {
            "epic": quote.epic,
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
        "mode": "ig_read_only_rest_snapshot",
        "order_authority": False,
        "account_id_suffix": account.account_id[-4:],
        "epic": epic,
        "quote_timestamp": quote.timestamp.isoformat(),
        "quote_delay_seconds": quote.delay_seconds,
        "spread_sample_rows": spread_rows,
        "margin_source": margin.source,
        "margin_rate_scope_max_size": margin.max_size,
        "open_position_rows": len(positions),
    }
    return IGReadOnlyRiskSnapshot(
        payload=payload,
        provenance=provenance,
        issues=tuple(dict.fromkeys(issues)),
    )


def positions_from_ig(
    response: Any,
    *,
    account: IGAccountSnapshot,
    epic_to_instrument: Mapping[str, str],
    value_per_price_unit_by_instrument: Mapping[str, float],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Parse IG open positions without inventing stops or per-position margin."""

    if isinstance(response, pd.DataFrame):
        rows = response.to_dict(orient="records")
    else:
        raw = _mapping(response)
        value = raw.get("positions", [])
        rows = list(value) if isinstance(value, list) else []
    if not rows:
        return [], ()

    issues: list[str] = []
    total_margin = max(account.equity - account.available, 0.0)
    output: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row)
        epic = _text(_value(row, "market.epic", "epic"))
        symbol = epic_to_instrument.get(epic or "")
        direction = _text(_value(row, "position.direction", "direction"))
        side = {"BUY": "long", "SELL": "short"}.get((direction or "").upper())
        bid = _optional_number(_value(row, "market.bid", "bid"))
        ask = _optional_number(_value(row, "market.offer", "market.ask", "offer", "ask"))
        current_price = bid if side == "long" else ask if side == "short" else None
        stop = _optional_number(_value(row, "position.stopLevel", "stopLevel", "stop"))
        size = _optional_number(_value(row, "position.size", "size"))
        point_value = value_per_price_unit_by_instrument.get(symbol or "")
        margin = total_margin if len(rows) == 1 else None
        if symbol is None:
            issues.append(f"position_{index}_unmapped_epic")
        if side is None:
            issues.append(f"position_{index}_invalid_direction")
        if current_price is None:
            issues.append(f"position_{index}_missing_liquidation_quote")
        if stop is None:
            issues.append(f"position_{index}_missing_stop")
        if size is None:
            issues.append(f"position_{index}_missing_size")
        if point_value is None:
            issues.append(f"position_{index}_missing_value_per_price_unit")
        if margin is None:
            issues.append("multiple_positions_prevent_margin_allocation")
        output.append(
            {
                "instrument": symbol,
                "side": side,
                "size": size,
                "current_price": current_price,
                "stop_price": stop,
                "value_per_price_unit": point_value,
                "margin_required": margin,
            }
        )
    return output, tuple(dict.fromkeys(issues))


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(row: Mapping[str, Any], *paths: str) -> Any:
    for path in paths:
        if path in row:
            return row[path]
        value: Any = row
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
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
