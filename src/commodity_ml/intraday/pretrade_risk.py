"""Offline quantitative pre-trade risk assessment for precious-metals trades.

This module contains no alpha model and submits no orders. Broker-specific
facts are explicit inputs so an absent contract value, margin rate, quote or
position stop produces an auditable INCOMPLETE decision instead of a guess.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass
import math
from typing import Any, Literal, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from commodity_ml.intraday.risk_calibration import CalibratedRiskLookup


DecisionStatus = Literal["ALLOW", "REDUCE", "BLOCK", "INCOMPLETE"]
Side = Literal["long", "short"]
OrderType = Literal["market", "limit", "stop"]

POLICY_VERSION = "pretrade-risk-v1-oanda-2026-08-04"
SUPPORTED_INSTRUMENTS = ("XAUUSD", "XAGUSD")
LONDON = ZoneInfo("Europe/London")


@dataclass(frozen=True)
class TradeIntent:
    intent_id: str | None
    instrument: str | None
    side: str | None
    order_type: str | None
    decision_time: pd.Timestamp | None
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    requested_size: float | None
    holding_period_minutes: int | None
    allow_overnight: bool | None
    max_slippage_points_per_side: float | None


@dataclass(frozen=True)
class AccountSnapshot:
    account_type: str | None
    status: str | None
    currency: str | None
    equity: float | None
    available_funds: float | None
    day_start_equity: float | None
    week_start_equity: float | None
    entries_today: int | None
    kill_switch_active: bool | None


@dataclass(frozen=True)
class InstrumentSpec:
    instrument: str | None
    broker_instrument: str | None
    instrument_type: str | None
    expiry: str | None
    account_currency: str | None
    min_size: float | None
    size_increment: float | None
    value_per_price_unit: float | None
    margin_rate: float | None
    margin_rate_max_size: float | None = None
    commission_per_size_per_side: float | None = 0.0
    notional_value_per_price_unit: float | None = None


@dataclass(frozen=True)
class MarketRiskSnapshot:
    broker_instrument: str | None
    as_of: pd.Timestamp | None
    bid: float | None
    ask: float | None
    market_open: bool | None
    quote_delay_seconds: int | None
    scheduled_event_blackout_active: bool | None
    median_spread_points: float | None = None
    p95_spread_points: float | None = None
    forecast_range_points: float | None = None
    forecast_horizon_minutes: int | None = None
    volatility_percentile: float | None = None
    volatility_artifact_approved: bool = False


@dataclass(frozen=True)
class ExistingPosition:
    instrument: str | None
    side: str | None
    size: float | None
    current_price: float | None
    stop_price: float | None
    value_per_price_unit: float | None
    margin_required: float | None


@dataclass(frozen=True)
class PreTradeRiskPolicy:
    risk_per_trade: float = 0.0025
    daily_loss_limit: float = 0.0075
    weekly_loss_limit: float = 0.015
    max_entries_per_day: int = 3
    max_open_positions: int = 1
    max_notional_fraction: float = 1.0
    max_total_stop_risk_fraction: float = 0.005
    max_margin_fraction: float = 0.30
    max_current_cost_r: float = 0.25
    max_median_cost_r: float = 0.10
    max_p95_cost_r: float = 0.25
    max_quote_age_seconds: int = 60
    entry_start_hour: int = 7
    entry_end_hour: int = 19
    entry_end_minute: int = 30
    flat_hour: int = 21
    flat_minute: int = 30
    max_holding_minutes: int = 8 * 60

    def __post_init__(self) -> None:
        fractions = (
            "risk_per_trade",
            "daily_loss_limit",
            "weekly_loss_limit",
            "max_notional_fraction",
            "max_total_stop_risk_fraction",
            "max_margin_fraction",
            "max_current_cost_r",
            "max_median_cost_r",
            "max_p95_cost_r",
        )
        for name in fractions:
            value = getattr(self, name)
            if not _positive(value):
                raise ValueError(f"{name} must be finite and positive")
        if self.weekly_loss_limit < self.daily_loss_limit:
            raise ValueError("weekly_loss_limit cannot be below daily_loss_limit")
        for name in (
            "max_entries_per_day",
            "max_open_positions",
            "max_quote_age_seconds",
            "max_holding_minutes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class PreTradeRiskReport:
    schema_version: int
    policy_version: str
    intent_id: str | None
    decision: DecisionStatus
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    trade: dict[str, Any]
    sizing: dict[str, Any]
    cash_risk: dict[str, Any]
    costs: dict[str, Any]
    market_risk: dict[str, Any]
    portfolio: dict[str, Any]
    scenarios: tuple[dict[str, Any], ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_pretrade_risk(
    intent: TradeIntent,
    *,
    account: AccountSnapshot,
    instrument: InstrumentSpec,
    market: MarketRiskSnapshot,
    existing_positions: Sequence[ExistingPosition] = (),
    policy: PreTradeRiskPolicy | None = None,
    historical_calibration: CalibratedRiskLookup | None = None,
) -> PreTradeRiskReport:
    """Evaluate one proposed trade under the deterministic version-1 policy."""

    policy = policy or PreTradeRiskPolicy()
    missing = _missing_required_inputs(
        intent, account, instrument, market, existing_positions
    )
    warnings: list[str] = []
    if market.median_spread_points is None:
        warnings.append("median_spread_history_unavailable")
    if market.p95_spread_points is None:
        warnings.append("p95_spread_history_unavailable")
    if not market.volatility_artifact_approved:
        warnings.append("volatility_forecast_not_approved_for_sizing")
    if historical_calibration is None:
        warnings.append("historical_range_calibration_unavailable_or_stale")

    if missing:
        return _empty_report(
            intent,
            decision="INCOMPLETE",
            reasons=tuple(f"missing:{item}" for item in missing),
            warnings=tuple(warnings),
        )

    # The missing-input gate above narrows these optional annotations.
    decision_time = _utc(intent.decision_time)
    quote_time = _utc(market.as_of)
    entry = float(intent.entry_price)
    stop = float(intent.stop_price)
    target = float(intent.target_price)
    equity = float(account.equity)
    available_funds = float(account.available_funds)
    value_per_point = float(instrument.value_per_price_unit)
    min_size = float(instrument.min_size)
    size_increment = float(instrument.size_increment)
    margin_rate = float(instrument.margin_rate)
    commission = float(instrument.commission_per_size_per_side)
    slippage = float(intent.max_slippage_points_per_side)
    bid = float(market.bid)
    ask = float(market.ask)

    reasons: list[str] = []
    if intent.instrument not in SUPPORTED_INSTRUMENTS:
        reasons.append("unsupported_instrument")
    if intent.side not in ("long", "short"):
        reasons.append("invalid_side")
    if intent.order_type not in ("market", "limit", "stop"):
        reasons.append("invalid_order_type")
    if not isinstance(intent.allow_overnight, bool):
        reasons.append("invalid_allow_overnight_flag")
    if not isinstance(market.market_open, bool):
        reasons.append("invalid_market_open_flag")
    if (
        isinstance(market.quote_delay_seconds, bool)
        or not isinstance(market.quote_delay_seconds, int)
        or market.quote_delay_seconds < 0
    ):
        reasons.append("invalid_quote_delay")
    elif market.quote_delay_seconds != 0:
        reasons.append("delayed_quote")
    if not isinstance(market.scheduled_event_blackout_active, bool):
        reasons.append("invalid_scheduled_event_blackout_flag")
    if not isinstance(market.volatility_artifact_approved, bool):
        reasons.append("invalid_volatility_approval_flag")
    if instrument.instrument != intent.instrument:
        reasons.append("instrument_mismatch")
    if market.broker_instrument != instrument.broker_instrument:
        reasons.append("quote_instrument_mismatch")
    if historical_calibration is not None and not _calibration_matches(
        historical_calibration,
        intent,
        decision_time,
    ):
        reasons.append("historical_calibration_context_mismatch")
    if account.currency != instrument.account_currency:
        reasons.append("account_currency_mismatch")
    if str(account.status).upper() != "ENABLED":
        reasons.append("account_not_enabled")
    if str(account.account_type).upper() != "SPREADBET":
        reasons.append("unsupported_account_type")
    if str(instrument.instrument_type).upper() not in ("METAL", "COMMODITIES"):
        reasons.append("unsupported_instrument_type")
    if str(instrument.expiry).upper() not in ("DFB", "-"):
        reasons.append("unsupported_expiring_instrument")
    if not all(_positive(value) for value in (entry, stop, target, equity)):
        reasons.append("non_positive_trade_or_equity_value")
    if not all(_positive(value) for value in (bid, ask)) or ask < bid:
        reasons.append("invalid_or_crossed_quote")
    if not all(_positive(value) for value in (min_size, size_increment, value_per_point)):
        reasons.append("invalid_instrument_sizing_fact")
    if not _finite_between(margin_rate, 0, 1, include_lower=False):
        reasons.append("invalid_margin_rate")
    if instrument.margin_rate_max_size is not None and not _positive(
        instrument.margin_rate_max_size
    ):
        reasons.append("invalid_margin_rate_max_size")
    if not _non_negative(commission) or not _non_negative(slippage):
        reasons.append("invalid_cost_allowance")
    if available_funds < 0:
        reasons.append("negative_available_funds")
    if not isinstance(intent.holding_period_minutes, int) or isinstance(
        intent.holding_period_minutes, bool
    ) or intent.holding_period_minutes <= 0:
        reasons.append("invalid_holding_period")
    if not isinstance(account.entries_today, int) or isinstance(
        account.entries_today, bool
    ) or account.entries_today < 0:
        reasons.append("invalid_entries_today")
    if not isinstance(account.kill_switch_active, bool):
        reasons.append("invalid_kill_switch_flag")
    if not _positive(account.day_start_equity):
        reasons.append("invalid_day_start_equity")
    if not _positive(account.week_start_equity):
        reasons.append("invalid_week_start_equity")

    if intent.side == "long" and not (stop < entry < target):
        reasons.append("invalid_long_price_geometry")
    if intent.side == "short" and not (target < entry < stop):
        reasons.append("invalid_short_price_geometry")
    current_mid = (bid + ask) / 2
    if intent.order_type == "market" and abs(entry - current_mid) > _spread_or_floor(
        ask - bid
    ):
        reasons.append("market_entry_reference_not_current")
    if intent.order_type == "stop" and (
        (intent.side == "long" and entry < ask)
        or (intent.side == "short" and entry > bid)
    ):
        reasons.append("invalid_stop_entry_level")

    # Do not attempt monetary arithmetic from malformed foundations. Policy
    # blockers below still receive a full report when the numeric contract is
    # valid.
    if reasons:
        return _empty_report(
            intent,
            decision="BLOCK",
            reasons=tuple(dict.fromkeys(reasons)),
            warnings=tuple(warnings),
        )

    quote_age = (decision_time - quote_time).total_seconds()
    if quote_age < -1:
        reasons.append("quote_timestamp_after_decision")
    elif quote_age > policy.max_quote_age_seconds:
        reasons.append("stale_quote")
    if market.market_open is not True:
        reasons.append("market_not_open")
    if market.scheduled_event_blackout_active is True:
        reasons.append("scheduled_event_blackout")
    if market.median_spread_points is None:
        reasons.append("median_spread_history_unavailable")
    if market.p95_spread_points is None:
        reasons.append("p95_spread_history_unavailable")

    _append_time_policy_reasons(reasons, intent, decision_time, policy)
    daily_drawdown = max(
        0.0, 1.0 - equity / float(account.day_start_equity)
    )
    weekly_drawdown = max(
        0.0, 1.0 - equity / float(account.week_start_equity)
    )
    if daily_drawdown >= policy.daily_loss_limit:
        reasons.append("daily_loss_limit")
    if weekly_drawdown >= policy.weekly_loss_limit:
        reasons.append("weekly_loss_limit")
    if int(account.entries_today) >= policy.max_entries_per_day:
        reasons.append("max_entries_per_day")
    if account.kill_switch_active is True:
        reasons.append("kill_switch_active")
    if len(existing_positions) >= policy.max_open_positions:
        reasons.append("max_open_positions")

    spread = ask - bid
    stop_distance = abs(entry - stop)
    target_distance = abs(target - entry)
    base_cost_per_size = _cost_per_size(
        spread, slippage, value_per_point, commission
    )
    gross_stop_per_size = stop_distance * value_per_point
    stop_loss_per_size = gross_stop_per_size + base_cost_per_size
    net_reward_per_size = target_distance * value_per_point - base_cost_per_size
    current_cost_r = _safe_ratio(base_cost_per_size, gross_stop_per_size)
    median_cost_r = _cost_r(
        market.median_spread_points,
        slippage,
        value_per_point,
        commission,
        gross_stop_per_size,
    )
    p95_cost_r = _cost_r(
        market.p95_spread_points,
        slippage,
        value_per_point,
        commission,
        gross_stop_per_size,
    )
    if current_cost_r > policy.max_current_cost_r:
        reasons.append("current_cost_above_0.25R")
    if median_cost_r is not None and median_cost_r > policy.max_median_cost_r:
        reasons.append("median_cost_above_0.10R")
    if p95_cost_r is not None and p95_cost_r > policy.max_p95_cost_r:
        reasons.append("p95_cost_above_0.25R")
    if net_reward_per_size <= 0:
        reasons.append("non_positive_net_target_reward")

    volatility_multiplier = 1.0
    volatility_state = "unavailable"
    if market.volatility_artifact_approved:
        percentile = market.volatility_percentile
        if percentile is None or not _finite_between(percentile, 0, 1):
            reasons.append("approved_volatility_percentile_invalid")
            volatility_multiplier = 0.0
            volatility_state = "invalid"
        elif percentile >= 0.95:
            reasons.append("volatility_at_or_above_p95")
            volatility_multiplier = 0.0
            volatility_state = "block"
        elif percentile >= 0.80:
            volatility_multiplier = 0.5
            volatility_state = "half_risk"
        else:
            volatility_state = "normal_risk"

    position_values = _aggregate_positions(existing_positions)
    if position_values["invalid_reasons"]:
        # Defensive fallback: values passed the missing gate but not validity.
        reasons.extend(position_values["invalid_reasons"])
    existing_stop_risk = position_values["stop_risk"]
    existing_gross = position_values["gross_exposure"]
    existing_margin = position_values["margin"]

    trade_risk_budget = equity * policy.risk_per_trade * volatility_multiplier
    portfolio_risk_room = max(
        0.0,
        equity * policy.max_total_stop_risk_fraction - existing_stop_risk,
    )
    effective_risk_budget = min(trade_risk_budget, portfolio_risk_room)
    risk_cap_size = (
        effective_risk_budget / stop_loss_per_size
        if stop_loss_per_size > 0
        else 0.0
    )
    notional_value_per_point = (
        float(instrument.notional_value_per_price_unit)
        if instrument.notional_value_per_price_unit is not None
        else value_per_point
    )
    if not _positive(notional_value_per_point):
        return _empty_report(
            intent,
            decision="BLOCK",
            reasons=("invalid_notional_conversion_fact",),
            warnings=tuple(warnings),
        )
    notional_per_size = entry * notional_value_per_point
    notional_room = max(
        0.0, equity * policy.max_notional_fraction - existing_gross
    )
    notional_cap_size = notional_room / notional_per_size
    margin_per_size = notional_per_size * margin_rate
    policy_margin_room = max(
        0.0, equity * policy.max_margin_fraction - existing_margin
    )
    margin_room = max(0.0, min(available_funds, policy_margin_room))
    margin_cap_size = margin_room / margin_per_size
    margin_schedule_cap_size = (
        float(instrument.margin_rate_max_size)
        if instrument.margin_rate_max_size is not None
        else math.inf
    )
    raw_cap = min(
        risk_cap_size,
        notional_cap_size,
        margin_cap_size,
        margin_schedule_cap_size,
    )
    recommended_size = _floor_broker_size(raw_cap, min_size, size_increment)
    binding_constraint = _binding_constraint(
        {
            "risk": risk_cap_size,
            "notional": notional_cap_size,
            "margin": margin_cap_size,
            "margin_schedule_scope": margin_schedule_cap_size,
        }
    )
    if recommended_size <= 0:
        reasons.append("minimum_size_exceeds_caps")

    requested_size = intent.requested_size
    size_adjustment_required = False
    if requested_size is None:
        assessed_size = recommended_size
        size_source = "recommended"
        recommended_action_size = recommended_size
    else:
        size_source = "requested"
        if not _positive(requested_size):
            reasons.append("invalid_requested_size")
            assessed_size = 0.0
            recommended_action_size = 0.0
        elif requested_size + 1e-12 < min_size:
            reasons.append("requested_size_below_minimum")
            assessed_size = float(requested_size)
            recommended_action_size = 0.0
        else:
            requested_on_step = _floor_broker_size(
                float(requested_size), min_size, size_increment
            )
            assessed_size = float(requested_size)
            recommended_action_size = min(requested_on_step, recommended_size)
            if not math.isclose(
                requested_on_step,
                float(requested_size),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                size_adjustment_required = True
                warnings.append("requested_size_not_on_broker_increment")
            if requested_size > recommended_size + 1e-12:
                size_adjustment_required = True

    proposed_cash = _cash_metrics(
        assessed_size,
        equity,
        stop_loss_per_size,
        net_reward_per_size,
        notional_per_size,
        margin_per_size,
        base_cost_per_size,
    )
    recommended_cash = _cash_metrics(
        recommended_size,
        equity,
        stop_loss_per_size,
        net_reward_per_size,
        notional_per_size,
        margin_per_size,
        base_cost_per_size,
    )
    if proposed_cash["net_reward_risk"] is not None and proposed_cash[
        "net_reward_risk"
    ] < 1:
        warnings.append("net_reward_risk_below_1")

    proposed_incremental_risk = proposed_cash["planned_stop_loss"]
    if existing_stop_risk + proposed_incremental_risk > (
        equity * policy.max_total_stop_risk_fraction + 1e-9
    ):
        size_adjustment_required = True

    blocking_reasons = tuple(dict.fromkeys(reasons))
    if blocking_reasons:
        status: DecisionStatus = "BLOCK"
    elif size_adjustment_required:
        status = "REDUCE"
        blocking_reasons = ("requested_size_exceeds_or_misses_safe_broker_cap",)
    else:
        status = "ALLOW"

    scenario_size = assessed_size if assessed_size > 0 else recommended_size
    scenarios = _build_scenarios(
        size=scenario_size,
        equity=equity,
        stop_distance=stop_distance,
        spread=spread,
        slippage=slippage,
        value_per_point=value_per_point,
        commission=commission,
        market=market,
        holding_period_minutes=int(intent.holding_period_minutes),
        warnings=warnings,
        historical_calibration=historical_calibration,
    )
    proposed_signed_one_pct = (
        (1.0 if intent.side == "long" else -1.0)
        * assessed_size
        * entry
        * value_per_point
        * 0.01
    )
    concentration = dict(position_values["by_instrument"])
    concentration[intent.instrument] = concentration.get(intent.instrument, 0.0) + (
        assessed_size * entry * value_per_point
    )

    return PreTradeRiskReport(
        schema_version=2,
        policy_version=POLICY_VERSION,
        intent_id=intent.intent_id,
        decision=status,
        reasons=blocking_reasons,
        warnings=tuple(dict.fromkeys(warnings)),
        trade=_trade_summary(intent, instrument.broker_instrument),
        sizing={
            "size_source": size_source,
            "requested_size": requested_size,
            "assessed_size": assessed_size,
            "recommended_max_size": recommended_size,
            "recommended_action_size": recommended_action_size,
            "raw_cap_size": raw_cap,
            "binding_constraint": binding_constraint,
            "normal_risk_budget": equity * policy.risk_per_trade,
            "effective_risk_budget": effective_risk_budget,
            "volatility_multiplier": volatility_multiplier,
        },
        cash_risk={
            "account_currency": account.currency,
            "proposed": proposed_cash,
            "at_recommended_max": recommended_cash,
            "daily_drawdown_fraction": daily_drawdown,
            "weekly_drawdown_fraction": weekly_drawdown,
            "stop_is_guaranteed_max_loss": False,
        },
        costs={
            "current_spread_points": spread,
            "slippage_points_per_side": slippage,
            "commission_per_size_per_side": commission,
            "base_round_trip_cost_per_size": base_cost_per_size,
            "current_cost_r": current_cost_r,
            "median_cost_r": median_cost_r,
            "p95_cost_r": p95_cost_r,
        },
        market_risk={
            "quote_age_seconds": quote_age,
            "volatility_state": volatility_state,
            "volatility_percentile": market.volatility_percentile,
            "forecast_range_points": market.forecast_range_points,
            "forecast_horizon_minutes": market.forecast_horizon_minutes,
            "scheduled_event_blackout_active": market.scheduled_event_blackout_active,
            "quote_delay_seconds": market.quote_delay_seconds,
            "historical_calibration": (
                {
                    "profile_as_of": historical_calibration.profile_as_of.isoformat(),
                    "sample_count": historical_calibration.sample_count,
                    "range_points": historical_calibration.range_points,
                    "adverse_points": historical_calibration.adverse_points,
                    "broker_spread_eligible": historical_calibration.broker_spread_eligible,
                    "volatility_sizing_approved": historical_calibration.volatility_sizing_approved,
                }
                if historical_calibration is not None
                else None
            ),
        },
        portfolio={
            "existing_position_count": len(existing_positions),
            "existing_stop_risk": existing_stop_risk,
            "proposed_total_stop_risk": existing_stop_risk + proposed_incremental_risk,
            "existing_margin": existing_margin,
            "proposed_total_margin": existing_margin + proposed_cash["initial_margin"],
            "existing_gross_exposure": existing_gross,
            "proposed_total_gross_exposure": existing_gross
            + proposed_cash["gross_exposure"],
            "existing_signed_one_pct_move": position_values[
                "signed_one_pct_move"
            ],
            "proposed_signed_one_pct_move": proposed_signed_one_pct,
            "combined_signed_one_pct_move": position_values[
                "signed_one_pct_move"
            ]
            + proposed_signed_one_pct,
            "gross_exposure_by_instrument": concentration,
        },
        scenarios=tuple(scenarios),
        limitations=_limitations(),
    )


def evaluate_pretrade_payload(
    payload: Mapping[str, Any],
    *,
    historical_calibration: CalibratedRiskLookup | None = None,
) -> PreTradeRiskReport:
    """Parse the documented JSON shape and evaluate it."""

    intent_data = dict(payload.get("trade_intent") or {})
    account_data = dict(payload.get("account") or {})
    instrument_data = dict(payload.get("instrument") or {})
    market_data = dict(payload.get("market") or {})
    position_data = payload.get("existing_positions") or []
    policy_data = dict(payload.get("policy") or {})
    if not isinstance(position_data, list):
        raise ValueError("existing_positions must be a list")

    intent_data["decision_time"] = _optional_timestamp(
        intent_data.get("decision_time"), "trade_intent.decision_time"
    )
    market_data["as_of"] = _optional_timestamp(
        market_data.get("as_of"), "market.as_of"
    )
    return evaluate_pretrade_risk(
        TradeIntent(**_fields_for(TradeIntent, intent_data)),
        account=AccountSnapshot(**_fields_for(AccountSnapshot, account_data)),
        instrument=InstrumentSpec(**_fields_for(InstrumentSpec, instrument_data)),
        market=MarketRiskSnapshot(**_fields_for(MarketRiskSnapshot, market_data)),
        existing_positions=tuple(
            ExistingPosition(**_fields_for(ExistingPosition, dict(item)))
            for item in position_data
        ),
        policy=PreTradeRiskPolicy(**policy_data) if policy_data else None,
        historical_calibration=historical_calibration,
    )


def _fields_for(cls, values: Mapping[str, Any]) -> dict[str, Any]:
    expected = cls.__dataclass_fields__
    unknown = sorted(set(values) - set(expected))
    if unknown:
        raise ValueError(f"Unexpected {cls.__name__} fields: {', '.join(unknown)}")
    parsed: dict[str, Any] = {}
    for name, field in expected.items():
        if name in values:
            parsed[name] = values[name]
        elif field.default is not MISSING:
            parsed[name] = field.default
        else:
            parsed[name] = None
    return parsed


def _missing_required_inputs(
    intent: TradeIntent,
    account: AccountSnapshot,
    instrument: InstrumentSpec,
    market: MarketRiskSnapshot,
    positions: Sequence[ExistingPosition],
) -> list[str]:
    required = {
        "trade_intent.intent_id": intent.intent_id,
        "trade_intent.instrument": intent.instrument,
        "trade_intent.side": intent.side,
        "trade_intent.order_type": intent.order_type,
        "trade_intent.decision_time": intent.decision_time,
        "trade_intent.entry_price": intent.entry_price,
        "trade_intent.stop_price": intent.stop_price,
        "trade_intent.target_price": intent.target_price,
        "trade_intent.holding_period_minutes": intent.holding_period_minutes,
        "trade_intent.allow_overnight": intent.allow_overnight,
        "trade_intent.max_slippage_points_per_side": intent.max_slippage_points_per_side,
        "account.currency": account.currency,
        "account.account_type": account.account_type,
        "account.status": account.status,
        "account.equity": account.equity,
        "account.available_funds": account.available_funds,
        "account.day_start_equity": account.day_start_equity,
        "account.week_start_equity": account.week_start_equity,
        "account.entries_today": account.entries_today,
        "account.kill_switch_active": account.kill_switch_active,
        "instrument.instrument": instrument.instrument,
        "instrument.broker_instrument": instrument.broker_instrument,
        "instrument.instrument_type": instrument.instrument_type,
        "instrument.expiry": instrument.expiry,
        "instrument.account_currency": instrument.account_currency,
        "instrument.min_size": instrument.min_size,
        "instrument.size_increment": instrument.size_increment,
        "instrument.value_per_price_unit": instrument.value_per_price_unit,
        "instrument.margin_rate": instrument.margin_rate,
        "instrument.commission_per_size_per_side": instrument.commission_per_size_per_side,
        "market.broker_instrument": market.broker_instrument,
        "market.as_of": market.as_of,
        "market.bid": market.bid,
        "market.ask": market.ask,
        "market.market_open": market.market_open,
        "market.quote_delay_seconds": market.quote_delay_seconds,
        "market.scheduled_event_blackout_active": market.scheduled_event_blackout_active,
    }
    missing = [name for name, value in required.items() if value is None or value == ""]
    for index, position in enumerate(positions):
        for name in (
            "instrument",
            "side",
            "size",
            "current_price",
            "stop_price",
            "value_per_price_unit",
            "margin_required",
        ):
            if getattr(position, name) is None:
                missing.append(f"existing_positions[{index}].{name}")
    return missing


def _append_time_policy_reasons(
    reasons: list[str],
    intent: TradeIntent,
    decision_time: pd.Timestamp,
    policy: PreTradeRiskPolicy,
) -> None:
    local = decision_time.tz_convert(LONDON)
    minute = local.hour * 60 + local.minute
    start = policy.entry_start_hour * 60
    end = policy.entry_end_hour * 60 + policy.entry_end_minute
    if minute < start or minute >= end:
        reasons.append("outside_entry_session")
    if intent.allow_overnight is not False:
        reasons.append("overnight_not_permitted")
    if intent.holding_period_minutes > policy.max_holding_minutes:
        reasons.append("holding_period_above_v1_limit")
    planned_exit = local + pd.Timedelta(minutes=int(intent.holding_period_minutes))
    cutoff = local.normalize() + pd.Timedelta(
        hours=policy.flat_hour, minutes=policy.flat_minute
    )
    if planned_exit > cutoff:
        reasons.append("planned_exit_after_daily_cutoff")


def _aggregate_positions(positions: Sequence[ExistingPosition]) -> dict[str, Any]:
    stop_risk = gross = margin = signed_one_pct = 0.0
    by_instrument: dict[str, float] = {}
    invalid: list[str] = []
    for index, position in enumerate(positions):
        values = (
            position.size,
            position.current_price,
            position.stop_price,
            position.value_per_price_unit,
        )
        if position.instrument not in SUPPORTED_INSTRUMENTS:
            invalid.append(f"existing_position_{index}_unsupported_instrument")
            continue
        if position.side not in ("long", "short"):
            invalid.append(f"existing_position_{index}_invalid_side")
            continue
        if not all(_positive(value) for value in values) or not _non_negative(
            position.margin_required
        ):
            invalid.append(f"existing_position_{index}_invalid_numeric_value")
            continue
        if (
            position.side == "long"
            and float(position.stop_price) >= float(position.current_price)
        ) or (
            position.side == "short"
            and float(position.stop_price) <= float(position.current_price)
        ):
            invalid.append(f"existing_position_{index}_non_protective_stop")
            continue
        size = float(position.size)
        price = float(position.current_price)
        vpp = float(position.value_per_price_unit)
        position_gross = size * price * vpp
        stop_risk += size * abs(price - float(position.stop_price)) * vpp
        gross += position_gross
        margin += float(position.margin_required)
        sign = 1.0 if position.side == "long" else -1.0
        signed_one_pct += sign * position_gross * 0.01
        by_instrument[position.instrument] = (
            by_instrument.get(position.instrument, 0.0) + position_gross
        )
    return {
        "stop_risk": stop_risk,
        "gross_exposure": gross,
        "margin": margin,
        "signed_one_pct_move": signed_one_pct,
        "by_instrument": by_instrument,
        "invalid_reasons": invalid,
    }


def _build_scenarios(
    *,
    size: float,
    equity: float,
    stop_distance: float,
    spread: float,
    slippage: float,
    value_per_point: float,
    commission: float,
    market: MarketRiskSnapshot,
    holding_period_minutes: int,
    warnings: list[str],
    historical_calibration: CalibratedRiskLookup | None,
) -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    for name, spread_multiplier in (
        ("planned_stop", 1.0),
        ("planned_stop_spread_1_5x", 1.5),
        ("planned_stop_spread_2x", 2.0),
    ):
        loss = size * (
            stop_distance * value_per_point
            + _cost_per_size(
                spread * spread_multiplier,
                slippage,
                value_per_point,
                commission,
            )
        )
        scenarios.append(_scenario(name, loss, equity))

    if historical_calibration is not None:
        for label in ("p50", "p80", "p95", "p99"):
            range_loss = size * (
                historical_calibration.range_points[label] * value_per_point
                + _cost_per_size(
                    spread, slippage, value_per_point, commission
                )
            )
            row = _scenario(f"historical_total_range_{label}", range_loss, equity)
            row["historical_quantile"] = int(label[1:]) / 100
            scenarios.append(row)
        for label in ("p80", "p95", "p99"):
            adverse_loss = size * (
                historical_calibration.adverse_points[label] * value_per_point
                + _cost_per_size(
                    spread, slippage, value_per_point, commission
                )
            )
            row = _scenario(f"historical_side_adverse_{label}", adverse_loss, equity)
            row["historical_quantile"] = int(label[1:]) / 100
            scenarios.append(row)

    if not market.volatility_artifact_approved:
        return scenarios
    if market.forecast_range_points is None or market.forecast_horizon_minutes is None:
        warnings.append("approved_forecast_range_or_horizon_missing")
        return scenarios
    if not _positive(market.forecast_range_points) or not isinstance(
        market.forecast_horizon_minutes, int
    ):
        warnings.append("approved_forecast_range_or_horizon_invalid")
        return scenarios
    if market.forecast_horizon_minutes != holding_period_minutes:
        warnings.append("forecast_horizon_does_not_match_trade_horizon")
        return scenarios

    base_cost = _cost_per_size(spread, slippage, value_per_point, commission)
    forecast_range = float(market.forecast_range_points)
    for multiple in (1, 2, 3):
        loss = size * (multiple * forecast_range * value_per_point + base_cost)
        scenarios.append(_scenario(f"adverse_{multiple}x_forecast_range", loss, equity))
    gap_loss = size * (
        (stop_distance + 0.5 * forecast_range) * value_per_point + base_cost
    )
    scenarios.append(_scenario("gap_half_range_beyond_stop", gap_loss, equity))
    return scenarios


def _cash_metrics(
    size: float,
    equity: float,
    stop_loss_per_size: float,
    net_reward_per_size: float,
    notional_per_size: float,
    margin_per_size: float,
    cost_per_size: float,
) -> dict[str, Any]:
    stop_loss = size * stop_loss_per_size
    reward = size * net_reward_per_size
    return {
        "size": size,
        "planned_stop_loss": stop_loss,
        "planned_stop_loss_fraction": _safe_ratio(stop_loss, equity),
        "net_target_reward": reward,
        "net_reward_risk": _safe_ratio(reward, stop_loss) if stop_loss > 0 else None,
        "breakeven_win_rate_binary": (
            _safe_ratio(stop_loss, stop_loss + reward)
            if stop_loss > 0 and reward > 0
            else None
        ),
        "round_trip_cost": size * cost_per_size,
        "gross_exposure": size * notional_per_size,
        "initial_margin": size * margin_per_size,
    }


def _cost_r(
    spread: float | None,
    slippage: float,
    value_per_point: float,
    commission: float,
    gross_stop_per_size: float,
) -> float | None:
    if spread is None:
        return None
    if not _non_negative(spread):
        return math.inf
    return _safe_ratio(
        _cost_per_size(spread, slippage, value_per_point, commission),
        gross_stop_per_size,
    )


def _cost_per_size(
    spread: float,
    slippage: float,
    value_per_point: float,
    commission: float,
) -> float:
    return value_per_point * (spread + 2 * slippage) + 2 * commission


def _floor_broker_size(cap: float, minimum: float, increment: float) -> float:
    if not all(_positive(value) for value in (minimum, increment)):
        return 0.0
    if not np.isfinite(cap) or cap + 1e-12 < minimum:
        return 0.0
    steps = math.floor((cap - minimum) / increment + 1e-12)
    return float(round(minimum + steps * increment, 12))


def _binding_constraint(caps: Mapping[str, float]) -> str:
    minimum = min(caps.values())
    bound = [
        name
        for name, value in caps.items()
        if math.isclose(value, minimum, rel_tol=1e-12, abs_tol=1e-12)
    ]
    return "_and_".join(bound)


def _scenario(name: str, loss: float, equity: float) -> dict[str, Any]:
    return {
        "name": name,
        "cash_loss": loss,
        "equity_fraction": _safe_ratio(loss, equity),
        "probability_assigned": False,
    }


def _empty_report(
    intent: TradeIntent,
    *,
    decision: DecisionStatus,
    reasons: tuple[str, ...],
    warnings: tuple[str, ...],
) -> PreTradeRiskReport:
    return PreTradeRiskReport(
        schema_version=2,
        policy_version=POLICY_VERSION,
        intent_id=intent.intent_id,
        decision=decision,
        reasons=reasons,
        warnings=warnings,
        trade=_trade_summary(intent, None),
        sizing={},
        cash_risk={},
        costs={},
        market_risk={},
        portfolio={},
        scenarios=(),
        limitations=_limitations(),
    )


def _limitations() -> tuple[str, ...]:
    return (
        "This report evaluates risk conditional on the supplied trade; it does not predict direction or profitability.",
        "A protective stop is not a guaranteed maximum loss because gaps, slippage, delay and closure can bypass it.",
        "Forecast scenarios are deterministic stresses with no assigned probability and are not value-at-risk.",
        "Historical quantiles describe development-sample coverage and are not guaranteed future probabilities.",
        "Broker contract, currency, margin and quote facts must be verified against the intended account and instrument.",
    )


def _trade_summary(intent: TradeIntent, broker_instrument: str | None) -> dict[str, Any]:
    return {
        "instrument": intent.instrument,
        "broker_instrument": broker_instrument,
        "side": intent.side,
        "order_type": intent.order_type,
        "decision_time": (
            _utc(intent.decision_time).isoformat()
            if intent.decision_time is not None
            else None
        ),
        "entry_price": intent.entry_price,
        "stop_price": intent.stop_price,
        "target_price": intent.target_price,
        "holding_period_minutes": intent.holding_period_minutes,
        "allow_overnight": intent.allow_overnight,
    }


def _calibration_matches(
    calibration: CalibratedRiskLookup,
    intent: TradeIntent,
    decision_time: pd.Timestamp,
) -> bool:
    local = decision_time.tz_convert(LONDON)
    slot = local.hour * 60 + local.minute
    return bool(
        calibration.instrument == intent.instrument
        and calibration.side == intent.side
        and calibration.horizon_minutes == intent.holding_period_minutes
        and calibration.london_slot_minute == slot
        and calibration.profile_as_of < decision_time
        and decision_time - calibration.profile_as_of <= pd.Timedelta(days=90)
        and calibration.sample_count >= 126
        and calibration.broker_spread_eligible is False
        and all(
            _positive(value)
            for values in (calibration.range_points, calibration.adverse_points)
            for value in values.values()
        )
    )


def _optional_timestamp(value: Any, field: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _utc(value: pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        raise ValueError("timestamp is missing")
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _positive(value: Any) -> bool:
    return _non_negative(value) and float(value) > 0


def _non_negative(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return bool(np.isfinite(value) and float(value) >= 0)
    except (TypeError, ValueError):
        return False


def _finite_between(
    value: Any,
    lower: float,
    upper: float,
    *,
    include_lower: bool = True,
) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        if not np.isfinite(value):
            return False
        lower_ok = float(value) >= lower if include_lower else float(value) > lower
        return lower_ok and float(value) <= upper
    except (TypeError, ValueError):
        return False


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else math.inf


def _spread_or_floor(spread: float) -> float:
    """Tolerance for a market-order midpoint reference."""

    return max(float(spread), 1e-9)
