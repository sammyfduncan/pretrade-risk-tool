"""Fail-closed paper order-intent construction.

No function in this module submits an order. It combines broker reconciliation,
fresh quotes, spread history, portfolio loss gates, and stop-based sizing into
an auditable intent that a later order adapter may submit exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

from commodity_ml.data.ig import (
    IGAccountSnapshot,
    IGMarketMetadata,
    IGMarketQuote,
)
from commodity_ml.intraday.market_data import (
    SpreadAssessment,
    assess_spread_history,
    closed_quote_window,
)
from commodity_ml.intraday.risk import (
    InstrumentConstraints,
    RiskDecision,
    RiskLimits,
    RiskState,
    SizingDecision,
    assess_risk,
    size_for_stop,
)

PaperSide = Literal["long", "short"]


@dataclass(frozen=True)
class PaperCandidate:
    intent_id: str
    strategy: str
    side: PaperSide
    decision_time: pd.Timestamp
    stop_distance_pct: float
    target_distance_pct: float

    def __post_init__(self) -> None:
        if not self.intent_id or not self.strategy:
            raise ValueError("intent_id and strategy must be non-empty")
        if self.side not in ("long", "short"):
            raise ValueError("side must be long or short")
        if self.stop_distance_pct <= 0 or self.target_distance_pct <= 0:
            raise ValueError("stop and target distances must be positive")
        _utc(self.decision_time)


@dataclass(frozen=True)
class PaperRuntimeState:
    day_start_equity: float
    week_start_equity: float
    entries_today: int
    expected_open_positions: int
    expected_open_position_ids: tuple[str, ...] = ()
    submitted_intent_ids: tuple[str, ...] = ()
    kill_switch_active: bool = False


@dataclass(frozen=True)
class PaperOrderIntent:
    intent_id: str
    strategy: str
    epic: str
    account_id: str
    account_currency: str
    direction: str
    size: float
    reference_price: float
    stop_level: float
    target_level: float
    created_at: pd.Timestamp
    dry_run: bool = True


@dataclass(frozen=True)
class PaperDecision:
    allowed: bool
    reasons: tuple[str, ...]
    flatten_required: bool
    risk: RiskDecision | None
    spread: SpreadAssessment | None
    sizing: SizingDecision | None
    intent: PaperOrderIntent | None


def build_paper_intent(
    candidate: PaperCandidate,
    *,
    account: IGAccountSnapshot,
    market: IGMarketMetadata,
    quote: IGMarketQuote,
    quote_candles: pd.DataFrame,
    runtime: PaperRuntimeState,
    broker_open_position_ids: Sequence[str],
    bar_duration: pd.Timedelta = pd.Timedelta(minutes=30),
    quote_history_bars: int = 100,
    max_snapshot_age: pd.Timedelta = pd.Timedelta(seconds=60),
    slippage_pct_per_side: float = 0.00005,
    limits: RiskLimits | None = None,
) -> PaperDecision:
    """Construct one dry order intent only when every release blocker passes."""

    limits = limits or RiskLimits()
    reasons: list[str] = []
    now = _utc(candidate.decision_time)

    if runtime.kill_switch_active:
        reasons.append("kill_switch_active")
    if candidate.intent_id in runtime.submitted_intent_ids:
        reasons.append("duplicate_intent_id")
    actual_position_ids = tuple(sorted(set(broker_open_position_ids)))
    actual_open_positions = len(actual_position_ids)
    if actual_open_positions != runtime.expected_open_positions:
        reasons.append("position_reconciliation_mismatch")
    if (
        runtime.expected_open_position_ids
        and actual_position_ids != tuple(sorted(set(runtime.expected_open_position_ids)))
    ):
        reasons.append("position_id_reconciliation_mismatch")
    if account.currency != market.account_currency:
        reasons.append("account_currency_mismatch")
    if account.status.upper() != "ENABLED":
        reasons.append(f"account_not_enabled:{account.status or 'UNKNOWN'}")
    if account.account_type.upper() != "CFD":
        reasons.append(f"unsupported_account_type:{account.account_type or 'UNKNOWN'}")
    if account.equity <= 0:
        reasons.append("non_positive_equity")
    if market.instrument_type != "COMMODITIES":
        reasons.append(f"unsupported_instrument_type:{market.instrument_type}")
    if market.unit != "CONTRACTS":
        reasons.append(f"unsupported_instrument_unit:{market.unit}")
    if not market.streaming_prices_available:
        reasons.append("streaming_prices_unavailable")
    if quote.epic != market.epic:
        reasons.append("quote_epic_mismatch")

    try:
        quote.validate_for_entry(now=now, max_age=max_snapshot_age)
    except RuntimeError as exc:
        reasons.append(str(exc))

    risk = assess_risk(
        RiskState(
            current_equity=account.equity,
            day_start_equity=runtime.day_start_equity,
            week_start_equity=runtime.week_start_equity,
            open_positions=actual_open_positions,
            entries_today=runtime.entries_today,
        ),
        limits,
    )
    reasons.extend(risk.reasons)
    if reasons:
        return PaperDecision(
            allowed=False,
            reasons=tuple(dict.fromkeys(reasons)),
            flatten_required=risk.flatten_positions,
            risk=risk,
            spread=None,
            sizing=None,
            intent=None,
        )

    try:
        closed_quotes = closed_quote_window(
            quote_candles,
            bar_duration=bar_duration,
            now=now,
            min_bars=quote_history_bars,
        )
        spread = assess_spread_history(
            closed_quotes,
            stop_distance_pct=candidate.stop_distance_pct,
            slippage_pct_per_side=slippage_pct_per_side,
        )
    except (RuntimeError, ValueError) as exc:
        return PaperDecision(
            allowed=False,
            reasons=(str(exc),),
            flatten_required=False,
            risk=risk,
            spread=None,
            sizing=None,
            intent=None,
        )

    current_cost_r = (
        quote.spread_fraction + 2 * slippage_pct_per_side
    ) / candidate.stop_distance_pct
    spread_reasons = list(spread.reasons)
    if current_cost_r > 0.25:
        spread_reasons.append("snapshot_cost_above_0.25R")
    if spread_reasons:
        return PaperDecision(
            allowed=False,
            reasons=tuple(dict.fromkeys(spread_reasons)),
            flatten_required=False,
            risk=risk,
            spread=spread,
            sizing=None,
            intent=None,
        )

    entry_price = quote.ask if candidate.side == "long" else quote.bid
    direction = "BUY" if candidate.side == "long" else "SELL"
    direction_sign = 1 if candidate.side == "long" else -1
    stop_price = entry_price * (1 - direction_sign * candidate.stop_distance_pct)
    target_price = entry_price * (1 + direction_sign * candidate.target_distance_pct)
    explicit_loss_per_size = (
        2
        * entry_price
        * market.value_per_price_unit
        * slippage_pct_per_side
    )
    sizing = size_for_stop(
        equity=account.equity,
        entry_price=entry_price,
        stop_price=stop_price,
        notional_per_size_unit=market.notional_per_size_unit(entry_price),
        instrument=InstrumentConstraints(
            min_size=market.min_size,
            size_increment=market.size_increment,
            value_per_price_unit=market.value_per_price_unit,
        ),
        limits=limits,
        additional_loss_per_size_unit=explicit_loss_per_size,
    )
    if not sizing.allowed:
        return PaperDecision(
            allowed=False,
            reasons=(sizing.reason or "sizing_rejected",),
            flatten_required=False,
            risk=risk,
            spread=spread,
            sizing=sizing,
            intent=None,
        )

    intent = PaperOrderIntent(
        intent_id=candidate.intent_id,
        strategy=candidate.strategy,
        epic=market.epic,
        account_id=account.account_id,
        account_currency=account.currency,
        direction=direction,
        size=sizing.size,
        reference_price=entry_price,
        stop_level=stop_price,
        target_level=target_price,
        created_at=now,
    )
    return PaperDecision(
        allowed=True,
        reasons=(),
        flatten_required=False,
        risk=risk,
        spread=spread,
        sizing=sizing,
        intent=intent,
    )


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    return timestamp.tz_convert("UTC")
