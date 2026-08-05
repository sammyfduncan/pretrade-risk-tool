"""Deterministic version-1 intraday risk gates and stop-based sizing.

This module contains policy, not alpha. It deliberately takes account-currency
values as inputs so broker contract sizes and FX conversions cannot be guessed.
The same functions can be called by research, paper, and live paths.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VolatilityRiskDecision:
    """Downside-only multiplier from a retained volatility forecast."""

    allowed: bool
    multiplier: float
    forecast: float
    threshold_80: float | None
    threshold_95: float | None
    reference_count: int
    reason: str


def assess_volatility_risk(
    *,
    forecast: float,
    decision_time: pd.Timestamp,
    forecast_history: pd.Series,
    artifact_approved: bool,
    minimum_history: int = 500,
) -> VolatilityRiskDecision:
    """Map a causal forecast to the frozen 1.0/0.5/0.0 risk ladder.

    Only strictly earlier OOF or prospective forecasts may be supplied in
    ``forecast_history``.  The function cannot verify their provenance, so the
    caller must separately validate the artifact and history manifest; the
    explicit approval flag makes that boundary fail closed.
    """

    if isinstance(minimum_history, bool) or not isinstance(minimum_history, int):
        raise ValueError("minimum_history must be an integer")
    if minimum_history < 1:
        raise ValueError("minimum_history must be positive")
    now = pd.Timestamp(decision_time)
    if now.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    now = now.tz_convert("UTC")
    if not artifact_approved:
        return _blocked_volatility_decision(forecast, 0, "forecast_artifact_unapproved")
    if not np.isfinite(forecast):
        return _blocked_volatility_decision(forecast, 0, "forecast_non_finite")
    if not isinstance(forecast_history, pd.Series):
        raise ValueError("forecast_history must be a pandas Series")
    if not isinstance(forecast_history.index, pd.DatetimeIndex):
        raise ValueError("forecast_history requires a DatetimeIndex")
    if forecast_history.index.tz is None:
        raise ValueError("forecast_history timestamps must be timezone-aware")
    history = forecast_history.copy()
    history.index = history.index.tz_convert("UTC")
    if not history.index.is_monotonic_increasing or not history.index.is_unique:
        raise ValueError("forecast_history must be sorted and unique")
    if len(history) and history.index[-1] >= now:
        return _blocked_volatility_decision(
            forecast,
            len(history),
            "forecast_history_not_strictly_prior",
        )
    values = history.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return _blocked_volatility_decision(
            forecast,
            len(history),
            "forecast_history_non_finite",
        )
    if len(values) < minimum_history:
        return _blocked_volatility_decision(
            forecast,
            len(values),
            "insufficient_forecast_history",
        )

    threshold_80 = float(np.quantile(values, 0.80, method="higher"))
    threshold_95 = float(np.quantile(values, 0.95, method="higher"))
    if forecast >= threshold_95:
        multiplier, reason = 0.0, "forecast_at_or_above_p95"
    elif forecast >= threshold_80:
        multiplier, reason = 0.5, "forecast_at_or_above_p80"
    else:
        multiplier, reason = 1.0, "forecast_below_p80"
    return VolatilityRiskDecision(
        allowed=multiplier > 0,
        multiplier=multiplier,
        forecast=float(forecast),
        threshold_80=threshold_80,
        threshold_95=threshold_95,
        reference_count=len(values),
        reason=reason,
    )


@dataclass(frozen=True)
class RiskLimits:
    """Frozen portfolio limits from ``docs/archive/intraday_rebuild_plan.md`` §2.3."""

    risk_per_trade: float = 0.0025
    daily_loss_limit: float = 0.0075
    weekly_loss_limit: float = 0.015
    max_open_positions: int = 1
    max_entries_per_day: int = 3
    max_notional_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in ("risk_per_trade", "daily_loss_limit", "weekly_loss_limit"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        if self.weekly_loss_limit < self.daily_loss_limit:
            raise ValueError("weekly_loss_limit cannot be below daily_loss_limit")
        for name in ("max_open_positions", "max_entries_per_day"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not np.isfinite(self.max_notional_fraction)
            or self.max_notional_fraction <= 0
        ):
            raise ValueError("max_notional_fraction must be finite and positive")


@dataclass(frozen=True)
class RiskState:
    """Portfolio state at one decision instant.

    ``current_equity`` must include realized and open P&L. Day/week starts are
    reset by the caller on the Europe/London trading boundaries.
    """

    current_equity: float
    day_start_equity: float
    week_start_equity: float
    open_positions: int = 0
    entries_today: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.current_equity):
            raise ValueError("current_equity must be finite")
        for name in ("day_start_equity", "week_start_equity"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("open_positions", "entries_today"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class RiskDecision:
    """Auditable entry/flatten decision for the supplied portfolio state."""

    allow_entry: bool
    flatten_positions: bool
    reasons: tuple[str, ...]
    daily_drawdown: float
    weekly_drawdown: float


def assess_risk(
    state: RiskState,
    limits: RiskLimits | None = None,
) -> RiskDecision:
    """Apply loss, overlap, and entry-count gates without recovery sizing."""

    limits = limits or RiskLimits()
    daily_drawdown = max(0.0, 1 - state.current_equity / state.day_start_equity)
    weekly_drawdown = max(0.0, 1 - state.current_equity / state.week_start_equity)

    reasons: list[str] = []
    if state.current_equity <= state.day_start_equity * (1 - limits.daily_loss_limit):
        reasons.append("daily_loss_limit")
    if state.current_equity <= state.week_start_equity * (1 - limits.weekly_loss_limit):
        reasons.append("weekly_loss_limit")

    flatten_positions = bool(reasons)
    if state.open_positions >= limits.max_open_positions:
        reasons.append("max_open_positions")
    if state.entries_today >= limits.max_entries_per_day:
        reasons.append("max_entries_per_day")

    return RiskDecision(
        allow_entry=not reasons,
        flatten_positions=flatten_positions,
        reasons=tuple(reasons),
        daily_drawdown=daily_drawdown,
        weekly_drawdown=weekly_drawdown,
    )


@dataclass(frozen=True)
class InstrumentConstraints:
    """Broker sizing facts expressed in the trading account's currency.

    ``value_per_price_unit`` is the P&L change for one unit of position when
    price changes by one quoted price point.
    """

    min_size: float
    size_increment: float
    value_per_price_unit: float

    def __post_init__(self) -> None:
        for name in ("min_size", "size_increment", "value_per_price_unit"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class SizingDecision:
    """Result of applying both stop-risk and gross-notional caps."""

    allowed: bool
    size: float
    risk_budget: float
    planned_stop_loss: float
    gross_notional: float
    binding_constraint: str | None
    reason: str | None


def size_for_stop(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    notional_per_size_unit: float,
    instrument: InstrumentConstraints,
    limits: RiskLimits | None = None,
    additional_loss_per_size_unit: float = 0.0,
    risk_multiplier: float = 1.0,
) -> SizingDecision:
    """Size a position without exceeding stop-risk or notional limits.

    Every monetary input must already be converted to the account currency.
    ``additional_loss_per_size_unit`` reserves room for explicit commission
    and adverse stop slippage. Bid/ask spread should already be reflected in
    executable ``entry_price`` and ``stop_price``. ``risk_multiplier`` is a
    downside-only policy input in ``[0, 1]``; zero refuses minimum size.
    """

    limits = limits or RiskLimits()
    for name, value in (
        ("equity", equity),
        ("entry_price", entry_price),
        ("stop_price", stop_price),
        ("notional_per_size_unit", notional_per_size_unit),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if entry_price == stop_price:
        raise ValueError("entry_price and stop_price must differ")
    if (
        not np.isfinite(additional_loss_per_size_unit)
        or additional_loss_per_size_unit < 0
    ):
        raise ValueError(
            "additional_loss_per_size_unit must be finite and non-negative"
        )
    if not np.isfinite(risk_multiplier) or not 0 <= risk_multiplier <= 1:
        raise ValueError("risk_multiplier must be finite and between 0 and 1")

    risk_budget = equity * limits.risk_per_trade * risk_multiplier
    risk_per_size_unit = (
        abs(entry_price - stop_price) * instrument.value_per_price_unit
        + additional_loss_per_size_unit
    )
    risk_cap_size = risk_budget / risk_per_size_unit
    notional_cap_size = (
        equity * limits.max_notional_fraction / notional_per_size_unit
    )
    cap_size = min(risk_cap_size, notional_cap_size)

    if cap_size + 1e-12 < instrument.min_size:
        return SizingDecision(
            allowed=False,
            size=0.0,
            risk_budget=risk_budget,
            planned_stop_loss=0.0,
            gross_notional=0.0,
            binding_constraint=_binding_constraint(
                risk_cap_size,
                notional_cap_size,
            ),
            reason="minimum_size_exceeds_caps",
        )

    steps = math.floor(
        (cap_size - instrument.min_size) / instrument.size_increment + 1e-12
    )
    size = instrument.min_size + steps * instrument.size_increment
    # Normalize tiny binary floating-point tails without changing broker steps.
    size = float(round(size, 12))
    planned_stop_loss = size * risk_per_size_unit
    gross_notional = size * notional_per_size_unit

    return SizingDecision(
        allowed=True,
        size=size,
        risk_budget=risk_budget,
        planned_stop_loss=planned_stop_loss,
        gross_notional=gross_notional,
        binding_constraint=_binding_constraint(
            risk_cap_size,
            notional_cap_size,
        ),
        reason=None,
    )


def _binding_constraint(risk_cap_size: float, notional_cap_size: float) -> str:
    if math.isclose(risk_cap_size, notional_cap_size, rel_tol=1e-12, abs_tol=1e-12):
        return "risk_and_notional"
    return "risk" if risk_cap_size < notional_cap_size else "notional"


def _blocked_volatility_decision(
    forecast: float,
    reference_count: int,
    reason: str,
) -> VolatilityRiskDecision:
    return VolatilityRiskDecision(
        allowed=False,
        multiplier=0.0,
        forecast=float(forecast),
        threshold_80=None,
        threshold_95=None,
        reference_count=reference_count,
        reason=reason,
    )
