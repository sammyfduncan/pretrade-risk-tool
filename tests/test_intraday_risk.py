"""Synthetic tests for frozen intraday risk and sizing policy."""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.intraday.risk import (
    InstrumentConstraints,
    RiskLimits,
    RiskState,
    assess_risk,
    assess_volatility_risk,
    size_for_stop,
)


def test_default_risk_state_allows_an_entry():
    decision = assess_risk(
        RiskState(
            current_equity=100_000,
            day_start_equity=100_000,
            week_start_equity=100_000,
        )
    )

    assert decision.allow_entry
    assert not decision.flatten_positions
    assert decision.reasons == ()


def test_daily_loss_gate_includes_open_pnl_and_requires_flattening():
    decision = assess_risk(
        RiskState(
            current_equity=99_250,
            day_start_equity=100_000,
            week_start_equity=100_000,
            open_positions=1,
        )
    )

    assert not decision.allow_entry
    assert decision.flatten_positions
    assert decision.daily_drawdown == pytest.approx(0.0075)
    assert "daily_loss_limit" in decision.reasons


def test_weekly_loss_gate_pauses_even_when_daily_gate_is_clear():
    decision = assess_risk(
        RiskState(
            current_equity=98_500,
            day_start_equity=99_000,
            week_start_equity=100_000,
        )
    )

    assert not decision.allow_entry
    assert decision.flatten_positions
    assert decision.weekly_drawdown == pytest.approx(0.015)
    assert decision.reasons == ("weekly_loss_limit",)


def test_catastrophic_non_positive_equity_still_produces_a_flatten_decision():
    decision = assess_risk(
        RiskState(
            current_equity=-100,
            day_start_equity=100_000,
            week_start_equity=100_000,
            open_positions=1,
        )
    )

    assert not decision.allow_entry
    assert decision.flatten_positions
    assert "daily_loss_limit" in decision.reasons
    assert "weekly_loss_limit" in decision.reasons


@pytest.mark.parametrize(
    ("open_positions", "entries_today", "expected_reason"),
    [
        (1, 0, "max_open_positions"),
        (0, 3, "max_entries_per_day"),
    ],
)
def test_overlap_and_entry_count_block_without_forced_flatten(
    open_positions,
    entries_today,
    expected_reason,
):
    decision = assess_risk(
        RiskState(
            current_equity=100_000,
            day_start_equity=100_000,
            week_start_equity=100_000,
            open_positions=open_positions,
            entries_today=entries_today,
        )
    )

    assert not decision.allow_entry
    assert not decision.flatten_positions
    assert decision.reasons == (expected_reason,)


def test_stop_sizing_uses_point_value_and_floors_to_broker_increment():
    result = size_for_stop(
        equity=100_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=InstrumentConstraints(
            min_size=0.1,
            size_increment=0.1,
            value_per_price_unit=1.0,
        ),
    )

    assert result.allowed
    assert result.binding_constraint == "risk"
    assert result.risk_budget == pytest.approx(250)
    assert result.size == pytest.approx(25.0)
    assert result.planned_stop_loss == pytest.approx(250)
    assert result.gross_notional == pytest.approx(62_500)


def test_notional_cap_can_bind_before_stop_risk():
    result = size_for_stop(
        equity=10_000,
        entry_price=2_500,
        stop_price=2_499,
        notional_per_size_unit=2_500,
        instrument=InstrumentConstraints(
            min_size=0.1,
            size_increment=0.1,
            value_per_price_unit=1.0,
        ),
    )

    assert result.allowed
    assert result.binding_constraint == "notional"
    assert result.size == pytest.approx(4.0)
    assert result.gross_notional == pytest.approx(10_000)
    assert result.planned_stop_loss < result.risk_budget


def test_minimum_size_is_rejected_when_it_would_break_a_cap():
    result = size_for_stop(
        equity=1_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=InstrumentConstraints(
            min_size=1.0,
            size_increment=0.1,
            value_per_price_unit=1.0,
        ),
    )

    assert not result.allowed
    assert result.size == 0
    assert result.reason == "minimum_size_exceeds_caps"


def test_explicit_stop_cost_reserve_reduces_size():
    instrument = InstrumentConstraints(
        min_size=0.1,
        size_increment=0.1,
        value_per_price_unit=1.0,
    )
    without_cost = size_for_stop(
        equity=100_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=instrument,
    )
    with_cost = size_for_stop(
        equity=100_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=instrument,
        additional_loss_per_size_unit=2.0,
    )

    assert with_cost.size < without_cost.size
    assert with_cost.planned_stop_loss <= with_cost.risk_budget


def test_non_multiple_minimum_size_uses_minimum_plus_broker_steps():
    result = size_for_stop(
        equity=1_000,
        entry_price=100,
        stop_price=99,
        notional_per_size_unit=100,
        instrument=InstrumentConstraints(
            min_size=0.25,
            size_increment=0.1,
            value_per_price_unit=1.0,
        ),
        limits=RiskLimits(risk_per_trade=0.00065),
    )

    assert result.allowed
    assert result.size == pytest.approx(0.65)
    assert result.planned_stop_loss <= result.risk_budget


def forecast_history(n: int = 500) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq="30min", tz="UTC")
    return pd.Series(np.arange(n, dtype=float), index=index)


def test_volatility_policy_uses_frozen_expanding_rank_ladder():
    history = forecast_history()
    decision_time = history.index[-1] + pd.Timedelta(minutes=30)

    normal = assess_volatility_risk(
        forecast=100,
        decision_time=decision_time,
        forecast_history=history,
        artifact_approved=True,
    )
    reduced = assess_volatility_risk(
        forecast=450,
        decision_time=decision_time,
        forecast_history=history,
        artifact_approved=True,
    )
    blocked = assess_volatility_risk(
        forecast=490,
        decision_time=decision_time,
        forecast_history=history,
        artifact_approved=True,
    )

    assert normal.multiplier == 1.0
    assert reduced.multiplier == 0.5
    assert blocked.multiplier == 0.0
    assert not blocked.allowed
    assert normal.threshold_80 == 400
    assert normal.threshold_95 == 475


def test_volatility_policy_fails_closed_without_approval_or_history():
    history = forecast_history(499)
    decision_time = history.index[-1] + pd.Timedelta(minutes=30)

    unapproved = assess_volatility_risk(
        forecast=0.0,
        decision_time=decision_time,
        forecast_history=history,
        artifact_approved=False,
    )
    warming = assess_volatility_risk(
        forecast=0.0,
        decision_time=decision_time,
        forecast_history=history,
        artifact_approved=True,
    )

    assert unapproved.reason == "forecast_artifact_unapproved"
    assert warming.reason == "insufficient_forecast_history"
    assert unapproved.multiplier == warming.multiplier == 0.0


def test_volatility_policy_rejects_future_dated_reference_history():
    history = forecast_history()
    result = assess_volatility_risk(
        forecast=0.0,
        decision_time=history.index[-1],
        forecast_history=history,
        artifact_approved=True,
    )
    assert result.reason == "forecast_history_not_strictly_prior"
    assert not result.allowed


def test_risk_multiplier_can_only_reduce_sizing_budget():
    instrument = InstrumentConstraints(
        min_size=0.1,
        size_increment=0.1,
        value_per_price_unit=1.0,
    )
    full = size_for_stop(
        equity=100_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=instrument,
    )
    half = size_for_stop(
        equity=100_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=instrument,
        risk_multiplier=0.5,
    )
    blocked = size_for_stop(
        equity=100_000,
        entry_price=2_500,
        stop_price=2_490,
        notional_per_size_unit=2_500,
        instrument=instrument,
        risk_multiplier=0.0,
    )

    assert half.risk_budget == pytest.approx(full.risk_budget / 2)
    assert half.size <= full.size
    assert not blocked.allowed
    assert blocked.risk_budget == 0
