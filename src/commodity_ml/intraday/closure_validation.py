"""Deterministic closure scenarios for the completed risk-framework prototype."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload


VALIDATION_VERSION = "risk-framework-closure-v1-2026-08-10"


def baseline_payload() -> dict[str, Any]:
    """Return a complete, synthetic account/trade snapshot with no future data."""
    return {
        "trade_intent": {
            "intent_id": "closure-baseline",
            "instrument": "XAUUSD",
            "side": "long",
            "order_type": "market",
            "decision_time": "2026-08-03T10:00:00Z",
            "entry_price": 2400.0,
            "stop_price": 2380.0,
            "target_price": 2430.0,
            "requested_size": 10.0,
            "holding_period_minutes": 180,
            "allow_overnight": False,
            "max_slippage_points_per_side": 0.05,
        },
        "account": {
            "account_type": "SPREADBET",
            "status": "ENABLED",
            "currency": "GBP",
            "equity": 100_000.0,
            "available_funds": 80_000.0,
            "day_start_equity": 100_000.0,
            "week_start_equity": 100_000.0,
            "entries_today": 0,
            "kill_switch_active": False,
        },
        "instrument": {
            "instrument": "XAUUSD",
            "broker_instrument": "XAU_USD",
            "instrument_type": "METAL",
            "expiry": "DFB",
            "account_currency": "GBP",
            "min_size": 0.1,
            "size_increment": 0.1,
            "value_per_price_unit": 1.0,
            "notional_value_per_price_unit": 1.0,
            "margin_rate": 0.05,
            "margin_rate_max_size": None,
            "commission_per_size_per_side": 0.0,
        },
        "market": {
            "broker_instrument": "XAU_USD",
            "as_of": "2026-08-03T09:59:30Z",
            "bid": 2399.9,
            "ask": 2400.1,
            "market_open": True,
            "quote_delay_seconds": 0,
            "scheduled_event_blackout_active": False,
            "median_spread_points": 0.2,
            "p95_spread_points": 0.4,
            "forecast_range_points": None,
            "forecast_horizon_minutes": None,
            "volatility_percentile": None,
            "volatility_artifact_approved": False,
        },
        "existing_positions": [],
    }


def run_closure_scenarios() -> dict[str, Any]:
    """Exercise representative decisions and fail closed if behavior drifts."""
    cases = [
        _scenario("gold_allow", lambda value: None, "ALLOW", (), 6),
        _scenario(
            "oversized_reduce",
            lambda value: value["trade_intent"].update(requested_size=20.0),
            "REDUCE",
            ("requested_size_exceeds_or_misses_safe_broker_cap",),
            7,
        ),
        _scenario(
            "missing_contract_value",
            lambda value: value["instrument"].update(value_per_price_unit=None),
            "INCOMPLETE",
            ("missing:instrument.value_per_price_unit",),
            None,
        ),
        _scenario(
            "invalid_directional_geometry",
            lambda value: value["trade_intent"].update(stop_price=2410.0),
            "BLOCK",
            ("invalid_long_price_geometry",),
            7,
        ),
        _scenario(
            "market_closed",
            lambda value: value["market"].update(market_open=False),
            "BLOCK",
            ("market_not_open",),
            7,
        ),
        _scenario(
            "stale_quote",
            lambda value: value["market"].update(as_of="2026-08-03T09:58:00Z"),
            "BLOCK",
            ("stale_quote",),
            7,
        ),
        _scenario(
            "delayed_quote",
            lambda value: value["market"].update(quote_delay_seconds=15),
            "BLOCK",
            ("delayed_quote",),
            7,
        ),
        _scenario(
            "scheduled_event_blackout",
            lambda value: value["market"].update(
                scheduled_event_blackout_active=True
            ),
            "BLOCK",
            ("scheduled_event_blackout",),
            7,
        ),
        _scenario(
            "daily_drawdown_limit",
            lambda value: value["account"].update(equity=99_000.0),
            "BLOCK",
            ("daily_loss_limit",),
            7,
        ),
        _scenario(
            "weekly_drawdown_limit",
            lambda value: value["account"].update(
                equity=98_000.0,
                day_start_equity=98_000.0,
            ),
            "BLOCK",
            ("weekly_loss_limit",),
            7,
        ),
        _scenario(
            "entry_count_limit",
            lambda value: value["account"].update(entries_today=3),
            "BLOCK",
            ("max_entries_per_day",),
            7,
        ),
        _scenario(
            "kill_switch",
            lambda value: value["account"].update(kill_switch_active=True),
            "BLOCK",
            ("kill_switch_active",),
            7,
        ),
        _scenario(
            "execution_cost_limit",
            _tight_cost_geometry,
            "BLOCK",
            (
                "current_cost_above_0.25R",
                "median_cost_above_0.10R",
                "p95_cost_above_0.25R",
            ),
            7,
        ),
        _scenario(
            "open_position_limit",
            _add_existing_position,
            "BLOCK",
            ("max_open_positions",),
            7,
        ),
        _scenario(
            "margin_constrained_reduce",
            _constrain_margin,
            "REDUCE",
            ("requested_size_exceeds_or_misses_safe_broker_cap",),
            7,
        ),
        _scenario("short_allow", _make_short, "ALLOW", (), 6),
        _scenario("silver_allow", _make_silver, "ALLOW", (), 3),
        _scenario(
            "approved_extreme_volatility",
            _extreme_volatility,
            "BLOCK",
            ("volatility_at_or_above_p95",),
            7,
        ),
    ]
    results = [_evaluate_case(case) for case in cases]
    return {
        "schema_version": 1,
        "validation_version": VALIDATION_VERSION,
        "uses_future_outcomes": False,
        "directional_signal_used": False,
        "strategy_pnl_used": False,
        "scenario_count": len(results),
        "passed_count": sum(bool(item["passed"]) for item in results),
        "passed": all(bool(item["passed"]) for item in results),
        "scenarios": results,
    }


def _scenario(
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    expected_decision: str,
    expected_reasons: tuple[str, ...],
    expected_score: int | None,
) -> dict[str, Any]:
    payload = deepcopy(baseline_payload())
    payload["trade_intent"]["intent_id"] = f"closure-{name}"
    mutate(payload)
    return {
        "name": name,
        "payload": payload,
        "expected_decision": expected_decision,
        "expected_reasons": expected_reasons,
        "expected_score": expected_score,
    }


def _evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    report = evaluate_pretrade_payload(case["payload"])
    reasons = tuple(report.reasons)
    score = report.assessment.get("policy_pressure_score")
    missing_reasons = sorted(set(case["expected_reasons"]).difference(reasons))
    score_matches = score == case["expected_score"]
    passed = (
        report.decision == case["expected_decision"]
        and not missing_reasons
        and score_matches
    )
    return {
        "name": case["name"],
        "expected_decision": case["expected_decision"],
        "actual_decision": report.decision,
        "expected_reasons": list(case["expected_reasons"]),
        "actual_reasons": list(reasons),
        "missing_expected_reasons": missing_reasons,
        "expected_policy_pressure_score": case["expected_score"],
        "actual_policy_pressure_score": score,
        "passed": passed,
    }


def _tight_cost_geometry(value: dict[str, Any]) -> None:
    value["trade_intent"].update(
        stop_price=2399.0,
        target_price=2403.0,
        max_slippage_points_per_side=0.1,
    )


def _add_existing_position(value: dict[str, Any]) -> None:
    value["existing_positions"] = [
        {
            "instrument": "XAGUSD",
            "side": "long",
            "size": 1.0,
            "current_price": 30.0,
            "stop_price": 29.0,
            "value_per_price_unit": 1.0,
            "margin_required": 1.5,
        }
    ]


def _constrain_margin(value: dict[str, Any]) -> None:
    value["account"].update(
        equity=10_000.0,
        available_funds=100.0,
        day_start_equity=10_000.0,
        week_start_equity=10_000.0,
    )


def _make_short(value: dict[str, Any]) -> None:
    value["trade_intent"].update(
        side="short",
        stop_price=2420.0,
        target_price=2370.0,
    )


def _make_silver(value: dict[str, Any]) -> None:
    value["trade_intent"].update(
        instrument="XAGUSD",
        entry_price=30.0,
        stop_price=29.0,
        target_price=32.0,
        requested_size=10.0,
        max_slippage_points_per_side=0.01,
    )
    value["instrument"].update(
        instrument="XAGUSD",
        broker_instrument="XAG_USD",
    )
    value["market"].update(
        broker_instrument="XAG_USD",
        bid=29.99,
        ask=30.01,
        median_spread_points=0.02,
        p95_spread_points=0.04,
    )


def _extreme_volatility(value: dict[str, Any]) -> None:
    value["market"].update(
        forecast_range_points=20.0,
        forecast_horizon_minutes=180,
        volatility_percentile=0.95,
        volatility_artifact_approved=True,
    )
