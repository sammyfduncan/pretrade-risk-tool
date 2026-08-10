from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload
from commodity_ml.intraday.risk_calibration import CalibratedRiskLookup


def valid_payload() -> dict:
    return {
        "trade_intent": {
            "intent_id": "gold-20260803-001",
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
            "instrument_type": "COMMODITIES",
            "expiry": "DFB",
            "account_currency": "GBP",
            "min_size": 0.1,
            "size_increment": 0.1,
            "value_per_price_unit": 1.0,
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


def test_complete_trade_is_allowed_and_risk_is_cost_inclusive():
    report = evaluate_pretrade_payload(valid_payload())

    assert report.decision == "ALLOW"
    assert report.reasons == ()
    assert report.trade["broker_instrument"] == "XAU_USD"
    assert report.sizing["recommended_max_size"] == pytest.approx(12.3)
    assert report.sizing["recommended_action_size"] == pytest.approx(10.0)
    assert report.sizing["binding_constraint"] == "risk"
    assert report.cash_risk["proposed"]["planned_stop_loss"] == pytest.approx(203.0)
    assert report.cash_risk["proposed"]["round_trip_cost"] == pytest.approx(3.0)
    assert report.cash_risk["proposed"]["gross_exposure"] == pytest.approx(24_000)
    assert report.cash_risk["proposed"]["initial_margin"] == pytest.approx(1_200)
    assert report.costs["current_cost_r"] == pytest.approx(0.015)
    assert report.schema_version == 3
    assert report.assessment["version"] == "policy-pressure-v1-2026-08-10"
    assert report.assessment["policy_pressure_score"] == 6
    assert report.assessment["policy_pressure_driver"] == "size_cap"
    assert report.assessment["maximum_pressure_ratio"] == pytest.approx(10 / 12.3)
    assert report.assessment["evidence_score"] == 1
    assert report.assessment["score_is_loss_probability"] is False
    assert [row["name"] for row in report.scenarios] == [
        "planned_stop",
        "planned_stop_spread_1_5x",
        "planned_stop_spread_2x",
    ]
    assert all(not row["probability_assigned"] for row in report.scenarios)
    json.dumps(report.to_dict(), allow_nan=False)


def test_oversized_request_is_reduced_to_broker_step():
    payload = valid_payload()
    payload["trade_intent"]["requested_size"] = 20.0

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "REDUCE"
    assert report.reasons == (
        "requested_size_exceeds_or_misses_safe_broker_cap",
    )
    assert report.sizing["recommended_max_size"] == pytest.approx(12.3)
    assert report.cash_risk["proposed"]["planned_stop_loss"] > 250
    assert report.cash_risk["at_recommended_max"]["planned_stop_loss"] <= 250
    assert report.assessment["policy_pressure_score"] == 7
    assert report.assessment["policy_pressure_driver"] == "size_cap"


def test_missing_broker_fact_is_incomplete_not_guessed():
    payload = valid_payload()
    payload["instrument"]["value_per_price_unit"] = None

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "INCOMPLETE"
    assert "missing:instrument.value_per_price_unit" in report.reasons
    assert report.sizing == {}
    assert report.assessment["policy_pressure_score"] is None


def test_missing_spread_history_blocks_shadow_admission():
    payload = valid_payload()
    payload["market"]["median_spread_points"] = None
    payload["market"]["p95_spread_points"] = None

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert "median_spread_history_unavailable" in report.reasons
    assert "p95_spread_history_unavailable" in report.reasons


def test_off_increment_request_recommends_a_smaller_valid_action_size():
    payload = valid_payload()
    payload["trade_intent"]["requested_size"] = 10.05

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "REDUCE"
    assert report.sizing["recommended_max_size"] == pytest.approx(12.3)
    assert report.sizing["recommended_action_size"] == pytest.approx(10.0)


def test_invalid_directional_geometry_is_blocked():
    payload = valid_payload()
    payload["trade_intent"]["stop_price"] = 2410.0

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert report.reasons == ("invalid_long_price_geometry",)
    assert report.assessment["policy_pressure_score"] == 7
    assert report.assessment["policy_pressure_driver"] == "hard_gate:invalid_long_price_geometry"


def test_quote_for_a_different_broker_instrument_is_blocked():
    payload = valid_payload()
    payload["market"]["broker_instrument"] = "XAG_USD"

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert report.reasons == ("quote_instrument_mismatch",)


def test_delayed_quote_is_blocked_even_when_fresh():
    payload = valid_payload()
    payload["market"]["quote_delay_seconds"] = 15

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert report.reasons == ("delayed_quote",)


def test_v1_refuses_wrong_account_or_product_contract():
    payload = valid_payload()
    payload["account"]["account_type"] = "CFD"
    payload["instrument"]["instrument_type"] = "CURRENCIES"
    payload["instrument"]["expiry"] = "SEP-26"

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert report.reasons == (
        "unsupported_account_type",
        "unsupported_instrument_type",
        "unsupported_expiring_instrument",
    )


def test_margin_rate_band_scope_is_a_hard_size_cap():
    payload = valid_payload()
    payload["trade_intent"]["requested_size"] = None
    payload["instrument"]["margin_rate_max_size"] = 5.05

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "ALLOW"
    assert report.sizing["recommended_max_size"] == pytest.approx(5.0)
    assert report.sizing["binding_constraint"] == "margin_schedule_scope"


def test_persistent_kill_switch_blocks_otherwise_valid_trade():
    payload = valid_payload()
    payload["account"]["kill_switch_active"] = True

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert "kill_switch_active" in report.reasons


def test_session_event_and_loss_gates_accumulate_reason_codes():
    payload = valid_payload()
    payload["trade_intent"]["decision_time"] = "2026-08-03T19:00:00Z"
    payload["market"]["as_of"] = "2026-08-03T19:00:00Z"
    payload["market"]["scheduled_event_blackout_active"] = True
    payload["account"]["equity"] = 99_000.0

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert "scheduled_event_blackout" in report.reasons
    assert "outside_entry_session" in report.reasons
    assert "planned_exit_after_daily_cutoff" in report.reasons
    assert "daily_loss_limit" in report.reasons


def test_approved_volatility_forecast_can_only_reduce_or_block():
    half = valid_payload()
    half["trade_intent"]["requested_size"] = None
    half["market"].update(
        {
            "volatility_artifact_approved": True,
            "volatility_percentile": 0.85,
            "forecast_range_points": 12.0,
            "forecast_horizon_minutes": 180,
        }
    )
    normal = deepcopy(half)
    normal["market"]["volatility_percentile"] = 0.50

    half_report = evaluate_pretrade_payload(half)
    normal_report = evaluate_pretrade_payload(normal)

    assert half_report.decision == "ALLOW"
    assert half_report.sizing["volatility_multiplier"] == 0.5
    assert half_report.sizing["recommended_max_size"] < normal_report.sizing[
        "recommended_max_size"
    ]
    assert "gap_half_range_beyond_stop" in {
        row["name"] for row in half_report.scenarios
    }

    blocked = deepcopy(half)
    blocked["market"]["volatility_percentile"] = 0.95
    blocked_report = evaluate_pretrade_payload(blocked)
    assert blocked_report.decision == "BLOCK"
    assert "volatility_at_or_above_p95" in blocked_report.reasons
    assert blocked_report.sizing["recommended_max_size"] == 0


def test_existing_position_requires_a_stop_and_then_hits_overlap_gate():
    payload = valid_payload()
    payload["existing_positions"] = [
        {
            "instrument": "XAGUSD",
            "side": "short",
            "size": 2.0,
            "current_price": 30.0,
            "stop_price": None,
            "value_per_price_unit": 1.0,
            "margin_required": 3.0,
        }
    ]

    incomplete = evaluate_pretrade_payload(payload)
    assert incomplete.decision == "INCOMPLETE"
    assert "missing:existing_positions[0].stop_price" in incomplete.reasons

    payload["existing_positions"][0]["stop_price"] = 31.0
    blocked = evaluate_pretrade_payload(payload)
    assert blocked.decision == "BLOCK"
    assert "max_open_positions" in blocked.reasons
    assert blocked.portfolio["existing_stop_risk"] == pytest.approx(2.0)
    assert blocked.portfolio["existing_signed_one_pct_move"] == pytest.approx(-0.6)


def test_cost_history_can_block_a_trade_even_when_current_quote_is_tight():
    payload = valid_payload()
    payload["market"]["median_spread_points"] = 3.0
    payload["market"]["p95_spread_points"] = 5.1

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "BLOCK"
    assert "median_cost_above_0.10R" in report.reasons
    assert "p95_cost_above_0.25R" in report.reasons


def test_unknown_json_field_is_refused():
    payload = valid_payload()
    payload["trade_intent"]["directional_confidence"] = 0.99

    with pytest.raises(ValueError, match="directional_confidence"):
        evaluate_pretrade_payload(payload)


def test_short_trade_uses_symmetric_geometry_and_signed_exposure():
    payload = valid_payload()
    payload["trade_intent"].update(
        {
            "side": "short",
            "stop_price": 2420.0,
            "target_price": 2370.0,
        }
    )

    report = evaluate_pretrade_payload(payload)

    assert report.decision == "ALLOW"
    assert report.portfolio["proposed_signed_one_pct_move"] == pytest.approx(-240.0)


def test_admitted_current_calibration_adds_quantile_scenarios():
    calibration = CalibratedRiskLookup(
        instrument="XAUUSD",
        side="long",
        horizon_minutes=180,
        london_slot_minute=11 * 60,
        profile_as_of=pd.Timestamp("2026-07-30T10:00:00Z"),
        sample_count=252,
        range_points={"p50": 8.0, "p80": 12.0, "p95": 18.0, "p99": 25.0},
        adverse_points={"p50": 4.0, "p80": 7.0, "p95": 11.0, "p99": 16.0},
        reference_spread_points={
            "p50": 0.2,
            "p80": 0.3,
            "p95": 0.4,
            "p99": 0.6,
        },
        admitted_for_current_use=True,
        policy_version="current_range_v1",
        release_id="release-2026-08-01",
    )

    report = evaluate_pretrade_payload(
        valid_payload(), historical_calibration=calibration
    )

    assert report.decision == "ALLOW"
    names = {scenario["name"] for scenario in report.scenarios}
    assert "historical_total_range_p95" in names
    assert "historical_side_adverse_p95" in names
    assert report.market_risk["historical_calibration"]["sample_count"] == 252
    assert report.market_risk["historical_calibration"]["admitted_for_current_use"] is True


def test_unadmitted_calibration_fails_closed_without_scenarios():
    calibration = CalibratedRiskLookup(
        instrument="XAUUSD",
        side="long",
        horizon_minutes=180,
        london_slot_minute=11 * 60,
        profile_as_of=pd.Timestamp("2026-08-01T10:00:00Z"),
        sample_count=252,
        range_points={"p50": 8.0, "p80": 12.0, "p95": 18.0, "p99": 25.0},
        adverse_points={"p50": 4.0, "p80": 7.0, "p95": 11.0, "p99": 16.0},
        reference_spread_points={"p50": 0.2, "p80": 0.3, "p95": 0.4, "p99": 0.6},
    )

    report = evaluate_pretrade_payload(valid_payload(), historical_calibration=calibration)

    assert "historical_calibration_context_mismatch" in report.reasons
    assert not any(row["name"].startswith("historical_") for row in report.scenarios)
