from __future__ import annotations

import pytest

from commodity_ml.intraday.pretrade_report import render_pretrade_report_html


def report() -> dict:
    return {
        "schema_version": 1,
        "policy_version": "test-policy",
        "intent_id": "risk-1",
        "decision": "REDUCE",
        "reasons": ["requested_size_exceeds_or_misses_safe_broker_cap"],
        "warnings": ["net_reward_risk_below_1"],
        "trade": {
            "instrument": "XAUUSD",
            "broker_instrument": "XAU_USD",
            "side": "long",
            "order_type": "market",
            "decision_time": "2026-08-03T10:00:00+00:00",
        },
        "sizing": {
            "requested_size": 20,
            "assessed_size": 20,
            "recommended_max_size": 12.3,
            "recommended_action_size": 12.3,
            "binding_constraint": "risk",
        },
        "cash_risk": {
            "account_currency": "GBP",
            "proposed": {
                "planned_stop_loss": 406,
                "planned_stop_loss_fraction": 0.00406,
                "net_target_reward": 594,
                "net_reward_risk": 1.46,
                "round_trip_cost": 6,
                "initial_margin": 2400,
            },
        },
        "costs": {
            "current_spread_points": 0.2,
            "current_cost_r": 0.015,
            "median_cost_r": 0.015,
            "p95_cost_r": 0.025,
        },
        "market_risk": {"quote_age_seconds": 10, "volatility_state": "unavailable"},
        "portfolio": {
            "existing_position_count": 0,
            "proposed_total_stop_risk": 406,
            "proposed_total_margin": 2400,
            "proposed_total_gross_exposure": 48_000,
            "combined_signed_one_pct_move": 480,
        },
        "scenarios": [
            {
                "name": "planned_stop",
                "cash_loss": 406,
                "equity_fraction": 0.00406,
                "probability_assigned": False,
            }
        ],
        "limitations": ["A stop is not a guaranteed maximum loss."],
    }


def test_renderer_outputs_standalone_auditable_report():
    html = render_pretrade_report_html(report())

    assert html.startswith("<!doctype html>")
    assert "REDUCE" in html
    assert "12.3" in html
    assert "planned_stop" in html
    assert "Order authority" in html
    assert "disabled" in html
    assert "<script" not in html
    assert "http://" not in html
    assert "https://" not in html


def test_renderer_accepts_oanda_wrapper_and_shows_provenance():
    wrapper = {
        "mode": "OANDA_V20_READ_ONLY",
        "order_authority": False,
        "adapter_issues": ["broker_reports_hedging_disabled"],
        "provenance": {
            "broker_instrument": "XAU_USD",
            "quote_timestamp": "2026-08-03T10:00:00+00:00",
            "margin_source": "account.instrument.marginRate",
            "spread_sample_rows": 18,
        },
        "risk_report": report(),
    }

    html = render_pretrade_report_html(wrapper)

    assert "OANDA_V20_READ_ONLY" in html
    assert "account.instrument.marginRate" in html
    assert "broker_reports_hedging_disabled" in html


def test_renderer_escapes_untrusted_text():
    value = report()
    value["warnings"] = ["<script>alert('x')</script>"]

    html = render_pretrade_report_html(value)

    assert "<script>alert" not in html
    assert "&lt;script&gt;alert" in html


def test_renderer_refuses_non_report_document():
    with pytest.raises(ValueError, match="does not contain"):
        render_pretrade_report_html({"hello": "world"})
