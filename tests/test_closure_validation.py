from __future__ import annotations

from commodity_ml.intraday.closure_validation import run_closure_scenarios


def test_closure_scenario_matrix_passes_without_future_outcomes():
    result = run_closure_scenarios()

    assert result["uses_future_outcomes"] is False
    assert result["directional_signal_used"] is False
    assert result["strategy_pnl_used"] is False
    assert result["scenario_count"] >= 18
    assert result["passed_count"] == result["scenario_count"]
    assert result["passed"] is True


def test_closure_matrix_covers_every_decision_and_both_metals():
    result = run_closure_scenarios()
    decisions = {item["actual_decision"] for item in result["scenarios"]}
    names = {item["name"] for item in result["scenarios"]}

    assert decisions == {"ALLOW", "REDUCE", "BLOCK", "INCOMPLETE"}
    assert {"gold_allow", "silver_allow"}.issubset(names)
