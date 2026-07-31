"""Economic retention tests for version-2 bare primary rules."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_primary_gate import evaluate_primary_gate


def trades(strategy: str, *, improved: bool, n: int = 500) -> pd.DataFrame:
    index = pd.date_range("2019-01-01", periods=n, freq="1D", tz="UTC")
    if improved:
        winning = np.arange(n) % 5 < 3
        gross = np.where(winning, 0.015, -0.010)
    else:
        winning = np.arange(n) % 5 < 2
        gross = np.where(winning, 0.010, -0.010)
    return pd.DataFrame(
        {
            "strategy": strategy,
            "gross_return": gross,
            "initial_r_pct": 0.01,
        },
        index=index,
    )


def test_positive_v2_rule_passes_all_frozen_conditions():
    result = evaluate_primary_gate(
        trades("P2V2", improved=True),
        trades("P2", improved=False),
        strategy="P2V2",
        control_strategy="P2",
        cost_pct_per_side=0.0001,
        min_trades=300,
    )

    assert result.passed
    assert result.profitable_blocks == 5
    assert result.metrics_by_cost[2.0].profit_factor > 1.0
    assert result.metrics_by_cost[1.0].profit_factor > result.control_metrics.profit_factor


def test_negative_v2_rule_fails_economic_and_control_gates():
    result = evaluate_primary_gate(
        trades("P3V2", improved=False, n=120),
        trades("P3", improved=True, n=120),
        strategy="P3V2",
        control_strategy="P3",
        cost_pct_per_side=0.0001,
        min_trades=100,
    )

    assert not result.passed
    assert any("profit_factor" in reason for reason in result.failure_reasons)
    assert any("pf_not_above_v1" in reason for reason in result.failure_reasons)
