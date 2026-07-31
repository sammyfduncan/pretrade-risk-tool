"""Economic and placebo gates for benchmark strategy B1."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_benchmark_gate import (
    BenchmarkGateConfig,
    evaluate_benchmark_gate,
)


def ledger(strategy: str, wins: float, losses: float, n: int = 100) -> pd.DataFrame:
    signal = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    gross = np.where(np.arange(n) % 2 == 0, wins, -losses)
    return pd.DataFrame(
        {
            "strategy": strategy,
            "status": "executed",
            "signal_time": signal,
            "gross_return": gross,
            "initial_r_pct": 0.01,
        }
    )


def test_benchmark_gate_passes_stable_edge_above_all_placebos():
    result = evaluate_benchmark_gate(
        ledger("B1", 0.015, 0.008),
        {
            name: ledger(name, 0.010, 0.010)
            for name in (
                "B1_PLACEBO_1300",
                "B1_PLACEBO_1400",
                "B1_PLACEBO_1600",
                "B1_PLACEBO_1700",
            )
        },
        strategy="B1",
        cost_pct_per_side=0.0,
        config=BenchmarkGateConfig(min_trades=100),
    )

    assert result.passed
    assert result.profitable_blocks == 5
    assert result.metrics_by_cost[1.0].profit_factor > result.maximum_placebo_pf


def test_benchmark_gate_fails_when_a_placebo_is_better():
    result = evaluate_benchmark_gate(
        ledger("B1", 0.012, 0.010),
        {
            "B1_PLACEBO_1300": ledger("B1_PLACEBO_1300", 0.015, 0.008),
            "B1_PLACEBO_1400": ledger("B1_PLACEBO_1400", 0.010, 0.010),
            "B1_PLACEBO_1600": ledger("B1_PLACEBO_1600", 0.010, 0.010),
            "B1_PLACEBO_1700": ledger("B1_PLACEBO_1700", 0.010, 0.010),
        },
        strategy="B1",
        cost_pct_per_side=0.0,
        config=BenchmarkGateConfig(min_trades=100),
    )

    assert not result.passed
    assert any(
        reason.startswith("profit_factor_not_above_placebos:")
        for reason in result.failure_reasons
    )
