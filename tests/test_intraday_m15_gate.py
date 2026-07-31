"""Frozen retention conditions for matched M15 execution."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_m15_gate import evaluate_m15_gate


def ledgers(n: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    alternating = np.arange(n) % 2 == 0
    control_gross = np.where(alternating, 0.012, -0.010)
    replay_gross = np.where(alternating, 0.015, -0.008)
    common = {
        "strategy": "P3V2",
        "status": "executed",
        "signal_time": signal,
        "initial_r_pct": 0.01,
    }
    control = pd.DataFrame(
        {
            **common,
            "gross_return": control_gross,
            "exit_reason": np.where(alternating, "target", "stop"),
        }
    )
    replay = pd.DataFrame(
        {
            **common,
            "gross_return": replay_gross,
            "exit_reason": np.where(alternating, "target", "stop"),
        }
    )
    return control, replay


def test_m15_gate_passes_consistent_matched_improvement():
    control, replay = ledgers()

    result = evaluate_m15_gate(
        control,
        replay,
        strategy="P3V2",
        cost_pct_per_side=0.0,
    )

    assert result.passed
    assert result.coverage == 1.0
    assert result.improved_blocks == 5
    assert result.m15_by_cost[1.0].profit_factor > result.m30_by_cost[1.0].profit_factor


def test_m15_gate_fails_incomplete_replay_and_no_improvement():
    control, replay = ledgers()
    replay = replay.iloc[:90].copy()
    replay["gross_return"] = control.iloc[:90]["gross_return"].to_numpy()

    result = evaluate_m15_gate(
        control,
        replay,
        strategy="P3V2",
        cost_pct_per_side=0.0,
    )

    assert not result.passed
    assert any(reason.startswith("coverage:") for reason in result.failure_reasons)
    assert any(
        reason.startswith("profit_factor_not_better:")
        for reason in result.failure_reasons
    )
