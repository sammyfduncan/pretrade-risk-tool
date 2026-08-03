"""Tests for the frozen P4 performance and placebo gate."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_macro_impulse_gate import (
    circular_shift_joint_state,
    evaluate_macro_impulse_gate,
    registered_shift_offsets,
)


def _ledger(winners: int, losers: int, *, win: float = 0.003, loss: float = -0.001) -> pd.DataFrame:
    returns = np.array([win, loss] * min(winners, losers), dtype=float)
    returns = np.concatenate(
        [returns, np.full(winners - min(winners, losers), win), np.full(losers - min(winners, losers), loss)]
    )
    times = pd.date_range("2020-01-01", periods=len(returns), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "strategy": "P4",
            "status": "executed",
            "signal_time": times,
            "gross_return": returns,
            "initial_r_pct": 0.002,
        }
    )


def test_gate_passes_only_when_observed_clears_all_registered_requirements() -> None:
    observed = _ledger(125, 75)
    placebos = {f"shift_{i}": _ledger(100, 100, win=0.001, loss=-0.001) for i in range(20)}

    result = evaluate_macro_impulse_gate(
        observed,
        placebos,
        coverage=0.95,
        cost_pct_per_side=0.0001,
    )

    assert result.passed
    assert result.profitable_blocks == 5
    assert result.metrics_by_cost[2.0].profit_factor > 1.0
    assert result.metrics_by_cost[1.0].profit_factor > result.maximum_placebo_pf


def test_gate_records_coverage_sample_and_placebo_failures() -> None:
    observed = _ledger(10, 10)
    placebos = {f"shift_{i}": _ledger(10, 10) for i in range(20)}

    result = evaluate_macro_impulse_gate(
        observed,
        placebos,
        coverage=0.50,
        cost_pct_per_side=0.0001,
    )

    assert not result.passed
    assert any(reason.startswith("coverage:") for reason in result.failure_reasons)
    assert any(reason.startswith("trades:") for reason in result.failure_reasons)
    assert any(reason.startswith("profit_factor_not_above_placebos:") for reason in result.failure_reasons)


def test_registered_joint_shifts_are_unique_and_preserve_row_pairs() -> None:
    state = pd.DataFrame(
        {"zn_z": np.arange(210), "euro_z": np.arange(210) + 1000}
    )
    offsets = registered_shift_offsets(len(state))

    assert len(offsets) == len(set(offsets)) == 20
    shifted = circular_shift_joint_state(state, offsets[0])
    assert ((shifted["euro_z"] - shifted["zn_z"]) == 1000).all()


def test_empty_observed_ledger_fails_without_crashing() -> None:
    empty = pd.DataFrame(columns=["status", "signal_time"])
    placebos = {f"shift_{i}": empty.copy() for i in range(20)}

    result = evaluate_macro_impulse_gate(
        empty,
        placebos,
        coverage=0.95,
        cost_pct_per_side=0.0001,
    )

    assert not result.passed
    assert result.metrics_by_cost[1.0].trades == 0
