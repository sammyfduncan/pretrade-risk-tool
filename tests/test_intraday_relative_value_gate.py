"""Tests for the frozen RV1 gate."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_relative_value_gate import (
    evaluate_relative_value_gate,
)


def _ledger(winners: int, losers: int, *, win: float, loss: float) -> pd.DataFrame:
    count = min(winners, losers)
    returns = np.array([win, loss] * count)
    returns = np.concatenate(
        [returns, np.full(winners - count, win), np.full(losers - count, loss)]
    )
    times = pd.date_range("2020-01-01", periods=len(returns), freq="D", tz="UTC")
    sides = np.resize(["long_residual", "short_residual"], len(returns))
    return pd.DataFrame(
        {
            "status": "executed",
            "signal_time": times,
            "side": sides,
            "initial_r": 0.002,
            "net_return_1_0": returns,
            "net_return_1_5": returns - 0.00005,
            "net_return_2_0": returns - 0.0001,
        }
    )


def test_rv1_gate_passes_only_complete_strong_result() -> None:
    observed = _ledger(130, 70, win=0.003, loss=-0.001)
    placebos = {
        f"shift_{i}": _ledger(100, 100, win=0.001, loss=-0.001)
        for i in range(20)
    }

    result = evaluate_relative_value_gate(observed, placebos, coverage=0.99)

    assert result.passed
    assert result.profitable_blocks == 5
    assert result.side_fractions == {"long_residual": 0.5, "short_residual": 0.5}


def test_rv1_gate_records_sample_side_and_placebo_failures() -> None:
    observed = _ledger(10, 5, win=0.001, loss=-0.001)
    observed["side"] = "long_residual"
    placebos = {f"shift_{i}": observed.copy() for i in range(20)}

    result = evaluate_relative_value_gate(observed, placebos, coverage=0.5)

    assert not result.passed
    assert any(reason.startswith("coverage:") for reason in result.failure_reasons)
    assert any(reason.startswith("side_fraction:short_residual") for reason in result.failure_reasons)
    assert any(reason.startswith("profit_factor_not_above_placebos") for reason in result.failure_reasons)
