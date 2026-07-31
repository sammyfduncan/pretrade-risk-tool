"""Synthetic tests for the matched volatility-risk economic gate."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_risk_policy import (
    RiskPolicyGateConfig,
    evaluate_risk_policy_gate,
)


def synthetic_inputs(n_forecasts: int = 1_200):
    index = pd.date_range("2021-01-01", periods=n_forecasts, freq="30min", tz="UTC")
    rng = np.random.default_rng(72)
    forecasts = pd.Series(rng.normal(size=n_forecasts), index=index)
    signal_index = index[600:1_100] - pd.Timedelta(minutes=30)
    selected = forecasts.loc[index[600:1_100]].to_numpy()
    gross = np.where(selected > 1.0, -0.012, 0.006)
    gross[::3] = -0.003
    candidates = pd.DataFrame(
        {"gross_return": gross, "initial_r_pct": 0.005},
        index=signal_index,
    )
    return candidates, forecasts


def test_risk_policy_gate_uses_strict_prior_forecasts_and_matched_zeroes():
    candidates, forecasts = synthetic_inputs()
    result = evaluate_risk_policy_gate(
        candidates,
        forecasts,
        cost_pct_per_side=0.0,
        config=RiskPolicyGateConfig(minimum_opportunities=200),
    )

    assert result.match_coverage == 1.0
    assert result.matched_opportunities == len(candidates)
    assert result.decisions["reference_count"].iloc[0] == 600
    assert (result.decisions["reference_count"].diff().dropna() == 1).all()
    assert result.multiplier_counts["0.0"] > 0
    blocked = result.decisions["multiplier"] == 0
    assert (result.decisions.loc[blocked, "policy_contribution"] == 0).all()
    assert (result.decisions.loc[blocked, "constant_contribution"] != 0).all()


def test_risk_policy_gate_can_pass_when_high_forecast_losses_are_reduced():
    candidates, forecasts = synthetic_inputs()
    result = evaluate_risk_policy_gate(
        candidates,
        forecasts,
        cost_pct_per_side=0.0,
    )

    assert result.passed
    assert result.policy.max_drawdown > result.constant.max_drawdown
    assert result.policy.profit_factor > result.constant.profit_factor
    assert result.policy.compounded_return > 0


def test_risk_policy_gate_fails_low_exact_timestamp_coverage():
    candidates, forecasts = synthetic_inputs()
    forecasts = forecasts.drop(forecasts.index[650:850])
    result = evaluate_risk_policy_gate(
        candidates,
        forecasts,
        cost_pct_per_side=0.0,
    )

    assert result.match_coverage < 0.90
    assert not result.passed
    assert any(
        reason.startswith("forecast_match_coverage:")
        for reason in result.failure_reasons
    )
