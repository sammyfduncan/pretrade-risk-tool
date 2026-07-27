"""Trading metrics against hand-computable cases."""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.evaluation.trading_metrics import (
    annualized_sharpe,
    max_drawdown,
    profit_factor,
    summarize,
    time_in_market,
    win_rate,
)


def test_sharpe_known_case():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.001, 0.01, 10_000))
    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert annualized_sharpe(returns, 252) == pytest.approx(expected)


def test_sharpe_zero_variance_is_zero():
    assert annualized_sharpe(pd.Series([0.01] * 50), 252) == 0.0


def test_max_drawdown_known_curve():
    # Equity: 1.0 -> 1.5 -> 0.75 -> 0.9. Peak 1.5, trough 0.75 -> -50%.
    returns = pd.Series([0.5, -0.5, 0.2])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(pd.Series([0.01] * 20)) == pytest.approx(0.0)


def test_profit_factor_and_win_rate():
    pnls = pd.Series([10.0, -5.0, 20.0, -10.0])  # 30 won / 15 lost
    assert profit_factor(pnls) == pytest.approx(2.0)
    assert win_rate(pnls) == pytest.approx(0.5)


def test_profit_factor_no_losses_is_inf():
    assert profit_factor(pd.Series([1.0, 2.0])) == float("inf")


def test_time_in_market():
    position = pd.Series([0, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert time_in_market(position) == pytest.approx(0.4)


def test_summarize_total_return_compounds():
    returns = pd.Series([0.10, -0.05, 0.02])
    summary = summarize(returns, periods_per_year=252)
    assert summary["total_return"] == pytest.approx(1.1 * 0.95 * 1.02 - 1)
    assert summary["n_bars"] == 3
