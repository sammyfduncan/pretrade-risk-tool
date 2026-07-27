"""Backtest engine: signal delay honesty and end-to-end plumbing."""

import numpy as np
import pandas as pd

from commodity_ml.backtest.engine import SIGNAL_DELAY, run_backtest, threshold_sweep
from commodity_ml.config import BacktestConfig, bars_per_year


def synthetic_market(n: int = 400, seed: int = 5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=idx)
    proba = pd.Series(rng.uniform(0.3, 0.8, n), index=idx)
    return close, proba


def test_backtest_runs_and_reports(ohlcv=None):
    close, proba = synthetic_market()
    cfg = BacktestConfig(prob_entry=0.6, prob_exit=0.5, fee_pct=0.0001, init_cash=10_000)
    result = run_backtest(close, proba, cfg, "H1", bars_per_year("H1"))
    for key in ("sharpe", "max_drawdown", "n_trades", "profit_factor", "time_in_market"):
        assert key in result.stats
    assert result.stats["n_trades"] > 0
    assert "sharpe" in result.buy_hold_stats


def test_execution_is_delayed_by_one_bar():
    """A probability spike at bar k must fill at bar k+SIGNAL_DELAY, never k —
    filling on the same bar the features closed on would be lookahead."""
    close, _ = synthetic_market()
    proba = pd.Series(0.0, index=close.index)
    k = 100
    proba.iloc[k : k + 50] = 1.0  # long conviction for 50 bars

    cfg = BacktestConfig(prob_entry=0.6, prob_exit=0.5)
    result = run_backtest(close, proba, cfg, "H1", bars_per_year("H1"))
    trades = result.portfolio.trades.records_readable
    assert len(trades) >= 1
    first_entry = pd.Timestamp(trades.iloc[0]["Entry Timestamp"])
    assert first_entry == close.index[k + SIGNAL_DELAY]


def test_threshold_sweep_shape():
    close, proba = synthetic_market()
    cfg = BacktestConfig(prob_entry=0.6, prob_exit=0.5)
    sweep = threshold_sweep(close, proba, cfg, "H1", bars_per_year("H1"),
                            entry_thresholds=(0.5, 0.6, 0.7))
    assert list(sweep.index) == [0.5, 0.6, 0.7]
    assert "sharpe" in sweep.columns and "n_trades" in sweep.columns
