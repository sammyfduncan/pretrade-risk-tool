"""Backtest engine: signal delay, quantile-threshold honesty, barrier exits."""

import numpy as np
import pandas as pd

from commodity_ml.backtest.engine import (
    QUANTILE_WARMUP,
    SIGNAL_DELAY,
    debounce_entries,
    expanding_quantile_entries,
    quantile_sweep,
    run_backtest,
)
from commodity_ml.config import BacktestConfig, bars_per_year

HORIZON = 12


def synthetic_market(n: int = 2000, seed: int = 5):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = pd.Series(2000 * np.exp(np.cumsum(rng.normal(0, 0.004, n))), index=idx)
    proba = pd.Series(rng.uniform(0.2, 0.8, n), index=idx)
    barrier = pd.Series(0.015, index=idx)
    return close, proba, barrier


def test_backtest_runs_and_reports():
    close, proba, barrier = synthetic_market()
    cfg = BacktestConfig(entry_quantile=0.90, fee_pct=0.0001, init_cash=10_000)
    result = run_backtest(close, proba, barrier, cfg, "H4", bars_per_year("H4"), HORIZON)
    for key in ("sharpe", "max_drawdown", "n_trades", "profit_factor", "time_in_market"):
        assert key in result.stats
    assert result.stats["n_trades"] > 0
    assert "sharpe" in result.buy_hold_stats


def test_quantile_threshold_uses_no_future_information():
    """The threshold at bar t must depend only on predictions up to t.
    Computing it over the whole series would let future predictions set
    today's entry bar — a subtle but decisive lookahead bug."""
    _, proba, _ = synthetic_market()
    _, threshold = expanding_quantile_entries(proba, 0.90)

    # Truncating the future must leave earlier thresholds untouched.
    cut = 1200
    _, truncated = expanding_quantile_entries(proba.iloc[:cut], 0.90)
    pd.testing.assert_series_equal(threshold.iloc[:cut], truncated, check_exact=False, rtol=1e-12)

    # And a threshold at t must equal the quantile of history through t.
    t = 1500
    expected = proba.iloc[: t + 1].quantile(0.90)
    assert np.isclose(threshold.iloc[t], expected)


def test_quantile_warmup_suppresses_early_entries():
    _, proba, _ = synthetic_market()
    entries, threshold = expanding_quantile_entries(proba, 0.90)
    assert not entries.iloc[: QUANTILE_WARMUP - 1].any()
    assert threshold.iloc[: QUANTILE_WARMUP - 1].isna().all()


def test_selectivity_controls_trade_count():
    """A stricter quantile must not produce more entries than a looser one."""
    _, proba, _ = synthetic_market()
    loose, _ = expanding_quantile_entries(proba, 0.70)
    strict, _ = expanding_quantile_entries(proba, 0.95)
    assert strict.sum() < loose.sum()


def test_execution_is_delayed_by_one_bar():
    """The first signal at bar k must fill at k+SIGNAL_DELAY, never k — filling
    on the bar whose close produced the features would be lookahead."""
    close, proba, barrier = synthetic_market()
    raw_entries, _ = expanding_quantile_entries(proba, 0.90)
    first_signal = close.index.get_loc(raw_entries.idxmax())

    cfg = BacktestConfig(entry_quantile=0.90)
    result = run_backtest(close, proba, barrier, cfg, "H4", bars_per_year("H4"), HORIZON)
    trades = result.portfolio.trades.records_readable
    assert len(trades) >= 1
    assert pd.Timestamp(trades.iloc[0]["Entry Timestamp"]) == close.index[
        first_signal + SIGNAL_DELAY
    ]


def test_entries_are_debounced_so_trades_never_overlap():
    """Overlapping trades inflate the sample without adding information, and
    let an exit collide with a fresh entry."""
    close, proba, _ = synthetic_market()
    raw_entries, _ = expanding_quantile_entries(proba, 0.90)
    kept = debounce_entries(raw_entries, HORIZON)

    assert kept.sum() < raw_entries.sum(), "debouncing should remove clustered signals"
    positions = np.flatnonzero(kept.to_numpy())
    assert (np.diff(positions) > HORIZON).all(), "kept entries must be > horizon apart"
    assert set(np.flatnonzero(kept.to_numpy())) <= set(np.flatnonzero(raw_entries.to_numpy()))


def test_positions_close_within_the_label_horizon():
    """Exits come from the label's barriers plus a vertical time barrier, so no
    trade may outlive the horizon the model was trained to predict."""
    close, proba, barrier = synthetic_market()
    cfg = BacktestConfig(entry_quantile=0.90)
    result = run_backtest(close, proba, barrier, cfg, "H4", bars_per_year("H4"), HORIZON)

    trades = result.portfolio.trades.records_readable
    held = pd.to_datetime(trades["Exit Timestamp"]) - pd.to_datetime(trades["Entry Timestamp"])
    max_allowed = pd.Timedelta(hours=4 * HORIZON)
    assert (held <= max_allowed).all(), f"longest hold {held.max()} exceeds {max_allowed}"


def test_wider_barriers_hold_trades_longer():
    close, proba, _ = synthetic_market()
    cfg = BacktestConfig(entry_quantile=0.90)
    durations = []
    for width in (0.004, 0.05):
        barrier = pd.Series(width, index=close.index)
        result = run_backtest(close, proba, barrier, cfg, "H4", bars_per_year("H4"), HORIZON)
        trades = result.portfolio.trades.records_readable
        durations.append(
            (pd.to_datetime(trades["Exit Timestamp"])
             - pd.to_datetime(trades["Entry Timestamp"])).mean()
        )
    assert durations[1] > durations[0], "wider stops should survive longer before triggering"


def test_quantile_sweep_shape():
    close, proba, barrier = synthetic_market()
    cfg = BacktestConfig(entry_quantile=0.90)
    sweep = quantile_sweep(
        close, proba, barrier, cfg, "H4", bars_per_year("H4"), HORIZON,
        quantiles=(0.80, 0.90, 0.95),
    )
    assert list(sweep.index) == [0.80, 0.90, 0.95]
    assert "sharpe" in sweep.columns and "n_trades" in sweep.columns
