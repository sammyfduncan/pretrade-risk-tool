"""Turn out-of-fold model probabilities into simulated trades with vectorbt.

Honesty rules baked in here:

1. Only out-of-fold probabilities may be passed in — every prediction comes
   from a model that never saw that bar. The backtest window is therefore
   restricted to the OOF index.

2. Signals are delayed by one bar (SIGNAL_DELAY). The model's features use
   the close of bar t, so the earliest realistic execution is *after* t
   closes. vectorbt fills at the close of the signal bar, so shifting the
   signal to t+1 means we fill at close(t+1) — slightly pessimistic, never
   optimistic. Removing this shift makes results look better and is a bug.

3. Spread/fee cost is charged on every side of every trade (fees=fee_pct),
   and a buy-and-hold portfolio is always computed alongside as the
   'did the model beat doing nothing' reference.

Strategy shape (step 1): long-only. Enter when P(up-move) >= prob_entry,
flatten when it decays to <= prob_exit. The entry/exit gap is deliberate
hysteresis — one threshold for both would flip in and out on noise (same
principle the README mandates for cross-asset rotation later).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import vectorbt as vbt

from commodity_ml.config import BacktestConfig
from commodity_ml.evaluation import trading_metrics

SIGNAL_DELAY = 1  # bars between prediction and execution; see module docstring

_GRANULARITY_FREQ = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D": "1D"}


def granularity_to_freq(granularity: str) -> str:
    return _GRANULARITY_FREQ[granularity]


@dataclass
class BacktestResult:
    portfolio: vbt.Portfolio
    buy_hold: vbt.Portfolio
    stats: dict
    buy_hold_stats: dict


def _summarize_portfolio(pf: vbt.Portfolio, periods_per_year: float) -> dict:
    trades = pf.trades.records_readable
    return trading_metrics.summarize(
        returns=pf.returns(),
        periods_per_year=periods_per_year,
        trade_pnls=trades["PnL"] if len(trades) else pd.Series(dtype=float),
        position=(pf.asset_value() > 0),
    )


def run_backtest(
    close: pd.Series,
    oof_proba: pd.Series,
    cfg: BacktestConfig,
    granularity: str,
    periods_per_year: float,
) -> BacktestResult:
    close = close.loc[oof_proba.index]  # OOF window only — see rule 1

    entries = (oof_proba >= cfg.prob_entry).shift(SIGNAL_DELAY, fill_value=False)
    exits = (oof_proba <= cfg.prob_exit).shift(SIGNAL_DELAY, fill_value=False)

    freq = granularity_to_freq(granularity)
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        fees=cfg.fee_pct,
        init_cash=cfg.init_cash,
        freq=freq,
    )
    bh = vbt.Portfolio.from_holding(close, init_cash=cfg.init_cash, fees=cfg.fee_pct, freq=freq)

    return BacktestResult(
        portfolio=pf,
        buy_hold=bh,
        stats=_summarize_portfolio(pf, periods_per_year),
        buy_hold_stats=_summarize_portfolio(bh, periods_per_year),
    )


def threshold_sweep(
    close: pd.Series,
    oof_proba: pd.Series,
    cfg: BacktestConfig,
    granularity: str,
    periods_per_year: float,
    entry_thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70),
) -> pd.DataFrame:
    """Sensitivity of results to the entry threshold (exit kept at the same
    hysteresis gap below entry as in the config).

    DIAGNOSTIC ONLY. Picking the best row of this table and reporting it as
    'the result' is multiple-comparisons overfitting — the README methodology
    forbids exactly that. Use it to check the configured threshold isn't a
    knife-edge fluke: a real effect degrades smoothly across neighbors.
    """
    gap = cfg.prob_entry - cfg.prob_exit
    rows = []
    for entry in entry_thresholds:
        variant = BacktestConfig(
            prob_entry=entry,
            prob_exit=entry - gap,
            fee_pct=cfg.fee_pct,
            init_cash=cfg.init_cash,
        )
        stats = run_backtest(close, oof_proba, variant, granularity, periods_per_year).stats
        rows.append(
            {
                "prob_entry": entry,
                "n_trades": stats.get("n_trades", 0),
                "total_return": round(stats["total_return"], 4),
                "sharpe": round(stats["sharpe"], 2),
                "profit_factor": round(stats.get("profit_factor", 0.0), 2),
                "max_drawdown": round(stats["max_drawdown"], 4),
                "time_in_market": round(stats.get("time_in_market", 0.0), 3),
            }
        )
    return pd.DataFrame(rows).set_index("prob_entry")
