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

3. The entry threshold is an **expanding-window quantile**, not a fixed
   probability. Two reasons. A fixed 0.60 is unreachable for a well-calibrated
   model on a ~35% base rate — on the first gold run the calibrated baseline
   crossed it twice in six years, while LightGBM crossed it constantly only
   because it was overconfident. And taking the quantile over the *whole* OOF
   period would set today's threshold using predictions from the future, so
   the quantile is computed only over predictions seen so far.

4. Exits come from the label's own barriers, so the simulated trade is the
   trade the model was trained to predict: take profit at +barrier_pct, stop
   out at -barrier_pct, and give up after the label horizon. Previously the
   model was graded on a 1-ATR/12-bar question but traded on a
   "hold until probability decays" rule — two different strategies.

5. Entries are debounced so trades never overlap (see `debounce_entries`).
   Each trade is then one independent bet on one labeled outcome, which is
   what the README's "100+ trades" minimum assumes — overlapping trades
   inflate the apparent sample without adding information.

6. Spread/fee cost is charged on every side of every trade, and a
   buy-and-hold portfolio is always computed alongside as the
   'did the model beat doing nothing' reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from commodity_ml.config import BacktestConfig
from commodity_ml.evaluation import trading_metrics

SIGNAL_DELAY = 1  # bars between prediction and execution; see module docstring
QUANTILE_WARMUP = 500  # bars of history before the expanding quantile is trusted

_GRANULARITY_FREQ = {"M5": "5min", "M15": "15min", "M30": "30min", "H1": "1h", "H4": "4h", "D": "1D"}


def granularity_to_freq(granularity: str) -> str:
    return _GRANULARITY_FREQ[granularity]


@dataclass
class BacktestResult:
    portfolio: vbt.Portfolio
    buy_hold: vbt.Portfolio
    stats: dict
    buy_hold_stats: dict
    threshold: pd.Series


def expanding_quantile_entries(proba: pd.Series, quantile: float) -> tuple[pd.Series, pd.Series]:
    """Long signal when a prediction ranks in the top (1-quantile) of history.

    Uses only predictions available up to and including each bar, so the
    threshold at time t never depends on anything after t.

    NaN entries (bars with no prediction, e.g. off-signal bars in meta mode)
    are skipped by the expanding window and can never produce an entry.
    """
    threshold = proba.expanding(min_periods=QUANTILE_WARMUP).quantile(quantile)
    entries = (proba >= threshold) & threshold.notna() & proba.notna()
    return entries, threshold


def debounce_entries(entries: pd.Series, min_gap: int) -> pd.Series:
    """Keep an entry only if `min_gap` bars have passed since the last kept one.

    Two reasons this matters. Economically, it stops the model firing a burst
    of signals across one move and counting it as many bets — 757 overlapping
    trades in the first gold run were only ~63 independent ones, well under the
    pre-registered 100-trade minimum. Mechanically, it guarantees the vertical
    time barrier (an exit `min_gap` bars after an entry) lands before the next
    possible entry, so entry and exit signals can never collide on the same bar
    and leave a position open past its horizon.

    NOTE: this keeps the FIRST qualifying bar in each cluster, not the strongest.
    That is the only causal choice — at bar t you cannot know that t+3 will score
    higher — but it has a consequence worth understanding (run 5). Kept bars
    score systematically lower than the pool they are drawn from (0.8909 vs
    0.8987 at q=0.90 on gold), and lowering the entry threshold makes it worse,
    because you fire on a mediocre bar and are then locked out of a strong one a
    few bars later. So trade sets at different thresholds are NOT nested: gold's
    q=0.90 and q=0.80 sets overlap on only 53 of 120 and 194 entries. Anything
    that compares across thresholds must account for this.
    """
    kept = np.zeros(len(entries), dtype=bool)
    last = -min_gap - 1
    for i in np.flatnonzero(entries.to_numpy()):
        if i - last > min_gap:
            kept[i] = True
            last = i
    return pd.Series(kept, index=entries.index)


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
    barrier_pct: pd.Series,
    cfg: BacktestConfig,
    granularity: str,
    periods_per_year: float,
    horizon_bars: int,
) -> BacktestResult:
    """The ML-filtered strategy: enter on the top slice of model predictions.

    `oof_proba` must be the RAW model score, not the calibrated probability.
    The entry rule is a quantile, which is a statement about ordering, and
    calibration is allowed to flatten regions where the score does not
    discriminate — ranking on it would change the trade set for reasons that
    have nothing to do with signal.
    """
    close, proba, barrier_pct = _align_to_window(close, oof_proba, barrier_pct)
    raw_entries, threshold = expanding_quantile_entries(proba, cfg.entry_quantile)
    return _simulate(
        close, raw_entries, barrier_pct, cfg, granularity, periods_per_year,
        horizon_bars, threshold,
    )


def run_rule_backtest(
    close: pd.Series,
    signal: pd.Series,
    barrier_pct: pd.Series,
    cfg: BacktestConfig,
    granularity: str,
    periods_per_year: float,
    horizon_bars: int,
    window: pd.DatetimeIndex,
) -> BacktestResult:
    """The primary rule alone, with no ML filter — the control that decides it.

    README methodology: a strategy that filters a rule must be compared against
    the bare rule. A filter that removes trades and keeps the same return has
    earned something; one that merely tracks its own primary signal is
    decoration wearing a model's clothes. Same barriers, same fees, same
    debouncing and same window as the filtered version, so the only difference
    is whether the model got a vote.
    """
    close = close.loc[window]
    # fill_value on the reindex rather than a chained fillna: an object-dtype
    # boolean column downcasts with a FutureWarning under pandas 2.x.
    entries = signal.reindex(window, fill_value=False).astype(bool)
    barrier_pct = barrier_pct.reindex(window).ffill()
    return _simulate(
        close, entries, barrier_pct, cfg, granularity, periods_per_year,
        horizon_bars, pd.Series(dtype=float),
    )


def _align_to_window(
    close: pd.Series, oof_proba: pd.Series, barrier_pct: pd.Series
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Restrict to the OOF *time range* (rule 1) but keep every bar in it.

    In meta mode predictions exist only on primary-signal bars; indexing the
    price series by that sparse set would silently compress the timeline and
    make barriers and holding periods meaningless.
    """
    window = close.loc[
        (close.index >= oof_proba.index.min()) & (close.index <= oof_proba.index.max())
    ]
    return (
        window,
        oof_proba.reindex(window.index),  # NaN on bars with no prediction
        barrier_pct.reindex(window.index).ffill(),
    )


def _simulate(
    close: pd.Series,
    raw_entries: pd.Series,
    barrier_pct: pd.Series,
    cfg: BacktestConfig,
    granularity: str,
    periods_per_year: float,
    horizon_bars: int,
    threshold: pd.Series,
) -> BacktestResult:
    """Shared execution model, so the strategy and its control cannot diverge."""
    entries = debounce_entries(raw_entries, horizon_bars).shift(
        SIGNAL_DELAY, fill_value=False
    )

    # Barriers are set from the ATR at the signal bar, so shift them alongside
    # the signal to keep each trade's stops matched to its own entry.
    stops = barrier_pct.shift(SIGNAL_DELAY).ffill().fillna(barrier_pct.median())

    freq = granularity_to_freq(granularity)
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        # The vertical barrier: give up on a trade after the label's horizon,
        # matching the label's own time limit. Debouncing guarantees the next
        # entry cannot land on this bar, so the exit is never lost to a conflict.
        entries.shift(horizon_bars, fill_value=False),
        sl_stop=stops,
        tp_stop=stops,
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
        threshold=threshold,
    )


def quantile_sweep(
    close: pd.Series,
    oof_proba: pd.Series,
    barrier_pct: pd.Series,
    cfg: BacktestConfig,
    granularity: str,
    periods_per_year: float,
    horizon_bars: int,
    quantiles: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95, 0.99),
) -> pd.DataFrame:
    """Sensitivity of results to how selective the entry rule is.

    DIAGNOSTIC ONLY, AND A WEAK ONE. Picking the best row and reporting it as
    'the result' is multiple-comparisons overfitting — the README methodology
    forbids exactly that.

    It is also NOT a robustness test, in either direction. Run 3 read a smooth
    profit-factor curve here as evidence the effect was not knife-edge; run 5
    retracted that. Changing the quantile changes which bar in each cluster
    survives `debounce_entries` (see its docstring), so this table varies
    cluster-entry timing *and* selectivity at once and cannot isolate either.
    A jagged row is not evidence of fragility and a smooth one is not evidence
    of robustness. For the signal question, test whether the score ranks label
    outcomes — deciles of the raw score against the base rate, deflated by
    effective sample size.
    """
    rows = []
    for q in quantiles:
        variant = BacktestConfig(
            entry_quantile=q, fee_pct=cfg.fee_pct, init_cash=cfg.init_cash
        )
        stats = run_backtest(
            close, oof_proba, barrier_pct, variant, granularity, periods_per_year, horizon_bars
        ).stats
        rows.append(
            {
                "entry_quantile": q,
                "n_trades": stats.get("n_trades", 0),
                "total_return": round(stats["total_return"], 4),
                "sharpe": round(stats["sharpe"], 2),
                "profit_factor": round(stats.get("profit_factor", 0.0), 2),
                "max_drawdown": round(stats["max_drawdown"], 4),
                "time_in_market": round(stats.get("time_in_market", 0.0), 3),
            }
        )
    return pd.DataFrame(rows).set_index("entry_quantile")
