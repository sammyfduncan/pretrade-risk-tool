"""Trading-quality metrics: does the strategy make money, and how painfully?

Pure pandas/numpy on a per-bar returns series — no vectorbt dependency, so
this module is unit-testable offline and reusable for the portfolio-level
evaluation in later steps (README section 4 requires these at both levels).

All ratios are annualized via periods_per_year (bars per year for the given
granularity) so results are comparable across timeframes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Below this many losing bars, downside deviation is too noisy to annualize.
MIN_DOWNSIDE_OBSERVATIONS = 30


def annualized_sharpe(returns: pd.Series, periods_per_year: float) -> float:
    """Mean/std of per-bar returns, scaled to annual. Risk-free rate ignored."""
    if returns.std(ddof=1) == 0 or len(returns) < 2:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(periods_per_year))


def annualized_sortino(returns: pd.Series, periods_per_year: float) -> float:
    """Like Sharpe but penalizes only downside deviation.

    Returns NaN — not 0.0 — when there are too few losing bars to estimate
    downside deviation. A strategy that is barely ever in the market divides
    by a near-zero denominator and produces an astronomically large ratio that
    reads as spectacular performance; NaN correctly says "undefined here".
    """
    downside = returns[returns < 0]
    if len(downside) < MIN_DOWNSIDE_OBSERVATIONS or downside.std(ddof=1) == 0:
        return float("nan")
    return float(returns.mean() / downside.std(ddof=1) * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drop of the compounded equity curve.

    Returned as a negative fraction (e.g. -0.23 = a 23% drawdown).
    """
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    return float((equity / peak - 1).min())


def profit_factor(trade_pnls: pd.Series) -> float:
    """Gross profit / gross loss over closed trades. >1 means net profitable;
    sustained values above ~1.3 on 100+ trades start to be interesting."""
    gross_win = trade_pnls[trade_pnls > 0].sum()
    gross_loss = -trade_pnls[trade_pnls < 0].sum()
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return float(gross_win / gross_loss)


def win_rate(trade_pnls: pd.Series) -> float:
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).mean())


def time_in_market(position: pd.Series) -> float:
    """Fraction of bars with a nonzero position. The README requires
    'time spent in cash' as an explicit portfolio metric — this is its inverse."""
    if len(position) == 0:
        return 0.0
    return float((position != 0).mean())


def summarize(
    returns: pd.Series,
    periods_per_year: float,
    trade_pnls: pd.Series | None = None,
    position: pd.Series | None = None,
) -> dict:
    """One dict with the headline numbers. returns = per-bar strategy returns."""
    total_return = float((1 + returns).prod() - 1)
    years = len(returns) / periods_per_year
    summary = {
        "n_bars": int(len(returns)),
        "years": round(years, 2),
        "total_return": total_return,
        "cagr": float((1 + total_return) ** (1 / years) - 1) if years > 0 else 0.0,
        "sharpe": annualized_sharpe(returns, periods_per_year),
        "sortino": annualized_sortino(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
    }
    if trade_pnls is not None:
        summary.update(
            {
                "n_trades": int(len(trade_pnls)),
                "profit_factor": profit_factor(trade_pnls),
                "win_rate": win_rate(trade_pnls),
            }
        )
    if position is not None:
        summary["time_in_market"] = time_in_market(position)
    return summary
