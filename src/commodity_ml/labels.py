"""Label construction — what the model is asked to predict.

Kept deliberately small and separate: this is the module most prone to
accidental lookahead bugs. The one rule: labels may look *forward* from bar t
(that's their job), but any scaling/threshold applied to them (ATR here) must
be computed from data available *at* bar t.

Why triple_barrier is the default
---------------------------------
The original `long_threshold` label ("did price rise more than k*ATR over the
next N bars?") was measurably contaminated by volatility. Its threshold is set
from ATR at bar t, so the label is easier to satisfy whenever volatility
*rises* after t. Volatility is highly forecastable (features correlate ~0.62
with future volatility, versus ~0.05 with future direction), so a model
maximizes that label's AUC by predicting volatility and ignoring direction
entirely — which is exactly what the first gold run did: AUC 0.52 with a
correlation of -0.001 between its probabilities and the actual forward return.

Triple barriers fix this structurally. Both barriers sit at +/-k*ATR, so a
rise in volatility makes *both* more likely to be touched. Predicting
volatility therefore earns no advantage, and only genuine directional
information can move AUC above 0.50. Expect the reported AUC to drop toward
0.50 — that is the honest directional number becoming visible, not a
regression.

The barriers also define the exit, so the label and the backtest finally
describe the same trade (take profit at +k*ATR, stop out at -k*ATR, give up
after `horizon_bars`).

Modes
-----
triple_barrier (default):
    1 if the upper barrier is touched first, 0 if the lower is touched first.
    If neither is touched within the horizon, the bar is labeled by the sign
    of its return at the horizon (`timeout: sign`) or dropped (`timeout: drop`).

meta:
    Identical barrier outcomes, but training is restricted to bars where the
    primary rule (see primary.py) wants to be long. The label then reads
    "would obeying the rule here have paid off?", turning the model from a
    direction predictor into a trade filter. Sample restriction happens in
    scripts/train.py; the label maths is the same.

updown:
    1 if the forward return over `horizon_bars` is positive. Symmetric and
    immune to the volatility channel, but noisier: most bars are chop.

long_threshold (legacy, kept for reproducing earlier results):
    1 if forward return > atr_mult * ATR(14)/close. Volatility-contaminated
    as described above; do not use for new work.

The final `horizon` bars get NaN labels (their future isn't known yet) and
must be dropped before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_ml.config import LabelConfig
from commodity_ml.features.technical import atr

LABEL_COLUMNS = ["label", "exit_bars", "barrier_pct"]


def make_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """Build labels for every bar.

    Returns a DataFrame with:
      label       1.0 / 0.0, NaN where the outcome isn't knowable
      exit_bars   how many bars until the trade resolved (barrier touch or
                  horizon). The backtest holds for exactly this long, so the
                  simulated trade matches the labeled one.
      barrier_pct the +/- barrier width at that bar, as a fraction of price.
                  Feeds the backtest's per-bar stop-loss / take-profit.
    """
    close = df["close"]
    atr_pct = (atr(df, length=14) / close).rename("barrier_pct")

    if cfg.mode in ("triple_barrier", "meta"):
        label, exit_bars = _triple_barrier(
            high=df["high"].to_numpy(),
            low=df["low"].to_numpy(),
            close=close.to_numpy(),
            atr_pct=atr_pct.to_numpy(),
            horizon=cfg.horizon_bars,
            k=cfg.atr_mult,
            timeout=cfg.timeout,
        )
        out = pd.DataFrame(
            {
                "label": label,
                "exit_bars": exit_bars,
                "barrier_pct": cfg.atr_mult * atr_pct.to_numpy(),
            },
            index=df.index,
        )
        return out

    fwd_ret = close.shift(-cfg.horizon_bars) / close - 1

    if cfg.mode == "updown":
        label = (fwd_ret > 0).astype(float)
    elif cfg.mode == "long_threshold":
        label = (fwd_ret > cfg.atr_mult * atr_pct).astype(float)
    else:
        raise ValueError(f"Unknown label mode: {cfg.mode!r}")

    # astype(float) turned NaN comparisons into 0.0 — restore NaN where the
    # forward return (or the ATR warmup) is undefined so those rows get dropped.
    label[fwd_ret.isna()] = np.nan
    if cfg.mode == "long_threshold":
        label[atr_pct.isna()] = np.nan

    return pd.DataFrame(
        {
            "label": label,
            "exit_bars": float(cfg.horizon_bars),
            "barrier_pct": cfg.atr_mult * atr_pct,
        },
        index=df.index,
    )


def _triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_pct: np.ndarray,
    horizon: int,
    k: float,
    timeout: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk forward from each bar until a barrier is touched.

    Scans bars t+1..t+horizon and reports which barrier came first. When a
    single bar's range spans *both* barriers we cannot tell from OHLC which
    was touched first, so we assume the adverse one — pessimistic by
    construction, because the optimistic assumption would inflate results.
    """
    n = len(close)
    labels = np.full(n, np.nan)
    exit_bars = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr_pct[i]) or i + horizon >= n:
            continue

        upper = close[i] * (1 + k * atr_pct[i])
        lower = close[i] * (1 - k * atr_pct[i])

        for j in range(i + 1, i + horizon + 1):
            touched_lower = low[j] <= lower
            touched_upper = high[j] >= upper
            if touched_lower:  # checked first: ambiguous bars resolve adversely
                labels[i], exit_bars[i] = 0.0, j - i
                break
            if touched_upper:
                labels[i], exit_bars[i] = 1.0, j - i
                break
        else:
            exit_bars[i] = horizon
            if timeout == "sign":
                labels[i] = 1.0 if close[i + horizon] > close[i] else 0.0
            elif timeout == "drop":
                labels[i] = np.nan
            else:
                raise ValueError(f"label.timeout must be 'sign' or 'drop', got {timeout!r}")

    return labels, exit_bars
