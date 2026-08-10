"""Label construction; what the model is asked to predict.

Main rule: labels may look forward from bar t, but any scaling applied to them 
must be computed from data available at bar t.

triple_barrier is default

Modes
triple_barrier (default):
    1 if the upper barrier is touched first, 0 if lower first.
    If neither is touched within the horizon, the bar labeled by sign
    of its return at the horizon (`timeout: sign`) or dropped (`timeout: drop`)

meta:
    training restricted to bars where primary rule (primary.py) set long.

updown:
    1 if forward return over horizon bar is positive. Symmetric and
    immune to the volatility channel but noisier

long_threshold (legacy):
    1 if forward return > atr_mult * ATR(14)/close

Final `horizon` bars get NaN labels, must be dropped before training
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_ml.config import LabelConfig
from commodity_ml.features.technical import atr

LABEL_COLUMNS = ["label", "exit_bars", "barrier_pct", "exit_time"]


def make_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.DataFrame:
    """Build labels for every bar.

    Returns DF with:
      label       1.0 / 0.0, NaN where outcome unknown
      exit_bars   how many bars until the labeled outcome resolved (barrier
                  touch or horizon). An execution simulator must reproduce
                  this contract explicitly; the historical H4 simulator did
                  not yet do so exactly.
      barrier_pct +/- barrier width at that bar, as fraction of price.
                  Feeds backtest per-bar SL/TP.
      exit_time   timestamp exit_bars later. Each label has forward
                  window [t, exit_time]; neighbouring windows overlap

    All four are metadata, never model inputs
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
        barrier_pct = cfg.atr_mult * atr_pct.to_numpy()
    else:
        fwd_ret = close.shift(-cfg.horizon_bars) / close - 1

        if cfg.mode == "updown":
            label = (fwd_ret > 0).astype(float)
        elif cfg.mode == "long_threshold":
            label = (fwd_ret > cfg.atr_mult * atr_pct).astype(float)
        else:
            raise ValueError(f"Unknown label mode: {cfg.mode!r}")

        # astype(float) turned NaN comparisons into 0.0 — restore NaN where the
        # forward return (or the ATR warmup) is undefined so those rows drop.
        label[fwd_ret.isna()] = np.nan
        if cfg.mode == "long_threshold":
            label[atr_pct.isna()] = np.nan

        label = label.to_numpy()
        exit_bars = np.where(np.isnan(label), np.nan, float(cfg.horizon_bars))
        barrier_pct = (cfg.atr_mult * atr_pct).to_numpy()

    return pd.DataFrame(
        {
            "label": label,
            "exit_bars": exit_bars,
            "barrier_pct": barrier_pct,
            "exit_time": _exit_times(df.index, exit_bars),
        },
        index=df.index,
    )


def _exit_times(index: pd.DatetimeIndex, exit_bars: np.ndarray) -> np.ndarray:
    """Timestamp of exit_bars ahead, NaT where label is unknown

    Works in int bar positions; not wall clock arithmetic; 
    "12 bars later" and "48 hours later" different instants 
    accounting for weekend closures
    """
    n = len(index)
    out = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    known = ~np.isnan(exit_bars)
    if known.any():
        target = (np.flatnonzero(known) + exit_bars[known]).astype(int)
        out[known] = index.values[np.clip(target, 0, n - 1)]
    return pd.DatetimeIndex(out, tz="UTC")

def _triple_barrier(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_pct: np.ndarray,
    horizon: int,
    k: float,
    timeout: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk forward from each bar until barrier touched

    Scans bars t+1..t+horizon and reports which barrier came first. When a
    single bar's range spans both, cannot tell from OHLC which
    was touched first, so assume adverse bar
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
