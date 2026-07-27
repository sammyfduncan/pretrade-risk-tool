"""Label construction — what the model is asked to predict.

Kept deliberately small and separate: this is the module most prone to
accidental lookahead bugs. The one rule: labels may look *forward* from bar t
(that's their job), but any scaling/threshold applied to them (ATR here) must
be computed from data available *at* bar t.

Modes
-----
long_threshold (default):
    y[t] = 1  if close[t + horizon] / close[t] - 1 > atr_mult * ATR14[t] / close[t]
    i.e. "did price rise by more than `atr_mult` ATRs over the next `horizon`
    bars?" Class 0 mixes flat and down moves — a long-only framing, matching
    the step-1 scope. Scaling the threshold by ATR keeps the meaning of a
    'meaningful move' stable across volatility regimes.

updown:
    y[t] = 1 if the forward return is positive. Symmetric and balanced
    (~50/50) but noisier — most bars are chop with no meaningful move.

The final `horizon` bars get NaN labels (their future isn't known yet) and
must be dropped before training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_ml.config import LabelConfig
from commodity_ml.features.technical import atr


def make_labels(df: pd.DataFrame, cfg: LabelConfig) -> pd.Series:
    """Return a float Series: 1.0 / 0.0 labels, NaN where the future is unknown."""
    close = df["close"]
    fwd_ret = close.shift(-cfg.horizon_bars) / close - 1

    if cfg.mode == "long_threshold":
        threshold = cfg.atr_mult * atr(df, length=14) / close
        label = (fwd_ret > threshold).astype(float)
    elif cfg.mode == "updown":
        label = (fwd_ret > 0).astype(float)
    else:
        raise ValueError(f"Unknown label mode: {cfg.mode!r}")

    # astype(float) turned NaN comparisons into 0.0 — restore NaN where the
    # forward return (or the ATR warmup) is undefined so those rows get dropped.
    label[fwd_ret.isna()] = np.nan
    if cfg.mode == "long_threshold":
        label[atr(df, length=14).isna()] = np.nan
    label.name = "label"
    return label
