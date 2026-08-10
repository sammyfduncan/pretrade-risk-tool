"""Primary signals; rule to decide direction, before ml.

Supports meta-labeling. Both gold and silver showed the same asymmetry: 
neither predicts direction (|corr| <= 0.09, noise level), some predict 
volatility strongly (|corr| ~ 0.51-0.54). 
"""

from __future__ import annotations

import pandas as pd
import pandas_ta_classic as ta

from commodity_ml.config import PrimaryConfig

SIGNAL_COLUMN = "primary_signal"


def make_primary_signal(df: pd.DataFrame, cfg: PrimaryConfig) -> pd.Series:
    """Boolean series: True on bars where the primary rule wants to be long.

    Uses only information available at each bar's close, like every feature.
    """
    close = df["close"]

    if cfg.rule == "trend":
        # Classic trend following: hold longs while price is above its own
        # medium-term average. Deliberately simple and not tuned.
        signal = close > ta.ema(close, length=cfg.ema_span)
    elif cfg.rule == "breakout":
        # Long on a push above the recent high. shift(1) so the current bar's
        # own high cannot be part of the level it must exceed.
        signal = close > close.rolling(cfg.lookback).max().shift(1)
    elif cfg.rule == "always":
        # Degenerate rule: every bar is a candidate. Reproduces the plain
        # direction-prediction setup, useful as a control.
        signal = pd.Series(True, index=df.index)
    else:
        raise ValueError(
            f"Unknown primary.rule: {cfg.rule!r} (expected 'trend', 'breakout' or 'always')"
        )

    signal = signal.fillna(False).astype(bool)
    signal.name = SIGNAL_COLUMN
    return signal
