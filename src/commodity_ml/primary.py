"""Primary signals — the rule that decides *direction*, before ML gets involved.

This exists to support meta-labeling. Both gold and silver showed the same
asymmetry: no feature predicts direction (|corr| <= 0.09, noise level) while
several predict volatility strongly (|corr| ~ 0.51-0.54). Asking a model
"which way will it go" therefore asks it to supply information nothing in the
dataset contains.

Meta-labeling restructures the question. A simple, transparent rule picks the
side; the model then answers a *different* question — "given this rule wants to
go long here, is this a good moment to obey it?" That second question is about
regime, and regime is exactly what our features do carry. The model can only
filter trades, never invent them, which is also why it cannot manufacture an
edge out of noise the way a direction model can.

Note what this does and does not assume. It does not assume the primary rule
has an edge on its own. It assumes the rule's *win rate varies with regime* in
a learnable way. If it doesn't, meta-labeling correctly reports no improvement.
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
