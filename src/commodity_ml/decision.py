"""The live trade/no-trade decision, separated from the CLI that prints it.

This is the code that will eventually place real orders, so it lives in the
package where it can be tested rather than inside `scripts/predict.py`.

Under meta-labeling the model never picks a side. The primary rule decides
whether to be long at all; the model only decides whether *that particular* long
is worth taking. A high score on a bar where the trend rule is flat means
nothing — there is no trade to filter. Both conditions must hold, which is why
`take_trade` is a conjunction and not a score comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Decision:
    bar_time: pd.Timestamp
    score: float
    calibrated_proba: float
    primary_long: bool
    threshold: float

    @property
    def model_agrees(self) -> bool:
        return self.score >= self.threshold

    @property
    def take_trade(self) -> bool:
        return self.primary_long and self.model_agrees


def decide(artifact: dict, features: pd.Series, primary_long: bool) -> Decision:
    """Score one bar's feature row against the stored entry threshold.

    `artifact` is what `train.py` saved. The threshold is read from it rather
    than recomputed, because the backtest's entry rule is an *expanding
    quantile* of the out-of-fold distribution — a single bar has no
    distribution of its own to take a quantile of, so a live path that tried to
    derive its own threshold would be answering a different question than the
    one the backtest measured.

    Two numbers with two jobs, and they are not interchangeable: `score` is the
    raw model output and is what the threshold was derived from, so it decides.
    `calibrated_proba` is what the score *means* and is for reporting and
    position sizing. Ranking on the calibrated value would change which bars
    trade, for reasons unrelated to signal (CLAUDE.md, run 4).
    """
    model = artifact["model"]
    row = features.to_frame().T

    return Decision(
        bar_time=features.name,
        score=float(model.raw_score(row)[0]),
        calibrated_proba=float(model.positive_proba(row)[0]),
        primary_long=bool(primary_long),
        threshold=float(artifact["entry_threshold"]),
    )
