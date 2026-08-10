"""The live trade/no trade decision.

Under meta labelling model never picks a side. Primary rule decides 
whether to go long; model decides whether that long is worth taking. 

Extensible for trade execution
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
    """Score one bars feature row against scored entry threshold. 
    artifact is what train.py saved, threshold read from it not recomputed;
    backtest entry rile is an expanding quantile of the out of fold dist.

    Score is raw model output (threshold derives from), it decides.
    calibrated_proba is what score means, used for reporting and position sizing. 
    Ranking on calibrated val changes which bars trade.
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
