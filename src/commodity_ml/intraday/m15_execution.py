"""Replay an unchanged M30 candidate on the finer M15 price path."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd

from commodity_ml.intraday.execution import (
    CandidateRejected,
    IntradayExecutionConfig,
    TradeCandidate,
    resolve_trade,
)

M15_DURATION = pd.Timedelta(minutes=15)
M30_DURATION = pd.Timedelta(minutes=30)


def replay_m30_candidates_on_m15(
    m15_candles: pd.DataFrame,
    candidates: Sequence[TradeCandidate],
    cfg: IntradayExecutionConfig,
) -> pd.DataFrame:
    """Replay candidates independently while preserving M30 decision timing.

    An M30 candidate timestamp identifies the bar open, so the signal becomes
    knowable 30 minutes later. ``resolve_trade`` expects a signal-bar open;
    anchoring the replay to the final M15 bar inside that M30 candle makes its
    next-bar entry occur at the identical timestamp as the M30 execution.
    """

    if cfg.bar_duration != M30_DURATION:
        raise ValueError("M15 replay requires an M30 source execution config")
    replay_cfg = replace(cfg, bar_duration=M15_DURATION)
    records: list[dict] = []

    for candidate in sorted(candidates, key=lambda item: _utc(item.signal_time)):
        original_signal = _utc(candidate.signal_time)
        anchor = original_signal + M15_DURATION
        replay_candidate = replace(
            candidate,
            signal_time=anchor,
            max_holding_bars=candidate.max_holding_bars * 2,
        )
        base = {
            "strategy": candidate.strategy,
            "side": candidate.side,
            "signal_time": original_signal,
            "m15_anchor_time": anchor,
        }
        try:
            outcome = resolve_trade(m15_candles, replay_candidate, replay_cfg)
        except CandidateRejected as exc:
            records.append(
                {
                    **base,
                    "status": "rejected",
                    "reject_reason": exc.reason,
                    "entry_time": pd.NaT,
                    "exit_time": pd.NaT,
                    "label": np.nan,
                    "net_return": np.nan,
                }
            )
            continue

        record = outcome.to_record()
        record["signal_time"] = original_signal
        record["m15_anchor_time"] = anchor
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "strategy",
                "side",
                "signal_time",
                "m15_anchor_time",
                "status",
                "reject_reason",
                "entry_time",
                "exit_time",
                "label",
                "net_return",
            ]
        )
    return pd.DataFrame.from_records(records)


def _utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        raise ValueError("Candidate timestamps must be timezone-aware")
    return value.tz_convert("UTC")
