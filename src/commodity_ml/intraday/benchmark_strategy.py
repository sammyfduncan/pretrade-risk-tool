"""Gold benchmark-window strategies that operate directly on M15 candles."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

from commodity_ml.data.schema import validate_candles
from commodity_ml.features.technical import atr
from commodity_ml.intraday.execution import TradeCandidate

LONDON = "Europe/London"
M15 = pd.Timedelta(minutes=15)


def benchmark_continuation_candidates(
    candles: pd.DataFrame,
    *,
    benchmark_time: time = time(15, 0),
    strategy: str = "B1",
) -> list[TradeCandidate]:
    """Create one-hour pre-window momentum candidates at a fixed London clock."""

    frame = validate_candles(candles, f"{strategy}:M15")
    _require_m15_grid(frame.index)
    if benchmark_time.minute % 15:
        raise ValueError("benchmark_time must align to M15")
    benchmark_minutes = benchmark_time.hour * 60 + benchmark_time.minute
    if not 60 <= benchmark_minutes <= 22 * 60:
        raise ValueError("benchmark_time must leave a complete prior hour")

    atr14 = atr(frame, length=14)
    local = frame.index.tz_convert(LONDON)
    session_dates = pd.Index(local.date)
    local_minutes = local.hour * 60 + local.minute
    candidates: list[TradeCandidate] = []

    for session_date in pd.unique(session_dates):
        window_mask = (
            (session_dates == session_date)
            & (local_minutes >= benchmark_minutes - 60)
            & (local_minutes < benchmark_minutes)
        )
        window = frame.loc[window_mask]
        if len(window) != 4 or not _is_contiguous(window.index):
            continue
        signal_time = window.index[-1]
        atr_value = float(atr14.loc[signal_time])
        if not np.isfinite(atr_value) or atr_value <= 0:
            continue
        move = float(window["close"].iloc[-1] - window["open"].iloc[0])
        if abs(move) < atr_value:
            continue
        candidates.append(
            TradeCandidate(
                signal_time=signal_time,
                side="long" if move > 0 else "short",
                stop_distance_pct=None,
                target_distance_pct=None,
                max_holding_bars=4,
                strategy=strategy,
                stop_distance_abs=atr_value,
                target_r_multiple=1.5,
            )
        )
    return candidates


def _require_m15_grid(index: pd.DatetimeIndex) -> None:
    if index.empty:
        raise ValueError("M15 candles are empty")
    offsets = index - index.normalize()
    if ((offsets % M15) != pd.Timedelta(0)).any():
        raise ValueError("M15 candle timestamps are not aligned")


def _is_contiguous(index: pd.DatetimeIndex) -> bool:
    return len(index) == 4 and (index.to_series().diff().dropna() == M15).all()
