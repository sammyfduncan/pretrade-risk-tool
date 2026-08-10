# Sample weighting by label uniqueness

from __future__ import annotations

import numpy as np
import pandas as pd


def label_concurrency(t_start: pd.DatetimeIndex, t_end: pd.Series) -> pd.Series:
    """Nr of each labels forward windows open at each sampled bar

    Uses a difference arr instead of nested loop: +1 when labels window opens;
    -1 past where closes, then cumsum. O(nlog(n)) for lookups instead of O(n * horizon)
    """
    starts, ends = _spans(t_start, t_end)
    n = len(starts)

    delta = np.zeros(n + 1, dtype=np.int64)
    np.add.at(delta, starts, 1)
    np.add.at(delta, ends + 1, -1)
    return pd.Series(np.cumsum(delta)[:n], index=t_start, name="concurrency")

def average_uniqueness(t_start: pd.DatetimeIndex, t_end: pd.Series) -> pd.Series:
    """Mean of 1/concurrency over each label's own forward window.

    1.0 means the label overlaps nothing; 0.1 means it shares its window with
    about nine others and is worth roughly a tenth of an observation.
    """
    starts, ends = _spans(t_start, t_end)
    concurrency = label_concurrency(t_start, t_end).to_numpy()

    # Prefix sums let each label's window-average be read off in O(1).
    inverse = 1.0 / concurrency
    prefix = np.concatenate([[0.0], np.cumsum(inverse)])
    span_totals = prefix[ends + 1] - prefix[starts]
    uniqueness = span_totals / (ends - starts + 1)

    return pd.Series(uniqueness, index=t_start, name="uniqueness")


def uniqueness_weights(t_start: pd.DatetimeIndex, t_end: pd.Series) -> pd.Series:
    """Average uniqueness rescaled to mean 1.
    """
    uniqueness = average_uniqueness(t_start, t_end)
    mean = uniqueness.mean()
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Degenerate uniqueness weights — check exit_time alignment.")
    return (uniqueness / mean).rename("weight")


def _spans(t_start: pd.DatetimeIndex, t_end: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Each label's window as [start_position, end_position] within the sample.
    """
    if not t_start.is_monotonic_increasing:
        raise ValueError("t_start must be sorted before computing label uniqueness.")

    t_end = pd.Series(t_end).reindex(t_start)
    n = len(t_start)
    starts = np.arange(n)

    # searchsorted on the sample's own timestamps: the window ends at the last
    # sampled bar at or before exit_time.
    ends = t_start.searchsorted(t_end.to_numpy(), side="right") - 1

    # A label whose exit_time is unknown (NaT sorts high) or lands before its
    # own bar collapses to a window of one — it can only cover itself.
    ends = np.where(pd.isna(t_end.to_numpy()), starts, ends)
    ends = np.clip(ends, starts, n - 1)
    return starts, ends.astype(np.int64)
