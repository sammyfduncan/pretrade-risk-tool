"""Sample weighting by label uniqueness (López de Prado, AFML ch. 4).

The problem
-----------
Our labels look 12 bars forward. So the label at bar 100 covers bars 100-112,
the label at 101 covers 101-113, and so on: consecutive labels share almost
all of their future. They are not 12 independent observations of what happens
next, they are roughly *one* observation described 12 times.

Standard classifiers assume rows are independent draws. Handed 12 near-copies
of one event, a model treats a single coincidence as twelve confirmations. The
consequences are exactly what runs 1-3 kept showing:

  * overconfidence — probabilities pushed toward 0 and 1, which is why
    LightGBM's Brier score came out *worse* than always predicting the base
    rate despite ranking bars no worse than the baseline;
  * the effective sample being far smaller than the row count, so apparent
    edges carry much wider error bars than n suggests (gold's meta-labeling
    AUC edge is ~1.1 SE once overlap is accounted for, not ~3).

The correction
--------------
Count, at every bar, how many labels' forward windows are open there
(*concurrency*). A label's *average uniqueness* is the mean of 1/concurrency
across its own window: 1.0 if nothing overlaps it, ~1/12 if it sits in a dense
run. Feed that in as `sample_weight` and the model stops double-counting.

This does not create information. It stops the model believing it has more
than it does — which usually makes reported metrics slightly *worse* and more
honest. It corrects statistical overlap in the labeled sample; it does not
validate the separate execution simulator.

Scope note
----------
Concurrency is computed over whichever sample is passed in. In meta mode that
is the primary-signal bars only, which is correct: those are the rows being
fitted, so their overlap with each other is what biases the fit. Weights are
computed per walk-forward fold, over training rows alone, so a training row's
weight can never be influenced by a test-period label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def label_concurrency(t_start: pd.DatetimeIndex, t_end: pd.Series) -> pd.Series:
    """How many labels' forward windows are open at each sampled bar.

    Uses a difference array rather than the naive nested loop: +1 where each
    label's window opens, -1 just past where it closes, then a cumulative sum.
    O(n log n) for the lookups instead of O(n * horizon).
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
    """Average uniqueness, rescaled to mean 1.

    Rescaling matters because LightGBM's `min_child_samples` and the logistic
    model's C both compare against weight totals. Leaving raw uniqueness of
    ~0.1 in place would shrink every leaf's effective count tenfold and
    silently change the regularization we chose, turning a weighting decision
    into an unintended hyperparameter change.
    """
    uniqueness = average_uniqueness(t_start, t_end)
    mean = uniqueness.mean()
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Degenerate uniqueness weights — check exit_time alignment.")
    return (uniqueness / mean).rename("weight")


def _spans(t_start: pd.DatetimeIndex, t_end: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Each label's window as [start_position, end_position] within the sample.

    Positions are indices into the *sample*, which in meta mode is sparse.
    That is the right frame of reference: an unsampled bar carries no label,
    so nothing is concurrent there.
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
