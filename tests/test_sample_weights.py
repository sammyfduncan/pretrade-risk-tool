"""Label uniqueness weighting: does it actually measure overlap?

If these are wrong the weights are silently meaningless — the model still
trains, the numbers still print, and nothing tells you the correction did
nothing. Hence the exact hand-computable cases.
"""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.modeling.sample_weights import (
    average_uniqueness,
    label_concurrency,
    uniqueness_weights,
)

BAR = pd.Timedelta(hours=4)


def spans(n: int, horizon: int) -> tuple[pd.DatetimeIndex, pd.Series]:
    """n bars, each labeled over the next `horizon` bars."""
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    return idx, pd.Series(idx + horizon * BAR, index=idx)


def test_non_overlapping_labels_are_fully_unique():
    """A label sharing its window with nothing is worth one observation."""
    idx = pd.date_range("2024-01-01", periods=5, freq="4h", tz="UTC")
    # Each label resolves before the next bar opens -> windows of one bar.
    t_end = pd.Series(idx, index=idx)
    assert (average_uniqueness(idx, t_end) == 1.0).all()


def test_concurrency_counts_open_windows():
    """With horizon 2, bar k is covered by labels k, k-1 and k-2.

    The tail stays at 3 rather than thinning: a label whose window runs past
    the end of the sample is truncated to the last observed bar, because that
    is the last bar on which anything about it is knowable.
    """
    idx, t_end = spans(10, horizon=2)
    conc = label_concurrency(idx, t_end)

    assert conc.iloc[0] == 1            # only label 0 has opened
    assert conc.iloc[1] == 2            # labels 0 and 1
    assert (conc.iloc[2:] == 3).all()   # steady state: three windows open


def test_uniqueness_falls_as_overlap_grows():
    """Longer horizons mean more sharing, so each label is worth less."""
    means = []
    for horizon in (1, 4, 12):
        idx, t_end = spans(400, horizon)
        means.append(average_uniqueness(idx, t_end).mean())
    assert means[0] > means[1] > means[2]
    # A 12-bar horizon should land near 1/13 in the steady state.
    assert 0.05 < means[-1] < 0.15


def test_uniqueness_matches_hand_computation():
    """Small enough to verify by hand: 4 bars, horizon 1.

    Windows: [0,1], [1,2], [2,3], [3,3]
    Concurrency: bar0=1, bar1=2, bar2=2, bar3=2
    Label 0 = mean(1/1, 1/2) = 0.75; label 1 = mean(1/2, 1/2) = 0.5;
    label 2 = 0.5; label 3 = mean(1/2) = 0.5
    """
    idx, t_end = spans(4, horizon=1)
    assert np.allclose(average_uniqueness(idx, t_end).to_numpy(), [0.75, 0.5, 0.5, 0.5])


def test_weights_are_rescaled_to_mean_one():
    """Otherwise uniqueness of ~0.08 would shrink every LightGBM leaf's
    effective count 12-fold, silently changing min_child_samples."""
    idx, t_end = spans(300, horizon=12)
    weights = uniqueness_weights(idx, t_end)
    assert np.isclose(weights.mean(), 1.0)
    assert (weights > 0).all()


def test_sparse_sample_only_counts_sampled_bars():
    """Meta mode trains on a subset of bars. Labels on unsampled bars do not
    exist, so they cannot be concurrent with anything."""
    idx, t_end = spans(100, horizon=12)
    sparse = idx[::13]  # roughly one sampled bar per label window
    conc = label_concurrency(sparse, t_end.loc[sparse])
    assert conc.max() <= 2, "widely spaced labels should barely overlap"


def test_unsorted_index_rejected():
    idx, t_end = spans(20, horizon=3)
    shuffled = idx[::-1]
    with pytest.raises(ValueError, match="sorted"):
        average_uniqueness(shuffled, t_end)


def test_unknown_exit_time_collapses_to_a_single_bar():
    """A NaT exit time must not open a window running to the end of the sample.

    NaT sorts *above* every real timestamp, so a naive searchsorted would put
    the window's end at the last bar — one unlabeled row would then appear
    concurrent with everything, dragging every other weight down. The failure
    would be invisible: weights still sum, training still runs.
    """
    idx, t_end = spans(50, horizon=6)
    known = label_concurrency(idx, t_end)

    unknown = t_end.copy()
    unknown.iloc[10] = pd.NaT
    conc = label_concurrency(idx, unknown)

    # Label 10 now covers only its own bar, so it stops contributing from 11 on
    # and never reaches the far end of the sample.
    assert conc.iloc[10] == known.iloc[10]
    assert (conc.iloc[11:17] == known.iloc[11:17] - 1).all()
    assert (conc.iloc[17:] == known.iloc[17:]).all()

    uniqueness = average_uniqueness(idx, unknown)
    assert np.isfinite(uniqueness).all()
    # Its own uniqueness is exactly 1/concurrency at that single bar.
    assert np.isclose(uniqueness.iloc[10], 1.0 / conc.iloc[10])
