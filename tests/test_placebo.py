"""The placebo null used by the step-2 retention gate.

These are not incidental helpers — §6 condition 3 is the only part of the gate
with real evidential content, and its verdict is exactly as trustworthy as the
claim "a shifted series is a genuine null". Each test below pins one way that
claim could quietly become false: a shift that isn't really a shift, a null
that changed the data's distribution, a "random" offset that lands next to the
truth, or a p-value floor that overstates what 20 runs can resolve.
"""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.evaluation.placebo import (
    MIN_SHIFT_PERIODS,
    SHIFT_STEP_PERIODS,
    circular_shift,
    draw_shifts,
    empirical_p,
)


def frame(n: int = 1000) -> pd.DataFrame:
    index = pd.date_range("2019-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"tone": rng.normal(size=n), "vol": rng.normal(size=n)}, index=index
    )


def test_shift_keeps_the_index_and_moves_the_values():
    original = frame(100)
    shifted = circular_shift(original, 10)

    assert shifted.index.equals(original.index), "the dates must not move"
    assert not np.allclose(shifted["tone"], original["tone"])
    # Value at position i came from position i-10, wrapping.
    assert shifted["tone"].iloc[10] == pytest.approx(original["tone"].iloc[0])
    assert shifted["tone"].iloc[0] == pytest.approx(original["tone"].iloc[-10])


def test_shift_is_a_permutation_so_the_null_invents_nothing():
    """A placebo must differ from the truth in *alignment only*. If it also
    changed the distribution, a rejected null would be ambiguous between "the
    real series is informative" and "the placebo was an easier problem"."""
    original = frame(500)
    shifted = circular_shift(original, 137)

    for column in original.columns:
        assert sorted(shifted[column]) == pytest.approx(sorted(original[column]))
        assert shifted[column].mean() == pytest.approx(original[column].mean())
        assert shifted[column].std() == pytest.approx(original[column].std())


def test_shift_preserves_missingness():
    """COT's Feb-2023 outage and GDELT's zero-story days are part of what the
    pipeline has to cope with; the null must face them too, just elsewhere."""
    original = frame(200)
    original.iloc[50:60] = np.nan
    shifted = circular_shift(original, 30)

    assert shifted.isna().sum().equals(original.isna().sum())
    assert shifted["tone"].iloc[80:90].isna().all()


def test_zero_shift_is_the_identity():
    """The observed delta is computed through the same code path as every
    placebo, with shift 0 — so shift 0 must be an exact no-op, or the observed
    and placebo arms are not comparable."""
    original = frame(50)
    pd.testing.assert_frame_equal(circular_shift(original, 0), original)


@pytest.mark.parametrize("family", sorted(MIN_SHIFT_PERIODS))
def test_drawn_shifts_stay_clear_of_both_ends(family):
    """The trap this guards: a shift of n-3 wraps around to within three
    observations of no shift at all, which would enter the null distribution as
    an almost-perfectly-aligned "placebo" and make the gate trivially easy to
    pass."""
    n_obs, low = 3000, MIN_SHIFT_PERIODS[family]
    shifts = draw_shifts(n_obs, 20, family, np.random.default_rng(1))

    assert len(set(shifts)) == 20, "a repeated shift double-counts one null"
    for s in shifts:
        assert low <= s <= n_obs - low
        assert s % SHIFT_STEP_PERIODS[family] == 0


def test_daily_shifts_are_whole_weeks():
    """News volume has a weekday cycle. Shifting by a non-multiple of 7 would
    scramble that phase — destroying structure the real series has, and making
    the null easier to beat than it should be."""
    shifts = draw_shifts(3000, 20, "gdelt", np.random.default_rng(2))

    assert all(s % 7 == 0 for s in shifts)


def test_a_series_too_short_for_a_legal_shift_is_refused():
    """Silently falling back to a small shift would produce a null that quietly
    contains the truth."""
    with pytest.raises(ValueError, match="too short"):
        draw_shifts(400, 20, "gdelt", np.random.default_rng(3))


def test_too_few_legal_shifts_is_refused_rather_than_resampled():
    """115 weekly reports leave only 12 offsets that clear 52 weeks at both
    ends — fewer than the 20 the gate asks for. Sampling with replacement would
    quietly satisfy the count while shrinking the null."""
    with pytest.raises(ValueError, match="double-count"):
        draw_shifts(115, 20, "cot", np.random.default_rng(4))


def test_unknown_family_has_no_shift_policy():
    with pytest.raises(ValueError, match="no placebo shift policy|No placebo shift policy"):
        draw_shifts(3000, 5, "etf_flows", np.random.default_rng(5))


def test_empirical_p_counts_the_observed_run_in_the_reference_set():
    """With 20 placebos the floor is 1/21, not 0. Reporting p = 0.000 from 20
    runs would claim resolution the experiment does not have."""
    placebos = np.linspace(-0.01, 0.01, 20)

    assert empirical_p(0.5, placebos) == pytest.approx(1 / 21)
    assert empirical_p(-0.5, placebos) == pytest.approx(1.0)
    # Ties count against the observation: a placebo equalling it is not beaten.
    assert empirical_p(0.01, placebos) == pytest.approx(2 / 21)
