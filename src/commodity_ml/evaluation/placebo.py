"""Time-shifted placebo nulls for feature-family evaluation.

The question a placebo answers: when a feature family carries *no* information,
how large a metric improvement does this pipeline produce anyway? Analytic
standard errors cannot answer it here — overlapping labels, autocorrelated
features, uniqueness weighting, per-fold calibration and five time-ordered
folds each bend the null in a direction nobody can write down. Run 5's naive
p-value was 1,800x too small for exactly this reason.

So the null is generated empirically: keep the pipeline, break the alignment.

Used by `scripts/placebo_altdata.py` for the step-2 retention gate
(docs/step2_design.md §6, condition 3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A placebo is only a null if it is far from the truth. One year is the §6
# floor, and it applies at BOTH ends of the range — a shift of n−3 wraps around
# to within three observations of no shift at all, which would smuggle the real
# alignment back in as a "placebo".
#
# "gvz" and "calendar" are the step-2b sources (docs/step2b_vol_design.md §4):
# GVZ is a daily FRED series; calendar is the daily scheduled-event flag frame.
MIN_SHIFT_PERIODS = {"cot": 52, "gdelt": 365, "gvz": 365, "calendar": 365}

# GDELT news volume has a strong weekly cycle — `gdelt_vol_smooth7_z` exists
# precisely to smooth it. Shifting a daily series by a non-multiple of 7 would
# scramble weekday phase, destroying structure the real series *has* and making
# the null easier for the real features to beat. Whole weeks keep the placebo
# conservative. COT is already weekly, so every shift is a whole cycle. The
# same argument binds the new daily sources even harder: FOMC decisions land
# on Wednesdays and NFP on Fridays, so a non-multiple-of-7 shift would hand
# the real calendar an easy win over a weekday-scrambled null.
SHIFT_STEP_PERIODS = {"cot": 1, "gdelt": 7, "gvz": 7, "calendar": 7}


def circular_shift(frame: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Roll the values, keep the index.

    Rolling the *values* rather than reindexing is what makes this a null
    instead of a truncation: the returned frame holds exactly the same
    observations, in the same order, on the same dates — just starting from a
    different point. So it keeps

      * the marginal distribution (it is a permutation, nothing is invented),
      * the autocorrelation and spikiness of the real series,
      * the missingness pattern, including COT's Feb-2023 publication outage
        and GDELT's zero-story days,

    while destroying the one thing under test: temporal correspondence with
    price. Everything downstream — rolling z-windows, staleness bridges,
    publication lags — is then recomputed on it by the real code.
    """
    if periods == 0:
        return frame.copy()
    rolled = frame.copy()
    rolled[:] = np.roll(frame.to_numpy(), periods, axis=0)
    return rolled


def draw_shifts(
    n_obs: int,
    n_draws: int,
    family: str,
    rng: np.random.Generator,
) -> list[int]:
    """Distinct legal shift offsets for `family`, sorted.

    Legal means: at least `MIN_SHIFT_PERIODS[family]` from *either* end, and on
    the family's step. Sampling without replacement matters — a repeated shift
    would be the same placebo counted twice, quietly shrinking an
    already-small max-of-20 null.
    """
    if family not in MIN_SHIFT_PERIODS:
        raise ValueError(f"No placebo shift policy for family {family!r}")
    low = MIN_SHIFT_PERIODS[family]
    step = SHIFT_STEP_PERIODS[family]
    high = n_obs - low
    if high <= low:
        raise ValueError(
            f"{family}: {n_obs} observations cannot support a >= {low}-period shift "
            "at both ends, so no legal placebo exists. The snapshot is too short."
        )
    pool = np.arange(low, high + 1)
    pool = pool[pool % step == 0]
    if len(pool) < n_draws:
        raise ValueError(
            f"{family}: only {len(pool)} legal shifts available for {n_draws} "
            "placebos; drawing with replacement would double-count a null."
        )
    return sorted(int(s) for s in rng.choice(pool, size=n_draws, replace=False))


def empirical_p(observed: float, placebos: pd.Series | np.ndarray) -> float:
    """One-sided empirical p: P(placebo >= observed).

    The +1 in both numerator and denominator is not a fudge. If the family is
    uninformative then the observed run is itself just another draw from this
    null, so it belongs in the reference set. Without it, 20 placebos could
    report p = 0.000 — a claim no 20-run experiment can support. With it the
    floor is 1/21 ≈ 0.048, which is exactly the resolution 20 runs buy.
    """
    placebos = np.asarray(placebos, dtype=float)
    exceeded = int((placebos >= observed).sum())
    return (1 + exceeded) / (1 + len(placebos))
