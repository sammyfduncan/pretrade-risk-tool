"""Tier-1 diagnostic arithmetic (scripts/diagnose_altdata.py).

These functions produce the first real numbers step 2 reports, and their
failure mode is not an exception — it is a plausible table. The bug this file
exists for shipped a full IC table with n=936, sensible-looking ICs and three
cells at |z| > 2, all because observations older than the price series were
silently matched to the same window.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from diagnose_altdata import (  # noqa: E402
    forward_outcomes,
    spearman_ic_newey_west,
)


@pytest.fixture
def daily_close() -> pd.Series:
    idx = pd.date_range("2019-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)


def test_observation_older_than_the_price_series_has_no_outcome(daily_close):
    """The regression. An observation from before the candles cannot have a
    forward return; searchsorted clamps to 0 and would hand it the first
    window instead, making every such observation a duplicate of the others.
    """
    before = pd.Timestamp("2015-06-01", tz="UTC")  # COT history starts 2006
    assert all(np.isnan(v) for v in forward_outcomes(daily_close, before, 5))

    # Two distinct pre-history dates must not collapse onto one shared outcome.
    other = pd.Timestamp("2017-02-02", tz="UTC")
    assert all(np.isnan(v) for v in forward_outcomes(daily_close, other, 5))


def test_observation_past_the_end_has_no_outcome(daily_close):
    beyond = daily_close.index[-2]
    assert all(np.isnan(v) for v in forward_outcomes(daily_close, beyond, 20))

def test_forward_return_is_measured_strictly_forward(daily_close):
    t0 = daily_close.index[100]
    ret, vol = forward_outcomes(daily_close, t0, 10)

    expected = np.log(daily_close.iloc[110] / daily_close.iloc[100])
    assert ret == pytest.approx(expected)
    assert vol > 0

    # Truncating history after t0+horizon must not change the answer: the
    # window is forward-looking but bounded.
    truncated = daily_close.iloc[:111]
    assert forward_outcomes(truncated, t0, 10)[0] == pytest.approx(expected)


def test_tied_outcomes_do_not_masquerade_as_signal(daily_close):
    """What the bug produced downstream: a block of identical y-values. Rank
    correlation against a constant is undefined, and the diagnostic must not
    report it as a number."""
    x = pd.Series(np.arange(200.0))
    constant = pd.Series(np.ones(200))
    stats = spearman_ic_newey_west(x, constant, lag=1)

    assert np.isnan(stats["ic"]) or stats["ic"] == pytest.approx(0.0), (
        "a constant outcome carries no information and must not yield an IC"
    )


def test_newey_west_se_exceeds_the_naive_se_under_overlap():
    """Overlapping forward windows make neighbouring products dependent. The
    whole point of the NW correction is a LARGER se than 1/sqrt(n); if it ever
    came out smaller, every z in the table would be overstated."""
    rng = np.random.default_rng(3)
    n = 500
    base = rng.normal(size=n + 20)
    # Overlapping 20-period sums: strong positive autocorrelation by design.
    y = pd.Series([base[i : i + 20].sum() for i in range(n)])
    x = pd.Series(np.roll(y.to_numpy(), 1))

    naive = spearman_ic_newey_west(x, y, lag=1)
    corrected = spearman_ic_newey_west(x, y, lag=20)

    assert corrected["n"] == naive["n"] == n
    assert corrected["se"] > naive["se"]
    assert abs(corrected["z"]) < abs(naive["z"]), (
        "correcting for overlap must reduce significance, never inflate it"
    )


def test_too_few_paired_observations_is_refused(daily_close):
    x = pd.Series(np.arange(10.0))
    stats = spearman_ic_newey_west(x, x, lag=1)

    assert np.isnan(stats["ic"]) and stats["n"] == 10
