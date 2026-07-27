"""Label construction — the module where a bug silently invalidates everything."""

import numpy as np
import pandas as pd

from commodity_ml.config import LabelConfig
from commodity_ml.labels import make_labels


def constant_range_df(closes: list[float], half_range: float = 0.01) -> pd.DataFrame:
    """Candles with tiny fixed ranges so ATR is small and predictable."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + half_range, "low": c - half_range, "close": c, "volume": 1},
        index=idx,
    )


# ---------------------------------------------------------------- updown ----

def test_unlabeled_tail_is_nan(ohlcv):
    cfg = LabelConfig(mode="updown", horizon_bars=12)
    labels = make_labels(ohlcv, cfg)["label"]
    assert labels.iloc[-12:].isna().all(), "bars whose future is unknown must be NaN"
    assert labels.iloc[:-12].notna().all()


def test_updown_label_correctness():
    # Price: flat 100, then jumps to 110 at bar 30 and stays.
    closes = [100.0] * 30 + [110.0] * 30
    labels = make_labels(constant_range_df(closes), LabelConfig(mode="updown", horizon_bars=5))["label"]
    # Bars 25..29 see the jump within 5 bars -> 1; earlier flat bars -> 0.
    assert (labels.iloc[25:30] == 1.0).all()
    assert (labels.iloc[:25] == 0.0).all()
    assert (labels.iloc[30:55] == 0.0).all()


# -------------------------------------------------------- triple barrier ----

def test_triple_barrier_upper_hit_first():
    """A clean rally must resolve as label 1 at the bar the barrier is touched."""
    closes = [100.0] * 30 + [100.0, 101.0, 102.0, 105.0] + [105.0] * 10
    cfg = LabelConfig(mode="triple_barrier", horizon_bars=10, atr_mult=1.0, timeout="sign")
    out = make_labels(constant_range_df(closes), cfg)
    # ATR here is ~0.02 so barriers are ~+/-0.02%; bar 29's upper barrier is
    # cleared by the very next bar's rally.
    assert out["label"].iloc[29] == 1.0
    assert out["exit_bars"].iloc[29] >= 1


def test_triple_barrier_lower_hit_first():
    closes = [100.0] * 30 + [99.0, 98.0, 95.0] + [95.0] * 10
    cfg = LabelConfig(mode="triple_barrier", horizon_bars=10, atr_mult=1.0, timeout="sign")
    out = make_labels(constant_range_df(closes), cfg)
    assert out["label"].iloc[29] == 0.0


def test_triple_barrier_ambiguous_bar_resolves_adversely():
    """When one bar's range spans both barriers we can't tell which came first,
    so we must assume the loss — the optimistic guess inflates results."""
    idx = pd.date_range("2024-01-01", periods=40, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, index=idx
    )
    # One wildly wide bar that engulfs both barriers at once.
    df.loc[df.index[35], ["high", "low"]] = [200.0, 50.0]
    cfg = LabelConfig(mode="triple_barrier", horizon_bars=5, atr_mult=1.0, timeout="sign")
    out = make_labels(df, cfg)
    assert out["label"].iloc[34] == 0.0


def test_triple_barrier_timeout_modes():
    """Flat prices touch no barrier; behaviour then depends on the timeout mode."""
    closes = [100.0] * 60
    df = constant_range_df(closes, half_range=0.001)

    signed = make_labels(df, LabelConfig(mode="triple_barrier", horizon_bars=5, timeout="sign"))
    labelled = signed["label"].dropna()
    assert (labelled == 0.0).all(), "flat close means not-up, so sign mode labels 0"
    assert (signed["exit_bars"].dropna() == 5).all(), "no barrier touched -> full horizon"

    dropped = make_labels(df, LabelConfig(mode="triple_barrier", horizon_bars=5, timeout="drop"))
    assert dropped["label"].isna().all(), "drop mode discards undecided bars"


def test_triple_barrier_is_symmetric_under_volatility():
    """The whole point of symmetric barriers: a volatility spike must not bias
    the label toward 1. This is what the old long_threshold label got wrong."""
    rng = np.random.default_rng(0)
    n = 3000
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    # Driftless random walk with a deliberate regime shift in volatility.
    vol = np.where(np.arange(n) < n // 2, 0.001, 0.006)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 1, n) * vol))
    df = pd.DataFrame(
        {"open": close, "high": close * 1.0005, "low": close * 0.9995,
         "close": close, "volume": 1},
        index=idx,
    )
    cfg = LabelConfig(mode="triple_barrier", horizon_bars=12, atr_mult=1.0, timeout="sign")
    labels = make_labels(df, cfg)["label"].dropna()

    calm, wild = labels.iloc[: n // 2 - 50], labels.iloc[n // 2 + 50 :]
    # On a driftless walk both regimes should sit near a coin flip, and crucially
    # the volatile regime must not be systematically more positive.
    assert 0.35 < calm.mean() < 0.65
    assert 0.35 < wild.mean() < 0.65
    assert abs(wild.mean() - calm.mean()) < 0.12


def test_no_lookahead_in_barrier_scaling():
    """Barrier width at bar t comes from ATR at t, so labels for a shared past
    must not change when later, more volatile data is appended."""
    calm = [100.0 + 0.01 * i for i in range(60)]
    wild = list(np.linspace(100.6, 160, 40))
    cfg = LabelConfig(mode="triple_barrier", horizon_bars=5, atr_mult=1.0, timeout="sign")

    full = make_labels(constant_range_df(calm + wild), cfg)["label"]
    trunc = make_labels(constant_range_df(calm), cfg)["label"]

    overlap = trunc.dropna().index
    pd.testing.assert_series_equal(full.loc[overlap], trunc.loc[overlap])

# --------------------------------------------------- legacy long_threshold ----

def test_long_threshold_still_works_for_reproducing_old_runs():
    closes = [100.0] * 30 + [100.001] * 30
    cfg = LabelConfig(mode="long_threshold", horizon_bars=5, atr_mult=1.0)
    labels = make_labels(constant_range_df(closes), cfg)["label"].dropna()
    assert (labels == 0.0).all()

    closes = [100.0] * 30 + [105.0] * 30
    labels = make_labels(constant_range_df(closes), cfg)["label"]
    assert (labels.iloc[26:30] == 1.0).all()
