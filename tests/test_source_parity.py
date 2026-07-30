"""Synthetic tests for cross-provider timestamp and return parity."""

import numpy as np
import pandas as pd

from commodity_ml.data.parity import compare_price_sources


def prices(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-05", periods=n, freq="30min", tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    return pd.DataFrame({"close": close}, index=index)


def test_nearly_identical_feeds_pass():
    duka = prices()
    ig = duka.copy()
    ig["close"] *= 1.0001

    report = compare_price_sources(
        duka,
        ig,
        bar_duration=pd.Timedelta(minutes=30),
    )

    assert report.verdict == "PASS"
    assert report.return_correlation > 0.999
    assert report.best_shift_bars == 0


def test_one_bar_timestamp_shift_is_detected():
    duka = prices()
    ig = duka.copy()
    ig.index = ig.index + pd.Timedelta(minutes=30)

    report = compare_price_sources(
        duka,
        ig,
        bar_duration=pd.Timedelta(minutes=30),
    )

    assert report.verdict == "FAIL"
    assert report.best_shift_bars == -1
    assert any("timestamp_shift_suspected" in reason for reason in report.reasons)


def test_low_timestamp_coverage_cannot_pass():
    duka = prices()
    ig = duka.loc[np.arange(len(duka)) % 4 != 0].copy()

    report = compare_price_sources(
        duka,
        ig,
        bar_duration=pd.Timedelta(minutes=30),
    )

    assert report.verdict == "FAIL"
    assert report.dukascopy_coverage < 0.95
