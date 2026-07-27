"""Feature causality: the load-bearing test of the whole pipeline.

If any feature at bar t uses information from after bar t, walk-forward
validation is silently invalidated and every downstream metric becomes
fiction. Extend test_features_are_causal's coverage when adding features.
"""

import numpy as np
import pandas as pd

from commodity_ml.features.technical import atr, build_features


def test_features_are_causal(ohlcv):
    """Features computed on truncated history must equal the same rows
    computed on full history — a feature that peeks ahead will differ."""
    full, names = build_features(ohlcv)
    for cut in (400, 500, len(ohlcv) - 1):
        truncated, _ = build_features(ohlcv.iloc[:cut])
        pd.testing.assert_frame_equal(
            full.iloc[:cut],
            truncated,
            check_exact=False,
            rtol=1e-9,
            obj=f"features with history truncated at bar {cut}",
        )


def test_feature_names_match_columns(ohlcv):
    features, names = build_features(ohlcv)
    assert list(features.columns) == names


def test_warmup_rows_are_nan_then_complete(ohlcv):
    features, _ = build_features(ohlcv)
    assert features.iloc[0].isna().any(), "warmup rows should have NaNs"
    assert features.iloc[250:].notna().all().all(), "post-warmup rows must be complete"


def test_atr_flat_market():
    """Flat prices with constant 1.0 range -> ATR converges to exactly 1.0."""
    idx = pd.date_range("2024-01-01", periods=100, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, index=idx
    )
    assert np.isclose(atr(df, length=14).iloc[-1], 1.0)


def test_atr_captures_gaps():
    """A gap between prev close and today's range must enter the true range."""
    idx = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0, "volume": 1}, index=idx
    )
    gapped = df.copy()
    gapped.loc[gapped.index[20], ["open", "high", "low", "close"]] = [110.0, 110.5, 109.5, 110.0]
    assert atr(gapped, 14).iloc[21] > atr(df, 14).iloc[21], "gap must raise ATR"
