"""Label construction — the module where a bug silently invalidates everything."""

import numpy as np
import pandas as pd

from commodity_ml.config import LabelConfig
from commodity_ml.labels import make_labels


def constant_range_df(closes: list[float]) -> pd.DataFrame:
    """Candles with tiny fixed ranges so ATR is small and predictable."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c + 0.01, "low": c - 0.01, "close": c, "volume": 1}, index=idx
    )


def test_unlabeled_tail_is_nan(ohlcv):
    cfg = LabelConfig(mode="updown", horizon_bars=12)
    labels = make_labels(ohlcv, cfg)
    assert labels.iloc[-12:].isna().all(), "bars whose future is unknown must be NaN"
    assert labels.iloc[:-12].notna().all()


def test_updown_label_correctness():
    # Price: flat 100, then jumps to 110 at bar 30 and stays.
    closes = [100.0] * 30 + [110.0] * 30
    labels = make_labels(constant_range_df(closes), LabelConfig(mode="updown", horizon_bars=5))
    # Bars 25..29 see the jump within 5 bars -> 1; earlier flat bars -> 0.
    assert (labels.iloc[25:30] == 1.0).all()
    assert (labels.iloc[:25] == 0.0).all()
    # After the jump the price is flat again -> 0.
    assert (labels.iloc[30:55] == 0.0).all()


def test_long_threshold_requires_meaningful_move():
    # A rise much smaller than one ATR must NOT be a positive label.
    closes = [100.0] * 30 + [100.001] * 30
    cfg = LabelConfig(mode="long_threshold", horizon_bars=5, atr_mult=1.0)
    labels = make_labels(constant_range_df(closes), cfg).dropna()
    assert (labels == 0.0).all()

    # A rise far larger than one ATR must be positive just before the jump.
    closes = [100.0] * 30 + [105.0] * 30
    labels = make_labels(constant_range_df(closes), cfg)
    assert (labels.iloc[26:30] == 1.0).all()


def test_no_lookahead_in_threshold_scaling():
    """The ATR used to scale the threshold at bar t must not see the future.

    Labels for the shared past must be identical whether or not later
    (more volatile) data exists in the frame.
    """
    calm = [100.0 + 0.01 * i for i in range(60)]
    wild = list(np.linspace(100.6, 160, 40))  # volatility explosion afterwards
    cfg = LabelConfig(mode="long_threshold", horizon_bars=5, atr_mult=1.0)

    labels_full = make_labels(constant_range_df(calm + wild), cfg)
    labels_trunc = make_labels(constant_range_df(calm), cfg)

    # Compare the overlap that is fully labeled in both versions (first 55 bars).
    overlap = labels_trunc.dropna().index
    pd.testing.assert_series_equal(labels_full.loc[overlap], labels_trunc.loc[overlap])
