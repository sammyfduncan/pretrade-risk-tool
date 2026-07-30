"""The canonical candle contract both data providers must satisfy.

Two sources feed this project (Dukascopy for training, IG for execution).
These tests pin the contract so a provider quirk fails loudly at ingest
rather than quietly distorting a model.
"""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.config import load_config
from commodity_ml.data.ig import estimate_points
from commodity_ml.data.schema import (
    GRANULARITY_TO_DUKASCOPY,
    GRANULARITY_TO_IG,
    QUOTE_COLUMNS,
    validate_candles,
)


def make_frame(**overrides) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    base = {"open": 2000.0, "high": 2001.0, "low": 1999.0, "close": 2000.5, "volume": 1.0}
    base.update(overrides)
    return pd.DataFrame(base, index=idx)


def test_validate_accepts_good_candles():
    out = validate_candles(make_frame(), "test")
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out.index.name == "time" and str(out.index.tz) == "UTC"


def test_naive_index_is_localized_to_utc():
    df = make_frame()
    df.index = df.index.tz_localize(None)
    assert str(validate_candles(df, "test").index.tz) == "UTC"


def test_non_utc_index_is_converted():
    df = make_frame()
    df.index = df.index.tz_convert("America/New_York")
    out = validate_candles(df, "test")
    assert str(out.index.tz) == "UTC"


def test_missing_column_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_candles(make_frame().drop(columns=["volume"]), "test")


def test_inconsistent_high_low_rejected():
    df = make_frame()
    df.loc[df.index[3], "high"] = 1990.0  # below the low
    with pytest.raises(ValueError, match="high >= close >= low"):
        validate_candles(df, "test")


def test_non_positive_price_rejected():
    df = make_frame()
    df.loc[df.index[2], "close"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_candles(df, "test")


def test_non_numeric_or_negative_volume_rejected():
    bad_text = make_frame(volume="not-a-number")
    with pytest.raises(ValueError, match="non-finite volume"):
        validate_candles(bad_text, "test")

    negative = make_frame()
    negative.loc[negative.index[0], "volume"] = -1
    with pytest.raises(ValueError, match="negative volume"):
        validate_candles(negative, "test")


def test_duplicates_dropped_and_sorted():
    df = pd.concat([make_frame(), make_frame()]).sample(frac=1, random_state=0)
    out = validate_candles(df, "test")
    assert len(out) == 10
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()


def test_complete_bid_ask_quotes_are_preserved():
    df = make_frame()
    for field in ("open", "high", "low", "close"):
        df[f"bid_{field}"] = df[field] - 0.05
        df[f"ask_{field}"] = df[field] + 0.05

    out = validate_candles(df, "test")

    assert all(column in out.columns for column in QUOTE_COLUMNS)
    assert np.allclose(out["ask_close"].sub(out["bid_close"]), 0.1)


def test_partial_and_crossed_quotes_fail_closed():
    partial = make_frame()
    partial["bid_open"] = partial["open"] - 0.05
    with pytest.raises(ValueError, match="partial bid/ask"):
        validate_candles(partial, "test")

    crossed = make_frame()
    for field in ("open", "high", "low", "close"):
        crossed[f"bid_{field}"] = crossed[field] + 0.05
        crossed[f"ask_{field}"] = crossed[field] - 0.05
    with pytest.raises(ValueError, match="crossed bid/ask"):
        validate_candles(crossed, "test")


def test_both_sources_cover_the_same_granularities():
    assert set(GRANULARITY_TO_IG) == set(GRANULARITY_TO_DUKASCOPY)


def test_ig_allowance_estimate_flags_oversized_pulls():
    """Seven years of H1 must estimate well past IG's 10k/week allowance —
    this is what stops us silently burning a week of quota."""
    assert estimate_points("H1", "2019-01-01", "2026-01-01") > 10_000
    assert estimate_points("D", "2019-01-01", "2026-01-01") < 10_000


def test_shipped_gold_config_loads():
    cfg = load_config("config/gold.yaml")
    assert cfg.symbol == "XAUUSD"
    assert cfg.data.source == "dukascopy"       # training history, not the broker
    assert cfg.cv.gap_bars >= cfg.label.horizon_bars
    # Sources get separate files so compare_sources.py can diff them
    assert cfg.raw_path("ig") != cfg.raw_path("dukascopy")
