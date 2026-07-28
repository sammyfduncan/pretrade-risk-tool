"""The development/holdout fence.

The holdout only protects you if it is excluded *by default* — a fence you have
to remember to close is not a fence. These tests pin that.
"""

import pandas as pd
import pytest

from commodity_ml.config import AssetConfig, DataConfig, load_config


def frame(start="2024-01-01", periods=400, freq="4h") -> pd.DataFrame:
    idx = pd.date_range(start, periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame({"x": range(periods)}, index=idx)


def cfg_with_holdout(holdout_start: str | None) -> AssetConfig:
    return AssetConfig(
        asset="test",
        symbol="XAUUSD",
        granularity="H4",
        data=DataConfig(holdout_start=holdout_start),
    )


def test_development_and_holdout_partition_the_data():
    cfg = cfg_with_holdout("2024-02-01")
    df = frame()
    dev, hold = cfg.development_slice(df), cfg.holdout_slice(df)

    assert len(dev) + len(hold) == len(df), "the split must lose no rows"
    assert dev.index.max() < cfg.holdout_ts <= hold.index.min()
    assert dev.index.intersection(hold.index).empty


def test_no_holdout_configured_returns_everything():
    cfg = cfg_with_holdout(None)
    df = frame()
    assert len(cfg.development_slice(df)) == len(df)
    assert cfg.holdout_ts is None
    with pytest.raises(ValueError, match="No data.holdout_start"):
        cfg.holdout_slice(df)


def test_holdout_boundary_is_inclusive_of_the_start_date():
    """A bar exactly at the cutoff belongs to the holdout, not development."""
    cfg = cfg_with_holdout("2024-01-05")
    df = frame()
    boundary = pd.Timestamp("2024-01-05", tz="UTC")
    assert boundary in df.index
    assert boundary not in cfg.development_slice(df).index
    assert boundary in cfg.holdout_slice(df).index


def test_shipped_gold_config_has_the_fence_up():
    """Guards against the holdout being quietly removed from the real config."""
    cfg = load_config("config/gold.yaml")
    assert cfg.data.holdout_start is not None, "gold.yaml must keep a holdout fenced off"
    assert cfg.holdout_ts > pd.Timestamp(cfg.data.start, tz="UTC")
