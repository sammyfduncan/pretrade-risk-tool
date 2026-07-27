"""Shared synthetic OHLCV data — all tests run offline, no OANDA needed."""

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Random-walk hourly candles that look enough like XAUUSD to exercise
    every indicator (600 bars > the 200-bar EMA warmup)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    spread = np.abs(rng.normal(0, 0.001, n))
    return pd.DataFrame(
        {
            "open": np.roll(close, 1),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(100, 5000, n),
        },
        index=idx,
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()
