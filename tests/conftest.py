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


def make_cot(n_weeks: int = 500, seed: int = 7) -> pd.DataFrame:
    """Synthetic weekly COT reports shaped like the parsed CFTC snapshot.

    Tuesday-dated (2015-01-06 is a Tuesday) and long enough that the 104-week
    z-window is warm years before the bar ranges tests use — including the
    real Feb-2023 publication-interruption window, which 500 weeks spans.
    Positions random-walk as fractions of a drifting open interest so the
    net-of-OI features move every week.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-06", periods=n_weeks, freq="7D", name="report_date")
    oi = 400_000 * np.exp(np.cumsum(rng.normal(0, 0.01, n_weeks)))

    def frac_walk(start: float, vol: float) -> np.ndarray:
        return np.clip(start + np.cumsum(rng.normal(0, vol, n_weeks)), 0.02, 0.6)

    return pd.DataFrame(
        {
            "oi": oi,
            "mm_long": frac_walk(0.30, 0.01) * oi,
            "mm_short": frac_walk(0.10, 0.008) * oi,
            "prod_long": frac_walk(0.15, 0.008) * oi,
            "prod_short": frac_walk(0.35, 0.01) * oi,
            "swap_long": frac_walk(0.20, 0.008) * oi,
            "swap_short": frac_walk(0.25, 0.008) * oi,
        },
        index=idx,
    )


def make_gdelt(n_days: int = 900, seed: int = 7) -> pd.DataFrame:
    """Synthetic daily GDELT timelines shaped like the pinned snapshot:
    tone (drifting, negative-ish like real news), vol (share of all coverage)
    and n_articles (never zero here — tests poison days themselves)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="D", name="date")
    return pd.DataFrame(
        {
            "tone": np.clip(-2 + np.cumsum(rng.normal(0, 0.08, n_days)), -8, 4),
            "vol": np.abs(0.05 + np.cumsum(rng.normal(0, 0.0005, n_days))),
            "n_articles": rng.integers(120, 400, n_days).astype(float),
        },
        index=idx,
    )