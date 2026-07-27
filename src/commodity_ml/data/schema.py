"""The canonical candle contract.

Two providers feed this project (Dukascopy for training history, IG for
execution), and everything downstream — features, labels, backtest — assumes
one shape. Both sources are funnelled through `validate_candles` so a
provider quirk fails loudly here rather than silently poisoning a model.

Contract: UTC DatetimeIndex named 'time', sorted, unique, with float columns
open/high/low/close and a numeric volume.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]

# Canonical granularity -> per-source timeframe identifiers.
# IG strings are pandas offset aliases; trading-ig's conv_resol maps them to
# IG's own names (1h -> "HOUR", 5Min -> "MINUTE_5").
GRANULARITY_TO_IG = {
    "M5": "5Min",
    "M15": "15Min",
    "M30": "30Min",
    "H1": "1h",
    "H4": "4h",
    "D": "D",
}

GRANULARITY_TO_DUKASCOPY = {
    "M5": "INTERVAL_MIN_5",
    "M15": "INTERVAL_MIN_15",
    "M30": "INTERVAL_MIN_30",
    "H1": "INTERVAL_HOUR_1",
    "H4": "INTERVAL_HOUR_4",
    "D": "INTERVAL_DAY_1",
}


def validate_candles(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize and sanity-check a provider's candles. Returns the clean frame."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"[{source}] candles missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{source}] candles must have a DatetimeIndex, got {type(df.index)}")
    df.index = (
        df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    )
    df.index.name = "time"

    df = df[~df.index.duplicated(keep="first")].sort_index()

    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"[{source}] candles contain non-positive prices")

    bad_range = (df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])
    if bad_range.any():
        raise ValueError(
            f"[{source}] {int(bad_range.sum())} candles violate high >= close >= low"
        )

    return df


def describe_candles(df: pd.DataFrame) -> str:
    """One-line summary used by the fetch scripts."""
    span_days = (df.index[-1] - df.index[0]).days
    return (
        f"{len(df):,} candles | {df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M} "
        f"({span_days / 365.25:.1f} years)"
    )
