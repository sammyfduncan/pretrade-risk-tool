"""The canonical candle contract.

Two providers feed this project (Dukascopy for training history, OANDA for
execution), and everything downstream — features, labels, backtest — assumes
one shape. Both sources are funnelled through `validate_candles` so a
provider quirk fails loudly here rather than silently poisoning a model.

Contract: UTC DatetimeIndex named 'time', sorted, unique, with float columns
open/high/low/close and a numeric volume. A provider may additionally supply a
*complete* bid/ask OHLC set. Partial or crossed quote sets are rejected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
OHLC_COLUMNS = ["open", "high", "low", "close"]
QUOTE_COLUMNS = [
    f"{quote_side}_{field}"
    for quote_side in ("bid", "ask")
    for field in OHLC_COLUMNS
]

# Canonical granularity -> per-source timeframe identifiers.
# OANDA v20 uses the same stable codes as this project's canonical names.
GRANULARITY_TO_OANDA = {
    "M5": "M5",
    "M15": "M15",
    "M30": "M30",
    "H1": "H1",
    "H4": "H4",
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

    quotes_present = [column for column in QUOTE_COLUMNS if column in df.columns]
    if quotes_present and len(quotes_present) != len(QUOTE_COLUMNS):
        missing_quotes = sorted(set(QUOTE_COLUMNS) - set(quotes_present))
        raise ValueError(
            f"[{source}] partial bid/ask OHLC is invalid; missing {missing_quotes}"
        )

    columns = REQUIRED_COLUMNS + (QUOTE_COLUMNS if quotes_present else [])
    df = df[columns].copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"[{source}] candles must have a DatetimeIndex, got {type(df.index)}")
    df.index = (
        df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    )
    df.index.name = "time"

    df = df[~df.index.duplicated(keep="first")].sort_index()

    for col in OHLC_COLUMNS + (QUOTE_COLUMNS if quotes_present else []):
        df[col] = df[col].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    price_columns = OHLC_COLUMNS + (QUOTE_COLUMNS if quotes_present else [])
    if not np.isfinite(df[price_columns].to_numpy()).all():
        raise ValueError(f"[{source}] candles contain non-finite prices")
    if (df[price_columns] <= 0).any().any():
        raise ValueError(f"[{source}] candles contain non-positive prices")
    if not np.isfinite(df["volume"].to_numpy()).all():
        raise ValueError(f"[{source}] candles contain non-finite volume")
    if (df["volume"] < 0).any():
        raise ValueError(f"[{source}] candles contain negative volume")

    for prefix in ("", "bid_", "ask_"):
        if prefix and not quotes_present:
            continue
        o, h, low, c = (f"{prefix}{field}" for field in OHLC_COLUMNS)
        bad_range = (
            (df[h] < df[[o, c]].max(axis=1))
            | (df[low] > df[[o, c]].min(axis=1))
            | (df[h] < df[low])
        )
        if bad_range.any():
            label = prefix.rstrip("_") or "mid"
            raise ValueError(
                f"[{source}] {int(bad_range.sum())} {label} candles violate OHLC "
                "ranges (high >= close >= low and open must also lie in range)"
            )

    if quotes_present:
        crossed = pd.DataFrame(
            {
                field: df[f"ask_{field}"] < df[f"bid_{field}"]
                for field in OHLC_COLUMNS
            }
        )
        if crossed.any().any():
            raise ValueError(f"[{source}] candles contain crossed bid/ask quotes")

    return df


def describe_candles(df: pd.DataFrame) -> str:
    """One-line summary used by the fetch scripts."""
    span_days = (df.index[-1] - df.index[0]).days
    return (
        f"{len(df):,} candles | {df.index[0]:%Y-%m-%d %H:%M} -> {df.index[-1]:%Y-%m-%d %H:%M} "
        f"({span_days / 365.25:.1f} years)"
    )
