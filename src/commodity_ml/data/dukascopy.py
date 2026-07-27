"""Bulk historical candles from Dukascopy — the model's training data.

Why not just use the broker? IG's REST API caps historical price data at
10,000 points per WEEK. Step 1 alone wants ~40,000 hourly bars, so pulling
training history from IG would take a month of waiting. Dukascopy (a Swiss
bank/broker) publishes its full tick archive for free with no account, which
is the standard way to get deep FX/metals history.

Dukascopy's XAU/USD is spot gold, same underlying as IG's spot gold market,
so features built on returns transfer. They are *not* byte-identical though
(different liquidity providers and feed timing) — scripts/compare_sources.py
quantifies the gap on the overlapping window before you trust a model trained
here to run there.
"""

from __future__ import annotations

import pandas as pd

from commodity_ml.config import DataConfig
from commodity_ml.data.schema import GRANULARITY_TO_DUKASCOPY, validate_candles

SOURCE = "dukascopy"


def fetch_candles(cfg: DataConfig, granularity: str, verbose: bool = True) -> pd.DataFrame:
    """Download candles for the configured instrument and date range."""
    import dukascopy_python
    from dukascopy_python import instruments as dk_instruments

    if granularity not in GRANULARITY_TO_DUKASCOPY:
        raise ValueError(f"Unsupported granularity for Dukascopy: {granularity}")
    interval = getattr(dukascopy_python, GRANULARITY_TO_DUKASCOPY[granularity])

    # Dukascopy names instruments as constants like INSTRUMENT_FX_METALS_XAU_USD;
    # resolve from the human-readable "XAU/USD" in config.
    instrument = _resolve_instrument(dk_instruments, cfg.dukascopy.instrument)

    offer_side = {
        "bid": dukascopy_python.OFFER_SIDE_BID,
        "ask": dukascopy_python.OFFER_SIDE_ASK,
    }.get(cfg.dukascopy.offer_side)
    if offer_side is None:
        raise ValueError(f"offer_side must be 'bid' or 'ask', got {cfg.dukascopy.offer_side!r}")

    start = pd.Timestamp(cfg.start).to_pydatetime()
    end = (pd.Timestamp(cfg.end) if cfg.end else pd.Timestamp.utcnow().tz_localize(None)).to_pydatetime()

    if verbose:
        print(f"Downloading {cfg.dukascopy.instrument} {granularity} "
              f"({cfg.dukascopy.offer_side}) from Dukascopy: {start:%Y-%m-%d} -> {end:%Y-%m-%d}")
        print("  (several years of hourly data takes a few minutes — it fetches month by month)")

    df = dukascopy_python.fetch(instrument, interval, offer_side, start, end)
    if df is None or df.empty:
        raise RuntimeError(
            f"Dukascopy returned no data for {cfg.dukascopy.instrument} {granularity}. "
            "Check the instrument name and date range."
        )
    return validate_candles(df, SOURCE)


def _resolve_instrument(dk_instruments, name: str) -> str:
    """Map 'XAU/USD' to the matching INSTRUMENT_* constant value."""
    for attr in dir(dk_instruments):
        if attr.startswith("INSTRUMENT_") and getattr(dk_instruments, attr) == name:
            return getattr(dk_instruments, attr)
    available = sorted(
        getattr(dk_instruments, a)
        for a in dir(dk_instruments)
        if a.startswith("INSTRUMENT_FX_METALS")
    )
    raise ValueError(f"Unknown Dukascopy instrument {name!r}. Metals available: {available}")
