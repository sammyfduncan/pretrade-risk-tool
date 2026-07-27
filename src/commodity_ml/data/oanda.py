"""Historical candle download from OANDA via oandapyV20.

OANDA caps each InstrumentsCandles request at 5000 candles, so long histories
are fetched by paginating forward with `from` + `count` until we reach `end`.
Only complete candles are kept — the currently-forming candle would otherwise
change on re-download and make runs non-reproducible.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
from oandapyV20 import API
from oandapyV20.endpoints import instruments as v20_instruments

MAX_CANDLES_PER_REQUEST = 5000


def get_client(api_key: str | None = None, environment: str | None = None) -> API:
    """Build an OANDA API client from args or env (OANDA_API_KEY / OANDA_ENV)."""
    api_key = api_key or os.environ.get("OANDA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OANDA_API_KEY not set. Create a free practice account at oanda.com, "
            "generate a token under 'Manage API Access', and put it in .env "
            "(see .env.example)."
        )
    environment = environment or os.environ.get("OANDA_ENV", "practice")
    return API(access_token=api_key, environment=environment)


def _parse_candles(candles: list[dict]) -> pd.DataFrame:
    rows = [
        {
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": int(c["volume"]),
        }
        for c in candles
        if c["complete"]
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.set_index("time")


def fetch_candles(
    client: API,
    instrument: str,
    granularity: str,
    start: str | datetime,
    end: str | datetime | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch complete mid-price candles for [start, end] as a UTC-indexed DataFrame."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz=timezone.utc)

    chunks: list[pd.DataFrame] = []
    cursor = start_ts
    while cursor < end_ts:
        params = {
            "granularity": granularity,
            "price": "M",
            "count": MAX_CANDLES_PER_REQUEST,
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        request = v20_instruments.InstrumentsCandles(instrument=instrument, params=params)
        client.request(request)
        chunk = _parse_candles(request.response.get("candles", []))
        if chunk.empty:
            break
        chunks.append(chunk)
        if verbose:
            print(f"  fetched {len(chunk):>5} candles up to {chunk.index[-1]}")
        last = chunk.index[-1]
        if last <= cursor:  # no forward progress -> end of available data
            break
        cursor = last + pd.Timedelta(seconds=1)

    if not chunks:
        raise RuntimeError(
            f"No candles returned for {instrument} {granularity} from {start_ts}. "
            "Check the instrument name and your account's instrument access."
        )

    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.loc[df.index <= end_ts]
