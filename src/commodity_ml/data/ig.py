"""IG Markets API access — epic lookup, recent candles, and (later) execution.

IG is the *execution* broker for this project, and its historical price
endpoint is deliberately not used for bulk training data: the REST API allows
only **10,000 historical price points per week**, which resets weekly. Step 1
needs roughly 40,000 hourly bars, so a full training pull would take about a
month. Training history therefore comes from Dukascopy (see dukascopy.py),
and IG is used for:

  * finding the epic (IG's market identifier — not a ticker symbol),
  * pulling a recent window to verify IG's prices agree with Dukascopy's,
  * fetching the latest bars at prediction time,
  * placing paper/live orders in a later step.

`fetch_candles` therefore *estimates the cost before spending it* and refuses
to silently burn the weekly allowance. Every response's remaining allowance is
printed so the budget is always visible.

Credentials come from .env: IG_API_KEY, IG_USERNAME, IG_PASSWORD, IG_ACC_TYPE.
"""

from __future__ import annotations

import os

import pandas as pd

from commodity_ml.config import DataConfig
from commodity_ml.data.schema import GRANULARITY_TO_IG, validate_candles

SOURCE = "ig"

# IG's published REST limit; see labs.ig.com FAQ. Used only to warn before spending.
WEEKLY_HISTORICAL_ALLOWANCE = 10_000


def get_service():
    """Authenticate against IG and return a live IGService."""
    from trading_ig import IGService

    missing = [
        var
        for var in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD")
        if not os.environ.get(var)
    ]
    if missing:
        raise RuntimeError(
            f"Missing IG credentials in environment: {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in — create the API key from "
            "IG's web platform under My Account -> Settings -> API keys "
            "(use a DEMO account key while developing)."
        )

    service = IGService(
        username=os.environ["IG_USERNAME"],
        password=os.environ["IG_PASSWORD"],
        api_key=os.environ["IG_API_KEY"],
        acc_type=os.environ.get("IG_ACC_TYPE", "demo"),
        acc_number=os.environ.get("IG_ACC_NUMBER") or None,
        use_rate_limiter=True,  # trading-ig throttles us below IG's per-minute cap
    )
    service.create_session()
    return service


def search_epics(service, search_term: str) -> pd.DataFrame:
    """Find candidate epics by name. IG epics look like 'CS.D.USCGC.TODAY.IP'
    and vary by account region, so they must be looked up, never guessed."""
    results = service.search_markets(search_term)
    columns = [c for c in ("epic", "instrumentName", "instrumentType", "expiry") if c in results]
    return results[columns]


def estimate_points(granularity: str, start: str, end: str | None = None) -> int:
    """Roughly how many price points a date range will cost against the allowance."""
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    hours = (end_ts - start_ts).total_seconds() / 3600
    per_hour = {"M5": 12, "M15": 4, "M30": 2, "H1": 1, "H4": 0.25, "D": 1 / 24}[granularity]
    return int(hours * per_hour * (5 / 7))  # markets closed at weekends


def fetch_candles(
    cfg: DataConfig,
    granularity: str,
    service=None,
    allow_large_download: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch candles from IG as mid prices, guarding the weekly allowance.

    Raises before making the request if the range would exceed the weekly
    allowance, unless allow_large_download=True.
    """
    if not cfg.ig.epic:
        raise ValueError(
            "No IG epic configured. Find it with:\n"
            '    python scripts/find_epic.py --search "spot gold"\n'
            "then set data.ig.epic in your config yaml."
        )
    if granularity not in GRANULARITY_TO_IG:
        raise ValueError(f"Unsupported granularity for IG: {granularity}")

    estimated = estimate_points(granularity, cfg.start, cfg.end)
    if estimated > WEEKLY_HISTORICAL_ALLOWANCE and not allow_large_download:
        raise RuntimeError(
            f"This request needs roughly {estimated:,} price points, but IG allows only "
            f"{WEEKLY_HISTORICAL_ALLOWANCE:,} per week.\n"
            "Use Dukascopy for training history (data.source: dukascopy) and keep IG for "
            "execution and recent bars. Pass allow_large_download=True to override."
        )

    service = service or get_service()
    if verbose:
        print(f"Fetching {cfg.ig.epic} {granularity} from IG "
              f"(~{estimated:,} points of the {WEEKLY_HISTORICAL_ALLOWANCE:,}/week allowance)")

    response = service.fetch_historical_prices_by_epic(
        epic=cfg.ig.epic,
        resolution=GRANULARITY_TO_IG[granularity],
        start_date=pd.Timestamp(cfg.start).strftime("%Y-%m-%dT%H:%M:%S"),
        end_date=(pd.Timestamp(cfg.end) if cfg.end else pd.Timestamp.utcnow()).strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    )

    if verbose and isinstance(response, dict) and "allowance" in response:
        allowance = response["allowance"]
        print(f"  allowance remaining: {allowance.get('remainingAllowance'):,} "
              f"of {allowance.get('totalAllowance'):,} "
              f"(resets in {allowance.get('allowanceExpiry')}s)")

    return _to_canonical(response["prices"])


def _to_canonical(prices: pd.DataFrame) -> pd.DataFrame:
    """IG returns hierarchical bid/ask columns; collapse to mid-price OHLCV.

    Mid (rather than bid or ask) keeps IG comparable with the Dukascopy
    training data and keeps trading costs in one place — the backtest's
    fee_pct — instead of being half-baked into the price series.
    """
    bid, ask = prices["bid"], prices["ask"]
    mid = pd.DataFrame(
        {
            "open": (bid["Open"] + ask["Open"]) / 2,
            "high": (bid["High"] + ask["High"]) / 2,
            "low": (bid["Low"] + ask["Low"]) / 2,
            "close": (bid["Close"] + ask["Close"]) / 2,
        }
    )
    # Spot gold is a CFD with no real traded volume; IG reports it when available.
    if "last" in prices.columns.get_level_values(0) and "Volume" in prices["last"].columns:
        mid["volume"] = prices["last"]["Volume"]
    else:
        mid["volume"] = 0.0

    return validate_candles(mid, SOURCE)
