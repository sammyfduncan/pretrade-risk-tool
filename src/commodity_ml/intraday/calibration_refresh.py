"""Pure validation helpers for the manual current-calibration refresh."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from commodity_ml.intraday.risk_calibration import HORIZONS_MINUTES, QUANTILES, sha256_file


MIN_MATCHED_PARITY_BARS = 100
MIN_PARITY_COVERAGE = 0.95
MIN_PARITY_CORRELATION = 0.98
MAX_PARITY_AGE_DAYS = 90
MAX_SOURCE_LAG_DAYS = 7
MIN_P95_COVERAGE = 0.93
MIN_P99_COVERAGE = 0.97


def synchronize_quote_legs(legs: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-align gold/silver bid/ask candles without filling a timestamp."""

    expected = {"gold_bid", "gold_ask", "silver_bid", "silver_ask"}
    if set(legs) != expected:
        raise ValueError(f"Quote legs must be exactly {sorted(expected)}")
    renamed: list[pd.DataFrame] = []
    for name in sorted(expected):
        frame = legs[name]
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
            raise ValueError(f"{name} requires a timezone-aware DatetimeIndex")
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{name} timestamps must be sorted and unique")
        required = {"open", "high", "low", "close"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} is missing OHLC columns")
        selected = frame.loc[:, ["open", "high", "low", "close"]].copy()
        selected.columns = [f"{name}_{column}" for column in selected.columns]
        renamed.append(selected)
    synchronized = pd.concat(renamed, axis=1, join="inner").sort_index()
    if synchronized.empty:
        raise ValueError("The four quote legs have no shared timestamps")
    if synchronized.isna().any().any():
        raise ValueError("Synchronized quotes contain missing values")
    return synchronized


def validate_parity_report(path: Path, *, as_of: pd.Timestamp) -> dict[str, Any]:
    """Validate one bounded Dukascopy/OANDA M30 parity artifact and its inputs."""

    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    reasons: list[str] = []
    if payload.get("granularity") != "M30":
        reasons.append("parity_not_m30")
    if payload.get("verdict") != "PASS":
        reasons.append("parity_verdict_not_pass")
    if int(payload.get("matched_bars", 0)) < MIN_MATCHED_PARITY_BARS:
        reasons.append("parity_too_few_matched_bars")
    for field in ("broker_coverage", "dukascopy_coverage"):
        if float(payload.get(field, 0)) < MIN_PARITY_COVERAGE:
            reasons.append(f"{field}_too_low")
    if float(payload.get("return_correlation", 0)) < MIN_PARITY_CORRELATION:
        reasons.append("parity_return_correlation_too_low")
    if payload.get("best_shift_bars") != 0:
        reasons.append("parity_best_shift_not_zero")

    latest: list[pd.Timestamp] = []
    sources = payload.get("sources")
    if not isinstance(sources, dict):
        reasons.append("parity_sources_missing")
    else:
        for provider in ("dukascopy", "oanda"):
            item = sources.get(provider)
            if not isinstance(item, dict):
                reasons.append(f"parity_{provider}_source_missing")
                continue
            source_path = Path(str(item.get("path", ""))).expanduser()
            if not source_path.is_file():
                reasons.append(f"parity_{provider}_artifact_missing")
                continue
            if sha256_file(source_path) != item.get("sha256"):
                reasons.append(f"parity_{provider}_hash_mismatch")
                continue
            frame = pd.read_parquet(source_path)
            if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
                reasons.append(f"parity_{provider}_timestamps_invalid")
                continue
            latest.append(frame.index.max().tz_convert("UTC"))
    now = _utc(as_of)
    if not latest or now - min(latest) > pd.Timedelta(days=MAX_PARITY_AGE_DAYS):
        reasons.append("parity_artifact_stale")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "passed": not reasons,
        "reasons": sorted(set(reasons)),
        "latest_source_timestamp": min(latest).isoformat() if latest else None,
    }


def profile_is_complete(profile: pd.DataFrame) -> bool:
    expected = {
        (instrument, horizon, slot)
        for instrument in ("XAUUSD", "XAGUSD")
        for horizon in HORIZONS_MINUTES
        for slot in range(7 * 60, 19 * 60 + 30, 30)
        if slot + horizon <= 21 * 60 + 30
    }
    observed = set(
        profile.loc[:, ["instrument", "horizon_minutes", "london_slot_minute"]]
        .itertuples(index=False, name=None)
    )
    return observed == expected and not profile.duplicated(
        ["instrument", "horizon_minutes", "london_slot_minute"]
    ).any()


def quantiles_are_valid(profile: pd.DataFrame) -> bool:
    for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps"):
        columns = [f"{metric}_p{int(q * 100):02d}" for q in QUANTILES]
        if not set(columns).issubset(profile.columns):
            return False
        values = profile[columns].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            return False
        if (np.diff(values, axis=1) < 0).any():
            return False
    return True


def coverage_gate(cells: Mapping[str, Mapping[str, Any]]) -> bool:
    for cell in cells.values():
        coverage = cell.get("coverage", {})
        for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps"):
            metric_coverage = coverage.get(metric, {})
            if metric_coverage.get("p95", {}).get("actual_coverage", 0) < MIN_P95_COVERAGE:
                return False
            if metric_coverage.get("p99", {}).get("actual_coverage", 0) < MIN_P99_COVERAGE:
                return False
    return bool(cells)


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    return timestamp.tz_convert("UTC")
