"""Causal COMEX GC market state for the registered intraday experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NEW_YORK = "America/New_York"
LONDON = "Europe/London"
FUTURES_FEATURES = (
    "gc_side_return_30m",
    "gc_side_return_2h",
    "gc_volume_surprise_30m",
    "gc_side_spot_gap_30m",
    "gc_open_interest_log_change",
    "gc_basis_deviation",
)


@dataclass(frozen=True)
class FuturesFeatureResult:
    features: pd.DataFrame
    eligible: pd.Series
    report: dict


def load_cme_partitions(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the normalized, development-only CME artifact."""

    root = Path(root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"Missing CME import manifest: {manifest}")
    ohlcv_paths = sorted((root / "ohlcv_1m").glob("*.parquet"))
    statistics_paths = sorted((root / "statistics").glob("*.parquet"))
    if not ohlcv_paths or not statistics_paths:
        raise ValueError("CME import is missing monthly OHLCV/statistics partitions")
    definitions = pd.read_parquet(root / "definitions.parquet")
    ohlcv = pd.concat((pd.read_parquet(path) for path in ohlcv_paths)).sort_index()
    statistics = pd.concat(
        (pd.read_parquet(path) for path in statistics_paths)
    ).sort_index()
    return ohlcv, statistics, definitions


def build_futures_candidate_features(
    spot_m30: pd.DataFrame,
    candidates: pd.DataFrame,
    ohlcv: pd.DataFrame,
    statistics: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    degraded_dates: tuple[str, ...] | list[str] = (),
) -> FuturesFeatureResult:
    """Build the frozen F-GC family using only information available at decision."""

    _require_utc_index(spot_m30, "spot M30")
    _require_utc_index(candidates, "candidate")
    required_spot = {"open", "close", "decision_time"}
    required_candidates = {"is_long"}
    if missing := sorted(required_spot - set(spot_m30)):
        raise ValueError(f"Spot M30 frame missing columns: {missing}")
    if missing := sorted(required_candidates - set(candidates)):
        raise ValueError(f"Candidate frame missing columns: {missing}")
    if len(candidates.index.difference(spot_m30.index)):
        raise ValueError("Candidate timestamps are missing from the spot M30 frame")

    active = build_active_gc_m30(ohlcv, statistics, definitions)
    spot = spot_m30.loc[:, ["open", "close", "decision_time"]].copy()
    joined = spot.join(active, how="left")
    joined["spot_return_30m"] = joined["close"] / joined["open"] - 1.0
    joined["gc_basis"] = np.log(joined["gc_close"] / joined["close"])
    london = joined.index.tz_convert(LONDON)
    joined["_london_slot"] = london.hour * 60 + london.minute
    basis_reference = joined.groupby(
        ["raw_symbol", "_london_slot"], dropna=False
    )["gc_basis"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).median()
    )
    joined["gc_basis_deviation"] = joined["gc_basis"] - basis_reference

    rows = joined.reindex(candidates.index)
    side = np.where(candidates["is_long"].astype(int).to_numpy() == 1, 1.0, -1.0)
    features = pd.DataFrame(index=candidates.index)
    features["gc_side_return_30m"] = side * rows["gc_return_30m"].to_numpy()
    features["gc_side_return_2h"] = side * rows["gc_return_2h"].to_numpy()
    features["gc_volume_surprise_30m"] = rows[
        "gc_volume_surprise_30m"
    ].to_numpy()
    features["gc_side_spot_gap_30m"] = side * (
        rows["gc_return_30m"].to_numpy() - rows["spot_return_30m"].to_numpy()
    )
    features["gc_open_interest_log_change"] = rows[
        "gc_open_interest_log_change"
    ].to_numpy()
    features["gc_basis_deviation"] = rows["gc_basis_deviation"].to_numpy()

    finite = pd.Series(
        np.isfinite(features.to_numpy(dtype=float)).all(axis=1),
        index=features.index,
    )
    quality = _quality_mask(features.index, degraded_dates)
    eligible = finite & quality
    report = {
        "candidate_rows": len(candidates),
        "eligible_rows": int(eligible.sum()),
        "coverage": float(eligible.mean()) if len(eligible) else 0.0,
        "incomplete_rows": int((~finite).sum()),
        "quality_excluded_rows": int((finite & ~quality).sum()),
        "degraded_source_dates": list(degraded_dates),
        "active_m30_rows": len(active),
        "active_contracts": int(active["raw_symbol"].nunique()),
        "feature_names": list(FUTURES_FEATURES),
    }
    return FuturesFeatureResult(features, eligible, report)


def build_active_gc_m30(
    ohlcv: pd.DataFrame,
    statistics: pd.DataFrame,
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """Construct causal M30 bars from the prior completed session's volume leader."""

    _require_utc_index(ohlcv, "GC OHLCV")
    _require_utc_index(statistics, "GC statistics")
    required = {"raw_symbol", "open", "high", "low", "close", "volume"}
    if missing := sorted(required - set(ohlcv)):
        raise ValueError(f"GC OHLCV missing columns: {missing}")
    selections = completed_session_selections(ohlcv, definitions)

    left = ohlcv.reset_index().rename(columns={ohlcv.index.name: "event_time"})
    left = left.sort_values("event_time")
    right = selections.loc[:, ["available_time", "active_symbol"]].sort_values(
        "available_time"
    )
    mapped = pd.merge_asof(
        left,
        right,
        left_on="event_time",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    mapped = mapped.loc[mapped["raw_symbol"] == mapped["active_symbol"]].copy()
    if mapped.empty:
        raise ValueError("No GC bars map to a causally selected active contract")
    mapped["bucket"] = mapped["event_time"].dt.floor("30min")
    symbol_count = mapped.groupby("bucket")["raw_symbol"].nunique()
    if (symbol_count > 1).any():
        raise ValueError("Active GC contract changes inside an M30 bucket")
    bars = mapped.groupby("bucket", sort=True).agg(
        raw_symbol=("raw_symbol", "first"),
        gc_open=("open", "first"),
        gc_high=("high", "max"),
        gc_low=("low", "min"),
        gc_close=("close", "last"),
        gc_volume=("volume", "sum"),
        gc_trade_minutes=("volume", "size"),
    )
    bars.index = pd.DatetimeIndex(bars.index, name="time")
    bars["gc_return_30m"] = bars["gc_close"] / bars["gc_open"] - 1.0
    log_return = np.log(bars["gc_close"] / bars["gc_open"])
    contiguous = (
        bars.index.to_series().diff(3).eq(pd.Timedelta(minutes=90))
        & bars["raw_symbol"].eq(bars["raw_symbol"].shift(3))
    )
    bars["gc_return_2h"] = log_return.rolling(4, min_periods=4).sum().where(
        contiguous
    )
    london = bars.index.tz_convert(LONDON)
    slot = pd.Series(london.hour * 60 + london.minute, index=bars.index)
    log_volume = np.log1p(bars["gc_volume"].astype(float))
    reference = log_volume.groupby(slot).transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).median()
    )
    bars["gc_volume_surprise_30m"] = log_volume - reference
    bars["gc_open_interest_log_change"] = _asof_open_interest(
        bars.index + pd.Timedelta(minutes=30),
        bars["raw_symbol"],
        statistics,
    )
    return bars


def completed_session_selections(
    ohlcv: pd.DataFrame,
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """Select each completed session's volume leader for subsequent use."""

    required_definitions = {"raw_symbol", "instrument_class", "expiration"}
    if missing := sorted(required_definitions - set(definitions)):
        raise ValueError(f"GC definitions missing columns: {missing}")
    local = ohlcv.index.tz_convert(NEW_YORK)
    session_date = pd.DatetimeIndex(local.date)
    session_date = session_date + pd.to_timedelta((local.hour >= 18).astype(int), unit="D")
    volume = ohlcv.assign(_session_date=session_date).groupby(
        ["_session_date", "raw_symbol"], sort=True
    )["volume"].sum().reset_index()
    volume = volume.sort_values(
        ["_session_date", "volume", "raw_symbol"],
        ascending=[True, False, True],
    )
    winners = volume.drop_duplicates("_session_date", keep="first").copy()
    naive_available = pd.DatetimeIndex(winners["_session_date"]) + pd.Timedelta(
        hours=17
    )
    winners["available_time"] = naive_available.tz_localize(NEW_YORK).tz_convert(
        "UTC"
    )
    winners = winners.rename(columns={"raw_symbol": "active_symbol"})

    outright = definitions.loc[definitions["instrument_class"] == "F"].copy()
    expiry = outright.groupby("raw_symbol")["expiration"].max()
    winners["expiration"] = winners["active_symbol"].map(expiry)
    if winners["expiration"].isna().any():
        raise ValueError("Active GC selection has no contract expiration")
    if (winners["expiration"] <= winners["available_time"]).any():
        raise ValueError("Completed-session volume selected an expired GC contract")
    return winners.loc[
        :, ["_session_date", "available_time", "active_symbol", "volume", "expiration"]
    ].reset_index(drop=True)


def _asof_open_interest(
    decision_times: pd.DatetimeIndex,
    symbols: pd.Series,
    statistics: pd.DataFrame,
) -> pd.Series:
    required = {"raw_symbol", "stat_type", "ts_ref", "quantity"}
    if missing := sorted(required - set(statistics)):
        raise ValueError(f"GC statistics missing columns: {missing}")
    oi = statistics.loc[
        (statistics["stat_type"] == 9)
        & statistics["ts_ref"].notna()
        & (statistics["quantity"] > 0)
    ].copy()
    oi = oi.reset_index().rename(columns={statistics.index.name: "available_time"})
    oi = oi.sort_values(["raw_symbol", "ts_ref", "available_time"])
    # Keeping only the final update for a reference date is conservative: that
    # value is not exposed until its own receive timestamp.
    oi = oi.drop_duplicates(["raw_symbol", "ts_ref"], keep="last")
    oi["gc_open_interest_log_change"] = oi.groupby("raw_symbol")["quantity"].transform(
        lambda values: np.log(values.astype(float)).diff()
    )
    left = pd.DataFrame(
        {
            "decision_time": decision_times,
            "raw_symbol": symbols.to_numpy(),
            "_row": np.arange(len(symbols)),
        }
    )
    chunks: list[pd.DataFrame] = []
    for symbol, group in left.groupby("raw_symbol", sort=False, dropna=False):
        right = oi.loc[
            oi["raw_symbol"] == symbol,
            ["available_time", "gc_open_interest_log_change"],
        ].sort_values("available_time")
        current = group.sort_values("decision_time")
        if right.empty or pd.isna(symbol):
            current["gc_open_interest_log_change"] = np.nan
        else:
            current = pd.merge_asof(
                current,
                right,
                left_on="decision_time",
                right_on="available_time",
                direction="backward",
                allow_exact_matches=True,
            )
        chunks.append(current)
    merged = pd.concat(chunks).sort_values("_row")
    return pd.Series(
        merged["gc_open_interest_log_change"].to_numpy(),
        index=symbols.index,
    )


def _quality_mask(
    index: pd.DatetimeIndex,
    degraded_dates: tuple[str, ...] | list[str],
) -> pd.Series:
    dates = index.normalize()
    valid = np.ones(len(index), dtype=bool)
    for value in degraded_dates:
        start = pd.Timestamp(value, tz="UTC")
        valid &= ~((dates >= start) & (dates <= start + pd.Timedelta(days=28)))
    return pd.Series(valid, index=index)


def _require_utc_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{label} requires a timezone-aware index")
    if str(frame.index.tz) != "UTC":
        raise ValueError(f"{label} index must be UTC")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be sorted")
