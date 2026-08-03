"""Causal Treasury/currency state for the frozen P4 gold experiment.

The driver layer is deliberately separate from the gold frame.  It can be
built and quality-checked without exposing a gold label or strategy outcome.
Every value timestamped ``t`` uses only driver records in ``[t-30m, t)`` and
same-clock reference observations strictly before ``t``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_ml.intraday.data import LONDON, M30_DURATION
from commodity_ml.intraday.execution import EntryBlackout, TradeCandidate
from commodity_ml.intraday.futures_state import completed_session_selections

REFERENCE_OBSERVATIONS = 60
MIN_TRADED_MINUTES = 25
Z_THRESHOLD = 1.0


@dataclass(frozen=True)
class MacroDriverResult:
    """One driver's causal M30 state and data-quality report."""

    frame: pd.DataFrame
    report: dict


@dataclass(frozen=True)
class MacroStateResult:
    """Joint ZN/6E state; ``eligible`` means both legs are usable."""

    frame: pd.DataFrame
    report: dict


def load_macro_driver(root: str | Path, leg: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load one verified normalized P4 driver leg and its source manifest."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing P4 driver manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("strategy_evaluated"):
        raise ValueError("P4 driver artifact is incomplete or not outcome-blind")
    leg_root = root / leg
    definitions_path = leg_root / "definitions.parquet"
    partitions = sorted((leg_root / "ohlcv_1m").glob("*.parquet"))
    if not definitions_path.is_file() or not partitions:
        raise ValueError(f"P4 driver leg {leg!r} is incomplete")
    definitions = pd.read_parquet(definitions_path)
    ohlcv = pd.concat((pd.read_parquet(path) for path in partitions)).sort_index()
    return ohlcv, definitions, manifest


def build_driver_state(
    ohlcv: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    prefix: str,
    degraded_dates: tuple[str, ...] | list[str] = (),
) -> MacroDriverResult:
    """Build active-contract M30 returns and causal same-clock robust z-scores."""

    _validate_ohlcv(ohlcv, prefix)
    if not prefix or not prefix.replace("_", "").isalnum():
        raise ValueError("Driver prefix must be a non-empty identifier")

    selections = completed_session_selections(ohlcv, definitions)
    events = ohlcv.reset_index()
    index_name = ohlcv.index.name or "index"
    events = events.rename(columns={index_name: "event_time"}).sort_values(
        "event_time", kind="stable"
    )
    events = pd.merge_asof(
        events,
        selections.loc[:, ["available_time", "active_symbol"]].sort_values(
            "available_time"
        ),
        left_on="event_time",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    events = events.loc[events["raw_symbol"] == events["active_symbol"]].copy()
    if events.empty:
        raise ValueError(f"No {prefix} bars map to a causal active contract")

    events["bucket"] = events["event_time"].dt.floor("30min")
    grouped = events.groupby("bucket", sort=True)
    bars = grouped.agg(
        raw_symbol=("raw_symbol", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_minutes=("event_time", "nunique"),
        symbol_count=("raw_symbol", "nunique"),
    )
    bars.index = pd.DatetimeIndex(bars.index, name="bucket_start")
    bars["return_30m"] = np.log(bars["close"] / bars["open"])
    degraded = pd.Series(
        bars.index.strftime("%Y-%m-%d").isin(set(degraded_dates)),
        index=bars.index,
    )
    bars["source_degraded"] = degraded
    bars["bar_eligible"] = (
        (bars["trade_minutes"] >= MIN_TRADED_MINUTES)
        & (bars["symbol_count"] == 1)
        & ~bars["source_degraded"]
        & np.isfinite(bars["return_30m"])
    )
    bars["return_30m"] = bars["return_30m"].where(bars["bar_eligible"])

    decision_time = bars.index + M30_DURATION
    slot = pd.Series(
        decision_time.tz_convert(LONDON).hour * 60
        + decision_time.tz_convert(LONDON).minute,
        index=bars.index,
    )
    median, mad, count = _prior_observed_reference(bars["return_30m"], slot)
    denominator = 1.4826 * mad
    bars["z"] = ((bars["return_30m"] - median) / denominator).where(
        (count >= REFERENCE_OBSERVATIONS) & denominator.gt(0)
    )
    bars["eligible"] = bars["bar_eligible"] & bars["z"].notna()
    bars.index = pd.DatetimeIndex(decision_time, name="decision_time")

    renamed = bars.rename(
        columns={column: f"{prefix}_{column}" for column in bars.columns}
    )
    report = {
        "prefix": prefix,
        "source_rows": len(ohlcv),
        "active_m30_rows": len(renamed),
        "first_decision": renamed.index.min().isoformat(),
        "last_decision": renamed.index.max().isoformat(),
        "insufficient_minutes": int(
            (renamed[f"{prefix}_trade_minutes"] < MIN_TRADED_MINUTES).sum()
        ),
        "contract_change_buckets": int(
            (renamed[f"{prefix}_symbol_count"] != 1).sum()
        ),
        "degraded_buckets": int(renamed[f"{prefix}_source_degraded"].sum()),
        "eligible_rows": int(renamed[f"{prefix}_eligible"].sum()),
    }
    return MacroDriverResult(frame=renamed, report=report)


def combine_driver_states(
    zn: MacroDriverResult,
    euro: MacroDriverResult,
) -> MacroStateResult:
    """Inner-align ZN and 6E decision times without filling either market."""

    joint = zn.frame.join(euro.frame, how="outer").sort_index()
    required = {"zn_eligible", "euro_eligible", "zn_z", "euro_z"}
    if missing := sorted(required - set(joint)):
        raise ValueError(f"Joint driver state missing columns: {missing}")
    joint["eligible"] = (
        joint["zn_eligible"].eq(True)
        & joint["euro_eligible"].eq(True)
        & joint[["zn_z", "euro_z"]].notna().all(axis=1)
    )
    report = {
        "zn": zn.report,
        "euro": euro.report,
        "joint_rows": len(joint),
        "joint_eligible_rows": int(joint["eligible"].sum()),
        "first_joint_eligible": (
            joint.index[joint["eligible"]].min().isoformat()
            if joint["eligible"].any()
            else None
        ),
        "last_joint_eligible": (
            joint.index[joint["eligible"]].max().isoformat()
            if joint["eligible"].any()
            else None
        ),
    }
    return MacroStateResult(frame=joint, report=report)


def generate_macro_impulse_candidates(
    gold: pd.DataFrame,
    state: pd.DataFrame,
    *,
    entry_blackouts: tuple[EntryBlackout, ...] = (),
) -> list[TradeCandidate]:
    """Apply the exact frozen P4 rule and take its first signal per London day."""

    _validate_gold(gold)
    required = {"zn_z", "euro_z", "eligible"}
    if missing := sorted(required - set(state)):
        raise ValueError(f"Macro state missing columns: {missing}")
    if not isinstance(state.index, pd.DatetimeIndex) or state.index.tz is None:
        raise ValueError("Macro state requires a timezone-aware DatetimeIndex")
    if not state.index.is_monotonic_increasing or not state.index.is_unique:
        raise ValueError("Macro state index must be sorted and unique")

    decision = pd.DatetimeIndex(pd.to_datetime(gold["decision_time"], utc=True))
    aligned = state.reindex(decision)
    aligned.index = gold.index
    valid = aligned["eligible"].eq(True)
    long_macro = valid & (aligned["zn_z"] >= Z_THRESHOLD) & (
        aligned["euro_z"] >= Z_THRESHOLD
    )
    short_macro = valid & (aligned["zn_z"] <= -Z_THRESHOLD) & (
        aligned["euro_z"] <= -Z_THRESHOLD
    )
    body = gold["close"] - gold["open"]
    confirmation = (long_macro & body.gt(0)) | (short_macro & body.lt(0))
    no_chase = body.abs() <= gold["m30_atr14"]
    signal = (
        _entry_eligible(gold.index)
        & ~_entry_blackout(gold.index, entry_blackouts)
        & confirmation
        & no_chase
    )

    entry_local = (gold.index + M30_DURATION).tz_convert(LONDON)
    session_dates = np.asarray(entry_local.date)
    candidates: list[TradeCandidate] = []
    for session_date in pd.unique(session_dates):
        times = gold.index[signal.fillna(False) & (session_dates == session_date)]
        if len(times) == 0:
            continue
        signal_time = times[0]
        distance = float(gold.at[signal_time, "m30_atr14"])
        if not np.isfinite(distance) or distance <= 0:
            continue
        candidates.append(
            TradeCandidate(
                signal_time=signal_time,
                side="long" if bool(long_macro.loc[signal_time]) else "short",
                stop_distance_pct=None,
                target_distance_pct=None,
                max_holding_bars=6,
                strategy="P4",
                stop_distance_abs=distance,
                target_r_multiple=1.5,
            )
        )
    return candidates


def driver_coverage(gold: pd.DataFrame, state: pd.DataFrame) -> dict:
    """Measure joint state coverage in the P4 entry universe after warm-up."""

    _validate_gold(gold)
    eligible_state = state.index[state["eligible"].eq(True)]
    if len(eligible_state) == 0:
        return {"eligible_gold_rows": 0, "matched_rows": 0, "coverage": 0.0}
    decision = pd.DatetimeIndex(pd.to_datetime(gold["decision_time"], utc=True))
    universe = _entry_eligible(gold.index) & (decision >= eligible_state.min())
    matched = pd.Series(decision.isin(eligible_state), index=gold.index) & universe
    denominator = int(universe.sum())
    return {
        "eligible_gold_rows": denominator,
        "matched_rows": int(matched.sum()),
        "coverage": float(matched.sum() / denominator) if denominator else 0.0,
        "warmup_end": eligible_state.min().isoformat(),
    }


def _prior_observed_reference(
    values: pd.Series,
    slot: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = pd.Series(np.nan, index=values.index, dtype=float)
    mad = pd.Series(np.nan, index=values.index, dtype=float)
    count = pd.Series(0, index=values.index, dtype="int64")
    for _, positions in slot.groupby(slot, sort=False).groups.items():
        observed = values.loc[positions].dropna()
        if observed.empty:
            continue
        prior = observed.shift(1)
        med = prior.rolling(
            REFERENCE_OBSERVATIONS,
            min_periods=REFERENCE_OBSERVATIONS,
        ).median()
        deviation = prior.rolling(
            REFERENCE_OBSERVATIONS,
            min_periods=REFERENCE_OBSERVATIONS,
        ).apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
        median.loc[observed.index] = med
        mad.loc[observed.index] = deviation
        count.loc[observed.index] = prior.expanding().count().clip(
            upper=REFERENCE_OBSERVATIONS
        ).astype("int64")
    return median, mad, count


def _entry_eligible(index: pd.DatetimeIndex) -> pd.Series:
    entry = (index + M30_DURATION).tz_convert(LONDON)
    minute = entry.hour * 60 + entry.minute
    return pd.Series(
        (minute >= 7 * 60) & (minute < 19 * 60 + 30),
        index=index,
    )


def _entry_blackout(
    index: pd.DatetimeIndex,
    blackouts: tuple[EntryBlackout, ...],
) -> pd.Series:
    entry = index + M30_DURATION
    blocked = np.zeros(len(index), dtype=bool)
    for blackout in blackouts:
        start = pd.Timestamp(blackout.start).tz_convert("UTC")
        end = pd.Timestamp(blackout.end).tz_convert("UTC")
        blocked |= (entry >= start) & (entry < end)
    return pd.Series(blocked, index=index)


def _validate_ohlcv(frame: pd.DataFrame, prefix: str) -> None:
    required = {"raw_symbol", "open", "high", "low", "close", "volume"}
    if missing := sorted(required - set(frame)):
        raise ValueError(f"{prefix} OHLCV missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{prefix} OHLCV requires a timezone-aware index")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{prefix} OHLCV index must be sorted")
    numeric = frame.loc[:, ["open", "high", "low", "close", "volume"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{prefix} OHLCV contains non-finite values")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"{prefix} OHLCV contains non-positive prices")
    if (frame["volume"] <= 0).any():
        raise ValueError(f"{prefix} OHLCV contains non-positive volume")
    bad = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    )
    if bad.any():
        raise ValueError(f"{prefix} OHLCV contains {int(bad.sum())} invalid ranges")


def _validate_gold(frame: pd.DataFrame) -> None:
    required = {"open", "close", "m30_atr14", "decision_time"}
    if missing := sorted(required - set(frame)):
        raise ValueError(f"Gold frame missing P4 columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Gold frame requires a timezone-aware DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Gold frame index must be sorted and unique")
