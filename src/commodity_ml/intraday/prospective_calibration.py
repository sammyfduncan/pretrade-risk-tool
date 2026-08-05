"""Immutable weekly forecasts and delayed outcomes for calibration validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from commodity_ml.intraday.calibration_refresh import (
    profile_is_complete,
    quantiles_are_valid,
)
from commodity_ml.intraday.calibration_release import (
    MAX_PROFILE_AGE_DAYS,
    POLICY_VERSION,
)
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.risk_calibration import (
    QUANTILES,
    measure_calibration_path,
    sha256_file,
)


SCHEMA_VERSION = 1
LONDON = ZoneInfo("Europe/London")
PRIMARY_DATES = 40
MIN_RESOLUTION_FRACTION = 0.80
FORECAST_METRICS = ("range_bps", "long_adverse_bps", "short_adverse_bps")


class ProspectiveCalibrationError(RuntimeError):
    """An immutable prospective boundary would be violated."""


class ProspectiveCalibrationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    registered_at TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    maturity_time TEXT NOT NULL,
                    london_date TEXT NOT NULL,
                    instrument TEXT NOT NULL,
                    horizon_minutes INTEGER NOT NULL,
                    release_id TEXT NOT NULL,
                    release_status TEXT NOT NULL,
                    forecast_sha256 TEXT NOT NULL,
                    forecast_json TEXT NOT NULL,
                    inserted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    forecast_id TEXT PRIMARY KEY REFERENCES forecasts(forecast_id),
                    resolved_at TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL,
                    outcome_sha256 TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    inserted_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS forecasts_maturity_idx
                    ON forecasts(maturity_time);
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def record_forecasts(
        self,
        forecasts: list[dict[str, Any]],
        *,
        inserted_at: pd.Timestamp,
    ) -> dict[str, int]:
        if not forecasts:
            raise ValueError("forecast schedule is empty")
        inserted = _utc(inserted_at, "inserted_at")
        self.initialize()
        created = 0
        repeated = 0
        with self._connect() as connection:
            for forecast in forecasts:
                registered = _utc(forecast["registered_at"], "registered_at")
                entry = _utc(forecast["entry_time"], "entry_time")
                if inserted < registered or registered >= entry:
                    raise ValueError("forecasts must be inserted after registration and before entry")
                canonical = _canonical(forecast)
                digest = _digest(canonical)
                existing = connection.execute(
                    "SELECT forecast_sha256 FROM forecasts WHERE forecast_id = ?",
                    (forecast["forecast_id"],),
                ).fetchone()
                if existing:
                    if existing[0] != digest:
                        raise ProspectiveCalibrationError("forecast_id_collision")
                    repeated += 1
                    continue
                connection.execute(
                    """INSERT INTO forecasts
                       (forecast_id, registered_at, entry_time, maturity_time,
                        london_date, instrument, horizon_minutes, release_id,
                        release_status, forecast_sha256, forecast_json, inserted_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        forecast["forecast_id"],
                        registered.isoformat(),
                        entry.isoformat(),
                        _utc(forecast["maturity_time"], "maturity_time").isoformat(),
                        forecast["london_date"],
                        forecast["instrument"],
                        int(forecast["horizon_minutes"]),
                        forecast["release_id"],
                        forecast["release_status"],
                        digest,
                        canonical,
                        inserted.isoformat(),
                    ),
                )
                created += 1
        return {"created": created, "already_present": repeated}

    def pending_matured(self, as_of: pd.Timestamp) -> list[dict[str, Any]]:
        now = _utc(as_of, "as_of")
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT f.forecast_json FROM forecasts f
                   LEFT JOIN outcomes o USING(forecast_id)
                   WHERE o.forecast_id IS NULL AND f.maturity_time <= ?
                   ORDER BY f.entry_time, f.instrument, f.horizon_minutes""",
                (now.isoformat(),),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def record_outcome(
        self,
        forecast_id: str,
        outcome: Mapping[str, Any],
        *,
        source_sha256: str,
        resolved_at: pd.Timestamp,
    ) -> bool:
        resolved = _utc(resolved_at, "resolved_at")
        canonical = _canonical(outcome)
        digest = _digest(canonical)
        self.initialize()
        with self._connect() as connection:
            forecast = connection.execute(
                "SELECT maturity_time FROM forecasts WHERE forecast_id = ?",
                (forecast_id,),
            ).fetchone()
            if forecast is None:
                raise ProspectiveCalibrationError("outcome_requires_prior_forecast")
            if resolved < pd.Timestamp(forecast[0]):
                raise ValueError("outcome cannot be resolved before maturity")
            existing = connection.execute(
                "SELECT outcome_sha256, source_sha256 FROM outcomes WHERE forecast_id = ?",
                (forecast_id,),
            ).fetchone()
            if existing:
                if existing == (digest, source_sha256):
                    return False
                raise ProspectiveCalibrationError("forecast_outcome_collision")
            connection.execute(
                """INSERT INTO outcomes
                   (forecast_id, resolved_at, source_sha256, outcome_sha256,
                    outcome_json, inserted_at) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    forecast_id,
                    resolved.isoformat(),
                    source_sha256,
                    digest,
                    canonical,
                    resolved.isoformat(),
                ),
            )
        return True

    def summary(self) -> dict[str, Any]:
        self.initialize()
        with self._connect() as connection:
            total = int(connection.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0])
            rows = connection.execute(
                """SELECT f.forecast_json, o.outcome_json
                   FROM forecasts f JOIN outcomes o USING(forecast_id)
                   ORDER BY f.entry_time"""
            ).fetchall()
            scheduled_dates = int(
                connection.execute("SELECT COUNT(DISTINCT london_date) FROM forecasts").fetchone()[0]
            )
            release_rows = connection.execute(
                "SELECT release_status, COUNT(*) FROM forecasts GROUP BY release_status"
            ).fetchall()
        paired = [(json.loads(f), json.loads(o)) for f, o in rows]
        resolved = len(paired)
        resolved_dates = len({forecast["london_date"] for forecast, _ in paired})
        fraction = resolved / total if total else 0.0
        return {
            "schema_version": SCHEMA_VERSION,
            "state": (
                "PRIMARY_CHECKPOINT_REACHED"
                if resolved_dates >= PRIMARY_DATES and fraction >= MIN_RESOLUTION_FRACTION
                else "COLLECTING"
            ),
            "scheduled_forecasts": total,
            "resolved_forecasts": resolved,
            "pending_forecasts": total - resolved,
            "resolution_fraction": fraction,
            "scheduled_london_dates": scheduled_dates,
            "resolved_london_dates": resolved_dates,
            "primary_checkpoint_dates": PRIMARY_DATES,
            "minimum_resolution_fraction": MIN_RESOLUTION_FRACTION,
            "release_status_counts": {str(k): int(v) for k, v in release_rows},
            "coverage": _coverage_summary(paired),
            "limitations": [
                "Research-only calibration coverage; no strategy, P&L or direction claim.",
                "Overlapping intraday rows are not independent; London dates are the conservative blocks.",
                "Checkpoint completion does not automatically admit a calibration release.",
            ],
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def build_weekly_forecasts(
    profile_path: Path,
    release_path: Path,
    *,
    start_date: str,
    registered_at: pd.Timestamp,
    weekdays: int = 5,
) -> list[dict[str, Any]]:
    registered = _utc(registered_at, "registered_at")
    profile_path = Path(profile_path).expanduser().resolve()
    release_path = Path(release_path).expanduser().resolve()
    release = json.loads(release_path.read_text(encoding="utf-8"))
    profile = pd.read_parquet(profile_path)
    _validate_candidate_release(release, profile_path, profile)
    if _utc(release.get("created_at"), "release.created_at") > registered:
        raise ValueError("candidate release cannot postdate forecast registration")
    profile_ends = pd.to_datetime(profile["sample_end"], utc=True)
    if (profile_ends >= registered).any():
        raise ValueError("candidate profile must be fully known before registration")

    start = pd.Timestamp(start_date)
    if start.tzinfo is not None or start.dayofweek >= 5:
        raise ValueError("start_date must be a timezone-naive weekday date")
    if isinstance(weekdays, bool) or not isinstance(weekdays, int) or not 1 <= weekdays <= 5:
        raise ValueError("weekdays must be an integer from 1 through 5")
    dates: list[pd.Timestamp] = []
    cursor = start
    while len(dates) < weekdays:
        if cursor.dayofweek < 5:
            dates.append(cursor)
        cursor += pd.Timedelta(days=1)
    earliest = pd.Timestamp(dates[0].date(), tz=LONDON) + pd.Timedelta(hours=7)
    if registered >= earliest.tz_convert("UTC"):
        raise ValueError("weekly schedule must be registered before its first entry")
    end_exclusive = pd.Timestamp((dates[-1] + pd.Timedelta(days=1)).date(), tz=LONDON)
    blackouts = scheduled_entry_blackouts(earliest.tz_convert("UTC"), end_exclusive.tz_convert("UTC"))

    rows: list[dict[str, Any]] = []
    release_hash = sha256_file(release_path)
    profile_hash = sha256_file(profile_path)
    for date in dates:
        local_midnight = pd.Timestamp(date.date(), tz=LONDON)
        for _, profile_row in profile.iterrows():
            entry = local_midnight + pd.Timedelta(minutes=int(profile_row["london_slot_minute"]))
            entry_utc = entry.tz_convert("UTC")
            if any(item.start <= entry_utc < item.end for item in blackouts):
                continue
            if entry_utc - _utc(profile_row["sample_end"], "profile.sample_end") > pd.Timedelta(days=MAX_PROFILE_AGE_DAYS):
                raise ValueError("profile would be stale before the weekly schedule ends")
            horizon = int(profile_row["horizon_minutes"])
            forecasts = {
                metric: {
                    f"p{int(q * 100):02d}": float(profile_row[f"{metric}_p{int(q * 100):02d}"])
                    for q in QUANTILES
                }
                for metric in FORECAST_METRICS
            }
            identity = f"{release['release_id']}|{entry_utc.isoformat()}|{profile_row['instrument']}|{horizon}"
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "forecast_id": hashlib.sha256(identity.encode()).hexdigest(),
                    "registered_at": registered.isoformat(),
                    "entry_time": entry_utc.isoformat(),
                    "maturity_time": (entry_utc + pd.Timedelta(minutes=horizon)).isoformat(),
                    "london_date": str(date.date()),
                    "instrument": str(profile_row["instrument"]),
                    "horizon_minutes": horizon,
                    "release_id": str(release["release_id"]),
                    "release_status": str(release["release_status"]),
                    "release_sha256": release_hash,
                    "profile_sha256": profile_hash,
                    "release_failed_checks": sorted(
                        key for key, passed in release["admission_checks"].items() if not passed
                    ),
                    "forecasts_bps": forecasts,
                    "directional_signal_used": False,
                    "strategy_pnl_used": False,
                    "trade_eligible": False,
                }
            )
    return rows


def resolve_matured_forecasts(
    store: ProspectiveCalibrationStore,
    source: pd.DataFrame,
    *,
    source_sha256: str,
    as_of: pd.Timestamp,
) -> dict[str, Any]:
    now = _utc(as_of, "as_of")
    created = 0
    repeated = 0
    unavailable: list[dict[str, str]] = []
    for forecast in store.pending_matured(now):
        try:
            outcome = measure_calibration_path(
                source,
                instrument=forecast["instrument"],
                entry_time=pd.Timestamp(forecast["entry_time"]),
                horizon_minutes=int(forecast["horizon_minutes"]),
            )
        except ValueError as exc:
            unavailable.append({"forecast_id": forecast["forecast_id"], "reason": str(exc)})
            continue
        document = {
            "schema_version": SCHEMA_VERSION,
            "forecast_id": forecast["forecast_id"],
            "entry_time": forecast["entry_time"],
            "maturity_time": forecast["maturity_time"],
            **outcome,
            "source_sha256": source_sha256,
            "resolved_at": now.isoformat(),
        }
        if store.record_outcome(
            forecast["forecast_id"],
            document,
            source_sha256=source_sha256,
            resolved_at=now,
        ):
            created += 1
        else:
            repeated += 1
    return {"created": created, "already_present": repeated, "unavailable": unavailable}


def _validate_candidate_release(release: Mapping[str, Any], profile_path: Path, profile: pd.DataFrame) -> None:
    if release.get("policy_version") != POLICY_VERSION:
        raise ValueError("unsupported candidate calibration policy")
    if release.get("release_status") not in ("ADMITTED", "REJECTED"):
        raise ValueError("candidate release must be admitted or rejected after a complete build")
    if release.get("artifacts", {}).get("profile_sha256") != sha256_file(profile_path):
        raise ValueError("candidate profile hash does not match release")
    if not profile_is_complete(profile) or not quantiles_are_valid(profile):
        raise ValueError("candidate profile is structurally invalid")
    declarations = release.get("declarations", {})
    if any(
        declarations.get(key) is not False
        for key in (
            "directional_signal_used",
            "strategy_pnl_used",
            "outcome_selected_parameters",
            "broker_spread_eligible",
        )
    ):
        raise ValueError("candidate release used a prohibited input")


def _coverage_summary(paired: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for forecast, outcome in paired:
        grouped.setdefault((forecast["instrument"], int(forecast["horizon_minutes"])), []).append((forecast, outcome))
    result: dict[str, Any] = {}
    for (instrument, horizon), rows in sorted(grouped.items()):
        cell: dict[str, Any] = {"observations": len(rows), "metrics": {}}
        for metric in FORECAST_METRICS:
            cell["metrics"][metric] = {
                f"p{int(q * 100):02d}": {
                    "nominal": q,
                    "actual_coverage": float(
                        np.mean(
                            [
                                float(outcome[metric])
                                <= float(forecast["forecasts_bps"][metric][f"p{int(q * 100):02d}"])
                                for forecast, outcome in rows
                            ]
                        )
                    ),
                }
                for q in QUANTILES
            }
        result[f"{instrument}_{horizon}m"] = cell
    return result


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
