"""Immutable prospective assessment/outcome ledger for paper risk validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
OUTCOME_ELIGIBLE_DECISIONS = ("ALLOW", "REDUCE")


class ObservationLedgerError(RuntimeError):
    """The prospective ledger would become ambiguous or non-immutable."""


@dataclass(frozen=True)
class PaperTradeOutcome:
    intent_id: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    filled_size: float
    expected_entry_price: float
    actual_entry_price: float
    actual_exit_price: float
    max_adverse_excursion_points: float
    max_favorable_excursion_points: float
    entry_spread_points: float
    exit_spread_points: float
    entry_adverse_slippage_points: float
    exit_adverse_slippage_points: float
    fees: float
    peak_margin: float
    realized_pnl: float
    source: str
    source_reference: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.intent_id or not self.source or not self.source_reference:
            raise ValueError(
                "outcome intent_id, source and source_reference must be non-empty"
            )
        entry = _utc(self.entry_time, "entry_time")
        exit_time = _utc(self.exit_time, "exit_time")
        if exit_time <= entry:
            raise ValueError("exit_time must be after entry_time")
        for name in (
            "filled_size",
            "expected_entry_price",
            "actual_entry_price",
            "actual_exit_price",
        ):
            if not _positive(getattr(self, name)):
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "max_adverse_excursion_points",
            "max_favorable_excursion_points",
            "entry_spread_points",
            "exit_spread_points",
            "entry_adverse_slippage_points",
            "exit_adverse_slippage_points",
            "fees",
            "peak_margin",
        ):
            if not _non_negative(getattr(self, name)):
                raise ValueError(f"{name} must be finite and non-negative")
        if not _finite(self.realized_pnl):
            raise ValueError("realized_pnl must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PaperTradeOutcome":
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - fields)
        if unknown:
            raise ValueError(f"unexpected outcome fields: {', '.join(unknown)}")
        missing = sorted(name for name in fields if name != "notes" and value.get(name) is None)
        if missing:
            raise ValueError(f"missing outcome fields: {', '.join(missing)}")
        payload = dict(value)
        payload["entry_time"] = _utc(payload["entry_time"], "entry_time")
        payload["exit_time"] = _utc(payload["exit_time"], "exit_time")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["entry_time"] = self.entry_time.isoformat()
        value["exit_time"] = self.exit_time.isoformat()
        return value


class PretradeObservationStore:
    """SQLite store that never updates or deletes assessments/outcomes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS assessments (
                    intent_id TEXT PRIMARY KEY,
                    assessed_at TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    instrument TEXT,
                    side TEXT,
                    source_mode TEXT NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    inserted_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    intent_id TEXT PRIMARY KEY REFERENCES assessments(intent_id),
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    outcome_sha256 TEXT NOT NULL,
                    outcome_json TEXT NOT NULL,
                    inserted_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assessments_decision_idx
                    ON assessments(decision);
                """
            )
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def record_assessment(
        self,
        document: Mapping[str, Any],
        *,
        inserted_at: pd.Timestamp | None = None,
    ) -> bool:
        """Insert a pre-outcome report; exact retries are idempotent."""

        report, source_mode = _extract_report(document)
        intent_id = str(report.get("intent_id") or "")
        if not intent_id:
            raise ValueError("assessment report requires intent_id")
        decision = str(report.get("decision") or "").upper()
        if decision not in ("ALLOW", "REDUCE", "BLOCK", "INCOMPLETE"):
            raise ValueError(f"unsupported assessment decision: {decision!r}")
        trade = _mapping(report.get("trade"))
        assessed_at = trade.get("decision_time")
        if assessed_at is None:
            raise ValueError("assessment report requires trade.decision_time")
        assessed_timestamp = _utc(assessed_at, "trade.decision_time")
        canonical = _canonical(document)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        inserted = _utc(
            inserted_at if inserted_at is not None else pd.Timestamp.now(tz="UTC"),
            "inserted_at",
        )
        if inserted < assessed_timestamp:
            raise ValueError("inserted_at cannot precede the assessment decision time")

        self.initialize()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT report_sha256 FROM assessments WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] == digest:
                    return False
                raise ObservationLedgerError("intent_id_assessment_collision")
            connection.execute(
                """INSERT INTO assessments
                   (intent_id, assessed_at, decision, instrument, side,
                    source_mode, report_sha256, report_json, inserted_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    intent_id,
                    assessed_timestamp.isoformat(),
                    decision,
                    trade.get("instrument"),
                    trade.get("side"),
                    source_mode,
                    digest,
                    canonical,
                    inserted.isoformat(),
                ),
            )
        return True

    def assessment_result(
        self, intent_id: str
    ) -> tuple[str, tuple[str, ...]] | None:
        """Return an existing immutable decision without re-assessing it."""

        if not intent_id:
            raise ValueError("intent_id must be non-empty")
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT decision, report_json FROM assessments WHERE intent_id = ?",
                (intent_id,),
            ).fetchone()
        if row is None:
            return None
        document = json.loads(row[1])
        report, _ = _extract_report(document)
        reasons = report.get("reasons") or []
        if not isinstance(reasons, (list, tuple)):
            raise ObservationLedgerError("stored_assessment_reasons_are_invalid")
        return str(row[0]), tuple(str(value) for value in reasons)

    def record_outcome(
        self,
        outcome: PaperTradeOutcome,
        *,
        inserted_at: pd.Timestamp | None = None,
    ) -> bool:
        """Attach one immutable realised observation to a prior eligible report."""

        canonical = _canonical(outcome.to_dict())
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        inserted = _utc(
            inserted_at if inserted_at is not None else pd.Timestamp.now(tz="UTC"),
            "inserted_at",
        )
        if inserted < outcome.exit_time:
            raise ValueError("inserted_at cannot precede outcome exit_time")
        self.initialize()
        with self._connect() as connection:
            assessment = connection.execute(
                "SELECT decision, assessed_at FROM assessments WHERE intent_id = ?",
                (outcome.intent_id,),
            ).fetchone()
            if assessment is None:
                raise ObservationLedgerError("outcome_requires_prior_assessment")
            if assessment[0] not in OUTCOME_ELIGIBLE_DECISIONS:
                raise ObservationLedgerError("outcome_not_allowed_for_blocked_assessment")
            if outcome.entry_time < pd.Timestamp(assessment[1]):
                raise ValueError("outcome entry_time precedes its assessment")
            existing = connection.execute(
                "SELECT outcome_sha256 FROM outcomes WHERE intent_id = ?",
                (outcome.intent_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] == digest:
                    return False
                raise ObservationLedgerError("intent_id_outcome_collision")
            connection.execute(
                """INSERT INTO outcomes
                   (intent_id, entry_time, exit_time, outcome_sha256,
                    outcome_json, inserted_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    outcome.intent_id,
                    outcome.entry_time.isoformat(),
                    outcome.exit_time.isoformat(),
                    digest,
                    canonical,
                    inserted.isoformat(),
                ),
            )
        return True

    def summary(self) -> dict[str, Any]:
        """Return descriptive validation metrics without an expectancy claim."""

        self.initialize()
        with self._connect() as connection:
            assessment_rows = connection.execute(
                "SELECT decision, COUNT(*) FROM assessments GROUP BY decision"
            ).fetchall()
            joined = connection.execute(
                """SELECT a.report_json, o.outcome_json
                   FROM assessments a JOIN outcomes o USING(intent_id)
                   ORDER BY o.exit_time"""
            ).fetchall()
        outcomes: list[Mapping[str, Any]] = []
        planned: list[float] = []
        stop_exceeded: list[bool] = []
        for report_raw, outcome_raw in joined:
            report_document = json.loads(report_raw)
            report, _ = _extract_report(report_document)
            outcome = _mapping(json.loads(outcome_raw))
            outcomes.append(outcome)
            planned_loss = _planned_loss_for_filled_size(report, outcome.get("filled_size"))
            if planned_loss is not None:
                planned.append(planned_loss)
                stop_exceeded.append(float(outcome["realized_pnl"]) < -planned_loss)
        return {
            "schema_version": SCHEMA_VERSION,
            "assessment_counts": {str(key): int(value) for key, value in assessment_rows},
            "completed_outcomes": len(outcomes),
            "planned_stop_comparisons": len(planned),
            "realized_stop_loss_exceedance_rate": (
                float(np.mean(stop_exceeded)) if stop_exceeded else None
            ),
            "realized_pnl": _distribution(outcomes, "realized_pnl"),
            "max_adverse_excursion_points": _distribution(
                outcomes, "max_adverse_excursion_points"
            ),
            "entry_spread_points": _distribution(outcomes, "entry_spread_points"),
            "exit_spread_points": _distribution(outcomes, "exit_spread_points"),
            "entry_adverse_slippage_points": _distribution(
                outcomes, "entry_adverse_slippage_points"
            ),
            "exit_adverse_slippage_points": _distribution(
                outcomes, "exit_adverse_slippage_points"
            ),
            "limitations": [
                "Descriptive prospective observations only; no causal or profitability claim.",
                "A planned-stop exceedance may reflect gaps, slippage, fees, sizing differences or bad data and requires review.",
            ],
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _extract_report(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    nested = document.get("risk_report")
    if nested is not None:
        report = _mapping(nested)
        mode = str(document.get("mode") or "wrapped")
    else:
        report = _mapping(document)
        mode = "offline_json"
    if not report or "decision" not in report:
        raise ValueError("document does not contain a pre-trade risk report")
    return report, mode


def _planned_loss_for_filled_size(
    report: Mapping[str, Any], filled_size: Any
) -> float | None:
    sizing = _mapping(report.get("sizing"))
    cash = _mapping(report.get("cash_risk"))
    proposed = _mapping(cash.get("proposed"))
    assessed_size = _number(sizing.get("assessed_size"))
    proposed_loss = _number(proposed.get("planned_stop_loss"))
    filled = _number(filled_size)
    if assessed_size and assessed_size > 0 and proposed_loss is not None and filled is not None:
        return proposed_loss * filled / assessed_size
    recommended = _mapping(cash.get("at_recommended_max"))
    recommended_size = _number(sizing.get("recommended_max_size"))
    recommended_loss = _number(recommended.get("planned_stop_loss"))
    if recommended_size and recommended_size > 0 and recommended_loss is not None and filled is not None:
        return recommended_loss * filled / recommended_size
    return None


def _distribution(rows: list[Mapping[str, Any]], field: str) -> dict[str, Any] | None:
    values = [_number(row.get(field)) for row in rows]
    clean = np.asarray([value for value in values if value is not None], dtype=float)
    if len(clean) == 0:
        return None
    return {
        "count": int(len(clean)),
        "mean": float(np.mean(clean)),
        "median": float(np.median(clean)),
        "p95": float(np.quantile(clean, 0.95)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
    }


def _canonical(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite(value: Any) -> bool:
    return _number(value) is not None


def _positive(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0


def _non_negative(value: Any) -> bool:
    number = _number(value)
    return number is not None and number >= 0


def _utc(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
