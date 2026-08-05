"""Fail-closed admission for current pre-trade range-risk calibration.

The release artifact is deliberately separate from the profile. Recency alone
does not make a set of quantiles eligible: the frozen source, parity, coverage,
provenance and non-directional-use gates must all have passed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd

from commodity_ml.intraday.risk_calibration import (
    CalibratedRiskLookup,
    lookup_calibrated_risk,
    sha256_file,
)


POLICY_VERSION = "current_range_v1"
RELEASE_SCHEMA_VERSION = 1
MAX_PROFILE_AGE_DAYS = 10
REQUIRED_CHECKS = (
    "source_gate_passed",
    "event_calendar_gate_passed",
    "parity_gate_passed",
    "data_quality_gate_passed",
    "profile_completeness_gate_passed",
    "coverage_gate_passed",
    "quantile_monotonicity_gate_passed",
    "prohibited_inputs_gate_passed",
)


@dataclass(frozen=True)
class CalibrationRelease:
    release_id: str
    created_at: pd.Timestamp
    profile_sha256: str
    profile_max_age_days: int
    report_path: Path


def load_admitted_calibrated_risk(
    profile_path: Path,
    release_path: Path,
    *,
    instrument: str,
    decision_time: pd.Timestamp,
    horizon_minutes: int,
    entry_price: float,
    side: str,
) -> CalibratedRiskLookup:
    """Validate a release and return one currently admitted lookup row."""

    profile_path = Path(profile_path).expanduser().resolve()
    release = validate_calibration_release(
        profile_path,
        release_path,
        decision_time=decision_time,
    )
    lookup = lookup_calibrated_risk(
        pd.read_parquet(profile_path),
        instrument=instrument,
        decision_time=decision_time,
        horizon_minutes=horizon_minutes,
        entry_price=entry_price,
        side=side,
        max_age_days=release.profile_max_age_days,
    )
    return replace(
        lookup,
        admitted_for_current_use=True,
        policy_version=POLICY_VERSION,
        release_id=release.release_id,
    )


def validate_calibration_release(
    profile_path: Path,
    release_path: Path,
    *,
    decision_time: pd.Timestamp,
) -> CalibrationRelease:
    """Validate release metadata and its exact profile without trusting age alone."""

    profile_path = Path(profile_path).expanduser().resolve()
    release_path = Path(release_path).expanduser().resolve()
    payload = _read_release(release_path)
    now = _utc(decision_time, "decision_time")

    if payload.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError("Calibration release schema is unsupported")
    if payload.get("policy_version") != POLICY_VERSION:
        raise ValueError("Calibration release policy version is unsupported")
    if payload.get("release_status") != "ADMITTED":
        raise ValueError("Calibration release is not admitted")

    release_id = payload.get("release_id")
    if not isinstance(release_id, str) or not release_id.strip():
        raise ValueError("Calibration release_id is missing")
    created_at = _utc(payload.get("created_at"), "release.created_at")
    if created_at >= now:
        raise ValueError("Calibration release must predate the assessment")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Calibration release artifacts are missing")
    expected_hash = artifacts.get("profile_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Calibration release profile hash is invalid")
    if not profile_path.is_file() or sha256_file(profile_path) != expected_hash:
        raise ValueError("Calibration profile does not match its admitted release")

    checks = payload.get("admission_checks")
    if not isinstance(checks, dict):
        raise ValueError("Calibration admission checks are missing")
    failed = [name for name in REQUIRED_CHECKS if checks.get(name) is not True]
    if failed:
        raise ValueError(f"Calibration admission checks failed: {', '.join(failed)}")

    declarations = payload.get("declarations")
    expected_declarations = {
        "directional_signal_used": False,
        "strategy_pnl_used": False,
        "outcome_selected_parameters": False,
        "broker_spread_eligible": False,
    }
    if not isinstance(declarations, dict) or any(
        declarations.get(name) is not expected
        for name, expected in expected_declarations.items()
    ):
        raise ValueError("Calibration release violates prohibited-input declarations")

    max_age = payload.get("profile_max_age_days")
    if max_age != MAX_PROFILE_AGE_DAYS:
        raise ValueError("Calibration release has an unexpected freshness policy")
    return CalibrationRelease(
        release_id=release_id,
        created_at=created_at,
        profile_sha256=expected_hash,
        profile_max_age_days=max_age,
        report_path=release_path,
    )


def _read_release(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read calibration release: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Calibration release must be a JSON object")
    return payload


def _utc(value: Any, name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a timestamp") from exc
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
