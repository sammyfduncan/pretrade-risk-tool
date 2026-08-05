from __future__ import annotations

import json

import pandas as pd
import pytest

from commodity_ml.intraday.calibration_release import (
    REQUIRED_CHECKS,
    load_admitted_calibrated_risk,
)
from commodity_ml.intraday.risk_calibration import sha256_file


def _profile() -> pd.DataFrame:
    row: dict[str, object] = {
        "instrument": "XAUUSD",
        "horizon_minutes": 120,
        "london_slot_minute": 10 * 60 + 30,
        "sample_count": 252,
        "sample_start": pd.Timestamp("2025-08-01T09:00:00Z"),
        "sample_end": pd.Timestamp("2026-08-01T09:00:00Z"),
    }
    for metric, scale in (
        ("range_bps", 1.0),
        ("long_adverse_bps", 0.6),
        ("short_adverse_bps", 0.7),
        ("spread_points", 0.01),
        ("spread_bps", 0.02),
    ):
        for label, value in (("p50", 10), ("p80", 15), ("p95", 20), ("p99", 30)):
            row[f"{metric}_{label}"] = value * scale
    return pd.DataFrame([row])


def _write_release(tmp_path, *, status: str = "ADMITTED", failed_check: str | None = None):
    profile_path = tmp_path / "profile.parquet"
    _profile().to_parquet(profile_path, index=False)
    checks = {name: True for name in REQUIRED_CHECKS}
    if failed_check:
        checks[failed_check] = False
    payload = {
        "schema_version": 1,
        "policy_version": "current_range_v1",
        "release_id": "current-range-20260802",
        "release_status": status,
        "created_at": "2026-08-02T08:00:00Z",
        "profile_max_age_days": 10,
        "artifacts": {"profile_sha256": sha256_file(profile_path)},
        "admission_checks": checks,
        "declarations": {
            "directional_signal_used": False,
            "strategy_pnl_used": False,
            "outcome_selected_parameters": False,
            "broker_spread_eligible": False,
        },
    }
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(payload), encoding="utf-8")
    return profile_path, release_path


def test_admitted_release_marks_lookup_current(tmp_path):
    profile_path, release_path = _write_release(tmp_path)

    lookup = load_admitted_calibrated_risk(
        profile_path,
        release_path,
        instrument="XAUUSD",
        decision_time=pd.Timestamp("2026-08-03T09:30:00Z"),
        horizon_minutes=120,
        entry_price=2_000,
        side="long",
    )

    assert lookup.admitted_for_current_use
    assert lookup.policy_version == "current_range_v1"
    assert lookup.release_id == "current-range-20260802"
    assert lookup.range_points["p95"] == pytest.approx(4.0)


def test_release_rejects_failed_gate_and_hash_mismatch(tmp_path):
    profile_path, release_path = _write_release(
        tmp_path, failed_check="coverage_gate_passed"
    )
    with pytest.raises(ValueError, match="coverage_gate_passed"):
        load_admitted_calibrated_risk(
            profile_path,
            release_path,
            instrument="XAUUSD",
            decision_time=pd.Timestamp("2026-08-03T09:30:00Z"),
            horizon_minutes=120,
            entry_price=2_000,
            side="long",
        )

    payload = json.loads(release_path.read_text(encoding="utf-8"))
    payload["admission_checks"]["coverage_gate_passed"] = True
    payload["artifacts"]["profile_sha256"] = "0" * 64
    release_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_admitted_calibrated_risk(
            profile_path,
            release_path,
            instrument="XAUUSD",
            decision_time=pd.Timestamp("2026-08-03T09:30:00Z"),
            horizon_minutes=120,
            entry_price=2_000,
            side="long",
        )


def test_admitted_profile_expires_after_ten_days(tmp_path):
    profile_path, release_path = _write_release(tmp_path)
    with pytest.raises(ValueError, match="stale"):
        load_admitted_calibrated_risk(
            profile_path,
            release_path,
            instrument="XAUUSD",
            decision_time=pd.Timestamp("2026-08-12T09:30:00Z"),
            horizon_minutes=120,
            entry_price=2_000,
            side="short",
        )
