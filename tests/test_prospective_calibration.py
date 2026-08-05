from __future__ import annotations

from copy import deepcopy
import json

import pandas as pd
import pytest

from commodity_ml.intraday.prospective_calibration import (
    ProspectiveCalibrationError,
    ProspectiveCalibrationStore,
    build_weekly_forecasts,
    resolve_matured_forecasts,
)
from commodity_ml.intraday.risk_calibration import HORIZONS_MINUTES, sha256_file


def _profile() -> pd.DataFrame:
    rows = []
    for instrument in ("XAUUSD", "XAGUSD"):
        for horizon in HORIZONS_MINUTES:
            for slot in range(7 * 60, 19 * 60 + 30, 30):
                if slot + horizon > 21 * 60 + 30:
                    continue
                row = {
                    "instrument": instrument,
                    "horizon_minutes": horizon,
                    "london_slot_minute": slot,
                    "sample_count": 252,
                    "sample_start": pd.Timestamp("2025-08-01T00:00:00Z"),
                    "sample_end": pd.Timestamp("2026-08-06T23:00:00Z"),
                }
                for metric, scale in (
                    ("range_bps", 1.0),
                    ("long_adverse_bps", 0.6),
                    ("short_adverse_bps", 0.7),
                ):
                    for label, value in (("p50", 10), ("p80", 15), ("p95", 20), ("p99", 30)):
                        row[f"{metric}_{label}"] = value * scale
                rows.append(row)
    return pd.DataFrame(rows)


def _release(tmp_path):
    profile_path = tmp_path / "profile.parquet"
    _profile().to_parquet(profile_path, index=False)
    release = {
        "schema_version": 1,
        "policy_version": "current_range_v1",
        "release_id": "rejected-research-release",
        "release_status": "REJECTED",
        "created_at": "2026-08-07T09:30:00Z",
        "admission_checks": {
            "source_gate_passed": True,
            "coverage_gate_passed": False,
        },
        "declarations": {
            "directional_signal_used": False,
            "strategy_pnl_used": False,
            "outcome_selected_parameters": False,
            "broker_spread_eligible": False,
        },
        "artifacts": {"profile_sha256": sha256_file(profile_path)},
    }
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    return profile_path, release_path


def test_rejected_release_can_only_build_future_research_schedule(tmp_path):
    profile_path, release_path = _release(tmp_path)
    registered = pd.Timestamp("2026-08-09T00:00:00Z")

    rows = build_weekly_forecasts(
        profile_path,
        release_path,
        start_date="2026-08-10",
        registered_at=registered,
    )

    assert rows
    assert {row["instrument"] for row in rows} == {"XAUUSD", "XAGUSD"}
    assert {row["horizon_minutes"] for row in rows} == set(HORIZONS_MINUTES)
    assert all(row["release_status"] == "REJECTED" for row in rows)
    assert all(row["trade_eligible"] is False for row in rows)
    assert all(row["directional_signal_used"] is False for row in rows)
    assert all("range_bps" in row["forecasts_bps"] for row in rows)
    assert all(pd.Timestamp(row["entry_time"]) > registered for row in rows)

    partial = build_weekly_forecasts(
        profile_path,
        release_path,
        start_date="2026-08-13",
        registered_at=pd.Timestamp("2026-08-12T00:00:00Z"),
        weekdays=2,
    )
    assert {row["london_date"] for row in partial} == {"2026-08-13", "2026-08-14"}

    with pytest.raises(ValueError, match="before its first entry"):
        build_weekly_forecasts(
            profile_path,
            release_path,
            start_date="2026-08-10",
            registered_at=pd.Timestamp("2026-08-10T07:00:00Z"),
        )


def test_release_and_profile_must_exist_before_registration(tmp_path):
    profile_path, release_path = _release(tmp_path)
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["created_at"] = "2026-08-10T00:00:00Z"
    release_path.write_text(json.dumps(release), encoding="utf-8")
    with pytest.raises(ValueError, match="postdate"):
        build_weekly_forecasts(
            profile_path,
            release_path,
            start_date="2026-08-10",
            registered_at=pd.Timestamp("2026-08-09T00:00:00Z"),
        )


def _forecast() -> dict:
    forecasts = {
        metric: {"p50": 50.0, "p80": 60.0, "p95": 70.0, "p99": 80.0}
        for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps")
    }
    return {
        "schema_version": 1,
        "forecast_id": "forecast-1",
        "registered_at": "2026-08-09T00:00:00Z",
        "entry_time": "2026-08-10T06:00:00Z",
        "maturity_time": "2026-08-10T07:00:00Z",
        "london_date": "2026-08-10",
        "instrument": "XAUUSD",
        "horizon_minutes": 60,
        "release_id": "candidate-1",
        "release_status": "REJECTED",
        "release_sha256": "1" * 64,
        "profile_sha256": "2" * 64,
        "release_failed_checks": ["coverage_gate_passed"],
        "forecasts_bps": forecasts,
        "directional_signal_used": False,
        "strategy_pnl_used": False,
        "trade_eligible": False,
    }


def _source() -> pd.DataFrame:
    index = pd.date_range("2026-08-10T06:00:00Z", periods=2, freq="30min")
    data = {}
    for asset, level, spread in (("gold", 100.0, 0.2), ("silver", 25.0, 0.04)):
        for side, offset in (("bid", 0.0), ("ask", spread)):
            data[f"{asset}_{side}_open"] = [level + offset, level + 1 + offset]
            data[f"{asset}_{side}_high"] = [level + 1 + offset, level + 2 + offset]
            data[f"{asset}_{side}_low"] = [level - 1 + offset, level - 0.5 + offset]
            data[f"{asset}_{side}_close"] = [level + 0.5 + offset, level + 1.5 + offset]
    return pd.DataFrame(data, index=index)


def test_ledger_is_immutable_and_resolves_only_matured_exact_paths(tmp_path):
    store = ProspectiveCalibrationStore(tmp_path / "prospective.sqlite")
    forecast = _forecast()
    inserted = pd.Timestamp("2026-08-09T00:00:00Z")
    assert store.record_forecasts([forecast], inserted_at=inserted) == {
        "created": 1,
        "already_present": 0,
    }
    assert store.record_forecasts([forecast], inserted_at=inserted)["already_present"] == 1
    changed = deepcopy(forecast)
    changed["forecasts_bps"]["range_bps"]["p95"] = 999
    with pytest.raises(ProspectiveCalibrationError, match="forecast_id_collision"):
        store.record_forecasts([changed], inserted_at=inserted)

    assert store.pending_matured(pd.Timestamp("2026-08-10T06:59:00Z")) == []
    result = resolve_matured_forecasts(
        store,
        _source(),
        source_sha256="3" * 64,
        as_of=pd.Timestamp("2026-08-10T07:00:00Z"),
    )
    assert result == {"created": 1, "already_present": 0, "unavailable": []}

    summary = store.summary()
    assert summary["state"] == "COLLECTING"
    assert summary["resolved_forecasts"] == 1
    assert summary["coverage"]["XAUUSD_60m"]["observations"] == 1
    assert "profitability" not in " ".join(summary["limitations"]).lower()


def test_missing_path_remains_pending_and_visible(tmp_path):
    store = ProspectiveCalibrationStore(tmp_path / "prospective.sqlite")
    store.record_forecasts(
        [_forecast()], inserted_at=pd.Timestamp("2026-08-09T00:00:00Z")
    )
    result = resolve_matured_forecasts(
        store,
        _source().iloc[:1],
        source_sha256="3" * 64,
        as_of=pd.Timestamp("2026-08-10T07:00:00Z"),
    )
    assert result["created"] == 0
    assert result["unavailable"][0]["reason"] == "prospective_path_missing_or_non_contiguous"
    assert store.summary()["pending_forecasts"] == 1
