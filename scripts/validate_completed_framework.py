"""Build the final evidence-bounded validation record for the risk prototype."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from commodity_ml.intraday.closure_validation import run_closure_scenarios
from commodity_ml.intraday.prospective_calibration import ProspectiveCalibrationStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/final_framework_validation.json",
    )
    args = parser.parse_args()

    scenarios = run_closure_scenarios()
    historical = _historical_evidence()
    current_releases = _current_releases()
    prospective = _prospective_state()
    admitted = any(item["release_status"] == "ADMITTED" for item in current_releases)
    prospective_ready = prospective.get("state") == "PRIMARY_CHECKPOINT_READY"
    document = {
        "schema_version": 1,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "product_status": "COMPLETED_RESEARCH_PROTOTYPE",
        "scenario_validation": scenarios,
        "historical_evidence": historical,
        "current_calibration_releases": current_releases,
        "prospective_calibration": prospective,
        "completion": {
            "engineering_validation_passed": scenarios["passed"],
            "historical_calibration_evidence_present": bool(historical),
            "current_calibration_admitted": admitted,
            "prospective_primary_checkpoint_reached": prospective_ready,
            "research_prototype_complete": bool(scenarios["passed"] and historical),
            "production_validated": bool(
                scenarios["passed"] and historical and admitted and prospective_ready
            ),
        },
        "non_claims": [
            "No directional edge or profitability is claimed.",
            "No current calibration release is treated as admitted unless every frozen gate passed.",
            "Unresolved prospective forecasts are retained and are not scored early.",
            "The framework is not approved for unattended execution or real-money reliance.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(document["completion"], indent=2, sort_keys=True))
    print(f"Scenarios: {scenarios['passed_count']}/{scenarios['scenario_count']} passed")
    print(f"Report -> {args.output.resolve()}")
    if not scenarios["passed"]:
        raise SystemExit("Closure validation failed")


def _historical_evidence() -> dict[str, Any]:
    path = PROJECT_ROOT / "reports/pretrade_risk_calibration.json"
    if not path.is_file():
        return {}
    source = json.loads(path.read_text(encoding="utf-8"))
    cells = source.get("cells", {})
    return {
        "artifact": str(path.relative_to(PROJECT_ROOT)),
        "sha256": _sha256(path),
        "development_start": source.get("development_start"),
        "development_end_exclusive": source.get("development_end_exclusive"),
        "source_rows": source.get("source_rows"),
        "cell_count": len(cells),
        "calibrated_rows": sum(
            int(cell.get("calibrated_rows", 0)) for cell in cells.values()
        ),
        "data_quality_passed": source.get("data_quality_passed"),
        "directional_signal_used": source.get("directional_signal_used"),
        "strategy_pnl_used": source.get("strategy_pnl_used"),
        "paper_sizing_enabled": source.get("paper_sizing_enabled"),
        "broker_spread_eligible": source.get("broker_spread_eligible"),
    }


def _current_releases() -> list[dict[str, Any]]:
    root = PROJECT_ROOT / "data/processed/current_risk_calibration"
    releases: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/release.json")) if root.is_dir() else ():
        try:
            source = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        checks = source.get("admission_checks", {})
        releases.append(
            {
                "release_id": source.get("release_id"),
                "created_at": source.get("created_at"),
                "release_status": source.get("release_status"),
                "failed_checks": sorted(
                    name for name, passed in checks.items() if passed is not True
                ),
                "sha256": _sha256(path),
            }
        )
    return releases


def _prospective_state() -> dict[str, Any]:
    path = PROJECT_ROOT / "data/runtime/prospective_calibration.sqlite"
    if not path.is_file():
        return {"state": "NOT_STARTED"}
    return {
        **ProspectiveCalibrationStore(path).summary(),
        "ledger_sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
