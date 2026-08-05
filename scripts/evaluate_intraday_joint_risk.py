"""Run the frozen joint B0 plus GVZ plus calendar volatility experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.evaluation.placebo import draw_shifts
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.volatility import (
    ALL_FEATURES,
    TARGET_COLUMN,
    build_volatility_dataset,
    clean_gvz_observations,
    evaluate_joint_volatility,
    fit_linear_volatility_model,
)

DESIGN = "docs/archive/intraday_joint_risk_design.md"
HARNESS = (
    "scripts/evaluate_intraday_joint_risk.py",
    "src/commodity_ml/intraday/volatility.py",
)
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2025-07-01", tz="UTC")
N_PLACEBOS = 20
SEED = 20260731


def main() -> None:
    _require_committed_harness()
    cfg = load_config("config/gold.yaml")
    prerequisite_path = cfg.reports_dir / "intraday_risk_gate.json"
    prerequisite = _load_prerequisite(prerequisite_path)

    source = cfg.raw_path("dukascopy", granularity="M30")
    gvz_path = Path(prerequisite["gvz_snapshot"])
    event_path = Path(prerequisite["event_snapshot"])
    _require_hash(source, prerequisite["source_sha256"], "M30 source")
    _require_hash(gvz_path, prerequisite["gvz_sha256"], "GVZ snapshot")
    _require_hash(event_path, prerequisite["event_sha256"], "event snapshot")

    candles = pd.read_parquet(
        source,
        filters=[("time", ">=", START), ("time", "<", END)],
    )
    gvz_frame = pd.read_parquet(gvz_path)
    if list(gvz_frame.columns) != ["gvz"]:
        raise RuntimeError("Pinned GVZ snapshot schema mismatch")
    gvz = clean_gvz_observations(gvz_frame["gvz"])
    releases = pd.read_parquet(event_path)
    blackouts = scheduled_entry_blackouts(START, END)
    observed = build_volatility_dataset(
        candles,
        gvz,
        development_start=START,
        development_end=END,
        entry_blackouts=blackouts,
        releases=releases,
    )
    if observed.matched_coverage < 0.90:
        raise RuntimeError("Joint experiment matched coverage is below 90%")

    comparison = evaluate_joint_volatility(observed.frame)
    rng = np.random.default_rng(SEED)
    calendar_days = (END.normalize() - START.normalize()).days
    shifts = draw_shifts(calendar_days, N_PLACEBOS, "calendar", rng)
    placebo_deltas: list[float] = []
    for shift in shifts:
        shifted_releases = _shift_releases(releases, shift, START, END)
        placebo = build_volatility_dataset(
            candles,
            gvz,
            development_start=START,
            development_end=END,
            entry_blackouts=blackouts,
            releases=shifted_releases,
        )
        fixed = _fixed_rows(placebo.frame, observed.frame.index)
        placebo_deltas.append(evaluate_joint_volatility(fixed).delta_r2_oos)

    placebo_max = float(max(placebo_deltas))
    passed = (
        comparison.joint.pooled_r2_oos > 0
        and comparison.delta_r2_oos > 0
        and comparison.folds_improved >= 4
        and comparison.delta_r2_oos > placebo_max
        and observed.matched_coverage >= 0.90
    )
    selection = "b0_iv_calendar" if passed else "b0_only"
    report = {
        "schema_version": 1,
        "design": DESIGN,
        "prerequisite_report": str(prerequisite_path),
        "source_sha256": prerequisite["source_sha256"],
        "gvz_sha256": prerequisite["gvz_sha256"],
        "event_sha256": prerequisite["event_sha256"],
        "development_start": START.isoformat(),
        "development_end_exclusive": END.isoformat(),
        "matched_rows": observed.matched_rows,
        "matched_coverage": observed.matched_coverage,
        "target": TARGET_COLUMN,
        "reference": "b0_plus_iv",
        "reference_r2_oos": comparison.reference.pooled_r2_oos,
        "joint_r2_oos": comparison.joint.pooled_r2_oos,
        "delta_r2": comparison.delta_r2_oos,
        "per_fold_delta": list(comparison.per_fold_delta),
        "folds_improved": comparison.folds_improved,
        "placebo_seed": SEED,
        "placebo_shifts": shifts,
        "placebo_deltas": placebo_deltas,
        "placebo_max": placebo_max,
        "passed": passed,
        "selection": selection,
        "paper_enabled": False,
    }
    report_path = cfg.reports_dir / "intraday_joint_risk_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if passed:
        final_model = fit_linear_volatility_model(
            observed.frame.loc[:, ALL_FEATURES],
            observed.frame[TARGET_COLUMN],
        )
        oof = pd.DataFrame(
            {
                "forecast_log_rv_2h": comparison.joint.oof_prediction,
                "observed_log_rv_2h": observed.frame.loc[
                    comparison.joint.oof_prediction.index, TARGET_COLUMN
                ],
            }
        )
        oof_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "gold_intraday_joint_risk_oof.parquet"
        )
        oof.to_parquet(oof_path)
        artifact = {
            "schema_version": 1,
            "model": final_model,
            "feature_names": ALL_FEATURES,
            "selection": selection,
            "forecast_gate_passed": True,
            "paper_enabled": False,
            "development_end_exclusive": END,
            "design": DESIGN,
        }
        joint_path = PROJECT_ROOT / "models" / "gold_intraday_joint_volatility.joblib"
        canonical_path = PROJECT_ROOT / "models" / "gold_intraday_volatility.joblib"
        joblib.dump(artifact, joint_path)
        joblib.dump(artifact, canonical_path)
        print(f"OOF -> {oof_path}")
        print(f"Model -> {joint_path}")
        print(f"Canonical model -> {canonical_path}")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report -> {report_path}")
    print("Paper sizing remains disabled.")


def _load_prerequisite(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError("Run the separate-family risk gate first")
    report = json.loads(path.read_text(encoding="utf-8"))
    families = report.get("families", {})
    if not (
        report.get("baseline", {}).get("passed") is True
        and families.get("iv", {}).get("passed") is True
        and families.get("calendar", {}).get("passed") is True
        and report.get("selection") == "b0_only"
        and report.get("paper_enabled") is False
        and report.get("development_end_exclusive") == END.isoformat()
    ):
        raise RuntimeError("Separate-family prerequisite does not match joint design")
    return report


def _fixed_rows(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    fixed = frame.reindex(index)
    if fixed.isna().any().any():
        raise RuntimeError("Joint placebo changed the registered matched rows")
    return fixed


def _shift_releases(
    releases: pd.DataFrame,
    shift_days: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    local_start = start.tz_convert("America/New_York").normalize().tz_localize(None)
    local_end = end.tz_convert("America/New_York").normalize().tz_localize(None)
    span = (local_end - local_start).days
    shifted: list[tuple[pd.Timestamp, str]] = []
    for timestamp, row in releases.iterrows():
        local = timestamp.tz_convert("America/New_York")
        offset = (local.normalize().tz_localize(None) - local_start).days
        day = local_start + pd.Timedelta(days=(offset + shift_days) % span)
        moved = pd.Timestamp.combine(
            day.date(), local.time().replace(tzinfo=None)
        ).tz_localize("America/New_York").tz_convert("UTC")
        shifted.append((moved, str(row["kind"])))
    shifted.sort()
    return pd.DataFrame(
        {"kind": [kind for _, kind in shifted]},
        index=pd.DatetimeIndex(
            [timestamp for timestamp, _ in shifted],
            tz="UTC",
            name="release_time",
        ),
    )


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.exists() or _sha256(path) != expected:
        raise RuntimeError(f"{label} is missing or its checksum changed")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_committed_harness() -> None:
    guarded = (DESIGN,) + HARNESS
    for item in guarded:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", item],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode:
            raise SystemExit(f"{item} must be committed before the joint evaluation")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *guarded],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise SystemExit("Joint design or harness is dirty; commit it before evaluation")


if __name__ == "__main__":
    main()
