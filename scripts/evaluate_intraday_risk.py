"""Run the frozen intraday volatility-risk information gate.

This command intentionally refuses to run while either registration document
is uncommitted.  It reads development data only, produces OOF forecasts, and
does not activate paper sizing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import joblib
import numpy as np
import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.evaluation.placebo import circular_shift, draw_shifts
from commodity_ml.features.vol_layer import fetch_gvz
from commodity_ml.intraday.event_blackouts import (
    scheduled_entry_blackouts,
    scheduled_release_times,
)
from commodity_ml.intraday.volatility import (
    BASELINE_FEATURES,
    CALENDAR_FEATURES,
    IV_FEATURES,
    TARGET_COLUMN,
    build_volatility_dataset,
    clean_gvz_observations,
    evaluate_volatility_ablation,
    fit_linear_volatility_model,
    walk_forward_volatility,
)

DESIGN_DOCS = (
    "docs/archive/intraday_risk_design.md",
    "docs/archive/intraday_risk_policy.md",
)
HARNESS_FILES = (
    "scripts/evaluate_intraday_risk.py",
    "src/commodity_ml/intraday/volatility.py",
)
GVZ_SNAPSHOT = PROJECT_ROOT / "data" / "raw" / "GVZCLS_intraday_risk.parquet"
START = pd.Timestamp("2021-01-01", tz="UTC")
END = pd.Timestamp("2025-07-01", tz="UTC")
N_PLACEBOS = 20
SEED = 0


def main() -> None:
    _require_committed_design()
    cfg = load_config("config/gold.yaml")
    if cfg.asset != "gold":
        raise SystemExit("The registered first risk experiment is gold only")

    source = cfg.raw_path("dukascopy", granularity="M30")
    candles = pd.read_parquet(
        source,
        filters=[("time", ">=", START), ("time", "<", END)],
    )
    gvz_raw, gvz_path, gvz_hash = _load_or_fetch_gvz()
    releases = scheduled_release_times(START, END)
    gvz = clean_gvz_observations(gvz_raw)
    event_path, event_hash = _pin_snapshot(
        releases,
        PROJECT_ROOT / "data" / "raw" / "gold_intraday_risk_events.parquet",
        {
            "sources": [
                "https://www.bls.gov/schedule/",
                "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
            ],
            "verified_at": "2026-07-30",
        },
    )
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
        raise SystemExit(
            f"Matched coverage {observed.matched_coverage:.1%} is below 90%; "
            "data-quality failure, no result"
        )

    baseline = walk_forward_volatility(observed.frame, BASELINE_FEATURES)
    baseline_pass = (
        baseline.pooled_r2_oos > 0
        and sum(fold.r2_oos > 0 for fold in baseline.folds) >= 4
    )
    rng = np.random.default_rng(SEED)
    evaluations = {}

    iv_observed = evaluate_volatility_ablation(observed.frame, IV_FEATURES)
    iv_placebos = []
    for shift in draw_shifts(len(gvz), N_PLACEBOS, "gvz", rng):
        shifted = circular_shift(gvz.to_frame(), shift)[gvz.name or "gvz"]
        shifted.name = "gvz"
        placebo = build_volatility_dataset(
            candles,
            shifted,
            development_start=START,
            development_end=END,
            entry_blackouts=blackouts,
            releases=releases,
        )
        fixed = _fixed_rows(placebo.frame, observed.frame.index)
        iv_placebos.append(
            evaluate_volatility_ablation(fixed, IV_FEATURES).delta_r2_oos
        )
    evaluations["iv"] = _family_report(iv_observed, iv_placebos, observed)

    cal_observed = evaluate_volatility_ablation(observed.frame, CALENDAR_FEATURES)
    calendar_days = (END.normalize() - START.normalize()).days
    cal_placebos = []
    for shift in draw_shifts(calendar_days, N_PLACEBOS, "calendar", rng):
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
        cal_placebos.append(
            evaluate_volatility_ablation(fixed, CALENDAR_FEATURES).delta_r2_oos
        )
    evaluations["calendar"] = _family_report(cal_observed, cal_placebos, observed)

    passing = [name for name, item in evaluations.items() if item["passed"]]
    if baseline_pass and len(passing) == 1:
        active_features = BASELINE_FEATURES + (
            IV_FEATURES if passing[0] == "iv" else CALENDAR_FEATURES
        )
        selection = f"b0_plus_{passing[0]}"
    elif baseline_pass:
        active_features = BASELINE_FEATURES
        selection = "b0_only"
    else:
        active_features = ()
        selection = "none"

    report = {
        "schema_version": 1,
        "design_docs": list(DESIGN_DOCS),
        "source": str(source),
        "source_sha256": _sha256(source),
        "source_rows": len(candles),
        "source_first": candles.index.min().isoformat(),
        "source_last": candles.index.max().isoformat(),
        "source_duplicate_timestamps": int(candles.index.duplicated().sum()),
        "source_non_30m_intervals": int(
            candles.index.to_series().diff().dropna().ne(pd.Timedelta(minutes=30)).sum()
        ),
        "gvz_snapshot": str(gvz_path),
        "gvz_sha256": gvz_hash,
        "event_snapshot": str(event_path),
        "event_sha256": event_hash,
        "development_start": START.isoformat(),
        "development_end_exclusive": END.isoformat(),
        "target": TARGET_COLUMN,
        "eligible_price_rows": observed.eligible_price_rows,
        "matched_rows": observed.matched_rows,
        "matched_coverage": observed.matched_coverage,
        "exclusions": observed.exclusions,
        "baseline": {
            "r2_oos": baseline.pooled_r2_oos,
            "fold_r2": [fold.r2_oos for fold in baseline.folds],
            "passed": baseline_pass,
        },
        "families": evaluations,
        "selection": selection,
        "paper_enabled": False,
    }
    report_path = cfg.reports_dir / "intraday_risk_gate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    if active_features:
        forecast = walk_forward_volatility(observed.frame, active_features)
        final_model = fit_linear_volatility_model(
            observed.frame.loc[:, active_features],
            observed.frame[TARGET_COLUMN],
        )
        oof_path = PROJECT_ROOT / "data" / "processed" / "gold_intraday_risk_oof.parquet"
        pd.DataFrame(
            {
                "forecast_log_rv_2h": forecast.oof_prediction,
                "observed_log_rv_2h": observed.frame.loc[
                    forecast.oof_prediction.index, TARGET_COLUMN
                ],
            }
        ).to_parquet(oof_path)
        artifact_path = PROJECT_ROOT / "models" / "gold_intraday_volatility.joblib"
        joblib.dump(
            {
                "schema_version": 1,
                "model": final_model,
                "feature_names": active_features,
                "selection": selection,
                "forecast_gate_passed": True,
                "paper_enabled": False,
                "development_end_exclusive": END,
            },
            artifact_path,
        )
        print(f"OOF -> {oof_path}\nModel -> {artifact_path}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Report -> {report_path}\nPaper sizing remains disabled.")


def _family_report(ablation, placebos, dataset) -> dict:
    maximum = float(max(placebos))
    passed = (
        ablation.augmented.pooled_r2_oos > 0
        and ablation.delta_r2_oos > 0
        and ablation.folds_improved >= 4
        and ablation.delta_r2_oos > maximum
        and dataset.matched_coverage >= 0.90
    )
    return {
        "r2_with": ablation.augmented.pooled_r2_oos,
        "r2_without": ablation.baseline.pooled_r2_oos,
        "delta_r2": ablation.delta_r2_oos,
        "per_fold_delta": list(ablation.per_fold_delta),
        "folds_improved": ablation.folds_improved,
        "placebo_max": maximum,
        "placebo_deltas": [float(value) for value in placebos],
        "passed": passed,
    }


def _fixed_rows(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    fixed = frame.reindex(index)
    if fixed.isna().any().any():
        raise RuntimeError(
            "A placebo cannot reproduce the registered matched rows; "
            "refusing a sample-composition change"
        )
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
    shifted = []
    for timestamp, row in releases.iterrows():
        local = timestamp.tz_convert("America/New_York")
        offset = (local.normalize().tz_localize(None) - local_start).days
        day = local_start + pd.Timedelta(days=(offset + shift_days) % span)
        clock = local.time().replace(tzinfo=None)
        moved = pd.Timestamp.combine(day.date(), clock).tz_localize(
            "America/New_York"
        ).tz_convert("UTC")
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


def _require_committed_design() -> None:
    guarded = DESIGN_DOCS + HARNESS_FILES
    for document in guarded:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", document],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if tracked.returncode:
            raise SystemExit(
                f"{document} is not committed. Commit the risk registrations "
                "and evaluation harness before generating a real-data forecast."
            )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *guarded],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise SystemExit(
            "Risk registration or evaluation harness has uncommitted changes. "
            "Commit it before generating a real-data forecast."
        )


def _load_or_fetch_gvz() -> tuple[pd.Series, Path, str]:
    sidecar = GVZ_SNAPSHOT.with_suffix(".meta.json")
    if GVZ_SNAPSHOT.exists():
        if not sidecar.exists():
            raise RuntimeError(
                f"Pinned GVZ snapshot has no provenance sidecar: {sidecar}"
            )
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        actual = _sha256(GVZ_SNAPSHOT)
        expected = metadata.get("sha256")
        if not expected or actual != expected:
            raise RuntimeError("Pinned GVZ snapshot checksum mismatch")
        frame = pd.read_parquet(GVZ_SNAPSHOT)
        if list(frame.columns) != ["gvz"]:
            raise RuntimeError(
                f"Pinned GVZ snapshot schema mismatch: {list(frame.columns)}"
            )
        return frame["gvz"], GVZ_SNAPSHOT, actual

    gvz = fetch_gvz(end="2025-06-30")
    path, digest = _pin_snapshot(
        gvz.to_frame(),
        GVZ_SNAPSHOT,
        {
            "source": "https://fred.stlouisfed.org/series/GVZCLS",
            "series": "GVZCLS",
            "retrieved_at": pd.Timestamp.now(tz="UTC").isoformat(),
        },
    )
    return gvz, path, digest


def _pin_snapshot(
    frame: pd.DataFrame,
    path: Path,
    metadata: dict,
) -> tuple[Path, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    digest = _sha256(path)
    sidecar = path.with_suffix(".meta.json")
    sidecar.write_text(
        json.dumps(
            metadata
            | {
                "snapshot": str(path),
                "sha256": digest,
                "rows": len(frame),
                "first": frame.index.min().isoformat() if len(frame) else None,
                "last": frame.index.max().isoformat() if len(frame) else None,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path, digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
