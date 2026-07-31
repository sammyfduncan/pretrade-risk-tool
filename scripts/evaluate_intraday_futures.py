"""Run the frozen P1 + logistic COMEX GC feature-family retention gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.evaluation.intraday_futures_gate import evaluate_futures_gate
from commodity_ml.intraday.futures_state import (
    FUTURES_FEATURES,
    build_futures_candidate_features,
    load_cme_partitions,
)
from commodity_ml.intraday.ml_dataset import load_candidate_bundle


def main() -> None:
    cfg = load_config("config/gold.yaml")
    bundle = load_candidate_bundle(
        cfg.intraday_candidate_path("P1"),
        expected_strategy="P1",
        expected_source="dukascopy",
        expected_cost_pct_per_side=cfg.backtest.fee_pct,
    )
    dataset = bundle.dataset.sort_index().copy()
    cutoff = cfg.holdout_ts
    if cutoff is None:
        raise SystemExit("Gold confirmation cutoff is not configured")
    if (dataset.index >= cutoff).any() or (
        pd.to_datetime(dataset["exit_time"], utc=True) >= cutoff
    ).any():
        raise SystemExit("P1 candidate bundle crosses the confirmation fence")

    cme_root = PROJECT_ROOT / "data" / "raw" / "cme_gc"
    cme_manifest_path = cme_root / "manifest.json"
    cme_manifest = json.loads(cme_manifest_path.read_text(encoding="utf-8"))
    if pd.Timestamp(cme_manifest["end_exclusive"]) != cutoff:
        raise SystemExit("CME artifact does not end exactly at the confirmation fence")
    if not cme_manifest.get("development_cutoff_applied"):
        raise SystemExit("CME artifact does not record the development cutoff")
    degraded_by_schema = cme_manifest["degraded_source_dates"]
    degraded_sets = {tuple(values) for values in degraded_by_schema.values()}
    if len(degraded_sets) != 1:
        raise SystemExit("CME schemas disagree on degraded source dates")
    degraded_dates = next(iter(degraded_sets))

    spot = pd.read_parquet(cfg.intraday_base_path).sort_index()
    if (spot.index >= cutoff).any() or (
        pd.to_datetime(spot["decision_time"], utc=True) >= cutoff
    ).any():
        raise SystemExit("Intraday spot base crosses the confirmation fence")
    ohlcv, statistics, definitions = load_cme_partitions(cme_root)
    features = build_futures_candidate_features(
        spot,
        dataset,
        ohlcv,
        statistics,
        definitions,
        degraded_dates=degraded_dates,
    )
    matched = dataset.loc[features.eligible].copy()
    matched.loc[:, FUTURES_FEATURES] = features.features.loc[
        features.eligible, FUTURES_FEATURES
    ]

    executed = bundle.ledger.loc[bundle.ledger["status"] == "executed"].copy()
    executed.index = pd.DatetimeIndex(
        pd.to_datetime(executed["signal_time"], utc=True), name="time"
    )
    matched["gross_return"] = executed.loc[
        matched.index, "gross_return"
    ].astype(float)
    result = evaluate_futures_gate(
        matched,
        bundle.feature_names,
        FUTURES_FEATURES,
        cost_pct_per_side=cfg.backtest.fee_pct,
        coverage=features.report["coverage"],
    )

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / "gold_p1_futures_gate.json"
    predictions_path = cfg.reports_dir / "gold_p1_futures_gate_predictions.parquet"
    report = result.report_dict() | {
        "feature_build": features.report,
        "cme_manifest": str(cme_manifest_path),
        "cme_manifest_sha256": _sha256(cme_manifest_path),
        "confirmation_cutoff": cutoff.isoformat(),
        "confirmation_read": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result.challenger.predictions.to_parquet(predictions_path)

    base = result.baseline.filtered_by_cost[1.0]
    challenge = result.challenger.filtered_by_cost[1.0]
    placebo_max = max(item["delta_auc"] for item in result.placebo_results)
    print(f"F-GC/P1+logistic: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  matched coverage: {result.coverage:.1%}\n"
        f"  pooled AUC: {result.baseline_auc:.4f} -> "
        f"{result.challenger_auc:.4f} (delta {result.delta_auc:+.4f})\n"
        f"  AUC-improved folds: {result.auc_improved_folds}/5; "
        f"placebo max delta: {placebo_max:+.4f}\n"
        f"  filtered trades: {base.trades} -> {challenge.trades}\n"
        f"  filtered PF: {base.profit_factor:.3f} -> "
        f"{challenge.profit_factor:.3f}; mean R {challenge.mean_r:+.3f}\n"
        f"  challenger PF at 2x cost: "
        f"{result.challenger.filtered_by_cost[2.0].profit_factor:.3f}"
    )
    if result.failure_reasons:
        print("  failures:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(
        f"  report -> {report_path}\n"
        f"  predictions -> {predictions_path}\n"
        "Confirmation data was not read and no deployment artifact was created."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
