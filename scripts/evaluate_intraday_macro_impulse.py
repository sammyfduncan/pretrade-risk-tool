"""Run the one-shot frozen P4 bare-rule and 20-shift placebo gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.evaluation.intraday_macro_impulse_gate import (
    circular_shift_joint_state,
    evaluate_macro_impulse_gate,
    registered_shift_offsets,
)
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.execution import IntradayExecutionConfig, simulate_candidates
from commodity_ml.intraday.macro_impulse import (
    driver_coverage,
    generate_macro_impulse_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument(
        "--state",
        default=str(PROJECT_ROOT / "data" / "processed" / "gold_macro_state.parquet"),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    cutoff = cfg.holdout_ts
    if cutoff is None:
        raise SystemExit("Gold confirmation cutoff is not configured")

    state_path = Path(args.state).expanduser().resolve()
    state_manifest_path = state_path.with_name(state_path.stem + "_manifest.json")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    if state_manifest.get("output_sha256") != _sha256(state_path):
        raise SystemExit("Macro state hash disagrees with its manifest")
    if state_manifest.get("strategy_evaluated") is not False:
        raise SystemExit("Macro state is not an outcome-blind artifact")
    state = pd.read_parquet(state_path).sort_index()
    if (state.index >= cutoff).any():
        raise SystemExit("Macro state crosses the unopened confirmation boundary")

    gold = pd.read_parquet(cfg.intraday_base_path).sort_index()
    if (gold.index >= cutoff).any() or (
        pd.to_datetime(gold["decision_time"], utc=True) >= cutoff
    ).any():
        raise SystemExit("Gold intraday base crosses the confirmation fence")
    gold = gold.loc[gold.index >= pd.Timestamp("2020-01-01T00:00:00Z")]
    coverage = driver_coverage(gold, state)
    execution_config = IntradayExecutionConfig(
        cost_pct_per_side=cfg.backtest.fee_pct,
        entry_blackouts=scheduled_entry_blackouts(
            gold.index.min(),
            pd.DatetimeIndex(pd.to_datetime(gold["decision_time"], utc=True)).max()
            + pd.Timedelta(minutes=1),
        ),
    )

    candidates = generate_macro_impulse_candidates(
        gold,
        state,
        entry_blackouts=execution_config.entry_blackouts,
    )
    observed = simulate_candidates(gold, candidates, execution_config)
    offsets = registered_shift_offsets(len(state))
    placebo_ledgers = {}
    placebo_counts = {}
    for number, offset in enumerate(offsets, start=1):
        shifted = circular_shift_joint_state(state, offset)
        placebo_candidates = generate_macro_impulse_candidates(
            gold,
            shifted,
            entry_blackouts=execution_config.entry_blackouts,
        )
        key = f"shift_{number:02d}_rows_{offset}"
        placebo_ledgers[key] = simulate_candidates(
            gold,
            placebo_candidates,
            execution_config,
        )
        placebo_counts[key] = len(placebo_candidates)

    result = evaluate_macro_impulse_gate(
        observed,
        placebo_ledgers,
        coverage=coverage["coverage"],
        cost_pct_per_side=cfg.backtest.fee_pct,
    )
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / "gold_p4_macro_impulse_gate.json"
    ledger_path = cfg.reports_dir / "gold_p4_macro_impulse_ledger.parquet"
    report = result.report_dict() | {
        "frozen_design": "docs/intraday_macro_impulse_design.md",
        "driver_coverage": coverage,
        "causal_candidates": len(candidates),
        "reason_coded_rejections": int(
            (observed["status"] == "rejected").sum()
        ) if len(observed) else 0,
        "placebo_candidate_counts": placebo_counts,
        "state": str(state_path),
        "state_sha256": _sha256(state_path),
        "state_manifest": str(state_manifest_path),
        "state_manifest_sha256": _sha256(state_manifest_path),
        "gold_source": str(cfg.intraday_base_path),
        "gold_source_sha256": _sha256(cfg.intraday_base_path),
        "confirmation_cutoff": cutoff.isoformat(),
        "confirmation_read": False,
        "ml_trained": False,
        "paper_enabled": False,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    observed.to_parquet(ledger_path, index=False)

    base = result.metrics_by_cost[1.0]
    stressed = result.metrics_by_cost[2.0]
    print(f"P4 macro impulse: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  driver coverage: {result.coverage:.1%}\n"
        f"  candidates: {len(candidates):,}; executed trades: {base.trades:,}\n"
        f"  PF: {base.profit_factor:.3f}; mean R: {base.mean_r:+.3f}; "
        f"win rate: {base.win_rate:.1%}\n"
        f"  profitable blocks: {result.profitable_blocks}/5; "
        f"double-cost PF: {stressed.profit_factor:.3f}\n"
        f"  maximum placebo PF: {result.maximum_placebo_pf:.3f}"
    )
    if result.failure_reasons:
        print("  failures:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(
        f"  report -> {report_path}\n"
        f"  ledger -> {ledger_path}\n"
        "Confirmation was not read, no ML was trained, and paper trading remains disabled."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
