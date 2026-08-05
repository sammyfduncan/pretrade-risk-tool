"""Run the one-shot frozen RV1 rule, stresses, and 20 time-shift placebos."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.evaluation.intraday_macro_impulse_gate import (
    circular_shift_joint_state,
    registered_shift_offsets,
)
from commodity_ml.evaluation.intraday_relative_value_gate import (
    evaluate_relative_value_gate,
)
from commodity_ml.evaluation.trading_metrics import profit_factor
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.relative_value import (
    RelativeValueExecutionConfig,
    generate_relative_value_candidates,
    remeasure_outcome,
    simulate_relative_value,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quotes",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "gold_silver_m30_quotes.parquet"
        ),
    )
    parser.add_argument(
        "--state",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "gold_silver_rv1_state.parquet"
        ),
    )
    args = parser.parse_args()
    quotes_path = Path(args.quotes).expanduser().resolve()
    state_path = Path(args.state).expanduser().resolve()
    quotes_manifest_path = quotes_path.with_name(quotes_path.stem + "_manifest.json")
    state_manifest_path = state_path.with_name(state_path.stem + "_manifest.json")
    quotes_manifest = _verified_manifest(quotes_path, quotes_manifest_path)
    state_manifest = _verified_manifest(state_path, state_manifest_path)
    if quotes_manifest.get("strategy_evaluated") is not False or state_manifest.get(
        "strategy_evaluated"
    ) is not False:
        raise SystemExit("RV1 inputs are not outcome-blind")
    if state_manifest.get("forward_outcomes_read") is not False:
        raise SystemExit("RV1 state manifest does not attest an outcome-blind build")

    quotes = pd.read_parquet(quotes_path).sort_index()
    state = pd.read_parquet(state_path).sort_index()
    cutoff = pd.Timestamp("2025-07-01T00:00:00Z")
    if (quotes.index >= cutoff).any() or (
        pd.to_datetime(state["decision_time"], utc=True) >= cutoff
    ).any():
        raise SystemExit("RV1 inputs cross the confirmation boundary")
    config = RelativeValueExecutionConfig(
        entry_blackouts=scheduled_entry_blackouts(
            quotes.index.min(),
            quotes.index.max() + pd.Timedelta(minutes=30),
        )
    )

    candidates = generate_relative_value_candidates(state, config=config)
    observed = _add_stresses(
        simulate_relative_value(quotes, candidates, config=config),
        quotes,
    )
    offsets = registered_shift_offsets(len(state))
    placebo_ledgers = {}
    placebo_counts = {}
    for number, offset in enumerate(offsets, start=1):
        shifted = circular_shift_joint_state(state, offset)
        shifted_candidates = generate_relative_value_candidates(shifted, config=config)
        key = f"shift_{number:02d}_rows_{offset}"
        placebo_ledgers[key] = _add_stresses(
            simulate_relative_value(quotes, shifted_candidates, config=config),
            quotes,
        )
        placebo_counts[key] = len(shifted_candidates)

    coverage = float(quotes_manifest["synchronization"]["shared_coverage"])
    result = evaluate_relative_value_gate(
        observed,
        placebo_ledgers,
        coverage=coverage,
    )
    executed = observed.loc[observed["status"] == "executed"].copy()
    report = result.report_dict() | {
        "frozen_design": "docs/archive/intraday_relative_value_design.md",
        "causal_candidates": len(candidates),
        "executed_trades": len(executed),
        "reason_coded_rejections": int((observed["status"] == "rejected").sum()),
        "rejection_reasons": observed.loc[
            observed["status"] == "rejected", "reject_reason"
        ].value_counts().to_dict(),
        "exit_reasons": executed["exit_reason"].value_counts().to_dict(),
        "holding_bars": _summary(executed["bars_held"]),
        "beta": _summary(executed["beta"]),
        "execution_audit": _execution_audit(executed, quotes),
        "placebo_candidate_counts": placebo_counts,
        "quotes": str(quotes_path),
        "quotes_sha256": _sha256(quotes_path),
        "quotes_manifest": str(quotes_manifest_path),
        "quotes_manifest_sha256": _sha256(quotes_manifest_path),
        "state": str(state_path),
        "state_sha256": _sha256(state_path),
        "state_manifest": str(state_manifest_path),
        "state_manifest_sha256": _sha256(state_manifest_path),
        "confirmation_cutoff": cutoff.isoformat(),
        "confirmation_read": False,
        "ml_trained": False,
        "paper_enabled": False,
    }
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "gold_silver_rv1_gate.json"
    ledger_path = reports / "gold_silver_rv1_ledger.parquet"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    observed.to_parquet(ledger_path, index=False)

    base = result.metrics_by_stress["1.0"]
    stressed = result.metrics_by_stress["2.0"]
    print(f"RV1 gold-silver residual reversal: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  synchronized coverage: {result.coverage:.2%}\n"
        f"  candidates: {len(candidates):,}; executed: {base.trades:,}\n"
        f"  side fractions: long {result.side_fractions['long_residual']:.1%}, "
        f"short {result.side_fractions['short_residual']:.1%}\n"
        f"  base PF: {base.profit_factor:.3f}; mean R: {base.mean_r:+.3f}; "
        f"win rate: {base.win_rate:.1%}\n"
        f"  profitable blocks: {result.profitable_blocks}/5; "
        f"2x spread/cost PF: {stressed.profit_factor:.3f}\n"
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


def _add_stresses(ledger: pd.DataFrame, quotes: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    for suffix in ("1_0", "1_5", "2_0"):
        out[f"gross_return_{suffix}"] = np.nan
        out[f"net_return_{suffix}"] = np.nan
    settings = {
        "1_0": (1.0, 0.00005),
        "1_5": (1.5, 0.000075),
        "2_0": (2.0, 0.0001),
    }
    for index, row in out.loc[out["status"] == "executed"].iterrows():
        for suffix, (spread, cost) in settings.items():
            gross, net = remeasure_outcome(
                quotes,
                row,
                spread_multiplier=spread,
                additional_cost_per_side=cost,
            )
            out.at[index, f"gross_return_{suffix}"] = gross
            out.at[index, f"net_return_{suffix}"] = net
    executed = out["status"] == "executed"
    if executed.any() and not np.allclose(
        out.loc[executed, "net_return"],
        out.loc[executed, "net_return_1_0"],
        rtol=0,
        atol=1e-15,
    ):
        raise ValueError("RV1 base replay diverges from its fixed-clock remeasurement")
    return out


def _verified_manifest(artifact: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != _sha256(artifact):
        raise SystemExit(f"Artifact hash disagrees with manifest: {artifact}")
    return manifest


def _execution_audit(executed: pd.DataFrame, quotes: pd.DataFrame) -> dict:
    """Separate weak midpoint tendency from executable two-leg spread drag."""

    if executed.empty:
        return {}
    fixed_exit = []
    next_bar = []
    for _, row in executed.iterrows():
        entry = quotes.loc[pd.Timestamp(row["entry_time"])]
        exit_bar = quotes.loc[pd.Timestamp(row["exit_bar_time"])]
        fixed_total = 0.0
        next_total = 0.0
        for leg, weight in (
            ("gold", float(row["gold_weight"])),
            ("silver", float(row["silver_weight"])),
        ):
            fixed_leg = exit_bar[f"{leg}_mid_close"] / entry[f"{leg}_mid_open"] - 1
            next_leg = entry[f"{leg}_mid_close"] / entry[f"{leg}_mid_open"] - 1
            direction = 1.0 if weight > 0 else -1.0
            fixed_total += abs(weight) * direction * fixed_leg
            next_total += abs(weight) * direction * next_leg
        fixed_exit.append(float(fixed_total))
        next_bar.append(float(next_total))
    fixed = pd.Series(fixed_exit, dtype=float)
    immediate = pd.Series(next_bar, dtype=float)
    spread_drag = fixed.to_numpy() - executed["gross_return_1_0"].to_numpy(dtype=float)
    return {
        "fixed_exit_midpoint_mean_return": float(fixed.mean()),
        "fixed_exit_midpoint_profit_factor": float(profit_factor(fixed)),
        "next_m30_midpoint_mean_return": float(immediate.mean()),
        "next_m30_midpoint_profit_factor": float(profit_factor(immediate)),
        "next_m30_midpoint_win_rate": float((immediate > 0).mean()),
        "mean_historical_quote_spread_drag": float(spread_drag.mean()),
        "median_historical_quote_spread_drag": float(np.median(spread_drag)),
        "additional_round_trip_allowance": 0.0001,
        "exit_clocks_changed": False,
    }


def _summary(values: pd.Series) -> dict:
    values = values.dropna().astype(float)
    if values.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
