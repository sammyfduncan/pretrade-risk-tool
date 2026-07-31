"""Evaluate the frozen risk policy only for one preregistered retained strategy."""

from __future__ import annotations

import json
import subprocess

import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.evaluation.intraday_risk_policy import evaluate_risk_policy_gate
from commodity_ml.intraday.ml_dataset import load_candidate_bundle

DESIGN = "docs/intraday_risk_economic_design.md"
HARNESS = (
    "scripts/evaluate_intraday_risk_policy.py",
    "src/commodity_ml/evaluation/intraday_risk_policy.py",
)
REGISTERED_STRATEGY: str | None = None


def main() -> None:
    _require_committed_harness()
    strategy = _registered_strategy()
    cfg = load_config("config/gold.yaml")
    primary_report_path = (
        cfg.reports_dir / f"gold_{strategy.lower()}_primary_gate.json"
    )
    primary = _load_passing_primary(primary_report_path, strategy)
    joint_path = cfg.reports_dir / "intraday_joint_risk_gate.json"
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    if not (
        joint.get("passed") is True
        and joint.get("selection") == "b0_iv_calendar"
        and joint.get("paper_enabled") is False
    ):
        raise RuntimeError("Joint volatility forecast is not eligible")

    bundle = load_candidate_bundle(
        cfg.intraday_candidate_path(strategy),
        expected_strategy=strategy,
        expected_source="dukascopy",
        expected_cost_pct_per_side=cfg.backtest.fee_pct,
    )
    candidates = bundle.dataset.sort_index().copy()
    executed = bundle.ledger.loc[bundle.ledger["status"] == "executed"].copy()
    executed.index = pd.DatetimeIndex(
        pd.to_datetime(executed["signal_time"], utc=True),
        name="time",
    )
    candidates["gross_return"] = executed.loc[
        candidates.index, "gross_return"
    ].astype(float)
    if cfg.holdout_ts is not None and (
        candidates.index.max() >= cfg.holdout_ts
        or pd.to_datetime(candidates["exit_time"], utc=True).max() >= cfg.holdout_ts
    ):
        raise RuntimeError("Candidate bundle crosses the confirmation fence")

    oof_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "gold_intraday_joint_risk_oof.parquet"
    )
    oof = pd.read_parquet(oof_path)["forecast_log_rv_2h"]
    result = evaluate_risk_policy_gate(
        candidates,
        oof,
        cost_pct_per_side=cfg.backtest.fee_pct,
    )
    report = result.report_dict() | {
        "schema_version": 1,
        "strategy": strategy,
        "primary_report": str(primary_report_path),
        "primary_gate_passed": primary["passed"],
        "joint_report": str(joint_path),
        "confirmation_opened": False,
        "paper_enabled": False,
    }
    report_path = cfg.reports_dir / f"gold_{strategy.lower()}_risk_policy_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    decisions_path = (
        cfg.reports_dir / f"gold_{strategy.lower()}_risk_policy_decisions.parquet"
    )
    result.decisions.to_parquet(decisions_path)
    print(f"gold {strategy} risk policy: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  matched={result.matched_opportunities}, "
        f"coverage={result.match_coverage:.1%}, multipliers={result.multiplier_counts}"
    )
    print(
        f"  constant return={result.constant.compounded_return:+.2%}, "
        f"maxDD={result.constant.max_drawdown:.2%}, "
        f"PF={result.constant.profit_factor:.3f}"
    )
    print(
        f"  policy return={result.policy.compounded_return:+.2%}, "
        f"maxDD={result.policy.max_drawdown:.2%}, "
        f"PF={result.policy.profit_factor:.3f}"
    )
    print(f"  report -> {report_path}\n  decisions -> {decisions_path}")
    print("Confirmation data was not read. Paper sizing remains disabled.")


def _registered_strategy() -> str:
    if REGISTERED_STRATEGY is None:
        raise SystemExit(
            "No strategy is registered for the risk-policy experiment. "
            "All current primary strategies failed; no trade P&L was read."
        )
    return REGISTERED_STRATEGY


def _load_passing_primary(path, strategy: str) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing primary-gate report for {strategy}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("strategy") != strategy or report.get("passed") is not True:
        raise RuntimeError(f"{strategy} did not pass its frozen primary gate")
    return report


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
            raise SystemExit(f"{item} must be committed before risk-policy evaluation")
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *guarded],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    if dirty.stdout.strip():
        raise SystemExit("Risk-policy design or harness is dirty; refusing evaluation")


if __name__ == "__main__":
    main()
