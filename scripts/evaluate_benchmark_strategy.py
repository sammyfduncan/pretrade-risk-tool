"""Evaluate frozen LBMA-PM strategy B1 and its four registered time placebos."""

import json
from datetime import time

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.intraday_benchmark_gate import evaluate_benchmark_gate
from commodity_ml.intraday.benchmark_strategy import benchmark_continuation_candidates
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.execution import IntradayExecutionConfig, simulate_candidates

REGISTERED_CLOCKS = {
    "B1": time(15, 0),
    "B1_PLACEBO_1300": time(13, 0),
    "B1_PLACEBO_1400": time(14, 0),
    "B1_PLACEBO_1600": time(16, 0),
    "B1_PLACEBO_1700": time(17, 0),
}


def main() -> None:
    cfg = load_config("config/gold.yaml")
    path = cfg.raw_path("dukascopy", granularity="M15")
    if not path.exists():
        raise SystemExit(f"Missing M15 data: {path}")
    candles = pd.read_parquet(path)
    cutoff = cfg.holdout_ts
    if cutoff is not None:
        candles = candles.loc[candles.index < cutoff]
    candles = candles.loc[candles.index >= pd.Timestamp("2020-01-01", tz="UTC")]
    if candles.empty:
        raise SystemExit("No development M15 candles remain")
    execution_cfg = IntradayExecutionConfig(
        bar_duration=pd.Timedelta(minutes=15),
        cost_pct_per_side=cfg.backtest.fee_pct,
        entry_blackouts=scheduled_entry_blackouts(
            candles.index.min(),
            candles.index.max() + pd.Timedelta(minutes=15),
        ),
    )

    ledgers: dict[str, pd.DataFrame] = {}
    for strategy, clock in REGISTERED_CLOCKS.items():
        candidates = benchmark_continuation_candidates(
            candles,
            benchmark_time=clock,
            strategy=strategy,
        )
        ledgers[strategy] = simulate_candidates(candles, candidates, execution_cfg)
    result = evaluate_benchmark_gate(
        ledgers["B1"],
        {name: ledger for name, ledger in ledgers.items() if name != "B1"},
        strategy="B1",
        cost_pct_per_side=cfg.backtest.fee_pct,
    )

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / "gold_b1_benchmark_gate.json"
    report_path.write_text(
        json.dumps(result.report_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    for name, ledger in ledgers.items():
        ledger.to_parquet(cfg.reports_dir / f"gold_{name.lower()}_ledger.parquet")

    base = result.metrics_by_cost[1.0]
    stressed = result.metrics_by_cost[2.0]
    print(f"gold B1 LBMA-PM continuation: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  trades={base.trades}, PF={base.profit_factor:.3f}, "
        f"mean R={base.mean_r:+.3f}, maxDD={base.max_drawdown:.1%}, "
        f"PF at 2x cost={stressed.profit_factor:.3f}"
    )
    print(
        "  chronological block PFs: "
        + ", ".join(f"{item.profit_factor:.3f}" for item in result.block_metrics)
    )
    print("  time placebos:")
    for name, metrics in result.placebo_metrics.items():
        print(f"    {name}: n={metrics.trades}, PF={metrics.profit_factor:.3f}")
    if result.failure_reasons:
        print("  failures:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(
        f"  report -> {report_path}\n"
        "Confirmation data was not read. No ML model was trained."
    )


if __name__ == "__main__":
    main()
