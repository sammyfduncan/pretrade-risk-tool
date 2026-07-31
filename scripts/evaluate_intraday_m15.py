"""Run the frozen matched-M15 execution gate without opening confirmation."""

import argparse
import json

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.intraday_m15_gate import evaluate_m15_gate
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.execution import IntradayExecutionConfig
from commodity_ml.intraday.m15_execution import replay_m30_candidates_on_m15
from commodity_ml.intraday.ml_dataset import load_candidate_bundle
from commodity_ml.intraday.strategies import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--strategy", choices=["P1V2", "P3V2"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    base = pd.read_parquet(cfg.intraday_base_path)
    m15_path = cfg.raw_path("dukascopy", granularity="M15")
    if not m15_path.exists():
        raise SystemExit(f"Missing M15 candles: {m15_path}")
    m15 = pd.read_parquet(m15_path)
    cutoff = cfg.holdout_ts
    if cutoff is not None:
        base = base.loc[base["decision_time"] < cutoff]
        m15 = m15.loc[m15.index < cutoff]
    coverage_start = pd.Timestamp("2020-01-01", tz="UTC")
    base = base.loc[base.index >= coverage_start]
    m15 = m15.loc[m15.index >= coverage_start]

    bundle = load_candidate_bundle(
        cfg.intraday_candidate_path(args.strategy),
        expected_strategy=args.strategy,
        expected_source="dukascopy",
        expected_cost_pct_per_side=cfg.backtest.fee_pct,
    )
    control = bundle.ledger.loc[bundle.ledger["status"] == "executed"].copy()
    if cutoff is not None and (
        pd.to_datetime(control["signal_time"], utc=True) >= cutoff
    ).any():
        raise SystemExit("M30 control crosses the confirmation fence")

    candidates = generate_candidates(base, args.strategy)
    by_signal = {pd.Timestamp(item.signal_time): item for item in candidates}
    control_signals = pd.DatetimeIndex(
        pd.to_datetime(control["signal_time"], utc=True)
    )
    missing = control_signals.difference(pd.DatetimeIndex(by_signal))
    if len(missing):
        raise SystemExit(
            "Could not reconstruct all M30 control candidates; rebuild the "
            f"candidate bundle. Missing examples: {list(missing[:5])}"
        )
    matched_candidates = [by_signal[timestamp] for timestamp in control_signals]
    execution_cfg = IntradayExecutionConfig(
        cost_pct_per_side=cfg.backtest.fee_pct,
        entry_blackouts=scheduled_entry_blackouts(
            m15.index.min(),
            m15.index.max() + pd.Timedelta(minutes=15),
        ),
    )
    replay = replay_m30_candidates_on_m15(m15, matched_candidates, execution_cfg)
    result = evaluate_m15_gate(
        control,
        replay,
        strategy=args.strategy,
        cost_pct_per_side=cfg.backtest.fee_pct,
    )

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.asset}_{args.strategy.lower()}_m15_execution_gate"
    report_path = cfg.reports_dir / f"{stem}.json"
    ledger_path = cfg.reports_dir / f"{stem}_ledger.parquet"
    report_path.write_text(
        json.dumps(result.report_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    replay.to_parquet(ledger_path)

    m30 = result.m30_by_cost[1.0]
    m15_metrics = result.m15_by_cost[1.0]
    stress = result.m15_by_cost[2.0]
    print(f"{cfg.asset} {args.strategy} M15: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  replayed={result.replayed_trades}/{result.control_trades} "
        f"({result.coverage:.1%})"
    )
    print(
        f"  matched PF M30={m30.profit_factor:.3f} -> "
        f"M15={m15_metrics.profit_factor:.3f}; mean R "
        f"{m30.mean_r:+.3f} -> {m15_metrics.mean_r:+.3f}"
    )
    print(
        f"  maxDD {m30.max_drawdown:.1%} -> {m15_metrics.max_drawdown:.1%}; "
        f"improved blocks={result.improved_blocks}/5; "
        f"M15 PF at 2x cost={stress.profit_factor:.3f}"
    )
    print("  exit transitions:")
    for transition, count in result.exit_transitions.items():
        print(f"    {transition}: {count}")
    if result.failure_reasons:
        print("  failures:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(
        f"  report -> {report_path}\n"
        f"  ledger -> {ledger_path}\n"
        "Confirmation data was not read. No M15 signal or ML model was created."
    )


if __name__ == "__main__":
    main()
