"""Run one frozen V2 bare-primary gate without opening confirmation data."""

import argparse
import json

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.intraday_primary_gate import evaluate_primary_gate
from commodity_ml.intraday.ml_dataset import load_candidate_bundle

CONTROLS = {"P1V2": "P1", "P2V2": "P2", "P3V2": "P3"}
MIN_TRADES = {"P1V2": 300, "P2V2": 300, "P3V2": 100}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--strategy", choices=sorted(CONTROLS), required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    control_strategy = CONTROLS[args.strategy]
    candidate = _load_with_gross(cfg, args.strategy)
    control = _load_with_gross(cfg, control_strategy)
    cutoff = cfg.holdout_ts
    for name, frame in ((args.strategy, candidate), (control_strategy, control)):
        if cutoff is not None and frame.index.max() >= cutoff:
            raise SystemExit(f"{name} crosses the confirmation fence")

    result = evaluate_primary_gate(
        candidate,
        control,
        strategy=args.strategy,
        control_strategy=control_strategy,
        cost_pct_per_side=cfg.backtest.fee_pct,
        min_trades=MIN_TRADES[args.strategy],
    )
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    output = cfg.reports_dir / f"{cfg.asset}_{args.strategy.lower()}_primary_gate.json"
    output.write_text(
        json.dumps(result.report_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    base = result.metrics_by_cost[1.0]
    stress = result.metrics_by_cost[2.0]
    print(f"{cfg.asset} {args.strategy}: {'PASS' if result.passed else 'FAIL'}")
    print(
        f"  trades={base.trades}, PF={base.profit_factor:.3f} "
        f"(V1={result.control_metrics.profit_factor:.3f}), "
        f"mean R={base.mean_r:+.3f}, maxDD={base.max_drawdown:.1%} "
        f"(V1={result.control_metrics.max_drawdown:.1%})"
    )
    print(
        f"  chronological block PFs: "
        + ", ".join(f"{item.profit_factor:.3f}" for item in result.block_metrics)
    )
    print(
        f"  profitable blocks={result.profitable_blocks}/5; "
        f"PF at 2x cost={stress.profit_factor:.3f}"
    )
    if result.failure_reasons:
        print("  failures:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(
        f"  report -> {output}\n"
        "Confirmation data was not read. ML was not trained."
    )


def _load_with_gross(cfg, strategy: str) -> pd.DataFrame:
    bundle = load_candidate_bundle(
        cfg.intraday_candidate_path(strategy),
        expected_strategy=strategy,
        expected_source="dukascopy",
        expected_cost_pct_per_side=cfg.backtest.fee_pct,
    )
    frame = bundle.dataset.sort_index().copy()
    executed = bundle.ledger.loc[bundle.ledger["status"] == "executed"].copy()
    executed.index = pd.DatetimeIndex(
        pd.to_datetime(executed["signal_time"], utc=True),
        name="time",
    )
    frame["gross_return"] = executed.loc[frame.index, "gross_return"].astype(float)
    return frame


if __name__ == "__main__":
    main()
