"""Profile cached OANDA bid/ask candle costs without a network request."""

import argparse
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.data.schema import validate_candles
from commodity_ml.intraday.cost_measurement import measure_quote_costs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--granularity", choices=["M15", "M30", "H1"], default="M30")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--stop-distance-pct", type=float)
    parser.add_argument("--slippage-bp-per-side", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.input or cfg.raw_path("oanda", granularity=args.granularity)
    if not source.exists():
        raise SystemExit(
            f"Missing cached OANDA quotes: {source}\n"
            "Fetch a bounded window first or pass --input."
        )
    candles = validate_candles(pd.read_parquet(source), "oanda-cost-profile")
    ledger = pd.read_parquet(args.ledger) if args.ledger else None
    report = measure_quote_costs(
        candles,
        slippage_pct_per_side=args.slippage_bp_per_side / 10_000,
        representative_stop_pct=args.stop_distance_pct,
        candidate_ledger=ledger,
    )
    output = args.output or (
        cfg.reports_dir / f"{cfg.asset}_{args.granularity.lower()}_oanda_cost_profile.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.report_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    overall = report.overall
    print(
        f"OANDA {args.granularity} cost profile: {overall.observations} bars, "
        f"median {overall.median_bp:.2f} bp, p95 {overall.p95_bp:.2f} bp, "
        f"p99 {overall.p99_bp:.2f} bp"
    )
    if report.median_cost_r is not None:
        print(
            f"Representative stop: median {report.median_cost_r:.3f}R, "
            f"p95 {report.p95_cost_r:.3f}R"
        )
    print(f"Report -> {output}\nNo network request or order was made.")


if __name__ == "__main__":
    main()
