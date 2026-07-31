"""Profile cached IG quote costs without making a network request.

Fetch or audit a bounded IG window first, then point this script at the saved
parquet. It never authenticates, consumes an allowance, or places an order.
"""

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
    parser.add_argument(
        "--input",
        type=Path,
        help="cached IG parquet; defaults to the config's IG granularity path",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        help="optional execution ledger whose entry timestamps overlap the quote file",
    )
    parser.add_argument(
        "--stop-distance-pct",
        type=float,
        help="optional representative entry-to-stop fraction for spread-in-R",
    )
    parser.add_argument(
        "--slippage-bp-per-side",
        type=float,
        default=0.5,
        help="explicit adverse slippage assumption in basis points per side",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.input or cfg.raw_path("ig", granularity=args.granularity)
    if not source.exists():
        raise SystemExit(
            f"Missing cached IG quotes: {source}\n"
            "Fetch a bounded window after configuring the confirmed demo epic, "
            "or pass --input with an existing canonical IG parquet."
        )
    candles = validate_candles(pd.read_parquet(source), "ig-cost-profile")
    ledger = pd.read_parquet(args.ledger) if args.ledger else None
    report = measure_quote_costs(
        candles,
        slippage_pct_per_side=args.slippage_bp_per_side / 10_000,
        representative_stop_pct=args.stop_distance_pct,
        candidate_ledger=ledger,
    )
    output = args.output or (
        cfg.reports_dir / f"{cfg.asset}_{args.granularity.lower()}_ig_cost_profile.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.report_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    overall = report.overall
    print(
        f"IG {args.granularity} cost profile: {overall.observations} bars, "
        f"median {overall.median_bp:.2f} bp, p95 {overall.p95_bp:.2f} bp, "
        f"p99 {overall.p99_bp:.2f} bp"
    )
    if report.median_cost_r is not None:
        print(
            f"Representative stop: median {report.median_cost_r:.3f}R, "
            f"p95 {report.p95_cost_r:.3f}R"
        )
    widest = sorted(
        report.by_london_slot.items(),
        key=lambda item: item[1].p95_bp,
        reverse=True,
    )[:5]
    print("Widest London slots by p95:")
    for slot, stats in widest:
        print(
            f"  {slot}: median {stats.median_bp:.2f} bp, "
            f"p95 {stats.p95_bp:.2f} bp (n={stats.observations})"
        )
    if report.candidate:
        item = report.candidate
        print(
            f"Candidate {item.strategy}: matched {item.matched_trades}/"
            f"{item.candidate_trades}, estimated PF {item.estimated_profit_factor:.3f}, "
            f"mean net {item.estimated_mean_net_bp:+.2f} bp, gross break-even "
            f"cost {item.gross_break_even_cost_bp:.2f} bp"
        )
    print(f"Report -> {output}\nNo network request or order was made.")


if __name__ == "__main__":
    main()
