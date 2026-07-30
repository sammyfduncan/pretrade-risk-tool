"""Check that Dukascopy training data and IG execution data agree.

The model is trained on Dukascopy history but will trade on IG prices. That is
standard practice, but it is only safe if the two feeds describe the same
market closely enough that patterns learned on one hold on the other. This
script measures that instead of assuming it.

What matters is NOT that prices match exactly — they won't. Different
liquidity providers quote slightly differently, and Dukascopy is configured
as bid while IG is mid, so expect a small roughly-constant offset. What
matters is that **bar-to-bar returns** match, because every model feature is
built from returns, not price levels.

Usage:
    python scripts/fetch_data.py --config config/gold.yaml                       # dukascopy
    python scripts/fetch_data.py --config config/gold.yaml --source ig \
        --start 2026-05-01                                                        # recent IG window
    python scripts/compare_sources.py --config config/gold.yaml
"""

import argparse
import json

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.data.parity import compare_price_sources

BAR_DURATIONS = {
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D": pd.Timedelta(days=1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument(
        "--granularity",
        choices=sorted(BAR_DURATIONS),
        help="override the historical config granularity, e.g. M30",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()
    cfg = load_config(args.config)
    granularity = args.granularity or cfg.granularity

    duka_path = cfg.raw_path("dukascopy", granularity=granularity)
    ig_path = cfg.raw_path("ig", granularity=granularity)
    for path, source in ((duka_path, "dukascopy"), (ig_path, "ig")):
        if not path.exists():
            raise SystemExit(
                f"Missing {source} data at {path}\n"
                f"Fetch it first: python scripts/fetch_data.py --config {args.config} "
                f"--source {source}"
            )

    duka = pd.read_parquet(duka_path)
    ig = pd.read_parquet(ig_path)
    try:
        report = compare_price_sources(
            duka,
            ig,
            bar_duration=BAR_DURATIONS[granularity],
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        return

    print(f"Granularity: {granularity}")
    print(f"  matched bars:             {report.matched_bars:,}")
    print(f"  contiguous return pairs:  {report.contiguous_return_pairs:,}")
    print(f"  IG timestamp coverage:    {report.ig_coverage:.1%}")
    print(f"  Duka timestamp coverage:  {report.dukascopy_coverage:.1%}")
    print("\nPrice level (expected to differ by ~half a spread — bid vs mid):")
    print(
        f"  median IG - Dukascopy:    {report.median_level_difference:+.3f} "
        f"({report.median_level_difference_pct:+.4%})"
    )
    print("\nBar-to-bar returns (this is what the model actually learns from):")
    print(f"  exact-alignment correlation: {report.return_correlation:.4f}")
    print(f"  tracking error:             {report.tracking_error:.6f}")
    print(
        f"  best timestamp shift:       {report.best_shift_bars:+d} bars "
        f"(corr {report.best_shift_correlation:.4f})"
    )

    print(f"\nVerdict: {report.verdict}")
    if report.verdict == "PASS":
        print("  PASS — the feeds track each other closely. Training on Dukascopy and")
        print("  executing on IG is sound for this asset.")
    elif report.verdict == "CAUTION":
        print("  CAUTION — returns broadly agree but not tightly. Check whether the gap")
        print("  concentrates in specific sessions (rollover, weekend opens) before trusting")
        print("  a model trained on one and run on the other.")
    else:
        print("  FAIL — these feeds disagree materially. Do not train on one and trade the")
        print("  other. Investigate bar alignment/timezone, or confirm the IG epic is spot")
        print("  gold rather than a futures or mini contract.")
    if report.reasons:
        print(f"  Reasons: {', '.join(report.reasons)}")


if __name__ == "__main__":
    main()
