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

import numpy as np
import pandas as pd

from commodity_ml.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    duka_path, ig_path = cfg.raw_path("dukascopy"), cfg.raw_path("ig")
    for path, source in ((duka_path, "dukascopy"), (ig_path, "ig")):
        if not path.exists():
            raise SystemExit(
                f"Missing {source} data at {path}\n"
                f"Fetch it first: python scripts/fetch_data.py --config {args.config} "
                f"--source {source}"
            )

    duka = pd.read_parquet(duka_path)["close"]
    ig = pd.read_parquet(ig_path)["close"]

    overlap = duka.index.intersection(ig.index)
    if len(overlap) < 50:
        raise SystemExit(
            f"Only {len(overlap)} timestamps overlap — too few to compare.\n"
            "Either the IG window doesn't overlap the Dukascopy range, or the two "
            "sources label bars differently (check timezone/session conventions)."
        )

    ig_window = ig.loc[(ig.index >= overlap[0]) & (ig.index <= overlap[-1])]
    coverage = len(overlap) / len(ig_window)

    d, i = duka.loc[overlap], ig.loc[overlap]
    level_diff = i - d
    ret_d, ret_i = np.log(d).diff().dropna(), np.log(i).diff().dropna()
    ret_corr = float(ret_d.corr(ret_i))
    tracking_error = float((ret_i - ret_d).std())

    print(f"Overlap window: {overlap[0]:%Y-%m-%d %H:%M} -> {overlap[-1]:%Y-%m-%d %H:%M}")
    print(f"  matched bars:        {len(overlap):,} ({coverage:.1%} of IG bars in range)")
    print("\nPrice level (expected to differ by ~half a spread — bid vs mid):")
    print(f"  median IG - Dukascopy: {level_diff.median():+.3f} "
          f"({level_diff.median() / d.median():+.4%})")
    print(f"  std of difference:     {level_diff.std():.3f}")
    print("\nBar-to-bar returns (this is what the model actually learns from):")
    print(f"  correlation:           {ret_corr:.4f}")
    print(f"  tracking error (std of return difference): {tracking_error:.6f}")

    print("\nVerdict:")
    if ret_corr >= 0.99 and coverage >= 0.95:
        print("  PASS — the feeds track each other closely. Training on Dukascopy and")
        print("  executing on IG is sound for this asset.")
    elif ret_corr >= 0.95:
        print("  CAUTION — returns broadly agree but not tightly. Check whether the gap")
        print("  concentrates in specific sessions (rollover, weekend opens) before trusting")
        print("  a model trained on one and run on the other.")
    else:
        print("  FAIL — these feeds disagree materially. Do not train on one and trade the")
        print("  other. Investigate bar alignment/timezone, or confirm the IG epic is spot")
        print("  gold rather than a futures or mini contract.")

    if coverage < 0.95:
        print(f"\n  NOTE: only {coverage:.1%} of IG bars found a Dukascopy match. Mismatched")
        print("  bar timestamps usually mean differing session or DST conventions.")


if __name__ == "__main__":
    main()
