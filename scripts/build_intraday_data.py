"""Build the causal H1-context/M30-decision base without opening the holdout.

Fetch both granularities first:
    python scripts/fetch_data.py --config config/gold.yaml \
        --source dukascopy --granularity H1
    python scripts/fetch_data.py --config config/gold.yaml \
        --source dukascopy --granularity M30
"""

import argparse

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.intraday.data import build_intraday_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument(
        "--source",
        choices=["dukascopy", "ig"],
        default="dukascopy",
        help=(
            "Dukascopy writes the training base; IG writes a separate recent "
            "diagnostic base and cannot overwrite training data"
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    paths = {
        granularity: cfg.raw_path(args.source, granularity=granularity)
        for granularity in ("H1", "M30")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing raw intraday inputs:\n  "
            + "\n  ".join(missing)
            + "\nFetch H1 and M30 with scripts/fetch_data.py --granularity ..."
        )

    h1 = pd.read_parquet(paths["H1"])
    m30 = pd.read_parquet(paths["M30"])
    cutoff = cfg.holdout_ts
    if cutoff is not None:
        h1 = h1.loc[h1.index < cutoff]
        m30 = m30.loc[m30.index < cutoff]

    frame = build_intraday_frame(h1, m30, source=args.source)
    if cutoff is not None:
        # A 23:30 M30 bar closes exactly at a midnight fence and therefore
        # belongs to the next evaluation period even though its open does not.
        frame = frame.loc[frame["decision_time"] < cutoff]
    if frame.empty:
        raise SystemExit("No development-period intraday rows remain after fencing")

    output = (
        cfg.intraday_base_path
        if args.source == "dukascopy"
        else cfg.intraday_base_path.with_name(
            cfg.intraday_base_path.stem + "_ig_diagnostic.parquet"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output)
    print(
        f"Built {len(frame):,} causal H1/M30 rows from {args.source}\n"
        f"  {frame.index[0]} -> {frame.index[-1]}\n"
        f"  holdout/confirmation remains excluded at {cfg.data.holdout_start}\n"
        f"  -> {output}"
    )


if __name__ == "__main__":
    main()
