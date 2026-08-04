"""Download historical candles and cache them as parquet.

Usage:
    python scripts/fetch_data.py --config config/gold.yaml
    python scripts/fetch_data.py --config config/gold.yaml --source oanda --start 2026-08-01

The default source is Dukascopy for deep research history. Use --source oanda
only for a bounded recent broker-parity window and executable bid/ask costs.
"""

import argparse
from dataclasses import replace

from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data.schema import describe_candles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--source", choices=["dukascopy", "oanda"],
                        help="override data.source from the config")
    parser.add_argument("--start", help="override data.start (YYYY-MM-DD)")
    parser.add_argument("--end", help="override data.end (YYYY-MM-DD)")
    parser.add_argument(
        "--granularity",
        choices=["M5", "M15", "M30", "H1", "H4", "D"],
        help="override the frozen config timeframe for an intraday data artifact",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    data_cfg = cfg.data
    if args.source:
        data_cfg = replace(data_cfg, source=args.source)
    if args.start:
        data_cfg = replace(data_cfg, start=args.start)
    if args.end:
        data_cfg = replace(data_cfg, end=args.end)

    if data_cfg.source == "dukascopy":
        from commodity_ml.data import dukascopy as provider
    else:
        from commodity_ml.data import oanda as provider

    granularity = args.granularity or cfg.granularity
    df = provider.fetch_candles(data_cfg, granularity)

    out_path = cfg.raw_path(data_cfg.source, granularity=granularity)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)

    print(f"\n{data_cfg.source}: {describe_candles(df)}")
    print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
