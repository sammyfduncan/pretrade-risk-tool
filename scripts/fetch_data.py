"""Download historical candles from OANDA and cache them as parquet.

Usage:
    python scripts/fetch_data.py --config config/gold.yaml

Requires OANDA_API_KEY in the environment or in .env (see .env.example).
Re-running overwrites the cached file with a fresh download.
"""

import argparse

from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data.oanda import fetch_candles, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    print(f"Fetching {cfg.instrument} {cfg.granularity} candles from {cfg.history.start} ...")
    df = fetch_candles(
        client=get_client(),
        instrument=cfg.instrument,
        granularity=cfg.granularity,
        start=cfg.history.start,
        end=cfg.history.end,
    )

    cfg.raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.raw_path)
    print(f"\nSaved {len(df):,} candles ({df.index[0]} -> {df.index[-1]})")
    print(f"  -> {cfg.raw_path}")


if __name__ == "__main__":
    main()
