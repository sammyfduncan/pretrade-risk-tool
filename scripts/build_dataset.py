"""Build the modeling dataset: features + labels from raw candles.

Usage:
    python scripts/build_dataset.py --config config/gold.yaml

Reads data/raw/, writes data/processed/<asset>_dataset.parquet and prints a
summary. Two lines to check:
  * class balance — should sit near 50% for triple_barrier (symmetric barriers);
    a large skew means the barriers or horizon are lopsided.
  * macro coverage — macro features are published with a lag, so a few early
    rows drop out. A large loss means the publication lags are misconfigured.
"""

import argparse

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.dataset import METADATA_COLUMNS, feature_columns
from commodity_ml.features.macro import fetch_macro
from commodity_ml.features.pipeline import (
    build_feature_frame,
    load_partner_candles,
    partner_symbols,
)
from commodity_ml.labels import make_labels
from commodity_ml.primary import SIGNAL_COLUMN, make_primary_signal


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw_path = cfg.raw_path()
    if not raw_path.exists():
        raise SystemExit(
            f"No raw candles at {raw_path}\n"
            f"Fetch them first: python scripts/fetch_data.py --config {args.config}"
        )
    raw = pd.read_parquet(raw_path)
    print(f"Loaded {len(raw):,} raw candles from {cfg.data.source} "
          f"({raw.index[0]} -> {raw.index[-1]})")

    macro_raw = None
    if cfg.features.macro:
        print("  macro features (FRED, no API key needed):")
        macro_raw = fetch_macro(start=cfg.data.start, end=cfg.data.end)

    partners = load_partner_candles(cfg)
    for symbol in partner_symbols(cfg):
        print(f"  cross-asset leg {symbol}: {len(partners[symbol]):,} candles")

    # One shared assembly path with scripts/predict.py — see features/pipeline.py.
    features, feature_names = build_feature_frame(
        cfg, raw, macro=macro_raw, partners=partners
    )
    print(f"  features assembled: {len(feature_names)}")

    labels = make_labels(raw, cfg.label)
    # Stored as metadata, never as a feature: in meta mode training is filtered
    # to these bars, so the column would be constant and useless anyway.
    primary = make_primary_signal(raw, cfg.primary)

    dataset = pd.concat([features, labels, primary, raw["close"]], axis=1)
    n_before = len(dataset)
    dataset = dataset.dropna()
    print(f"\nDropped {n_before - len(dataset):,} rows "
          "(indicator warmup, macro publication lag, unlabeled tail)")

    cfg.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(cfg.dataset_path)

    print(f"\nDataset: {len(dataset):,} rows x {len(feature_names)} features")
    print(f"  range:         {dataset.index[0]} -> {dataset.index[-1]}")
    print(f"  label mode:    {cfg.label.mode} (horizon={cfg.label.horizon_bars} bars, "
          f"atr_mult={cfg.label.atr_mult})")
    print(f"  class balance: {dataset['label'].mean():.1%} positive")
    if cfg.label.mode == "triple_barrier":
        print(f"  median bars held before a barrier is touched: "
              f"{dataset['exit_bars'].median():.0f} of {cfg.label.horizon_bars}")
        timed_out = (dataset["exit_bars"] >= cfg.label.horizon_bars).mean()
        print(f"  trades reaching the time limit without touching a barrier: {timed_out:.1%}")
    fires = dataset[SIGNAL_COLUMN].mean()
    print(f"  primary rule ({cfg.primary.rule}): fires on {fires:.1%} of bars")
    if cfg.label.mode == "meta":
        meta_rows = dataset[dataset[SIGNAL_COLUMN]]
        print(f"  meta-labeling trains on those {len(meta_rows):,} bars only; "
              f"rule win rate there = {meta_rows['label'].mean():.1%}")
    print(f"  -> {cfg.dataset_path}")

    # Both directions of the contract, checked where the file is written:
    # nothing unaccounted for, and nothing train.py will refuse to read.
    unused = [c for c in dataset.columns if c not in feature_names + METADATA_COLUMNS]
    if unused:
        raise SystemExit(f"Unexpected columns in dataset (would leak into training): {unused}")
    if sorted(feature_columns(dataset)) != sorted(feature_names):
        raise SystemExit(
            "Feature columns as written disagree with what train.py would infer "
            "from this file — the dataset contract in commodity_ml/dataset.py is out of sync."
        )


if __name__ == "__main__":
    main()
