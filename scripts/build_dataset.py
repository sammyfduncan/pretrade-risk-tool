"""Build the modeling dataset: features + labels from raw candles.

Usage:
    python scripts/build_dataset.py --config config/gold.yaml

Reads data/raw/, writes data/processed/<asset>_dataset.parquet and prints a
summary — check the class balance line: if it drifts far from ~25-45%
positive, revisit label.atr_mult before training.
"""

import argparse

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.features.technical import build_features
from commodity_ml.labels import make_labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    raw = pd.read_parquet(cfg.raw_path)
    print(f"Loaded {len(raw):,} raw candles ({raw.index[0]} -> {raw.index[-1]})")

    features, feature_names = build_features(raw)
    labels = make_labels(raw, cfg.label)

    dataset = features.join(labels).join(raw["close"])
    n_before = len(dataset)
    dataset = dataset.dropna()
    print(f"Dropped {n_before - len(dataset):,} rows (indicator warmup + unlabeled tail)")

    cfg.dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(cfg.dataset_path)

    print(f"\nDataset: {len(dataset):,} rows x {len(feature_names)} features")
    print(f"  range:         {dataset.index[0]} -> {dataset.index[-1]}")
    print(f"  label mode:    {cfg.label.mode} (horizon={cfg.label.horizon_bars} bars, "
          f"atr_mult={cfg.label.atr_mult})")
    print(f"  class balance: {dataset['label'].mean():.1%} positive")
    print(f"  features:      {', '.join(feature_names)}")
    print(f"  -> {cfg.dataset_path}")


if __name__ == "__main__":
    main()
