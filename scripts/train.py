"""Walk-forward train the model, report per-fold quality, save artifacts.

Usage:
    python scripts/train.py --config config/gold.yaml [--baseline]

--baseline additionally runs the LogisticRegression baseline on identical
folds, to check the boosted model is earning its complexity.

Outputs:
    models/<asset>_lightgbm.joblib          final model + metadata
    data/processed/<asset>_oof.parquet      out-of-fold predictions (backtest input)
"""

import argparse

import joblib
import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.ml_metrics import fold_metrics_table, overall_metrics
from commodity_ml.labels import LABEL_COLUMNS
from commodity_ml.modeling.walkforward import fit_final_model, walk_forward


def report(name: str, y: pd.Series, oof: pd.Series, folds) -> dict:
    print(f"\n=== {name}: per-fold out-of-fold metrics ===")
    print(fold_metrics_table(y, oof, folds).to_string())
    overall = overall_metrics(y, oof)
    print(f"\n=== {name}: pooled OOF ===")
    print(f"  n={overall['n']:,}  base_rate={overall['base_rate']:.3f}")
    print(f"  AUC   = {overall['auc']:.4f}   (0.5 = coin flip)")
    print(f"  Brier = {overall['brier']:.4f} (no-skill baseline: "
          f"{overall['brier_baseline']:.4f}, lower is better)")
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--baseline", action="store_true",
                        help="also run the LogisticRegression baseline for comparison")
    args = parser.parse_args()
    cfg = load_config(args.config)

    dataset = pd.read_parquet(cfg.dataset_path)
    # exit_bars / barrier_pct describe the label's own outcome — they are
    # backtest inputs, never model inputs. Feeding them in would be lookahead.
    feature_names = [c for c in dataset.columns if c not in LABEL_COLUMNS + ["close"]]
    X, y = dataset[feature_names], dataset["label"]
    print(f"Training on {len(X):,} bars, {len(feature_names)} features, "
          f"{cfg.cv.n_splits} walk-forward folds (gap={cfg.cv.gap_bars} bars)")

    wf = walk_forward(X, y, n_splits=cfg.cv.n_splits, gap=cfg.cv.gap_bars,
                      model_kind=cfg.model.type)
    overall = report(cfg.model.type, y, wf.oof_proba, wf.folds)

    if args.baseline:
        base_wf = walk_forward(X, y, n_splits=cfg.cv.n_splits, gap=cfg.cv.gap_bars,
                               model_kind="logistic")
        base_overall = report("logistic baseline", y, base_wf.oof_proba, base_wf.folds)
        delta = overall["auc"] - base_overall["auc"]
        print(f"\nAUC edge of {cfg.model.type} over baseline: {delta:+.4f}")

    # Feature importance from the last fold's model (most training data)
    last_model = wf.folds[-1].model
    if hasattr(last_model, "feature_importances_"):
        imp = (pd.Series(last_model.feature_importances_, index=feature_names)
               .sort_values(ascending=False))
        print("\n=== Top features (last fold, split importance) ===")
        print(imp.head(10).to_string())

    # OOF predictions -> the only backtest input. barrier_pct rides along so the
    # backtest can place each trade's stops at the same width the label used.
    oof_df = pd.DataFrame(
        {
            "proba": wf.oof_proba,
            "label": y.loc[wf.oof_proba.index],
            "barrier_pct": dataset["barrier_pct"].loc[wf.oof_proba.index],
        }
    )
    oof_df.to_parquet(cfg.oof_path)

    final_model = fit_final_model(X, y, cfg.model.type)
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "label_config": cfg.label,
            "symbol": cfg.symbol,
            "granularity": cfg.granularity,
            "trained_on_source": cfg.data.source,
            "trained_through": dataset.index[-1],
            "oof_auc": overall["auc"],
        },
        cfg.model_path,
    )
    print(f"\nSaved OOF predictions -> {cfg.oof_path}")
    print(f"Saved final model     -> {cfg.model_path}")


if __name__ == "__main__":
    main()
