"""Walk-forward train the model, report per-fold quality, save artifacts.

Usage:
    python scripts/train.py --config config/gold.yaml [--baseline]
    python scripts/train.py --config config/gold.yaml --exclude-family cot,gdelt

--baseline additionally runs the LogisticRegression baseline on identical
folds, to check the boosted model is earning its complexity.

--exclude-family drops an alt-data family's COLUMNS while keeping the same
dataset ROWS. This is the matched-row ablation device from
docs/archive/step2_design.md §6: toggling a family in the yaml would change which rows
survive dropna and therefore the fold boundaries, so yaml-level ablations
partly measure sample composition. Excluding columns on one both-families-on
dataset keeps rows and folds identical, so AUC deltas are attributable to the
columns alone. EVALUATION-ONLY: the artifact it saves is refused by the live
path — a retained family is expressed by yaml flags plus a retrain.

Outputs:
    models/<asset>_lightgbm.joblib          final model + metadata
    data/processed/<asset>_oof.parquet      out-of-fold predictions (backtest input)
"""

import argparse

import joblib
import pandas as pd

from commodity_ml.config import AssetConfig, load_config
from commodity_ml.dataset import feature_columns
from commodity_ml.features.cot import cot_feature_names
from commodity_ml.features.gdelt import gdelt_feature_names
from commodity_ml.evaluation.ml_metrics import fold_metrics_table, overall_metrics
from commodity_ml.modeling.calibration import IdentityCalibrator
from commodity_ml.modeling.walkforward import fit_final_model, walk_forward
from commodity_ml.primary import SIGNAL_COLUMN


def select_slice(cfg: AssetConfig, dataset: pd.DataFrame, use_holdout: bool) -> pd.DataFrame:
    """Restrict to the development period unless the holdout is explicitly asked for."""
    if cfg.holdout_ts is None:
        print("No holdout configured — every result below is in-sample for design choices.")
        return dataset
    if use_holdout:
        print(f"*** USING THE HOLDOUT ({cfg.data.holdout_start} onward). This is a "
              "single-use out-of-sample test — after this, it is development data. ***")
        return dataset
    development = cfg.development_slice(dataset)
    print(f"Holdout fenced off: {len(dataset) - len(development):,} rows from "
          f"{cfg.data.holdout_start} excluded (use --use-holdout to include).")
    return development


def report(name: str, y: pd.Series, wf, calibration: str) -> dict:
    """AUC on raw scores (ranking), Brier on calibrated ones (interpretation).

    Reporting Brier on the raw score would be measuring a number nobody uses,
    and AUC on the calibrated one lets a calibrator's ties move the metric.
    """
    oof, folds = wf.oof_proba, wf.folds
    print(f"\n=== {name}: per-fold out-of-fold metrics ===")
    print(fold_metrics_table(y, oof, folds).to_string())
    overall = overall_metrics(y, oof)
    calibrated = overall_metrics(y, wf.oof_calibrated)

    print(f"\n=== {name}: pooled OOF ===")
    print(f"  n={overall['n']:,}  base_rate={overall['base_rate']:.3f}")
    print(f"  AUC   = {overall['auc']:.4f}   (0.5 = coin flip; ranking, so raw scores)")
    print(f"  Brier = {calibrated['brier']:.4f} (no-skill baseline: "
          f"{overall['brier_baseline']:.4f}, lower is better)")
    if calibration != "none":
        print(f"          raw was {overall['brier']:.4f} before calibration; "
              f"{calibration} closed "
              f"{1 - (calibrated['brier'] - overall['brier_baseline']) / max(overall['brier'] - overall['brier_baseline'], 1e-9):.0%} "
              "of the gap to no-skill")
        skipped = sum(1 for f in folds if isinstance(f.calibrator, IdentityCalibrator))
        if skipped:
            print(f"          {skipped} of {len(folds)} folds left uncalibrated "
                  "(too few rows, or the fit would have inverted the model)")

    effective = [f.effective_n_train for f in folds if f.effective_n_train is not None]
    if effective:
        last, n_train = effective[-1], folds[-1].n_train
        print(f"  effective training sample (last fold): {last:,.0f} of {n_train:,} rows "
              f"({last / n_train:.0%}) — overlapping labels are not independent")
    overall["brier_calibrated"] = calibrated["brier"]
    return overall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--baseline", action="store_true",
                        help="also run the LogisticRegression baseline for comparison")
    parser.add_argument("--use-holdout", action="store_true",
                        help="INCLUDE the fenced-off final period. Single-use: once you "
                             "have seen it, it is no longer an out-of-sample test.")
    parser.add_argument("--exclude-family", default="",
                        help="comma-separated alt-data families (cot,gdelt) to drop as "
                             "COLUMNS while keeping identical rows/folds — the "
                             "matched-row ablation for step-2 evaluation ONLY")
    args = parser.parse_args()
    cfg = load_config(args.config)

    excluded_families = [f for f in args.exclude_family.split(",") if f]
    family_columns = {"cot": cot_feature_names(), "gdelt": gdelt_feature_names()}
    unknown = [f for f in excluded_families if f not in family_columns]
    if unknown:
        raise SystemExit(f"--exclude-family only knows {sorted(family_columns)}, got {unknown}")

    dataset = pd.read_parquet(cfg.dataset_path)
    dataset = select_slice(cfg, dataset, use_holdout=args.use_holdout)

    if cfg.label.mode == "meta":
        # Meta-labeling: the model only ever judges trades the primary rule
        # proposes, so it must be trained on exactly those bars.
        before = len(dataset)
        dataset = dataset[dataset[SIGNAL_COLUMN]]
        print(f"Meta-labeling on primary rule '{cfg.primary.rule}': "
              f"{len(dataset):,} of {before:,} bars where the rule wants to be long "
              f"(rule's raw win rate: {dataset['label'].mean():.1%})")

    # exit_bars / barrier_pct describe the label's own outcome, and
    # primary_signal is the rule being judged — all backtest metadata, never
    # model inputs. Feeding them in would be lookahead. dataset.feature_columns
    # owns that split and refuses a dataset that does not match this code.
    feature_names = feature_columns(dataset)
    if excluded_families:
        dropped = [c for f in excluded_families for c in family_columns[f]]
        absent = [c for c in dropped if c not in feature_names]
        if absent:
            raise SystemExit(
                f"Cannot exclude {absent} — the dataset was built without them. "
                "Matched-row ablations need one dataset built with BOTH families "
                "on; rebuild with the flags enabled first."
            )
        feature_names = [c for c in feature_names if c not in dropped]
        print(f"*** MATCHED-ROW ABLATION: excluding {excluded_families} "
              f"({len(dropped)} columns; rows and folds unchanged). The saved "
              "artifact is evaluation-only and the live path will refuse it. ***")
    X, y = dataset[feature_names], dataset["label"]
    print(f"Training on {len(X):,} bars, {len(feature_names)} features, "
          f"{cfg.cv.n_splits} walk-forward folds (gap={cfg.cv.gap_bars} bars)")
    print(f"  sample weighting: {cfg.model.sample_weighting}   "
          f"calibration: {cfg.model.calibration}")

    # Both corrections are fitted per fold, inside training data only.
    training_options = dict(
        n_splits=cfg.cv.n_splits,
        gap=cfg.cv.gap_bars,
        exit_time=dataset["exit_time"],
        sample_weighting=cfg.model.sample_weighting,
        calibration=cfg.model.calibration,
    )

    wf = walk_forward(X, y, model_kind=cfg.model.type, **training_options)
    overall = report(cfg.model.type, y, wf, cfg.model.calibration)

    if args.baseline:
        base_wf = walk_forward(X, y, model_kind="logistic", **training_options)
        base_overall = report("logistic baseline", y, base_wf, cfg.model.calibration)
        delta = overall["auc"] - base_overall["auc"]
        print(f"\nAUC edge of {cfg.model.type} over baseline: {delta:+.4f}")

    # Feature importance from the last fold's model (most training data).
    # Calibration is a monotone map applied after the model, so it leaves
    # importances — and AUC — untouched; it only moves the Brier score.
    last_model = wf.folds[-1].model
    if hasattr(last_model, "feature_importances_"):
        imp = (pd.Series(last_model.feature_importances_, index=feature_names)
               .sort_values(ascending=False))
        print("\n=== Top features (last fold, split importance) ===")
        print(imp.head(10).to_string())

    # OOF predictions -> the only backtest input. `proba` is the raw score
    # because entries rank on it; `proba_calibrated` rides along for reporting
    # and for position sizing later. barrier_pct lets the backtest place each
    # trade's stops at the same width the label used.
    oof_df = pd.DataFrame(
        {
            "proba": wf.oof_proba,
            "proba_calibrated": wf.oof_calibrated,
            "label": y.loc[wf.oof_proba.index],
            "barrier_pct": dataset["barrier_pct"].loc[wf.oof_proba.index],
        }
    )
    oof_df.to_parquet(cfg.oof_path)

    final_model = fit_final_model(
        X, y, cfg.model.type,
        exit_time=dataset["exit_time"],
        sample_weighting=cfg.model.sample_weighting,
        calibration=cfg.model.calibration,
        gap=cfg.cv.gap_bars,
    )

    # The artifact carries everything the live path needs to reproduce training
    # exactly: the model bundled with its calibrator and feature order, plus
    # the entry threshold. The threshold is a quantile of the *raw* OOF scores,
    # matching what the backtest ranks on — without it, predict.py would have a
    # number and no way to judge whether it is high, since the backtest's
    # expanding quantile means nothing on a single bar.
    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "label_config": cfg.label,
            "primary_config": cfg.primary,
            "symbol": cfg.symbol,
            "granularity": cfg.granularity,
            "trained_on_source": cfg.data.source,
            "trained_through": dataset.index[-1],
            "oof_auc": overall["auc"],
            "oof_proba": wf.oof_proba,
            "entry_quantile": cfg.backtest.entry_quantile,
            "entry_threshold": float(wf.oof_proba.quantile(cfg.backtest.entry_quantile)),
            # Non-empty only under --exclude-family; predict.py refuses such
            # artifacts so an ablation model can never reach the live path.
            "excluded_families": excluded_families,
        },
        cfg.model_path,
    )
    print(f"\nSaved OOF predictions -> {cfg.oof_path}")
    print(f"Saved final model     -> {cfg.model_path}")


if __name__ == "__main__":
    main()
