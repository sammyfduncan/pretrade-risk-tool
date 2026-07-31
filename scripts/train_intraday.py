"""Walk-forward candidate-only ML training for one frozen primary rule.

This is a research trainer, not a deployment command. It uses only executed
primary-rule candidates, purges any training label whose trade had not exited
before the test block, and saves raw OOF scores. No entry threshold is selected
here; that requires a separate pre-registered retention decision.
"""

import argparse

import joblib
import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.ml_metrics import overall_metrics
from commodity_ml.intraday.ml_dataset import load_candidate_bundle
from commodity_ml.modeling.walkforward import fit_final_model, walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--strategy", choices=["P1", "P2", "P3"], required=True)
    parser.add_argument(
        "--model",
        choices=["logistic", "lightgbm", "both"],
        default="both",
    )
    parser.add_argument("--min-candidates", type=int, default=100)
    args = parser.parse_args()
    cfg = load_config(args.config)
    path = cfg.intraday_candidate_path(args.strategy)
    if not path.exists():
        raise SystemExit(
            f"Missing {path}\nRun scripts/build_intraday_candidates.py first."
        )

    bundle = load_candidate_bundle(
        path,
        expected_strategy=args.strategy,
        expected_source="dukascopy",
        expected_cost_pct_per_side=cfg.backtest.fee_pct,
    )
    dataset = bundle.dataset.sort_index()
    if len(dataset) < args.min_candidates:
        raise SystemExit(
            f"Only {len(dataset)} executed candidates; at least "
            f"{args.min_candidates} are required before ML training."
        )
    if dataset["label"].nunique() != 2:
        raise SystemExit("Candidate labels contain only one class; classification is undefined")
    if cfg.holdout_ts is not None and (
        (dataset.index >= cfg.holdout_ts).any()
        or (pd.to_datetime(dataset["exit_time"], utc=True) >= cfg.holdout_ts).any()
    ):
        raise SystemExit(
            "Candidate dataset crosses the confirmation fence. Rebuild from the "
            "development-only intraday base."
        )

    feature_names = list(bundle.feature_names)
    X = dataset[feature_names]
    y = dataset["label"].astype("int8")
    exit_time = pd.to_datetime(dataset["exit_time"], utc=True)
    model_kinds = (
        ["logistic", "lightgbm"] if args.model == "both" else [args.model]
    )

    print(
        f"{cfg.asset} {args.strategy}: {len(dataset):,} executed candidates, "
        f"{len(feature_names)} pre-defined features, "
        f"{cfg.cv.n_splits} expanding walk-forward folds\n"
        "Purging rule: every training trade must exit before its test block starts."
    )
    scores: dict[str, float] = {}
    for model_kind in model_kinds:
        result = walk_forward(
            X,
            y,
            n_splits=cfg.cv.n_splits,
            gap=0,
            model_kind=model_kind,
            exit_time=exit_time,
            sample_weighting="uniqueness",
            calibration=cfg.model.calibration,
            purge_by_exit_time=True,
        )
        metrics = overall_metrics(y.loc[result.oof_proba.index], result.oof_proba)
        scores[model_kind] = metrics["auc"]
        print(
            f"\n{model_kind}: OOF n={metrics['n']:,}, "
            f"AUC={metrics['auc']:.4f}, base rate={metrics['base_rate']:.3f}"
        )
        for fold in result.folds:
            print(
                f"  fold {fold.fold}: train {fold.n_train:,} through "
                f"{fold.train_end}, test {fold.n_test:,} "
                f"{fold.test_start} -> {fold.test_end}"
            )

        oof_path = path.with_name(
            path.stem + f"_{model_kind}_oof.parquet"
        )
        pd.DataFrame(
            {
                "score": result.oof_proba,
                "calibrated_probability": result.oof_calibrated,
                "label": y.loc[result.oof_proba.index],
            }
        ).to_parquet(oof_path)

        final_model = fit_final_model(
            X,
            y,
            model_kind,
            exit_time=exit_time,
            sample_weighting="uniqueness",
            calibration=cfg.model.calibration,
            gap=0,
            purge_by_exit_time=True,
        )
        model_path = cfg.intraday_model_path(args.strategy, model_kind)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": final_model,
                "program": "intraday_v1",
                "strategy": args.strategy,
                "feature_names": feature_names,
                "symbol": cfg.symbol,
                "context_granularity": "H1",
                "decision_granularity": "M30",
                "trained_on_source": "dukascopy",
                "trained_through": dataset.index[-1],
                "oof_auc": metrics["auc"],
                "oof_scores": result.oof_proba,
                "deployment_threshold": None,
                "deployment_status": "research_only_threshold_not_registered",
            },
            model_path,
        )
        print(f"  OOF -> {oof_path}\n  model -> {model_path}")

    if len(scores) == 2:
        print(
            f"\nLightGBM minus logistic OOF AUC: "
            f"{scores['lightgbm'] - scores['logistic']:+.4f}"
        )
    print(
        "\nNo ML filter threshold was selected and this artifact cannot yet "
        "produce a paper-trading decision."
    )


if __name__ == "__main__":
    main()
