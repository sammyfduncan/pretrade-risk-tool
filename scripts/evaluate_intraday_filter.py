"""Run one frozen intraday ML filter-retention gate on development data.

Only the two registered leads are accepted:

* P1 + logistic
* P2 + LightGBM

The procedure is fixed in ``docs/archive/intraday_filter_gate.md``. It cannot select a
threshold from outer-test outcomes and never reads the confirmation period.
"""

import argparse
import json

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.intraday_filter_gate import evaluate_filter_gate
from commodity_ml.intraday.ml_dataset import load_candidate_bundle

REGISTERED_MODELS = {"P1": "logistic", "P2": "lightgbm"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--strategy", choices=["P1", "P2"], required=True)
    parser.add_argument(
        "--model",
        choices=["logistic", "lightgbm"],
        required=True,
    )
    args = parser.parse_args()
    expected_model = REGISTERED_MODELS[args.strategy]
    if args.model != expected_model:
        raise SystemExit(
            f"Unregistered filter: {args.strategy}+{args.model}. "
            f"The frozen choice is {args.strategy}+{expected_model}."
        )

    cfg = load_config(args.config)
    bundle = load_candidate_bundle(
        cfg.intraday_candidate_path(args.strategy),
        expected_strategy=args.strategy,
        expected_source="dukascopy",
        expected_cost_pct_per_side=cfg.backtest.fee_pct,
    )
    dataset = bundle.dataset.sort_index().copy()
    cutoff = cfg.holdout_ts
    if cutoff is not None and (
        (dataset.index >= cutoff).any()
        or (pd.to_datetime(dataset["exit_time"], utc=True) >= cutoff).any()
    ):
        raise SystemExit("Candidate bundle crosses the confirmation fence")

    executed = bundle.ledger.loc[bundle.ledger["status"] == "executed"].copy()
    executed.index = pd.DatetimeIndex(
        pd.to_datetime(executed["signal_time"], utc=True),
        name="time",
    )
    dataset["gross_return"] = executed.loc[dataset.index, "gross_return"].astype(float)
    result = evaluate_filter_gate(
        dataset,
        bundle.feature_names,
        strategy=args.strategy,
        model_kind=args.model,
        cost_pct_per_side=cfg.backtest.fee_pct,
    )

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.asset}_{args.strategy.lower()}_{args.model}_filter_gate"
    report_path = cfg.reports_dir / f"{stem}.json"
    predictions_path = cfg.reports_dir / f"{stem}_predictions.parquet"
    report_path.write_text(
        json.dumps(result.report_dict(), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    result.predictions.to_parquet(predictions_path)

    print(
        f"{cfg.asset} {args.strategy}+{args.model}: "
        f"{'PASS' if result.passed else 'FAIL'}"
    )
    for fold in result.folds:
        print(
            f"  fold {fold.fold}: fit={fold.fit_rows}, "
            f"reference={fold.reference_rows}, test={fold.test_rows}, "
            f"take={fold.filtered_rows} ({fold.retained_fraction:.1%}), "
            f"AUC={fold.auc:.3f}, "
            f"PF bare={fold.bare.profit_factor:.3f} -> "
            f"filter={fold.filtered.profit_factor:.3f}"
        )
    base_bare = result.bare_by_cost[1.0]
    base_filtered = result.filtered_by_cost[1.0]
    stress = result.filtered_by_cost[2.0]
    print(
        f"\n  pooled: trades {base_bare.trades} -> {base_filtered.trades}, "
        f"PF {base_bare.profit_factor:.3f} -> "
        f"{base_filtered.profit_factor:.3f}, "
        f"mean R {base_bare.mean_r:+.3f} -> {base_filtered.mean_r:+.3f}, "
        f"maxDD {base_bare.max_drawdown:.1%} -> "
        f"{base_filtered.max_drawdown:.1%}"
    )
    print(
        f"  PF-improved folds: {result.improved_folds}/5; "
        f"filtered PF at 2x cost: {stress.profit_factor:.3f}"
    )
    if result.failure_reasons:
        print("  failures:")
        for reason in result.failure_reasons:
            print(f"    - {reason}")
    print(
        f"\n  report -> {report_path}\n"
        f"  predictions -> {predictions_path}\n"
        "Confirmation data was not read. No deployment threshold was created."
    )


if __name__ == "__main__":
    main()
