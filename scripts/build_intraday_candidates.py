"""Generate and label one frozen bare-rule candidate family on development data.

This consumes the causal base from ``build_intraday_data.py`` and writes a
candidate-only ML artifact. It never reads or includes the confirmation window.
"""

import argparse

import numpy as np
import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.intraday.execution import IntradayExecutionConfig
from commodity_ml.intraday.ml_dataset import COMMON_FEATURES, build_candidate_dataset
from commodity_ml.intraday.strategies import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--strategy", choices=["P1", "P2", "P3"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.intraday_base_path.exists():
        raise SystemExit(
            f"Missing {cfg.intraday_base_path}\n"
            "Run scripts/build_intraday_data.py first."
        )

    frame = pd.read_parquet(cfg.intraday_base_path)
    cutoff = cfg.holdout_ts
    if cutoff is not None and (
        (frame.index >= cutoff).any()
        or (pd.DatetimeIndex(frame["decision_time"]) >= cutoff).any()
    ):
        raise SystemExit(
            "Intraday base crosses the confirmation fence. Rebuild it with the "
            "current build_intraday_data.py before generating outcomes."
        )

    finite_common = np.isfinite(frame[COMMON_FEATURES].astype(float)).all(axis=1)
    if not finite_common.any():
        raise SystemExit("No rows survive the deterministic intraday feature warmup")
    first_ready = finite_common[finite_common].index[0]
    frame = frame.loc[first_ready:]

    candidates = generate_candidates(frame, args.strategy)
    result = build_candidate_dataset(
        frame,
        candidates,
        strategy=args.strategy,
        execution_config=IntradayExecutionConfig(
            cost_pct_per_side=cfg.backtest.fee_pct,
        ),
    )
    output = cfg.intraday_candidate_path(args.strategy)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.dataset.to_parquet(output)
    ledger_path = output.with_name(output.stem + "_ledger.parquet")
    result.ledger.to_parquet(ledger_path)

    rejected = int((result.ledger["status"] == "rejected").sum()) if len(result.ledger) else 0
    print(
        f"{args.strategy}: {len(candidates):,} causal signals, "
        f"{len(result.dataset):,} executed labels, {rejected:,} reason-coded rejections\n"
        f"  features: {len(result.feature_names)} candidate-only inputs\n"
        f"  dataset -> {output}\n"
        f"  ledger  -> {ledger_path}\n"
        "No model was trained and no confirmation rows were read."
    )


if __name__ == "__main__":
    main()
