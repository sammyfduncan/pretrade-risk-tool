"""Generate and label one frozen bare-rule candidate family on development data.

This consumes the causal base from ``build_intraday_data.py`` and writes a
candidate-only ML artifact. It never reads or includes the confirmation window.
"""

import argparse

import numpy as np
import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.intraday.execution import IntradayExecutionConfig
from commodity_ml.intraday.event_blackouts import scheduled_entry_blackouts
from commodity_ml.intraday.ml_dataset import (
    COMMON_FEATURES,
    build_candidate_dataset,
    write_candidate_bundle,
)
from commodity_ml.intraday.strategies import generate_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument(
        "--strategy",
        choices=["P1", "P2", "P3", "P1V2", "P2V2", "P3V2"],
        required=True,
    )
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

    # The exact-time CPI/NFP/FOMC tables are verified from 2020 onward. Earlier
    # rows cannot honestly be treated as event-free, so they are excluded.
    event_coverage_start = pd.Timestamp("2020-01-01", tz="UTC")
    frame = frame.loc[frame.index >= event_coverage_start]

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
            entry_blackouts=scheduled_entry_blackouts(
                frame.index.min(),
                pd.DatetimeIndex(frame["decision_time"]).max()
                + pd.Timedelta(minutes=1),
            ),
        ),
    )
    output = cfg.intraday_candidate_path(args.strategy)
    manifest_path = write_candidate_bundle(
        result,
        output,
        strategy=args.strategy,
        source="dukascopy",
        cost_pct_per_side=cfg.backtest.fee_pct,
        confirmation_cutoff=cutoff,
    )
    ledger_path = output.with_name(output.stem + "_ledger.parquet")

    rejected = int((result.ledger["status"] == "rejected").sum()) if len(result.ledger) else 0
    print(
        f"{args.strategy}: {len(candidates):,} causal signals, "
        f"{len(result.dataset):,} executed labels, {rejected:,} reason-coded rejections\n"
        f"  features: {len(result.feature_names)} candidate-only inputs\n"
        f"  dataset -> {output}\n"
        f"  ledger  -> {ledger_path}\n"
        f"  manifest -> {manifest_path}\n"
        "No model was trained and no confirmation rows were read."
    )


if __name__ == "__main__":
    main()
