"""Candidate-only ML dataset contract for the intraday rebuild."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from commodity_ml.intraday.execution import (
    IntradayExecutionConfig,
    TradeCandidate,
    simulate_candidates,
)

COMMON_FEATURES = [
    "h1_ema20_dist",
    "h1_ema50_dist",
    "h1_ema50_slope3_atr",
    "h1_atr_pct",
    "m30_ema20_dist",
    "m30_atr_pct",
    "m30_bb_position",
    "m30_bb_width_pct",
    "m30_body_pct",
    "m30_range_pct",
    "previous_session_range_atr",
    "previous_session_return",
    "minutes_since_entry_start",
    "minutes_until_last_entry",
]

STRATEGY_FEATURES = {
    "P1": ["opening_range_compression", "setup_distance_atr"],
    "P2": ["setup_distance_atr", "h1_regime_distance_atr"],
    "P3": ["setup_distance_atr", "h1_regime_distance_atr"],
}

METADATA_COLUMNS = [
    "candidate_id",
    "strategy",
    "side",
    "entry_time",
    "exit_time",
    "exit_reason",
    "entry_price",
    "exit_price",
    "initial_r_pct",
    "net_return",
    "r_multiple",
    "label",
]


@dataclass(frozen=True)
class CandidateDatasetResult:
    dataset: pd.DataFrame
    ledger: pd.DataFrame
    feature_names: tuple[str, ...]


def build_candidate_dataset(
    frame: pd.DataFrame,
    candidates: list[TradeCandidate],
    *,
    strategy: str,
    execution_config: IntradayExecutionConfig | None = None,
) -> CandidateDatasetResult:
    """Resolve one bare rule and attach only features known at signal close."""

    if strategy not in STRATEGY_FEATURES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    wrong = [candidate.strategy for candidate in candidates if candidate.strategy != strategy]
    if wrong:
        raise ValueError("All candidates must belong to the requested strategy")

    ledger = simulate_candidates(frame, candidates, execution_config)
    executed = ledger.loc[ledger["status"] == "executed"].copy()
    feature_names = tuple(COMMON_FEATURES + STRATEGY_FEATURES[strategy] + ["is_long"])
    if executed.empty:
        empty = pd.DataFrame(columns=METADATA_COLUMNS + list(feature_names))
        empty.index = pd.DatetimeIndex([], name="time", tz="UTC")
        return CandidateDatasetResult(empty, ledger, feature_names)

    signal_index = pd.DatetimeIndex(executed["signal_time"])
    missing_signals = signal_index.difference(frame.index)
    if len(missing_signals):
        raise ValueError(f"Executed signals missing from feature frame: {list(missing_signals)}")
    rows = frame.loc[signal_index].copy()
    rows.index = signal_index
    sides = executed.set_index("signal_time")["side"].reindex(signal_index)
    rows["is_long"] = (sides == "long").astype("int8").to_numpy()
    direction = np.where(rows["is_long"].to_numpy() == 1, 1.0, -1.0)

    if strategy == "P1":
        boundary = np.where(
            direction > 0,
            rows["opening_range_high"],
            rows["opening_range_low"],
        )
        rows["setup_distance_atr"] = (
            direction * (rows["close"].to_numpy() - boundary)
            / rows["m30_atr14"].to_numpy()
        )
    elif strategy == "P2":
        rows["setup_distance_atr"] = (
            (rows["close"] - rows["m30_ema20"]).abs() / rows["m30_atr14"]
        )
        rows["h1_regime_distance_atr"] = (
            direction
            * (rows["h1_ema20"] - rows["h1_ema50"]).to_numpy()
            / rows["h1_atr14"].to_numpy()
        )
    else:
        band = np.where(
            direction > 0,
            rows["m30_bb_lower"],
            rows["m30_bb_upper"],
        )
        rows["setup_distance_atr"] = (
            direction * (rows["close"].to_numpy() - band)
            / rows["m30_atr14"].to_numpy()
        )
        rows["h1_regime_distance_atr"] = (
            (rows["h1_ema20"] - rows["h1_ema50"]).abs() / rows["h1_atr14"]
        )

    missing_features = [
        column for column in feature_names if column not in rows.columns
    ]
    if missing_features:
        raise ValueError(f"Candidate feature frame missing columns: {missing_features}")
    non_finite = ~np.isfinite(rows.loc[:, feature_names].astype(float).to_numpy())
    if non_finite.any():
        bad_rows, bad_columns = np.where(non_finite)
        examples = sorted(
            {
                f"{rows.index[row]}:{feature_names[column]}"
                for row, column in zip(bad_rows[:10], bad_columns[:10])
            }
        )
        raise ValueError(
            "Candidate features are incomplete; trim the deterministic warmup "
            f"before generating candidates. Examples: {examples}"
        )

    executed = executed.set_index("signal_time").reindex(signal_index)
    dataset = pd.DataFrame(index=signal_index)
    dataset.index.name = "time"
    dataset["candidate_id"] = [
        f"{strategy}:{timestamp.isoformat()}" for timestamp in signal_index
    ]
    for column in METADATA_COLUMNS[1:]:
        dataset[column] = executed[column].to_numpy()
    for column in feature_names:
        dataset[column] = rows[column].astype(float).to_numpy()
    return CandidateDatasetResult(dataset, ledger, feature_names)


def candidate_feature_columns(dataset: pd.DataFrame, strategy: str) -> list[str]:
    """Refuse stale candidate artifacts rather than infer feature columns."""

    expected = COMMON_FEATURES + STRATEGY_FEATURES[strategy] + ["is_long"]
    missing_metadata = [column for column in METADATA_COLUMNS if column not in dataset]
    missing_features = [column for column in expected if column not in dataset]
    unexpected = [
        column
        for column in dataset.columns
        if column not in METADATA_COLUMNS and column not in expected
    ]
    if missing_metadata or missing_features or unexpected:
        raise ValueError(
            "Candidate dataset contract mismatch: "
            f"missing metadata={missing_metadata}, missing features={missing_features}, "
            f"unexpected={unexpected}. Rebuild the intraday candidate dataset."
        )
    non_numeric = [
        column for column in expected if not ptypes.is_numeric_dtype(dataset[column])
    ]
    if non_numeric:
        raise ValueError(f"Candidate model features must be numeric: {non_numeric}")
    return expected
