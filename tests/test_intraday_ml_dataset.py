"""Candidate-only ML contract: exact executions, explicit features, no leakage."""

import pandas as pd
import pytest

from commodity_ml.intraday.execution import TradeCandidate
from commodity_ml.intraday.ml_dataset import (
    COMMON_FEATURES,
    METADATA_COLUMNS,
    build_candidate_dataset,
    candidate_feature_columns,
)


def frame_for_candidate() -> pd.DataFrame:
    index = pd.date_range("2026-01-05 07:00", periods=4, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 101.0],
            "high": [100.1, 101.6, 101.1, 101.1],
            "low": [99.9, 99.8, 100.9, 100.9],
            "close": [100.0, 101.5, 101.0, 101.0],
            "volume": 0.0,
        },
        index=index,
    )
    for column in COMMON_FEATURES:
        frame[column] = 0.1
    frame["opening_range_compression"] = 1.0
    frame["opening_range_high"] = 100.0
    frame["opening_range_low"] = 99.0
    frame["m30_atr14"] = 1.0
    frame["m30_ema20"] = 100.0
    frame["m30_bb_lower"] = 98.0
    frame["m30_bb_upper"] = 102.0
    frame["h1_ema20"] = 101.0
    frame["h1_ema50"] = 100.0
    frame["h1_atr14"] = 2.0
    return frame


def test_candidate_dataset_uses_exact_execution_label_and_explicit_features():
    frame = frame_for_candidate()
    candidate = TradeCandidate(
        signal_time=frame.index[0],
        side="long",
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=2,
        strategy="P1",
        stop_distance_abs=1.0,
        target_r_multiple=1.5,
    )

    result = build_candidate_dataset(frame, [candidate], strategy="P1")
    dataset = result.dataset

    assert len(dataset) == 1
    assert dataset["label"].iloc[0] == 1
    assert dataset["exit_reason"].iloc[0] == "target"
    assert dataset["candidate_id"].is_unique
    assert candidate_feature_columns(dataset, "P1") == list(result.feature_names)
    assert not set(METADATA_COLUMNS) & set(result.feature_names)


def test_incomplete_candidate_feature_is_refused_not_silently_dropped():
    frame = frame_for_candidate()
    frame.loc[frame.index[0], "h1_atr_pct"] = float("nan")
    candidate = TradeCandidate(
        signal_time=frame.index[0],
        side="long",
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=1,
        strategy="P1",
        stop_distance_abs=1.0,
        target_r_multiple=1.5,
    )

    with pytest.raises(ValueError, match="incomplete"):
        build_candidate_dataset(frame, [candidate], strategy="P1")
