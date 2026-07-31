"""Candidate-only ML contract: exact executions, explicit features, no leakage."""

import pandas as pd
import pytest

from commodity_ml.intraday.execution import TradeCandidate
from commodity_ml.intraday.ml_dataset import (
    COMMON_FEATURES,
    METADATA_COLUMNS,
    build_candidate_dataset,
    candidate_feature_columns,
    load_candidate_bundle,
    write_candidate_bundle,
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


def test_candidate_bundle_round_trip_verifies_ledger_and_provenance(tmp_path):
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
    output = tmp_path / "p1_candidates.parquet"
    cutoff = pd.Timestamp("2025-07-01", tz="UTC")

    manifest = write_candidate_bundle(
        result,
        output,
        strategy="P1",
        source="dukascopy",
        cost_pct_per_side=0.0001,
        confirmation_cutoff=cutoff,
    )
    loaded = load_candidate_bundle(output, expected_strategy="P1")

    assert manifest.exists()
    pd.testing.assert_frame_equal(loaded.dataset, result.dataset, check_freq=False)
    pd.testing.assert_frame_equal(loaded.ledger, result.ledger, check_freq=False)
    assert loaded.feature_names == result.feature_names
    with pytest.raises(ValueError, match="cost assumption mismatch"):
        load_candidate_bundle(
            output,
            expected_strategy="P1",
            expected_cost_pct_per_side=0.0002,
        )


def test_candidate_bundle_refuses_tampering_even_when_row_count_is_unchanged(tmp_path):
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
    output = tmp_path / "p1_candidates.parquet"
    write_candidate_bundle(
        result,
        output,
        strategy="P1",
        source="dukascopy",
        cost_pct_per_side=0.0001,
        confirmation_cutoff=pd.Timestamp("2025-07-01", tz="UTC"),
    )
    tampered = pd.read_parquet(output)
    tampered.loc[tampered.index[0], "label"] = 1 - tampered["label"].iloc[0]
    tampered.to_parquet(output)

    with pytest.raises(ValueError, match="checksum"):
        load_candidate_bundle(output, expected_strategy="P1")
