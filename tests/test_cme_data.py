"""CME GC import integrity tests; no network or proprietary sample required."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from commodity_ml.data.cme import (
    normalize_gc_definitions,
    normalize_gc_ohlcv,
    normalize_gc_statistics,
    verify_job,
)


def definition_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2025-05-01T00:00:00Z", "2025-05-02T00:00:00Z"],
        name="ts_recv",
    )
    return pd.DataFrame(
        {
            "instrument_id": [1, 2],
            "raw_symbol": ["GCM5", "GCM5-GCQ5"],
            "instrument_class": ["F", "S"],
            "exchange": "XCEC",
            "currency": "USD",
            "expiration": pd.to_datetime(["2025-06-26", "2025-06-26"], utc=True),
            "activation": pd.to_datetime(["2023-01-01", "2023-01-01"], utc=True),
            "asset": "GC",
            "security_type": "FUT",
            "min_price_increment": 0.1,
            "contract_multiplier": 100.0,
            "unit_of_measure": "TOZ",
        },
        index=index,
    )


def test_definitions_distinguish_outrights_from_spreads():
    result = normalize_gc_definitions(definition_frame())
    assert result.set_index("raw_symbol").loc["GCM5", "instrument_class"] == "F"
    assert result.set_index("raw_symbol").loc[
        "GCM5-GCQ5", "instrument_class"
    ] == "S"


def test_ohlcv_filters_spreads_and_pins_instrument_minute_uniqueness():
    index = pd.DatetimeIndex(
        ["2025-05-01T00:00:00Z", "2025-05-01T00:00:00Z"],
        name="ts_recv",
    )
    frame = pd.DataFrame(
        {
            "instrument_id": [1, 2],
            "open": [3300.0, 1.0],
            "high": [3301.0, 1.2],
            "low": [3299.0, 0.8],
            "close": [3300.5, 1.1],
            "volume": [25, 3],
        },
        index=index,
    )
    result = normalize_gc_ohlcv(
        frame,
        pd.Series({1: "GCM5", 2: "GCM5-GCQ5"}),
        pd.Series({1: "F", 2: "S"}),
    )
    assert len(result) == 1
    assert result.iloc[0]["raw_symbol"] == "GCM5"


def test_ohlcv_rejects_invalid_ranges():
    index = pd.DatetimeIndex(["2025-05-01T00:00:00Z"], name="ts_recv")
    frame = pd.DataFrame(
        {
            "instrument_id": [1],
            "open": [3300.0],
            "high": [3299.0],
            "low": [3298.0],
            "close": [3300.5],
            "volume": [1],
        },
        index=index,
    )
    with pytest.raises(ValueError, match="invalid ranges"):
        normalize_gc_ohlcv(
            frame,
            pd.Series({1: "GCM5"}),
            pd.Series({1: "F"}),
        )


def test_job_verification_refuses_manifest_tampering(tmp_path):
    data = tmp_path / "sample.ohlcv-1m.dbn.zst"
    data.write_bytes(b"not parsed in this manifest-only test")
    condition = tmp_path / "condition.json"
    condition.write_text('[{"date":"2025-05-01","condition":"available"}]')
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "query": {
                    "dataset": "GLBX.MDP3",
                    "schema": "ohlcv-1m",
                    "symbols": ["GC.FUT"],
                    "stype_in": "parent",
                    "stype_out": "instrument_id",
                    "start": pd.Timestamp("2025-05-01", tz="UTC").value,
                    "end": pd.Timestamp("2025-06-01", tz="UTC").value,
                    "encoding": "dbn",
                    "compression": "zstd",
                }
            }
        )
    )
    entries = []
    for path in (data, condition, metadata):
        entries.append(
            {
                "filename": path.name,
                "hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}))
    assert verify_job(tmp_path, expected_schema="ohlcv-1m").data_files == (data,)

    data.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify_job(tmp_path, expected_schema="ohlcv-1m")


def test_statistics_allow_missing_reference_only_for_intraday_types():
    index = pd.DatetimeIndex(
        ["2025-05-01T00:00:00Z", "2025-05-01T22:00:00Z"],
        name="ts_recv",
    )
    frame = pd.DataFrame(
        {
            "instrument_id": [1, 1],
            "ts_event": index,
            "ts_ref": [pd.NaT, pd.Timestamp("2025-05-01", tz="UTC")],
            "stat_type": [4, 9],
            "price": [3290.0, np.nan],
            "quantity": [0, 500_000],
            "update_action": [1, 1],
            "stat_flags": [0, 1],
            "sequence": [1, 2],
        },
        index=index,
    )
    result = normalize_gc_statistics(
        frame,
        pd.Series({1: "GCM5"}),
        pd.Series({1: "F"}),
    )
    assert list(result["stat_type_name"]) == [
        "trading_session_low_price",
        "open_interest",
    ]

    broken = frame.copy()
    broken.loc[broken["stat_type"] == 9, "ts_ref"] = pd.NaT
    with pytest.raises(ValueError, match="missing reference dates"):
        normalize_gc_statistics(
            broken,
            pd.Series({1: "GCM5"}),
            pd.Series({1: "F"}),
        )
