"""CME GC import integrity tests; no network or proprietary sample required."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from commodity_ml.data.cme import (
    development_files,
    normalize_gc_definitions,
    normalize_gc_ohlcv,
    normalize_futures_ohlcv,
    normalize_gc_statistics,
    verify_job,
)


def definition_frame() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        ["2025-05-01T00:00:00Z", "2025-05-01T00:00:00Z"],
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
        normalize_gc_definitions(definition_frame()),
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
            normalize_gc_definitions(definition_frame().iloc[[0]]),
        )


def test_ohlcv_maps_reused_instrument_id_as_of_definition_time():
    definitions = pd.concat(
        [
            definition_frame().iloc[[0]],
            definition_frame().iloc[[0]].assign(
                raw_symbol="GCQ5",
                expiration=pd.Timestamp("2025-08-27", tz="UTC"),
            ),
        ]
    )
    definitions.index = pd.DatetimeIndex(
        ["2025-05-01T00:00:00Z", "2025-06-01T00:00:00Z"],
        name="ts_recv",
    )
    bars = pd.DataFrame(
        {
            "instrument_id": [1, 1],
            "open": [3300.0, 3400.0],
            "high": [3301.0, 3401.0],
            "low": [3299.0, 3399.0],
            "close": [3300.5, 3400.5],
            "volume": [25, 30],
        },
        index=pd.DatetimeIndex(
            ["2025-05-15T12:00:00Z", "2025-06-15T12:00:00Z"],
            name="ts_event",
        ),
    )
    result = normalize_gc_ohlcv(bars, normalize_gc_definitions(definitions))
    assert list(result["raw_symbol"]) == ["GCM5", "GCQ5"]


def test_ohlcv_never_uses_a_future_definition():
    bars = pd.DataFrame(
        {
            "instrument_id": [1],
            "open": [3300.0],
            "high": [3301.0],
            "low": [3299.0],
            "close": [3300.5],
            "volume": [25],
        },
        index=pd.DatetimeIndex(["2025-04-30T23:59:00Z"], name="ts_event"),
    )
    with pytest.raises(ValueError, match="unmapped instrument IDs"):
        normalize_gc_ohlcv(bars, normalize_gc_definitions(definition_frame()))


def test_generic_normalizer_can_exclude_but_never_future_map_early_rows():
    bars = pd.DataFrame(
        {
            "instrument_id": [1, 1],
            "open": [3300.0, 3300.0],
            "high": [3301.0, 3301.0],
            "low": [3299.0, 3299.0],
            "close": [3300.5, 3300.5],
            "volume": [25, 25],
        },
        index=pd.DatetimeIndex(
            ["2025-04-30T23:59:00Z", "2025-05-01T00:01:00Z"],
            name="ts_event",
        ),
    )
    result = normalize_futures_ohlcv(
        bars,
        normalize_gc_definitions(definition_frame()),
        label="GC",
        drop_unmapped=True,
    )

    assert list(result.index) == [pd.Timestamp("2025-05-01T00:01:00Z")]
    assert result.attrs["unmapped_rows_dropped"] == 1


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


def test_job_verification_records_degraded_but_refuses_missing(tmp_path):
    data = tmp_path / "glbx-mdp3-20250501.ohlcv-1m.dbn.zst"
    data.write_bytes(b"payload")
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
                    "end": pd.Timestamp("2025-05-02", tz="UTC").value,
                    "encoding": "dbn",
                    "compression": "zstd",
                }
            }
        )
    )
    manifest_files = []
    condition = tmp_path / "condition.json"
    condition.write_text('[{"date":"2025-05-01","condition":"degraded"}]')
    for path in (data, condition, metadata):
        manifest_files.append(
            {
                "filename": path.name,
                "hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": manifest_files})
    )
    job = verify_job(tmp_path, expected_schema="ohlcv-1m")
    assert job.degraded_dates == ("2025-05-01",)

    condition.write_text('[{"date":"2025-05-01","condition":"missing"}]')
    # Keep manifest integrity valid so the condition code is the failing boundary.
    manifest_files[1]["hash"] = (
        "sha256:" + hashlib.sha256(condition.read_bytes()).hexdigest()
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": manifest_files})
    )
    with pytest.raises(ValueError, match="unavailable dates"):
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
        normalize_gc_definitions(definition_frame().iloc[[0]]),
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
            normalize_gc_definitions(definition_frame().iloc[[0]]),
        )


def test_development_file_fence_excludes_confirmation_without_reading_it(tmp_path):
    june = tmp_path / "glbx-mdp3-20250601-20250630.ohlcv-1m.dbn.zst"
    july = tmp_path / "glbx-mdp3-20250701.ohlcv-1m.dbn.zst"
    selected, excluded = development_files((june, july))
    assert selected == (june,)
    assert excluded == (july,)


def test_development_file_fence_refuses_a_straddling_partition(tmp_path):
    crossing = tmp_path / "glbx-mdp3-20250601-20250701.statistics.dbn.zst"
    with pytest.raises(ValueError, match="straddles development cutoff"):
        development_files((crossing,))
