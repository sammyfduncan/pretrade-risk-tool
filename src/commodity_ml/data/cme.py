"""Verified import of Databento CME/COMEX GC batch deliveries.

The importer preserves individual outright contracts.  It does not select a
front contract, back-adjust prices, fill no-trade minutes, or create features.
Those are research decisions, not file-format conveniences.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import databento as db
import numpy as np
import pandas as pd

EXPECTED_DATASET = "GLBX.MDP3"
EXPECTED_PARENT = "GC.FUT"
SCHEMAS = ("ohlcv-1m", "statistics", "definition")
DAILY_REFERENCE_STAT_TYPES = {3, 6, 9, 10}
STAT_TYPE_NAMES = {
    1: "opening_price",
    2: "indicative_opening_price",
    3: "settlement_price",
    4: "trading_session_low_price",
    5: "trading_session_high_price",
    6: "cleared_volume",
    7: "lowest_offer",
    8: "highest_bid",
    9: "open_interest",
    10: "fixing_price",
}


@dataclass(frozen=True)
class DatabentoJob:
    directory: Path
    schema: str
    start: pd.Timestamp
    end: pd.Timestamp
    data_files: tuple[Path, ...]
    hashes: dict[str, str]


@dataclass(frozen=True)
class GCImportResult:
    definitions: pd.DataFrame
    ohlcv: pd.DataFrame
    statistics: pd.DataFrame
    report: dict


def verify_job(directory: str | Path, *, expected_schema: str) -> DatabentoJob:
    """Verify one complete batch job against its Databento manifest."""

    root = Path(directory).expanduser().resolve()
    if expected_schema not in SCHEMAS:
        raise ValueError(f"Unsupported CME schema {expected_schema!r}")
    for name in ("manifest.json", "metadata.json", "condition.json"):
        if not (root / name).is_file():
            raise ValueError(f"Databento job is missing {name}: {root}")

    manifest = _read_json(root / "manifest.json")
    metadata = _read_json(root / "metadata.json")
    conditions = _read_json(root / "condition.json")
    query = metadata.get("query", {})
    expected_query = {
        "dataset": EXPECTED_DATASET,
        "schema": expected_schema,
        "symbols": [EXPECTED_PARENT],
        "stype_in": "parent",
        "stype_out": "instrument_id",
        "encoding": "dbn",
        "compression": "zstd",
    }
    mismatches = {
        key: (query.get(key), value)
        for key, value in expected_query.items()
        if query.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Unexpected Databento query fields: {mismatches}")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("Databento condition ledger is empty")
    unavailable = [
        item.get("date")
        for item in conditions
        if item.get("condition") != "available"
    ]
    if unavailable:
        raise ValueError(f"Databento reports unavailable dates: {unavailable}")

    hashes: dict[str, str] = {}
    data_files: list[Path] = []
    for item in manifest.get("files", []):
        filename = item.get("filename")
        expected_hash = str(item.get("hash", ""))
        if not filename or not expected_hash.startswith("sha256:"):
            raise ValueError("Malformed Databento manifest file entry")
        path = root / filename
        if not path.is_file():
            raise ValueError(f"Manifest file is missing: {path}")
        actual = _sha256(path)
        expected = expected_hash.removeprefix("sha256:")
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {path.name}")
        hashes[path.name] = actual
        if path.name.endswith(f".{expected_schema}.dbn.zst"):
            data_files.append(path)
    if not data_files:
        raise ValueError(f"No {expected_schema} DBN files in {root}")

    start = pd.Timestamp(int(query["start"]), unit="ns", tz="UTC")
    end = pd.Timestamp(int(query["end"]), unit="ns", tz="UTC")
    if start >= end:
        raise ValueError("Databento job has invalid time bounds")
    return DatabentoJob(
        directory=root,
        schema=expected_schema,
        start=start,
        end=end,
        data_files=tuple(sorted(data_files)),
        hashes=hashes,
    )


def import_gc_jobs(
    *,
    ohlcv_job: str | Path,
    statistics_job: str | Path,
    definition_job: str | Path,
) -> GCImportResult:
    """Load three verified jobs and retain only outright GC futures."""

    jobs = {
        "ohlcv-1m": verify_job(ohlcv_job, expected_schema="ohlcv-1m"),
        "statistics": verify_job(statistics_job, expected_schema="statistics"),
        "definition": verify_job(definition_job, expected_schema="definition"),
    }
    bounds = {(job.start, job.end) for job in jobs.values()}
    if len(bounds) != 1:
        raise ValueError("CME jobs do not share exact start/end bounds")

    raw_definitions = _read_dbn_files(jobs["definition"].data_files)
    definitions = normalize_gc_definitions(raw_definitions)
    symbol_map = definitions.set_index("instrument_id")["raw_symbol"]
    class_map = definitions.set_index("instrument_id")["instrument_class"]

    raw_ohlcv = _read_dbn_files(jobs["ohlcv-1m"].data_files)
    ohlcv = normalize_gc_ohlcv(raw_ohlcv, symbol_map, class_map)
    raw_statistics = _read_dbn_files(jobs["statistics"].data_files)
    statistics = normalize_gc_statistics(raw_statistics, symbol_map, class_map)
    start, end = next(iter(bounds))
    report = {
        "schema_version": 1,
        "dataset": EXPECTED_DATASET,
        "parent": EXPECTED_PARENT,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "definition_rows_raw": len(raw_definitions),
        "definition_instruments": len(definitions),
        "outright_instruments": int(
            (definitions["instrument_class"] == "F").sum()
        ),
        "spread_instruments": int(
            (definitions["instrument_class"] == "S").sum()
        ),
        "ohlcv_rows_raw": len(raw_ohlcv),
        "ohlcv_rows_outright": len(ohlcv),
        "ohlcv_outright_symbols": int(ohlcv["raw_symbol"].nunique()),
        "statistics_rows_raw": len(raw_statistics),
        "statistics_rows_outright": len(statistics),
        "input_sha256": {
            schema: job.hashes for schema, job in jobs.items()
        },
        "research_transformations": {
            "front_contract_selected": False,
            "back_adjusted": False,
            "missing_minutes_filled": False,
            "features_created": False,
        },
    }
    return GCImportResult(definitions, ohlcv, statistics, report)


def process_gc_jobs(
    *,
    ohlcv_job: str | Path,
    statistics_job: str | Path,
    definition_job: str | Path,
    output: str | Path | None,
) -> tuple[dict, Path | None]:
    """Validate and optionally write a full delivery one monthly file at a time.

    Definitions are small enough to combine for an unambiguous instrument map.
    OHLCV and statistics are normalized independently per source file so the
    multi-year delivery does not require holding millions of messages twice in
    memory. The returned manifest path is ``None`` for validation-only runs.
    """

    jobs = {
        "ohlcv-1m": verify_job(ohlcv_job, expected_schema="ohlcv-1m"),
        "statistics": verify_job(statistics_job, expected_schema="statistics"),
        "definition": verify_job(definition_job, expected_schema="definition"),
    }
    bounds = {(job.start, job.end) for job in jobs.values()}
    if len(bounds) != 1:
        raise ValueError("CME jobs do not share exact start/end bounds")
    start, end = next(iter(bounds))

    raw_definitions = _read_dbn_files(jobs["definition"].data_files)
    definitions = normalize_gc_definitions(raw_definitions)
    symbol_map = definitions.set_index("instrument_id")["raw_symbol"]
    class_map = definitions.set_index("instrument_id")["instrument_class"]
    root = Path(output).expanduser().resolve() if output is not None else None
    output_hashes: dict[str, object] = {}
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        definition_path = root / "definitions.parquet"
        _atomic_parquet(definitions, definition_path, index=False)
        output_hashes["definitions"] = _sha256(definition_path)

    ohlcv_rows_raw = 0
    ohlcv_rows_outright = 0
    ohlcv_symbols: set[str] = set()
    ohlcv_outputs: dict[str, str] = {}
    for path in jobs["ohlcv-1m"].data_files:
        raw = _read_dbn_files((path,))
        normalized = normalize_gc_ohlcv(raw, symbol_map, class_map)
        ohlcv_rows_raw += len(raw)
        ohlcv_rows_outright += len(normalized)
        ohlcv_symbols.update(normalized["raw_symbol"].astype(str).unique())
        if root is not None:
            target = root / "ohlcv_1m" / _parquet_name(path)
            _atomic_parquet(normalized, target, index=True)
            ohlcv_outputs[str(target.relative_to(root))] = _sha256(target)
    if root is not None:
        output_hashes["ohlcv_1m"] = ohlcv_outputs

    statistics_rows_raw = 0
    statistics_rows_outright = 0
    statistics_outputs: dict[str, str] = {}
    for path in jobs["statistics"].data_files:
        raw = _read_dbn_files((path,))
        normalized = normalize_gc_statistics(raw, symbol_map, class_map)
        statistics_rows_raw += len(raw)
        statistics_rows_outright += len(normalized)
        if root is not None:
            target = root / "statistics" / _parquet_name(path)
            _atomic_parquet(normalized, target, index=True)
            statistics_outputs[str(target.relative_to(root))] = _sha256(target)
    if root is not None:
        output_hashes["statistics"] = statistics_outputs

    report = {
        "schema_version": 1,
        "dataset": EXPECTED_DATASET,
        "parent": EXPECTED_PARENT,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "definition_rows_raw": len(raw_definitions),
        "definition_instruments": len(definitions),
        "outright_instruments": int(
            (definitions["instrument_class"] == "F").sum()
        ),
        "spread_instruments": int(
            (definitions["instrument_class"] == "S").sum()
        ),
        "ohlcv_rows_raw": ohlcv_rows_raw,
        "ohlcv_rows_outright": ohlcv_rows_outright,
        "ohlcv_outright_symbols": len(ohlcv_symbols),
        "statistics_rows_raw": statistics_rows_raw,
        "statistics_rows_outright": statistics_rows_outright,
        "input_sha256": {
            schema: job.hashes for schema, job in jobs.items()
        },
        "output_sha256": output_hashes,
        "research_transformations": {
            "front_contract_selected": False,
            "back_adjusted": False,
            "missing_minutes_filled": False,
            "features_created": False,
        },
    }
    if root is None:
        return report, None
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report, manifest_path


def normalize_gc_definitions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "instrument_id",
        "raw_symbol",
        "instrument_class",
        "exchange",
        "currency",
        "expiration",
        "activation",
        "asset",
        "security_type",
    }
    _require_columns(frame, required, "definition")
    ordered = frame.reset_index().sort_values("ts_recv")
    identities = ordered.groupby("instrument_id")["raw_symbol"].nunique()
    reused = identities[identities > 1]
    if len(reused):
        raise ValueError(
            "Databento instrument IDs map to multiple symbols; temporal mapping "
            f"is required for IDs {list(reused.index[:10])}"
        )
    latest = ordered.drop_duplicates("instrument_id", keep="last")
    if (latest["exchange"] != "XCEC").any():
        raise ValueError("GC definitions contain a non-COMEX exchange")
    if (latest["currency"] != "USD").any():
        raise ValueError("GC definitions contain a non-USD instrument")
    if (latest["asset"] != "GC").any() or (
        latest["security_type"] != "FUT"
    ).any():
        raise ValueError("GC definitions contain the wrong asset/security type")
    unknown_classes = set(latest["instrument_class"].astype(str)) - {"F", "S"}
    if unknown_classes:
        raise ValueError(f"Unknown GC instrument classes: {sorted(unknown_classes)}")
    columns = [
        "instrument_id",
        "raw_symbol",
        "instrument_class",
        "exchange",
        "currency",
        "expiration",
        "activation",
        "asset",
        "security_type",
        "min_price_increment",
        "contract_multiplier",
        "unit_of_measure",
    ]
    available = [column for column in columns if column in latest]
    return latest.loc[:, available].sort_values(
        ["instrument_class", "expiration", "raw_symbol"]
    ).reset_index(drop=True)


def normalize_gc_ohlcv(
    frame: pd.DataFrame,
    symbol_map: pd.Series,
    class_map: pd.Series,
) -> pd.DataFrame:
    required = {"instrument_id", "open", "high", "low", "close", "volume"}
    _require_columns(frame, required, "OHLCV")
    out = _attach_identity(frame, symbol_map, class_map, "OHLCV")
    out = out.loc[out["instrument_class"] == "F"].copy()
    values = out.loc[:, ["open", "high", "low", "close", "volume"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("CME OHLCV contains non-finite values")
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("CME OHLCV contains non-positive prices")
    if (out["volume"] <= 0).any():
        raise ValueError("CME OHLCV records must represent positive traded volume")
    bad = (
        (out["high"] < out[["open", "close", "low"]].max(axis=1))
        | (out["low"] > out[["open", "close", "high"]].min(axis=1))
    )
    if bad.any():
        raise ValueError(f"CME OHLCV has {int(bad.sum())} invalid ranges")
    if ((out.index.second != 0) | (out.index.microsecond != 0)).any():
        raise ValueError("CME OHLCV timestamps are not minute aligned")
    pairs = pd.MultiIndex.from_arrays([out.index, out["instrument_id"]])
    if pairs.duplicated().any():
        raise ValueError("CME OHLCV has duplicate instrument-minute records")
    return out.loc[
        :, ["instrument_id", "raw_symbol", "open", "high", "low", "close", "volume"]
    ].sort_index()


def normalize_gc_statistics(
    frame: pd.DataFrame,
    symbol_map: pd.Series,
    class_map: pd.Series,
) -> pd.DataFrame:
    required = {
        "instrument_id",
        "ts_event",
        "ts_ref",
        "stat_type",
        "update_action",
        "stat_flags",
    }
    _require_columns(frame, required, "statistics")
    out = _attach_identity(frame, symbol_map, class_map, "statistics")
    out = out.loc[out["instrument_class"] == "F"].copy()
    if out["ts_event"].isna().any():
        raise ValueError("CME statistics contain missing event timestamps")
    needs_reference = out["stat_type"].isin(DAILY_REFERENCE_STAT_TYPES)
    if out.loc[needs_reference, "ts_ref"].isna().any():
        raise ValueError("CME daily statistics contain missing reference dates")
    unknown_types = set(out["stat_type"].astype(int)) - set(STAT_TYPE_NAMES)
    if unknown_types:
        raise ValueError(f"Unknown CME statistic types: {sorted(unknown_types)}")
    out["stat_type_name"] = out["stat_type"].map(STAT_TYPE_NAMES)
    keep = [
        "instrument_id",
        "raw_symbol",
        "ts_event",
        "ts_ref",
        "stat_type",
        "stat_type_name",
        "price",
        "quantity",
        "update_action",
        "stat_flags",
        "sequence",
    ]
    return out.loc[:, [column for column in keep if column in out]].sort_index()


def write_gc_import(result: GCImportResult, output: str | Path) -> Path:
    """Write validated source-normalized artifacts; manifest is published last."""

    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "definitions": root / "definitions.parquet",
        "ohlcv_1m": root / "ohlcv_1m.parquet",
        "statistics": root / "statistics.parquet",
    }
    result.definitions.to_parquet(paths["definitions"], index=False)
    result.ohlcv.to_parquet(paths["ohlcv_1m"])
    result.statistics.to_parquet(paths["statistics"])
    output_hashes = {name: _sha256(path) for name, path in paths.items()}
    manifest = result.report | {
        "outputs": {name: str(path) for name, path in paths.items()},
        "output_sha256": output_hashes,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _read_dbn_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames = [db.DBNStore.from_file(path).to_df() for path in paths]
    if not frames:
        raise ValueError("No DBN frames to read")
    return pd.concat(frames).sort_index()


def _parquet_name(path: Path) -> str:
    suffix = ".dbn.zst"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected DBN filename: {path.name}")
    return path.name[: -len(suffix)] + ".parquet"


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=index)
    temporary.replace(path)


def _attach_identity(
    frame: pd.DataFrame,
    symbol_map: pd.Series,
    class_map: pd.Series,
    label: str,
) -> pd.DataFrame:
    out = frame.copy()
    out["raw_symbol"] = out["instrument_id"].map(symbol_map)
    out["instrument_class"] = out["instrument_id"].map(class_map)
    if out[["raw_symbol", "instrument_class"]].isna().any().any():
        missing = sorted(out.loc[out["raw_symbol"].isna(), "instrument_id"].unique())
        raise ValueError(f"{label} has unmapped instrument IDs: {missing[:10]}")
    return out


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CME {label} missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"CME {label} requires a timezone-aware receive index")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
