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
import re
from typing import Iterable

import databento as db
import numpy as np
import pandas as pd

EXPECTED_DATASET = "GLBX.MDP3"
EXPECTED_PARENT = "GC.FUT"
SCHEMAS = ("ohlcv-1m", "statistics", "definition")
DEVELOPMENT_END = pd.Timestamp("2025-07-01", tz="UTC")
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
    degraded_dates: tuple[str, ...]


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
    unknown_conditions = sorted(
        {
            str(item.get("condition"))
            for item in conditions
            if item.get("condition") not in {"available", "degraded", "pending", "missing"}
        }
    )
    if unknown_conditions:
        raise ValueError(f"Unknown Databento condition codes: {unknown_conditions}")
    unavailable = [
        item.get("date")
        for item in conditions
        if item.get("condition") in {"pending", "missing"}
    ]
    if unavailable:
        raise ValueError(f"Databento reports unavailable dates: {unavailable}")
    degraded = tuple(
        str(item.get("date"))
        for item in conditions
        if item.get("condition") == "degraded"
    )

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
        degraded_dates=degraded,
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
    raw_ohlcv = _read_dbn_files(jobs["ohlcv-1m"].data_files)
    ohlcv = normalize_gc_ohlcv(raw_ohlcv, definitions)
    raw_statistics = _read_dbn_files(jobs["statistics"].data_files)
    statistics = normalize_gc_statistics(raw_statistics, definitions)
    start, end = next(iter(bounds))
    report = {
        "schema_version": 1,
        "dataset": EXPECTED_DATASET,
        "parent": EXPECTED_PARENT,
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "definition_rows_raw": len(raw_definitions),
        "definition_instruments": int(definitions["instrument_id"].nunique()),
        "definition_history_periods": len(definitions),
        "outright_instruments": int(
            definitions.loc[
                definitions["instrument_class"] == "F", "instrument_id"
            ].nunique()
        ),
        "spread_instruments": int(
            definitions.loc[
                definitions["instrument_class"] == "S", "instrument_id"
            ].nunique()
        ),
        "ohlcv_rows_raw": len(raw_ohlcv),
        "ohlcv_rows_outright": len(ohlcv),
        "ohlcv_outright_symbols": int(ohlcv["raw_symbol"].nunique()),
        "statistics_rows_raw": len(raw_statistics),
        "statistics_rows_outright": len(statistics),
        "input_sha256": {
            schema: job.hashes for schema, job in jobs.items()
        },
        "degraded_source_dates": {
            schema: list(job.degraded_dates) for schema, job in jobs.items()
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
    development_end: pd.Timestamp = DEVELOPMENT_END,
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
    start, source_end = next(iter(bounds))
    cutoff = _utc(development_end, "development_end")
    if start >= cutoff:
        raise ValueError("CME delivery begins at or after the development cutoff")
    selected_files: dict[str, tuple[Path, ...]] = {}
    excluded_files: dict[str, list[str]] = {}
    for schema, job in jobs.items():
        selected, excluded = development_files(job.data_files, cutoff=cutoff)
        if not selected:
            raise ValueError(f"No pre-cutoff {schema} DBN files remain")
        selected_files[schema] = selected
        excluded_files[schema] = [path.name for path in excluded]

    raw_definitions = _read_dbn_files(selected_files["definition"])
    definitions = normalize_gc_definitions(raw_definitions)
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
    for path in selected_files["ohlcv-1m"]:
        raw = _read_dbn_files((path,))
        normalized = normalize_gc_ohlcv(raw, definitions)
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
    for path in selected_files["statistics"]:
        raw = _read_dbn_files((path,))
        normalized = normalize_gc_statistics(raw, definitions)
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
        "end_exclusive": min(source_end, cutoff).isoformat(),
        "source_end_exclusive": source_end.isoformat(),
        "development_cutoff_applied": source_end > cutoff,
        "excluded_post_cutoff_files": excluded_files,
        "degraded_source_dates": {
            schema: list(job.degraded_dates) for schema, job in jobs.items()
        },
        "definition_rows_raw": len(raw_definitions),
        "definition_instruments": int(definitions["instrument_id"].nunique()),
        "definition_history_periods": len(definitions),
        "outright_instruments": int(
            definitions.loc[
                definitions["instrument_class"] == "F", "instrument_id"
            ].nunique()
        ),
        "spread_instruments": int(
            definitions.loc[
                definitions["instrument_class"] == "S", "instrument_id"
            ].nunique()
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
    index_name = frame.index.name or "index"
    ordered = (
        frame.reset_index()
        .rename(columns={index_name: "definition_time"})
        .sort_values(["instrument_id", "definition_time"])
    )
    ordered["definition_time"] = pd.to_datetime(
        ordered["definition_time"], utc=True
    )
    if ordered["definition_time"].isna().any():
        raise ValueError("GC definitions contain missing receive timestamps")
    if (ordered["exchange"] != "XCEC").any():
        raise ValueError("GC definitions contain a non-COMEX exchange")
    if (ordered["currency"] != "USD").any():
        raise ValueError("GC definitions contain a non-USD instrument")
    if (ordered["asset"] != "GC").any() or (
        ordered["security_type"] != "FUT"
    ).any():
        raise ValueError("GC definitions contain the wrong asset/security type")
    unknown_classes = set(ordered["instrument_class"].astype(str)) - {"F", "S"}
    if unknown_classes:
        raise ValueError(f"Unknown GC instrument classes: {sorted(unknown_classes)}")

    # Definition snapshots repeat every day. Retain the first record and every
    # identity change for each numeric ID: IDs may be recycled over a long
    # delivery, so a single latest-value dictionary is incorrect.
    history_columns = [
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
    history_columns = [column for column in history_columns if column in ordered]
    previous = ordered.groupby("instrument_id")[history_columns].shift()
    identity_changed = previous.isna().all(axis=1) | ordered[
        history_columns
    ].ne(previous).any(axis=1)
    history = ordered.loc[identity_changed].copy()
    columns = [
        "definition_time",
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
    available = [column for column in columns if column in history]
    return history.loc[:, available].sort_values(
        ["definition_time", "instrument_id"]
    ).reset_index(drop=True)


def normalize_gc_ohlcv(
    frame: pd.DataFrame,
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    required = {"instrument_id", "open", "high", "low", "close", "volume"}
    _require_columns(frame, required, "OHLCV")
    out = _attach_identity(frame, definitions, "OHLCV")
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
    definitions: pd.DataFrame,
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
    out = _attach_identity(frame, definitions, "statistics")
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


def development_files(
    paths: Iterable[Path],
    *,
    cutoff: pd.Timestamp = DEVELOPMENT_END,
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Select only files wholly before cutoff without opening their records."""

    end = _utc(cutoff, "cutoff")
    selected: list[Path] = []
    excluded: list[Path] = []
    pattern = re.compile(r"^glbx-mdp3-(\d{8})(?:-(\d{8}))?\.")
    for path in sorted(paths):
        match = pattern.match(path.name)
        if match is None:
            raise ValueError(f"Cannot determine DBN file date range: {path.name}")
        first = pd.Timestamp(match.group(1), tz="UTC")
        last = pd.Timestamp(match.group(2) or match.group(1), tz="UTC")
        file_end = last + pd.Timedelta(days=1)
        if file_end <= end:
            selected.append(path)
        elif first >= end:
            excluded.append(path)
        else:
            raise ValueError(
                f"DBN file straddles development cutoff and cannot be sliced "
                f"silently: {path.name}"
            )
    return tuple(selected), tuple(excluded)


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=index)
    temporary.replace(path)


def _attach_identity(
    frame: pd.DataFrame,
    definitions: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    required = {
        "definition_time",
        "instrument_id",
        "raw_symbol",
        "instrument_class",
    }
    missing_columns = sorted(required - set(definitions.columns))
    if missing_columns:
        raise ValueError(
            "CME normalized definition history missing columns: "
            f"{missing_columns}"
        )
    index_name = frame.index.name or "index"
    left = frame.reset_index().rename(columns={index_name: "_record_time"})
    left["_row_order"] = np.arange(len(left))
    right = definitions.loc[
        :, ["definition_time", "instrument_id", "raw_symbol", "instrument_class"]
    ].copy()
    left["_record_time"] = pd.to_datetime(left["_record_time"], utc=True)
    right["definition_time"] = pd.to_datetime(right["definition_time"], utc=True)
    left = left.sort_values(["_record_time", "instrument_id"])
    right = right.sort_values(["definition_time", "instrument_id"])
    out = pd.merge_asof(
        left,
        right,
        left_on="_record_time",
        right_on="definition_time",
        by="instrument_id",
        direction="backward",
        allow_exact_matches=True,
    )
    if out[["raw_symbol", "instrument_class"]].isna().any().any():
        missing = sorted(out.loc[out["raw_symbol"].isna(), "instrument_id"].unique())
        raise ValueError(f"{label} has unmapped instrument IDs: {missing[:10]}")
    out = out.sort_values("_row_order").set_index("_record_time")
    out.index.name = frame.index.name
    return out.drop(columns=["_row_order", "definition_time"])


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


def _utc(value: pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
