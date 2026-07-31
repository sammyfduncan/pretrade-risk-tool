"""Verified Databento TBBO import for COMEX GC microstructure research."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import databento as db
import numpy as np
import pandas as pd

from commodity_ml.data.cme import (
    DEVELOPMENT_END,
    attach_gc_identity,
    development_files,
    verify_job,
)

TBBO_COLUMNS = (
    "instrument_id",
    "raw_symbol",
    "ts_event",
    "side",
    "price",
    "size",
    "bid_px_00",
    "ask_px_00",
    "bid_sz_00",
    "ask_sz_00",
    "bid_ct_00",
    "ask_ct_00",
    "sequence",
    "book_valid",
)


def normalize_gc_tbbo(
    frame: pd.DataFrame,
    definitions: pd.DataFrame,
) -> pd.DataFrame:
    """Temporally map and validate trade-space top-of-book records."""

    required = {
        "instrument_id",
        "ts_event",
        "action",
        "side",
        "price",
        "size",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"CME TBBO missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("CME TBBO requires a timezone-aware receive index")
    out = attach_gc_identity(frame, definitions, label="TBBO")
    out = out.loc[out["instrument_class"] == "F"].copy()
    if out.empty:
        return pd.DataFrame(columns=TBBO_COLUMNS, index=out.index)
    if (out["action"].astype(str) != "T").any():
        raise ValueError("TBBO contains a non-trade action")
    sides = set(out["side"].astype(str).unique())
    if unknown := sorted(sides - {"A", "B", "N"}):
        raise ValueError(f"TBBO contains unknown aggressor sides: {unknown}")
    numeric = out.loc[
        :, ["price", "size", "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00"]
    ].astype(float)
    if (out["size"] <= 0).any():
        raise ValueError("TBBO trade size must be positive")
    out["book_valid"] = (
        np.isfinite(numeric).all(axis=1)
        & (out["price"] > 0)
        & (out["bid_px_00"] > 0)
        & (out["ask_px_00"] > 0)
        & (out["bid_sz_00"] > 0)
        & (out["ask_sz_00"] > 0)
        & (out["bid_px_00"] < out["ask_px_00"])
    )
    if out["ts_event"].isna().any():
        raise ValueError("TBBO contains missing event timestamps")
    keep = [column for column in TBBO_COLUMNS if column in out]
    return out.loc[:, keep].sort_index()


def process_gc_tbbo_job(
    *,
    tbbo_job: str | Path,
    definitions_path: str | Path,
    output: str | Path | None,
    development_end: pd.Timestamp = DEVELOPMENT_END,
) -> tuple[dict, Path | None]:
    """Validate and optionally partition a TBBO delivery without creating features."""

    job = verify_job(tbbo_job, expected_schema="tbbo")
    cutoff = _utc(development_end)
    selected, excluded = development_files(job.data_files, cutoff=cutoff)
    if not selected:
        raise ValueError("No pre-cutoff TBBO files remain")
    definitions = pd.read_parquet(definitions_path)
    required = {"definition_time", "instrument_id", "raw_symbol", "instrument_class"}
    if missing := sorted(required - set(definitions)):
        raise ValueError(f"Normalized definitions missing columns: {missing}")
    if pd.Timestamp(definitions["definition_time"].min()) > job.start:
        raise ValueError("Definition history begins after the TBBO delivery")

    root = Path(output).expanduser().resolve() if output is not None else None
    output_hashes: dict[str, str] = {}
    raw_rows = 0
    outright_rows = 0
    sided_rows = 0
    valid_book_rows = 0
    valid_sided_rows = 0
    symbols: set[str] = set()
    for path in selected:
        raw = db.DBNStore.from_file(path).to_df()
        normalized = normalize_gc_tbbo(raw, definitions)
        raw_rows += len(raw)
        outright_rows += len(normalized)
        sided_rows += int(normalized["side"].isin(["A", "B"]).sum())
        valid_book_rows += int(normalized["book_valid"].sum())
        valid_sided_rows += int(
            (normalized["book_valid"] & normalized["side"].isin(["A", "B"])).sum()
        )
        symbols.update(normalized["raw_symbol"].astype(str).unique())
        if root is not None:
            target = root / "tbbo" / _parquet_name(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".tmp")
            normalized.to_parquet(temporary)
            temporary.replace(target)
            output_hashes[str(target.relative_to(root))] = _sha256(target)

    report = {
        "schema_version": 1,
        "dataset": "GLBX.MDP3",
        "parent": "GC.FUT",
        "schema": "tbbo",
        "start": job.start.isoformat(),
        "end_exclusive": min(job.end, cutoff).isoformat(),
        "source_end_exclusive": job.end.isoformat(),
        "development_cutoff_applied": job.end > cutoff,
        "excluded_post_cutoff_files": [path.name for path in excluded],
        "degraded_source_dates": list(job.degraded_dates),
        "raw_rows": raw_rows,
        "outright_rows": outright_rows,
        "sided_rows": sided_rows,
        "sided_coverage": sided_rows / outright_rows if outright_rows else 0.0,
        "valid_book_rows": valid_book_rows,
        "invalid_book_rows": outright_rows - valid_book_rows,
        "valid_book_coverage": valid_book_rows / outright_rows if outright_rows else 0.0,
        "valid_book_sided_coverage": (
            valid_sided_rows / valid_book_rows if valid_book_rows else 0.0
        ),
        "outright_symbols": len(symbols),
        "input_sha256": job.hashes,
        "output_sha256": output_hashes,
        "features_created": False,
        "strategy_evaluated": False,
    }
    if root is None:
        return report, None
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report, manifest_path


def _parquet_name(path: Path) -> str:
    suffix = ".dbn.zst"
    if not path.name.endswith(suffix):
        raise ValueError(f"Unexpected DBN filename: {path.name}")
    return path.name[: -len(suffix)] + ".parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("development_end must be timezone-aware")
    return timestamp.tz_convert("UTC")
