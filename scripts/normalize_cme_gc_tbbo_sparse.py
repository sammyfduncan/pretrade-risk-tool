"""Normalize the complete sparse GC TBBO delivery without reading outcomes."""

import argparse
import hashlib
import json
from pathlib import Path

import databento as db
import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.data.cme_microstructure import normalize_gc_tbbo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo_sparse"),
    )
    parser.add_argument(
        "--definitions",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc" / "definitions.parquet"),
    )
    parser.add_argument(
        "--cme-manifest",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc" / "manifest.json"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo_sparse_normalized"),
    )
    args = parser.parse_args()
    source = Path(args.source).resolve()
    ledger_path = source / "download_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not ledger.get("complete") or ledger.get("verified_windows") != 901:
        raise SystemExit("Sparse paid delivery is not complete")
    windows = ledger.get("windows", {})
    if len(windows) != ledger["verified_windows"]:
        raise SystemExit("Sparse download ledger row count is inconsistent")

    definitions_path = Path(args.definitions).resolve()
    definitions = pd.read_parquet(definitions_path)
    cme_manifest_path = Path(args.cme_manifest).resolve()
    cme_manifest = json.loads(cme_manifest_path.read_text(encoding="utf-8"))
    degraded_sets = {tuple(values) for values in cme_manifest["degraded_source_dates"].values()}
    if len(degraded_sets) != 1:
        raise SystemExit("CME source schemas disagree on degraded dates")
    degraded_dates = list(next(iter(degraded_sets)))

    output = Path(args.output).resolve()
    target_root = output / "tbbo"
    target_root.mkdir(parents=True, exist_ok=True)
    current_month = None
    chunks: list[pd.DataFrame] = []
    output_hashes = {}
    raw_rows = normalized_rows = sided_rows = valid_rows = 0
    first = last = None

    def flush(month: str | None) -> None:
        nonlocal chunks
        if month is None or not chunks:
            return
        frame = pd.concat(chunks).sort_index()
        target = target_root / f"{month}.parquet"
        temporary = target.with_suffix(target.suffix + ".tmp")
        frame.to_parquet(temporary)
        temporary.replace(target)
        output_hashes[str(target.relative_to(output))] = _sha256(target)
        chunks = []

    for key in sorted(windows):
        item = windows[key]
        path = Path(item["path"])
        if _sha256(path) != item["sha256"]:
            raise SystemExit(f"Sparse TBBO checksum mismatch for window {key}")
        raw = db.DBNStore.from_file(path).to_df()
        normalized = normalize_gc_tbbo(raw, definitions)
        start = pd.Timestamp(item["start"])
        end = pd.Timestamp(item["end_exclusive"])
        if len(raw) and (raw.index.min() < start or raw.index.max() >= end):
            raise SystemExit(f"Window {key} contains a receive time outside its plan")
        if len(normalized) and set(normalized["raw_symbol"]) != {item["symbol"]}:
            raise SystemExit(f"Window {key} contains an unplanned contract")
        month = start.strftime("%Y-%m")
        if current_month is not None and month != current_month:
            flush(current_month)
        current_month = month
        chunks.append(normalized)
        raw_rows += len(raw)
        normalized_rows += len(normalized)
        sided_rows += int(normalized["side"].isin(["A", "B"]).sum())
        valid_rows += int(normalized["book_valid"].sum())
        if len(normalized):
            first = normalized.index.min() if first is None else min(first, normalized.index.min())
            last = normalized.index.max() if last is None else max(last, normalized.index.max())
    flush(current_month)

    manifest = {
        "schema_version": 1,
        "program": "gc_tbbo_sparse_normalization",
        "dataset": "GLBX.MDP3",
        "schema": "tbbo",
        "start": min(pd.Timestamp(item["start"]) for item in windows.values()).isoformat(),
        "end_exclusive": max(pd.Timestamp(item["end_exclusive"]) for item in windows.values()).isoformat(),
        "source_ledger": str(ledger_path),
        "source_ledger_sha256": _sha256(ledger_path),
        "definitions": str(definitions_path),
        "definitions_sha256": _sha256(definitions_path),
        "source_windows": len(windows),
        "raw_rows": raw_rows,
        "outright_rows": normalized_rows,
        "sided_rows": sided_rows,
        "sided_coverage": sided_rows / normalized_rows if normalized_rows else 0.0,
        "valid_book_rows": valid_rows,
        "valid_book_coverage": valid_rows / normalized_rows if normalized_rows else 0.0,
        "first_record": first.isoformat() if first is not None else None,
        "last_record": last.isoformat() if last is not None else None,
        "degraded_source_dates": degraded_dates,
        "output_sha256": output_hashes,
        "features_created": False,
        "labels_read": False,
        "strategy_evaluated": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print("No label, return, model score, or strategy outcome was read.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
