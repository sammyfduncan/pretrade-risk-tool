"""Normalize the complete P4 ZN/6E delivery without reading gold outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import databento as db
import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.data.cme import (
    normalize_futures_definitions,
    normalize_futures_ohlcv,
)

LEGS = {"ZN.FUT": ("zn", "ZN"), "6E.FUT": ("euro", "6E")}
CUTOFF = pd.Timestamp("2025-07-01T00:00:00Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_macro_drivers_dbn"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_macro_drivers"),
    )
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    ledger_path = source / "download_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if not ledger.get("complete") or ledger.get("verified_requests") != 4:
        raise SystemExit("P4 driver paid delivery is not complete")
    requests = ledger.get("requests", {})
    if len(requests) != 4:
        raise SystemExit("P4 driver ledger row count is inconsistent")

    indexed = {(item["symbol"], item["schema"]): item for item in requests.values()}
    expected = {(symbol, schema) for symbol in LEGS for schema in ("definition", "ohlcv-1m")}
    if set(indexed) != expected:
        raise SystemExit("P4 driver ledger differs from the frozen requests")
    plan_path = Path(ledger["plan"])
    plan_manifest_path = plan_path.with_suffix(".manifest.json")
    plan_manifest = json.loads(plan_manifest_path.read_text(encoding="utf-8"))
    condition_path = Path(plan_manifest["condition_path"])
    if (
        not condition_path.is_file()
        or _sha256(condition_path) != plan_manifest.get("condition_sha256")
    ):
        raise SystemExit("P4 dataset-condition artifact fails its checksum")
    if plan_manifest.get("unavailable_source_dates"):
        raise SystemExit("Databento reports unavailable dates in the development period")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "program": "p4_macro_driver_normalization",
        "dataset": "GLBX.MDP3",
        "development_end_exclusive": CUTOFF.isoformat(),
        "source_ledger": str(ledger_path),
        "source_ledger_sha256": _sha256(ledger_path),
        "condition_path": str(condition_path),
        "condition_sha256": _sha256(condition_path),
        "degraded_source_dates": plan_manifest.get("degraded_source_dates", []),
        "legs": {},
        "output_sha256": {},
        "complete": False,
        "features_created": False,
        "gold_data_read": False,
        "labels_read": False,
        "strategy_evaluated": False,
    }

    for parent, (leg, asset) in LEGS.items():
        definition_item = indexed[(parent, "definition")]
        ohlcv_item = indexed[(parent, "ohlcv-1m")]
        definition_path = _verified_path(definition_item)
        ohlcv_path = _verified_path(ohlcv_item)
        raw_definitions = db.DBNStore.from_file(definition_path).to_df()
        definitions = normalize_futures_definitions(
            raw_definitions,
            expected_asset=asset,
        )
        exchanges = sorted(definitions["exchange"].astype(str).unique())
        if len(exchanges) != 1:
            raise SystemExit(f"{parent} definitions contain multiple exchanges: {exchanges}")

        raw_ohlcv = db.DBNStore.from_file(ohlcv_path).to_df()
        ohlcv = normalize_futures_ohlcv(
            raw_ohlcv,
            definitions,
            label=asset,
            drop_unmapped=True,
        )
        unmapped_rows_dropped = int(ohlcv.attrs.get("unmapped_rows_dropped", 0))
        if len(ohlcv) and ohlcv.index.max() >= CUTOFF:
            raise SystemExit(f"{parent} OHLCV crosses the unopened confirmation boundary")

        leg_root = output / leg
        definition_output = leg_root / "definitions.parquet"
        _atomic_parquet(definitions, definition_output, index=False)
        report["output_sha256"][str(definition_output.relative_to(output))] = _sha256(
            definition_output
        )
        partition_hashes = {}
        month = ohlcv.index.tz_convert("UTC").strftime("%Y-%m")
        for value in pd.unique(month):
            partition = ohlcv.loc[month == value]
            target = leg_root / "ohlcv_1m" / f"{value}.parquet"
            _atomic_parquet(partition, target, index=True)
            digest = _sha256(target)
            partition_hashes[str(target.relative_to(output))] = digest
            report["output_sha256"][str(target.relative_to(output))] = digest

        report["legs"][leg] = {
            "parent": parent,
            "asset": asset,
            "exchange": exchanges[0],
            "definition_rows_raw": len(raw_definitions),
            "definition_history_rows": len(definitions),
            "outright_symbols": int(
                definitions.loc[
                    definitions["instrument_class"] == "F", "raw_symbol"
                ].nunique()
            ),
            "ohlcv_rows_raw": len(raw_ohlcv),
            "ohlcv_rows_outright": len(ohlcv),
            "pre_definition_rows_excluded": unmapped_rows_dropped,
            "first_ohlcv": ohlcv.index.min().isoformat() if len(ohlcv) else None,
            "last_ohlcv": ohlcv.index.max().isoformat() if len(ohlcv) else None,
            "partitions": len(partition_hashes),
            "source_definition_sha256": _sha256(definition_path),
            "source_ohlcv_sha256": _sha256(ohlcv_path),
        }

    report["complete"] = True
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("No gold data, label, return, model score, or strategy outcome was read.")


def _verified_path(item: dict) -> Path:
    path = Path(item["path"])
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise SystemExit(f"Driver source fails its checksum: {path}")
    return path


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=index)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
