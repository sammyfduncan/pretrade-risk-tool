"""Stream normalized GC TBBO partitions into causal minute market state."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.intraday.order_flow import build_active_order_flow_partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tbbo-root",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo"),
    )
    parser.add_argument(
        "--cme-root",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "gold_gc_order_flow.parquet"),
    )
    args = parser.parse_args()
    tbbo_root = Path(args.tbbo_root).resolve()
    cme_root = Path(args.cme_root).resolve()
    source_manifest_path = tbbo_root / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("strategy_evaluated") or source_manifest.get("features_created"):
        raise SystemExit("TBBO source manifest is not a normalization-only artifact")
    paths = sorted((tbbo_root / "tbbo").glob("*.parquet"))
    if not paths:
        raise SystemExit(f"No normalized TBBO partitions in {tbbo_root / 'tbbo'}")
    ohlcv = pd.concat(
        pd.read_parquet(path)
        for path in sorted((cme_root / "ohlcv_1m").glob("*.parquet"))
    ).sort_index()
    definitions = pd.read_parquet(cme_root / "definitions.parquet")
    flow = build_active_order_flow_partitions(paths, ohlcv, definitions)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    flow.to_parquet(temporary)
    temporary.replace(output)
    manifest = {
        "schema_version": 1,
        "program": "gc_tbbo_order_flow_state",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_start": source_manifest["start"],
        "source_end_exclusive": source_manifest["end_exclusive"],
        "source_partitions": len(paths),
        "rows": len(flow),
        "first": flow.index.min().isoformat(),
        "last": flow.index.max().isoformat(),
        "columns": list(flow.columns),
        "output": str(output),
        "output_sha256": _sha256(output),
        "labels_read": False,
        "strategy_evaluated": False,
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"GC order-flow state: {len(flow):,} causal minute rows\n"
        f"  state -> {output}\n"
        f"  manifest -> {manifest_path}\n"
        "No label, candidate return, model, or strategy outcome was read."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
