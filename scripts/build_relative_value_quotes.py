"""Validate and synchronize the four outcome-blind RV1 quote streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.intraday.relative_value import synchronize_quotes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "data" / "raw" / "relative_value_quotes"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "gold_silver_m30_quotes.parquet"
        ),
    )
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("strategy_evaluated") is not False:
        raise SystemExit("RV1 source manifest is not outcome-blind")

    frames = {}
    for key in ("gold_bid", "gold_ask", "silver_bid", "silver_ask"):
        item = source_manifest["artifacts"][key]
        path = Path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise SystemExit(f"RV1 source checksum fails for {key}")
        frames[key] = pd.read_parquet(path)
    result = synchronize_quotes(
        frames["gold_bid"],
        frames["gold_ask"],
        frames["silver_bid"],
        frames["silver_ask"],
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    result.frame.to_parquet(temporary)
    temporary.replace(output)
    report = {
        "schema_version": 1,
        "program": "rv1_synchronized_quotes",
        "frozen_design": "docs/intraday_relative_value_design.md",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "synchronization": result.report,
        "output": str(output),
        "output_sha256": _sha256(output),
        "outcomes_calculated": False,
        "signals_calculated": False,
        "strategy_evaluated": False,
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("No residual, candidate, forward return, or strategy P&L was calculated.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
