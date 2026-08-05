"""Build the outcome-blind causal P4 ZN/6E state artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.intraday.macro_impulse import (
    build_driver_state,
    combine_driver_states,
    load_macro_driver,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_macro_drivers"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "gold_macro_state.parquet"),
    )
    args = parser.parse_args()
    source = Path(args.source).expanduser().resolve()
    zn_ohlcv, zn_definitions, source_manifest = load_macro_driver(source, "zn")
    euro_ohlcv, euro_definitions, second_manifest = load_macro_driver(source, "euro")
    if source_manifest != second_manifest:
        raise SystemExit("ZN and 6E loaders disagree on their source manifest")
    degraded = source_manifest.get("degraded_source_dates", [])

    zn = build_driver_state(
        zn_ohlcv,
        zn_definitions,
        prefix="zn",
        degraded_dates=degraded,
    )
    euro = build_driver_state(
        euro_ohlcv,
        euro_definitions,
        prefix="euro",
        degraded_dates=degraded,
    )
    joint = combine_driver_states(zn, euro)
    cutoff = pd.Timestamp(source_manifest["development_end_exclusive"])
    state = joint.frame.loc[joint.frame.index < cutoff].copy()
    if state.empty or (state.index >= cutoff).any():
        raise SystemExit("P4 state does not respect the development decision fence")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    state.to_parquet(temporary)
    temporary.replace(output)

    source_manifest_path = source / "manifest.json"
    report = {
        "schema_version": 1,
        "program": "p4_macro_driver_state",
        "frozen_design": "docs/archive/intraday_macro_impulse_design.md",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "development_end_exclusive": source_manifest["development_end_exclusive"],
        "degraded_source_dates": degraded,
        "feature_build": joint.report
        | {
            "written_joint_rows": len(state),
            "written_joint_eligible_rows": int(state["eligible"].sum()),
            "written_last_decision": state.index.max().isoformat(),
        },
        "output": str(output),
        "output_sha256": _sha256(output),
        "gold_data_read": False,
        "labels_read": False,
        "strategy_evaluated": False,
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"Macro state -> {output}\n"
        "No gold data, label, return, model score, or strategy outcome was read."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
