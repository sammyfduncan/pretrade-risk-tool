"""Build the frozen causal RV1 hedge/residual state without forward outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.intraday.relative_value import build_relative_value_state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quotes",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "gold_silver_m30_quotes.parquet"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "gold_silver_rv1_state.parquet"
        ),
    )
    args = parser.parse_args()
    quotes_path = Path(args.quotes).expanduser().resolve()
    quotes_manifest_path = quotes_path.with_name(quotes_path.stem + "_manifest.json")
    quotes_manifest = json.loads(quotes_manifest_path.read_text(encoding="utf-8"))
    if quotes_manifest.get("output_sha256") != _sha256(quotes_path):
        raise SystemExit("RV1 synchronized quote hash disagrees with its manifest")
    if quotes_manifest.get("strategy_evaluated") is not False:
        raise SystemExit("RV1 synchronized quotes are not outcome-blind")
    quotes = pd.read_parquet(quotes_path).sort_index()
    result = build_relative_value_state(quotes)
    cutoff = pd.Timestamp("2025-07-01T00:00:00Z")
    state = result.frame.loc[
        pd.to_datetime(result.frame["decision_time"], utc=True) < cutoff
    ].copy()
    if state.empty or (pd.to_datetime(state["decision_time"], utc=True) >= cutoff).any():
        raise SystemExit("RV1 state crosses the confirmation boundary")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    state.to_parquet(temporary)
    temporary.replace(output)
    report = {
        "schema_version": 1,
        "program": "rv1_causal_state",
        "frozen_design": "docs/archive/intraday_relative_value_design.md",
        "quotes": str(quotes_path),
        "quotes_sha256": _sha256(quotes_path),
        "quotes_manifest": str(quotes_manifest_path),
        "quotes_manifest_sha256": _sha256(quotes_manifest_path),
        "state_build": result.report
        | {
            "written_rows": len(state),
            "written_eligible_rows": int(state["eligible"].sum()),
            "written_last_decision": pd.Timestamp(state["decision_time"].max()).isoformat(),
        },
        "output": str(output),
        "output_sha256": _sha256(output),
        "confirmation_cutoff": cutoff.isoformat(),
        "forward_outcomes_read": False,
        "candidates_created": False,
        "strategy_evaluated": False,
    }
    manifest_path = output.with_name(output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("No candidate, forward return, or strategy P&L was calculated.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
