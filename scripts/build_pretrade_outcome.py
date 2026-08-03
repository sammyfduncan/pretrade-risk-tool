"""Build a validated paper outcome from broker facts and executable quotes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.intraday.quote_path import (
    TradeExecutionRecord,
    outcome_from_executable_quotes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution", type=Path, help="Observed broker execution JSON")
    parser.add_argument("quotes", type=Path, help="Timestamped bid/ask parquet or CSV")
    parser.add_argument("--output", type=Path, required=True, help="Outcome JSON")
    parser.add_argument("--audit-output", type=Path, help="Optional quote-path audit JSON")
    parser.add_argument("--max-boundary-gap-seconds", type=float, default=5.0)
    parser.add_argument("--max-internal-gap-seconds", type=float, default=120.0)
    args = parser.parse_args()

    try:
        execution = TradeExecutionRecord.from_mapping(
            json.loads(args.execution.read_text(encoding="utf-8"))
        )
        quotes = _read_quotes(args.quotes)
        outcome, audit = outcome_from_executable_quotes(
            execution,
            quotes,
            max_boundary_gap_seconds=args.max_boundary_gap_seconds,
            max_internal_gap_seconds=args.max_internal_gap_seconds,
        )
        outcome_json = json.dumps(
            outcome.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        audit_document = {
            "schema_version": 1,
            "quotes_path": str(args.quotes.resolve()),
            "quotes_sha256": hashlib.sha256(args.quotes.read_bytes()).hexdigest(),
            "execution_path": str(args.execution.resolve()),
            "execution_sha256": hashlib.sha256(args.execution.read_bytes()).hexdigest(),
            **audit.to_dict(),
        }
        audit_json = json.dumps(
            audit_document, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Outcome construction refused: {exc}") from exc

    audit_output = args.audit_output or args.output.with_name(
        args.output.stem + "_audit.json"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(outcome_json, encoding="utf-8")
    audit_output.write_text(audit_json, encoding="utf-8")
    print(f"Outcome -> {args.output.resolve()}")
    print(f"Audit -> {audit_output.resolve()}")
    print("No quote was interpolated or filled, and no broker action was performed.")


def _read_quotes(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("quotes must be parquet or CSV")
    if "time" in frame.columns:
        parsed = [pd.Timestamp(value) for value in frame.pop("time")]
        if any(value.tzinfo is None for value in parsed):
            raise ValueError("quote time values must include an explicit timezone")
        frame.index = pd.DatetimeIndex(
            [value.tz_convert("UTC") for value in parsed], name="time"
        )
    return frame


if __name__ == "__main__":
    main()
