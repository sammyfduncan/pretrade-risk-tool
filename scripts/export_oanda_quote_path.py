"""Export one outage-free OANDA bid/ask path from the raw stream database."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.intraday.quote_store import QuoteStoreError, StreamingQuoteStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    try:
        start, end = _timestamp(args.start), _timestamp(args.end)
        with StreamingQuoteStore(args.database) as store:
            frame = store.quote_path(instrument=args.instrument, start=start, end=end)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(args.output)
        audit = {
            "schema_version": 2,
            "source_database": str(args.database.resolve()),
            "broker": "OANDA_V20",
            "instrument": args.instrument,
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "first_broker_time": frame.index[0].isoformat(),
            "last_broker_time": frame.index[-1].isoformat(),
            "quote_rows": len(frame),
            "maximum_broker_gap_seconds": float(frame.index.to_series().diff().dt.total_seconds().max()) if len(frame) > 1 else None,
            "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        }
        encoded = json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, RuntimeError, TypeError, ValueError, QuoteStoreError) as exc:
        raise SystemExit(f"Quote-path export refused: {exc}") from exc
    audit_output = args.audit_output or args.output.with_name(args.output.stem + "_export_audit.json")
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(encoded, encoding="utf-8")
    print(f"Quote path -> {args.output.resolve()}")
    print(f"Export audit -> {audit_output.resolve()}")
    print("No outage was filled and no broker action was performed.")


def _timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return timestamp.tz_convert("UTC")


if __name__ == "__main__":
    main()
