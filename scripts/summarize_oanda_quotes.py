"""Summarize the prospective OANDA quote database without a network request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

from commodity_ml.intraday.quote_quality import summarize_quote_collection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/runtime/oanda_stream_quotes.sqlite"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.database.exists():
        raise SystemExit(f"Missing OANDA quote database: {args.database}")

    uri = f"file:{args.database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            schema_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            quotes = pd.read_sql_query(
                "SELECT instrument, broker_time, received_at, bid, ask FROM quotes",
                connection,
            )
            events = pd.read_sql_query(
                "SELECT event_type FROM stream_events", connection
            )
    except sqlite3.Error as exc:
        raise SystemExit(f"Cannot read OANDA quote database: {exc}") from exc

    report = summarize_quote_collection(
        quotes, events, schema_version=schema_version
    )
    payload = report.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    print(
        f"OANDA quote collection: {report.total_quotes:,} quotes, "
        f"{report.total_events:,} events, "
        f"{report.unresolved_invalidating_events} unresolved / "
        f"{report.recovered_invalidating_events} recovered invalidating events"
    )
    for item in report.instruments:
        print(
            f"  {item.instrument}: {item.quotes:,} quotes, "
            f"median/p95 spread {item.median_spread_bp:.2f}/{item.p95_spread_bp:.2f} bp, "
            f"duplicate times {item.duplicate_broker_times}, crossed {item.crossed_quotes}"
        )
    if args.output:
        print(f"Report -> {args.output.resolve()}")
    print("Descriptive collection quality only; no admission or profitability decision.")


if __name__ == "__main__":
    main()
