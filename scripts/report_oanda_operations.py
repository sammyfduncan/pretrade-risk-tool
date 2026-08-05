"""Combine OANDA shadow and quote evidence into an operational readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.intraday.operational_report import build_operational_readiness_report
from commodity_ml.intraday.quote_quality import summarize_quote_collection
from commodity_ml.intraday.shadow import ShadowCheckStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", default=[])
    parser.add_argument(
        "--strategy", action="append", choices=("P1", "P2", "P3"), default=[]
    )
    parser.add_argument(
        "--shadow-database",
        type=Path,
        default=Path("data/runtime/oanda_shadow_checks.sqlite"),
    )
    parser.add_argument(
        "--quote-database",
        type=Path,
        default=Path("data/runtime/oanda_stream_quotes.sqlite"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/oanda_operational_readiness.json"),
    )
    args = parser.parse_args()

    for path, label in (
        (args.shadow_database, "shadow"),
        (args.quote_database, "quote"),
    ):
        if not path.exists():
            raise SystemExit(f"Missing {label} database: {path}")
    config_paths = args.config or ["config/gold.yaml", "config/silver.yaml"]
    configs = [load_config(path) for path in config_paths]
    instrument_map = {
        cfg.symbol: cfg.data.oanda.instrument for cfg in configs
    }
    if any(not value for value in instrument_map.values()):
        raise SystemExit("Every report config must set data.oanda.instrument")

    checks = ShadowCheckStore(args.shadow_database).read_frame()
    uri = f"file:{args.quote_database.resolve()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            schema_version = int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            )
            quotes = pd.read_sql_query(
                "SELECT instrument, broker_time, received_at, bid, ask FROM quotes",
                connection,
            )
            events = pd.read_sql_query(
                "SELECT event_type FROM stream_events", connection
            )
    except sqlite3.Error as exc:
        raise SystemExit(f"Cannot read OANDA quote database: {exc}") from exc

    report = build_operational_readiness_report(
        checks,
        summarize_quote_collection(quotes, events, schema_version=schema_version),
        instrument_map=instrument_map,
        strategies=tuple(args.strategy or ("P1", "P2", "P3")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    shadow = report["shadow"]
    print(f"=== OANDA OPERATIONAL READINESS: {report['readiness']} ===")
    print(
        f"Shadow: {shadow['total_checks']} checks, {shadow['signal_count']} signals, "
        f"{shadow['london_session_days']} London session days"
    )
    print(
        f"Issues: {len(report['integrity_issues'])} integrity, "
        f"{len(report['collection_issues'])} collection"
    )
    print(f"Report -> {args.output.resolve()}")
    print("No network request or broker action was performed.")


if __name__ == "__main__":
    main()
