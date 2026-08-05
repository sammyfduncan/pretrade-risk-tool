"""Register, resolve and summarize prospective calibration forecasts offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from commodity_ml.intraday.prospective_calibration import (
    ProspectiveCalibrationError,
    ProspectiveCalibrationStore,
    build_weekly_forecasts,
    resolve_matured_forecasts,
)
from commodity_ml.intraday.risk_calibration import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/runtime/prospective_calibration.sqlite"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--profile", type=Path, required=True)
    register.add_argument("--release", type=Path, required=True)
    register.add_argument("--start-date", required=True, help="First future weekday YYYY-MM-DD")
    register.add_argument(
        "--weekdays",
        type=int,
        choices=range(1, 6),
        default=5,
        help="Consecutive weekdays to register; default 5",
    )
    register.add_argument(
        "--registered-at",
        help="Timezone-aware audit override; defaults to now UTC",
    )

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--source", type=Path, required=True)
    resolve.add_argument("--as-of", help="Timezone-aware audit override; defaults to now UTC")

    subparsers.add_parser("summary")
    args = parser.parse_args()
    store = ProspectiveCalibrationStore(args.database)

    try:
        if args.command == "register":
            registered_at = _timestamp(args.registered_at)
            forecasts = build_weekly_forecasts(
                args.profile,
                args.release,
                start_date=args.start_date,
                registered_at=registered_at,
                weekdays=args.weekdays,
            )
            result = store.record_forecasts(
                forecasts,
                inserted_at=registered_at,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            print(f"Scheduled rows: {len(forecasts):,}")
            print("Both hypothetical sides were frozen; no trade was created.")
        elif args.command == "resolve":
            as_of = _timestamp(args.as_of)
            source = pd.read_parquet(args.source)
            result = resolve_matured_forecasts(
                store,
                source,
                source_sha256=sha256_file(args.source),
                as_of=as_of,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            print("No missing path was filled or counted as covered.")
        else:
            print(json.dumps(store.summary(), indent=2, sort_keys=True, allow_nan=False))
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ProspectiveCalibrationError,
    ) as exc:
        raise SystemExit(f"Prospective calibration refused: {exc}") from exc
    print(f"Ledger -> {args.database.resolve()}")
    print("No broker or order action was performed.")


def _timestamp(value: str | None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value) if value else pd.Timestamp.now(tz="UTC")
    if timestamp.tzinfo is None:
        raise ValueError("audit timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC")


if __name__ == "__main__":
    main()
