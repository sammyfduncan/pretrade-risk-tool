"""Quote the frozen P4 ZN/6E development requests without downloading data.

This script has no time-series or batch client path.  Its only remote calls are
to ``Historical.metadata.get_cost``, which Databento documents as a metadata
quote rather than a data purchase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import PROJECT_ROOT

DATASET = "GLBX.MDP3"
START = pd.Timestamp("2020-01-01T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2025-07-01T00:00:00Z")
REQUESTS = (
    ("ZN.FUT", "ohlcv-1m"),
    ("ZN.FUT", "definition"),
    ("6E.FUT", "ohlcv-1m"),
    ("6E.FUT", "definition"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_macro_request_plan.csv"),
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Call only Databento metadata.get_cost; never submit a data request",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    plan = registered_plan()
    conditions = None
    if args.estimate_cost:
        import databento as db

        load_dotenv(PROJECT_ROOT / ".env")
        client = db.Historical()
        cached = cached_quotes(output, plan)
        plan["quoted_cost_usd"] = (
            cached if cached is not None else quote_costs(client.metadata, plan)
        )
        conditions = client.metadata.get_dataset_condition(
            dataset=DATASET,
            start_date=START.date(),
            end_date=(END_EXCLUSIVE - pd.Timedelta(days=1)).date(),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    report = {
        "schema_version": 1,
        "program": "p4_macro_driver_request_plan",
        "frozen_design": "docs/intraday_macro_impulse_design.md",
        "dataset": DATASET,
        "development_start": START.isoformat(),
        "development_end_exclusive": END_EXCLUSIVE.isoformat(),
        "requests": len(plan),
        "parents": sorted(plan["symbol"].unique()),
        "schemas": sorted(plan["schema"].unique()),
        "stype_in": "parent",
        "stype_out": "instrument_id",
        "encoding": "dbn",
        "compression": "zstd",
        "labels_read": False,
        "gold_data_read": False,
        "strategy_evaluated": False,
        "paid_endpoint_called": False,
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    if "quoted_cost_usd" in plan:
        report["quoted_cost_usd"] = float(plan["quoted_cost_usd"].sum())
        report["cost_by_request"] = {
            f"{row.symbol}:{row.schema}": float(row.quoted_cost_usd)
            for row in plan.itertuples(index=False)
        }
        report["cost_endpoint"] = "Historical.metadata.get_cost"
    if conditions is not None:
        condition_path = output.with_suffix(".conditions.json")
        condition_path.write_text(
            json.dumps(conditions, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        unknown = sorted(
            {
                str(item.get("condition"))
                for item in conditions
                if item.get("condition")
                not in {"available", "degraded", "pending", "missing"}
            }
        )
        if unknown:
            raise ValueError(f"Unknown Databento condition codes: {unknown}")
        report["condition_endpoint"] = "Historical.metadata.get_dataset_condition"
        report["condition_rows"] = len(conditions)
        report["degraded_source_dates"] = [
            str(item.get("date"))
            for item in conditions
            if item.get("condition") == "degraded"
        ]
        report["unavailable_source_dates"] = [
            str(item.get("date"))
            for item in conditions
            if item.get("condition") in {"pending", "missing"}
        ]
        report["condition_path"] = str(condition_path)
        report["condition_sha256"] = _sha256(condition_path)
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Request plan -> {output}\nNo market data was requested or downloaded.")


def registered_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "schema": schema,
                "stype_in": "parent",
                "stype_out": "instrument_id",
                "symbol": symbol,
                "start": START,
                "end_exclusive": END_EXCLUSIVE,
                "encoding": "dbn",
                "compression": "zstd",
            }
            for symbol, schema in REQUESTS
        ]
    )


def quote_costs(metadata, plan: pd.DataFrame) -> list[float]:
    """Call only the free cost-estimation method for the frozen request rows."""

    return [
        float(
            metadata.get_cost(
                dataset=row.dataset,
                symbols=row.symbol,
                schema=row.schema,
                stype_in=row.stype_in,
                start=row.start,
                end=row.end_exclusive,
            )
        )
        for row in plan.itertuples(index=False)
    ]


def cached_quotes(path: Path, plan: pd.DataFrame) -> list[float] | None:
    """Reuse an identical completed metadata quote; never cache market data."""

    if not path.is_file():
        return None
    cached = pd.read_csv(path, parse_dates=["start", "end_exclusive"])
    required = set(plan) | {"quoted_cost_usd"}
    if not required.issubset(cached.columns) or len(cached) != len(plan):
        return None
    comparison = cached.loc[:, plan.columns].copy()
    for column in ("start", "end_exclusive"):
        comparison[column] = pd.to_datetime(comparison[column], utc=True)
    expected = plan.reset_index(drop=True).copy()
    if not comparison.reset_index(drop=True).equals(expected):
        return None
    costs = cached["quoted_cost_usd"].astype(float)
    if costs.isna().any() or (costs < 0).any():
        return None
    return costs.tolist()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
