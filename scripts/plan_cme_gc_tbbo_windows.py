"""Plan minimal, outcome-blind TBBO requests around frozen P1 decisions.

The plan uses candidate timestamps but deliberately reads no label, return,
price, model score, or strategy outcome. Each request contains only the causal
30-minute feature window and only the GC contract selected from the prior
completed session's volume.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.intraday.futures_state import (
    completed_session_selections,
    load_cme_partitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--strategy", choices=["P1"], default="P1")
    parser.add_argument(
        "--cme-root",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc"),
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo_request_plan.csv"
        ),
    )
    parser.add_argument(
        "--estimate-cost",
        action="store_true",
        help=(
            "Call only Databento metadata.get_cost for each planned window. "
            "This does not submit or download market data."
        ),
    )
    parser.add_argument(
        "--cost-cache",
        default=str(
            PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo_cost_cache.csv"
        ),
        help="Resumable metadata-quote cache; contains no market data or secret",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    candidate_path = cfg.intraday_candidate_path(args.strategy)
    candidate_manifest = candidate_path.with_name(candidate_path.stem + "_manifest.json")
    if not candidate_path.is_file() or not candidate_manifest.is_file():
        raise SystemExit("Missing frozen candidate bundle; build P1 candidates first")
    _verify_candidate_bundle(candidate_path, candidate_manifest)

    # Only the index is consumed. Specifying one harmless structural column
    # prevents pandas from reading outcome columns from the Parquet artifact.
    candidate_index = pd.read_parquet(candidate_path, columns=["candidate_id"]).index
    base = pd.read_parquet(cfg.intraday_base_path, columns=["decision_time"])
    decisions = pd.DatetimeIndex(
        pd.to_datetime(base.loc[candidate_index, "decision_time"], utc=True),
        name="decision_time",
    )
    if cfg.holdout_ts is not None and (decisions >= cfg.holdout_ts).any():
        raise SystemExit("Request plan crosses the unopened confirmation boundary")

    ohlcv, _, definitions = load_cme_partitions(args.cme_root)
    selections = completed_session_selections(ohlcv, definitions)
    mapped = pd.merge_asof(
        pd.DataFrame({"decision_time": decisions}).sort_values("decision_time"),
        selections.loc[:, ["available_time", "active_symbol"]].sort_values(
            "available_time"
        ),
        left_on="decision_time",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    unmapped = mapped["active_symbol"].isna()
    # The first development candidate can precede the first completed CME
    # session in the delivery. It is structurally ineligible in both matched
    # arms and must not be assigned a contract using same-session volume.
    mapped = mapped.loc[~unmapped].copy()
    if mapped.empty:
        raise SystemExit("No decisions have a causal active-contract selection")

    plan = pd.DataFrame(
        {
            "dataset": "GLBX.MDP3",
            "schema": "tbbo",
            "stype_in": "raw_symbol",
            "symbol": mapped["active_symbol"].astype(str),
            "start": mapped["decision_time"] - pd.Timedelta(minutes=30),
            "end_exclusive": mapped["decision_time"],
        }
    )
    plan = _merge_overlapping_windows(plan)
    if args.estimate_cost:
        import databento as db

        load_dotenv(PROJECT_ROOT / ".env")
        # This object is deliberately passed to a helper that can reach only
        # its metadata namespace. No paid time-series or batch method appears
        # anywhere in this planning script.
        client = db.Historical()
        plan["quoted_cost_usd"] = _quote_plan_costs(
            client.metadata,
            plan,
            cache_path=Path(args.cost_cache).expanduser().resolve(),
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")

    requested = (
        pd.to_datetime(plan["end_exclusive"], utc=True)
        - pd.to_datetime(plan["start"], utc=True)
    ).sum()
    report = {
        "schema_version": 1,
        "program": "of_gc_candidate_window_request_plan",
        "strategy": args.strategy,
        "candidate_rows": len(candidate_index),
        "candidates_without_prior_session_selection": int(unmapped.sum()),
        "planned_candidate_rows": len(mapped),
        "planned_candidate_coverage": len(mapped) / len(candidate_index),
        "request_rows": len(plan),
        "unique_raw_symbols": int(plan["symbol"].nunique()),
        "requested_hours": requested.total_seconds() / 3600,
        "feature_lookback_minutes": 30,
        "candidate_sha256": _sha256(candidate_path),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "confirmation_end_exclusive": cfg.holdout_ts.isoformat()
        if cfg.holdout_ts is not None
        else None,
        "labels_read": False,
        "returns_read": False,
        "model_scores_read": False,
        "strategy_evaluated": False,
        "output": str(output),
        "output_sha256": _sha256(output),
    }
    if "quoted_cost_usd" in plan:
        report["quoted_cost_usd"] = float(plan["quoted_cost_usd"].sum())
        report["maximum_window_cost_usd"] = float(plan["quoted_cost_usd"].max())
        report["cost_endpoint"] = "Historical.metadata.get_cost"
        report["paid_endpoint_called"] = False
    manifest = output.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(
        f"Request plan -> {output}\n"
        "No label, return, model score, or strategy outcome was read."
    )


def _merge_overlapping_windows(plan: pd.DataFrame) -> pd.DataFrame:
    """Merge only overlapping/adjacent intervals for the same raw symbol."""

    rows: list[dict] = []
    for symbol, group in plan.groupby("symbol", sort=True):
        group = group.sort_values("start")
        for row in group.itertuples(index=False):
            if rows and rows[-1]["symbol"] == symbol and row.start <= rows[-1]["end_exclusive"]:
                rows[-1]["end_exclusive"] = max(rows[-1]["end_exclusive"], row.end_exclusive)
            else:
                rows.append(
                    {
                        "dataset": row.dataset,
                        "schema": row.schema,
                        "stype_in": row.stype_in,
                        "symbol": symbol,
                        "start": row.start,
                        "end_exclusive": row.end_exclusive,
                    }
                )
    return pd.DataFrame(rows).sort_values(["start", "symbol"]).reset_index(drop=True)


def _quote_plan_costs(
    metadata,
    plan: pd.DataFrame,
    *,
    cache_path: Path | None = None,
    max_attempts: int = 4,
    workers: int = 8,
) -> list[float]:
    """Quote disjoint windows through the free metadata namespace only."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    cache: dict[str, float] = {}
    if cache_path is not None and cache_path.is_file():
        cached = pd.read_csv(cache_path)
        if set(cached) != {"request_key", "quoted_cost_usd"}:
            raise ValueError("Cost cache has an unexpected schema")
        cache = dict(zip(cached["request_key"], cached["quoted_cost_usd"], strict=True))

    rows = list(plan.itertuples(index=False))
    missing = [(row, _request_key(row)) for row in rows if _request_key(row) not in cache]
    completed = len(rows) - len(missing)
    if cache_path is not None and completed:
        print(f"Cost metadata: resumed with {completed:,}/{len(rows):,} cached windows", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_quote_one, metadata, row, max_attempts): key
            for row, key in missing
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()
            completed += 1
            _write_cost_cache(cache_path, cache)
            if cache_path is not None and (completed % 50 == 0 or completed == len(rows)):
                print(f"Cost metadata: {completed:,}/{len(rows):,} windows quoted", flush=True)
    return [cache[_request_key(row)] for row in rows]


def _quote_one(metadata, row, max_attempts: int) -> float:
    for attempt in range(1, max_attempts + 1):
        try:
            return float(
                metadata.get_cost(
                    dataset=row.dataset,
                    symbols=row.symbol,
                    schema=row.schema,
                    stype_in=row.stype_in,
                    start=row.start,
                    end=row.end_exclusive,
                )
            )
        except requests.RequestException:
            if attempt == max_attempts:
                raise
            time.sleep(2 ** (attempt - 1))
    raise AssertionError("unreachable")


def _request_key(row) -> str:
    return "|".join(
        (
            str(row.dataset),
            str(row.schema),
            str(row.stype_in),
            str(row.symbol),
            pd.Timestamp(row.start).isoformat(),
            pd.Timestamp(row.end_exclusive).isoformat(),
        )
    )


def _write_cost_cache(cache_path: Path | None, cache: dict[str, float]) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    pd.DataFrame(
        {
            "request_key": list(cache),
            "quoted_cost_usd": list(cache.values()),
        }
    ).to_csv(temporary, index=False)
    temporary.replace(cache_path)


def _verify_candidate_bundle(candidate_path: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("dataset_sha256") or manifest.get("output_sha256")
    if expected is not None and expected != _sha256(candidate_path):
        raise SystemExit("Candidate bundle hash disagrees with its manifest")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
