"""Download the authorized sparse GC TBBO plan with a hard quote ceiling.

This is intentionally separate from the free planning script: importing this
module has no side effects, while running it calls Databento's paid historical
time-series endpoint exactly once per missing planned window. Failed or partial
requests are never retried automatically because duplicate streaming requests
can be billed twice.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import databento as db
import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo_request_plan.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo_sparse"),
    )
    parser.add_argument("--authorized-max-usd", type=float, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.authorized_max_usd <= 0:
        raise SystemExit("--authorized-max-usd must be positive")
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")

    plan_path = Path(args.plan).expanduser().resolve()
    manifest_path = plan_path.with_suffix(".manifest.json")
    plan, plan_manifest = _load_authorized_plan(
        plan_path, manifest_path, args.authorized_max_usd
    )
    root = Path(args.output).expanduser().resolve()
    data_root = root / "dbn"
    data_root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "download_ledger.json"
    ledger = _load_ledger(ledger_path, plan_path, plan_manifest)

    requests_to_run = []
    for request_id, row in enumerate(plan.itertuples(index=False), start=1):
        key = f"{request_id:04d}"
        final = data_root / f"{key}_{row.symbol}_{_stamp(row.start)}.tbbo.dbn.zst"
        partial = final.with_suffix(final.suffix + ".partial")
        recorded = ledger["windows"].get(key)
        if recorded is not None:
            if not final.is_file() or _sha256(final) != recorded["sha256"]:
                raise SystemExit(f"Downloaded window {key} fails its recorded checksum")
            continue
        if final.is_file() and not partial.exists():
            # A process interruption can occur after an atomic file rename but
            # before the batch ledger write. Validate and adopt that completed
            # file instead of issuing a duplicate paid request.
            ledger["windows"][key] = {
                "symbol": row.symbol,
                "start": pd.Timestamp(row.start).isoformat(),
                "end_exclusive": pd.Timestamp(row.end_exclusive).isoformat(),
                "quoted_cost_usd": float(row.quoted_cost_usd),
                "path": str(final),
                **_validate_downloaded_file(row, final),
            }
            continue
        if final.exists() or partial.exists():
            raise SystemExit(
                f"Unledgered or partial paid request exists for window {key}; "
                "refusing an automatic duplicate charge"
            )
        requests_to_run.append((key, row, final, partial))

    _write_json_atomic(ledger_path, ledger)

    if not requests_to_run:
        print("All sparse TBBO windows are already downloaded and verified.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    client = db.Historical()
    completed_before = len(ledger["windows"])
    print(
        f"Authorized quote ceiling: ${args.authorized_max_usd:.4f}\n"
        f"Frozen plan quote: ${plan['quoted_cost_usd'].sum():.4f}\n"
        f"Downloading {len(requests_to_run):,} missing windows; "
        f"{completed_before:,} already verified."
    )

    for offset in range(0, len(requests_to_run), args.workers):
        batch = requests_to_run[offset : offset + args.workers]
        successes = []
        errors = []
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(_download_one, client.timeseries, row, final, partial): (
                    key,
                    row,
                    final,
                )
                for key, row, final, partial in batch
            }
            for future in as_completed(futures):
                key, row, final = futures[future]
                try:
                    successes.append((key, row, final, future.result()))
                except Exception as exc:  # preserve successes, then stop all new spending
                    errors.append((key, exc))

        for key, row, final, report in successes:
            ledger["windows"][key] = {
                "symbol": row.symbol,
                "start": pd.Timestamp(row.start).isoformat(),
                "end_exclusive": pd.Timestamp(row.end_exclusive).isoformat(),
                "quoted_cost_usd": float(row.quoted_cost_usd),
                "path": str(final),
                **report,
            }
        _write_json_atomic(ledger_path, ledger)
        complete = len(ledger["windows"])
        if complete % 25 < len(successes) or complete == len(plan):
            print(f"Paid download: {complete:,}/{len(plan):,} windows verified", flush=True)
        if errors:
            detail = ", ".join(f"{key}: {exc}" for key, exc in errors)
            raise SystemExit(
                "A paid request failed. Successful peers were retained; no retry was "
                f"attempted. Resolve before resuming: {detail}"
            )

    ledger["complete"] = len(ledger["windows"]) == len(plan)
    ledger["verified_windows"] = len(ledger["windows"])
    ledger["quoted_cost_usd"] = float(plan["quoted_cost_usd"].sum())
    ledger["paid_endpoint"] = "Historical.timeseries.get_range"
    ledger["labels_read"] = False
    ledger["strategy_evaluated"] = False
    _write_json_atomic(ledger_path, ledger)
    print(
        f"Sparse TBBO delivery complete -> {root}\n"
        "No label, return, model score, or strategy outcome was read."
    )


def _load_authorized_plan(
    plan_path: Path,
    manifest_path: Path,
    authorized_max_usd: float,
) -> tuple[pd.DataFrame, dict]:
    if not plan_path.is_file() or not manifest_path.is_file():
        raise ValueError("Missing request plan or its manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != _sha256(plan_path):
        raise ValueError("Request plan hash disagrees with its manifest")
    if manifest.get("paid_endpoint_called") is not False:
        raise ValueError("Plan manifest does not attest an unpaid metadata quote")
    plan = pd.read_csv(plan_path, parse_dates=["start", "end_exclusive"])
    required = {
        "dataset",
        "schema",
        "stype_in",
        "symbol",
        "start",
        "end_exclusive",
        "quoted_cost_usd",
    }
    if missing := sorted(required - set(plan)):
        raise ValueError(f"Request plan is missing columns: {missing}")
    if len(plan) != manifest.get("request_rows"):
        raise ValueError("Request plan row count disagrees with its manifest")
    if not (plan["dataset"] == "GLBX.MDP3").all():
        raise ValueError("Only GLBX.MDP3 is authorized")
    if not (plan["schema"] == "tbbo").all():
        raise ValueError("Only TBBO is authorized")
    if not (plan["stype_in"] == "raw_symbol").all():
        raise ValueError("Only preselected raw symbols are authorized")
    if (plan["end_exclusive"] - plan["start"] > pd.Timedelta(minutes=30)).any():
        raise ValueError("A request exceeds the frozen 30-minute lookback")
    total = float(plan["quoted_cost_usd"].sum())
    if abs(total - float(manifest.get("quoted_cost_usd", -1))) > 1e-8:
        raise ValueError("Per-window quotes disagree with the plan manifest")
    if total > authorized_max_usd + 1e-9:
        raise ValueError(
            f"Plan quote ${total:.4f} exceeds authorization ${authorized_max_usd:.4f}"
        )
    return plan, manifest


def _load_ledger(ledger_path: Path, plan_path: Path, manifest: dict) -> dict:
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("plan_sha256") != manifest["output_sha256"]:
            raise ValueError("Download ledger belongs to a different request plan")
        return ledger
    return {
        "schema_version": 1,
        "program": "of_gc_sparse_tbbo_download",
        "plan": str(plan_path),
        "plan_sha256": manifest["output_sha256"],
        "windows": {},
        "complete": False,
    }


def _download_one(timeseries, row, final: Path, partial: Path) -> dict:
    # No retry here: Databento bills outbound streaming bytes, so an automatic
    # retry could exceed the user's authorization after a partial transfer.
    timeseries.get_range(
        dataset=row.dataset,
        symbols=row.symbol,
        schema=row.schema,
        stype_in=row.stype_in,
        stype_out="instrument_id",
        start=row.start,
        end=row.end_exclusive,
        path=partial,
    )
    report = _validate_downloaded_file(row, partial)
    partial.replace(final)
    return report


def _validate_downloaded_file(row, path: Path) -> dict:
    store = db.DBNStore.from_file(path)
    metadata = store.metadata
    expected_start = pd.Timestamp(row.start).value
    expected_end = pd.Timestamp(row.end_exclusive).value
    if (
        metadata.dataset != row.dataset
        or str(metadata.schema) != row.schema
        or str(metadata.stype_in) != row.stype_in
        or str(metadata.stype_out) != "instrument_id"
        or metadata.start != expected_start
        or metadata.end != expected_end
        or metadata.symbols != [row.symbol]
        or metadata.not_found
    ):
        raise ValueError("Downloaded DBN metadata disagrees with its planned window")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _stamp(value) -> str:
    return pd.Timestamp(value).strftime("%Y%m%dT%H%M%SZ")


def _write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
