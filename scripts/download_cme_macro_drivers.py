"""Download an explicitly authorized frozen P4 ZN/6E request plan.

The separate planner must first record the metadata quote.  This downloader
enforces the caller's dollar ceiling, makes no automatic paid retry, and keeps
an atomic checksum ledger so an interrupted completed request can be adopted
instead of purchased twice.
"""

from __future__ import annotations

import argparse
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
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_macro_request_plan.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_macro_drivers_dbn"),
    )
    parser.add_argument("--authorized-max-usd", type=float, required=True)
    args = parser.parse_args()
    if args.authorized_max_usd <= 0:
        raise SystemExit("--authorized-max-usd must be positive")

    plan_path = Path(args.plan).expanduser().resolve()
    manifest_path = plan_path.with_suffix(".manifest.json")
    plan, manifest = load_authorized_plan(
        plan_path,
        manifest_path,
        authorized_max_usd=args.authorized_max_usd,
    )
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "download_ledger.json"
    ledger = _load_ledger(ledger_path, plan_path, manifest)

    pending = []
    for request_id, row in enumerate(plan.itertuples(index=False), start=1):
        key = f"{request_id:02d}"
        symbol = row.symbol.replace(".", "_")
        final = output / f"{key}_{symbol}.{row.schema}.dbn.zst"
        partial = final.with_suffix(final.suffix + ".partial")
        recorded = ledger["requests"].get(key)
        if recorded is not None:
            if not final.is_file() or _sha256(final) != recorded["sha256"]:
                raise SystemExit(f"Downloaded request {key} fails its checksum")
            continue
        if final.is_file() and not partial.exists():
            ledger["requests"][key] = {
                **_request_record(row, final),
                **_validate_download(row, final),
            }
            _write_json_atomic(ledger_path, ledger)
            continue
        if final.exists() or partial.exists():
            raise SystemExit(
                f"Unledgered or partial paid request exists for {key}; refusing "
                "an automatic duplicate charge"
            )
        pending.append((key, row, final, partial))

    if not pending:
        print("All four P4 driver requests are already downloaded and verified.")
        return

    load_dotenv(PROJECT_ROOT / ".env")
    client = db.Historical()
    print(
        f"Authorized quote ceiling: ${args.authorized_max_usd:.4f}\n"
        f"Frozen plan quote: ${plan['quoted_cost_usd'].sum():.4f}\n"
        f"Downloading {len(pending)} missing request(s) with no automatic retry."
    )
    for key, row, final, partial in pending:
        # A request that raises is intentionally left as a .partial file.  The
        # operator must investigate it; rerunning will refuse a duplicate.
        client.timeseries.get_range(
            dataset=row.dataset,
            symbols=row.symbol,
            schema=row.schema,
            stype_in=row.stype_in,
            stype_out=row.stype_out,
            start=row.start,
            end=row.end_exclusive,
            path=partial,
        )
        validation = _validate_download(row, partial)
        partial.replace(final)
        ledger["requests"][key] = {
            **_request_record(row, final),
            **validation,
        }
        _write_json_atomic(ledger_path, ledger)
        print(f"Paid download: {len(ledger['requests'])}/4 verified", flush=True)

    ledger["complete"] = len(ledger["requests"]) == 4
    ledger["verified_requests"] = len(ledger["requests"])
    ledger["quoted_cost_usd"] = float(plan["quoted_cost_usd"].sum())
    ledger["paid_endpoint"] = "Historical.timeseries.get_range"
    _write_json_atomic(ledger_path, ledger)
    print(
        f"P4 driver delivery complete -> {output}\n"
        "No gold data, label, return, or strategy outcome was read."
    )


def load_authorized_plan(
    plan_path: Path,
    manifest_path: Path,
    *,
    authorized_max_usd: float,
) -> tuple[pd.DataFrame, dict]:
    if not plan_path.is_file() or not manifest_path.is_file():
        raise ValueError("Missing request plan or its manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != _sha256(plan_path):
        raise ValueError("Request plan hash disagrees with its manifest")
    if manifest.get("paid_endpoint_called") is not False:
        raise ValueError("Plan manifest does not attest an unpaid metadata quote")
    if manifest.get("cost_endpoint") != "Historical.metadata.get_cost":
        raise ValueError("Request plan has no verified metadata cost quote")

    plan = pd.read_csv(plan_path, parse_dates=["start", "end_exclusive"])
    required = {
        "dataset",
        "schema",
        "stype_in",
        "stype_out",
        "symbol",
        "start",
        "end_exclusive",
        "encoding",
        "compression",
        "quoted_cost_usd",
    }
    if missing := sorted(required - set(plan)):
        raise ValueError(f"Request plan is missing columns: {missing}")
    expected = {
        ("ZN.FUT", "ohlcv-1m"),
        ("ZN.FUT", "definition"),
        ("6E.FUT", "ohlcv-1m"),
        ("6E.FUT", "definition"),
    }
    if len(plan) != 4 or set(zip(plan["symbol"], plan["schema"], strict=True)) != expected:
        raise ValueError("Request plan differs from the four frozen P4 requests")
    if not (plan["dataset"] == "GLBX.MDP3").all():
        raise ValueError("Only GLBX.MDP3 is authorized")
    if not (plan["stype_in"] == "parent").all() or not (
        plan["stype_out"] == "instrument_id"
    ).all():
        raise ValueError("P4 requires parent input and instrument_id output")
    if not (plan["encoding"] == "dbn").all() or not (
        plan["compression"] == "zstd"
    ).all():
        raise ValueError("P4 requires DBN with Zstandard compression")
    if not (plan["start"] == pd.Timestamp("2020-01-01T00:00:00Z")).all() or not (
        plan["end_exclusive"] == pd.Timestamp("2025-07-01T00:00:00Z")
    ).all():
        raise ValueError("P4 request dates differ from the frozen development period")
    quoted = float(plan["quoted_cost_usd"].sum())
    if abs(quoted - float(manifest.get("quoted_cost_usd", -1))) > 1e-8:
        raise ValueError("Per-request quotes disagree with the plan manifest")
    if quoted > authorized_max_usd + 1e-9:
        raise ValueError(
            f"Plan quote ${quoted:.4f} exceeds authorization ${authorized_max_usd:.4f}"
        )
    return plan, manifest


def _validate_download(row, path: Path) -> dict:
    store = db.DBNStore.from_file(path)
    metadata = store.metadata
    if (
        metadata.dataset != row.dataset
        or str(metadata.schema) != row.schema
        or str(metadata.stype_in) != row.stype_in
        or str(metadata.stype_out) != row.stype_out
        or metadata.start != pd.Timestamp(row.start).value
        or metadata.end != pd.Timestamp(row.end_exclusive).value
        or metadata.symbols != [row.symbol]
        or metadata.not_found
    ):
        raise ValueError("Downloaded DBN metadata disagrees with its frozen request")
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_ledger(path: Path, plan_path: Path, manifest: dict) -> dict:
    if path.is_file():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("plan_sha256") != _sha256(plan_path):
            raise ValueError("Existing download ledger belongs to another plan")
        return ledger
    return {
        "schema_version": 1,
        "program": "p4_macro_driver_download",
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "plan_manifest_sha256": _sha256(plan_path.with_suffix(".manifest.json")),
        "quoted_cost_usd": manifest["quoted_cost_usd"],
        "requests": {},
        "complete": False,
        "gold_data_read": False,
        "labels_read": False,
        "strategy_evaluated": False,
    }


def _request_record(row, path: Path) -> dict:
    return {
        "symbol": row.symbol,
        "schema": row.schema,
        "start": pd.Timestamp(row.start).isoformat(),
        "end_exclusive": pd.Timestamp(row.end_exclusive).isoformat(),
        "quoted_cost_usd": float(row.quoted_cost_usd),
        "path": str(path),
    }


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
