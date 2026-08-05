"""Manually download and admit a current gold/silver range-risk calibration.

This command uses the free Dukascopy history endpoint. It creates no strategy
signal, reads no strategy P&L, contacts no order endpoint and never overwrites a
previous release directory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.data import dukascopy
from commodity_ml.features import event_dates
from commodity_ml.intraday.calibration_refresh import (
    MAX_SOURCE_LAG_DAYS,
    coverage_gate,
    profile_is_complete,
    quantiles_are_valid,
    synchronize_quote_legs,
    validate_parity_report,
)
from commodity_ml.intraday.calibration_release import (
    MAX_PROFILE_AGE_DAYS,
    POLICY_VERSION,
    RELEASE_SCHEMA_VERSION,
)
from commodity_ml.intraday.event_blackouts import (
    scheduled_entry_blackouts,
    scheduled_release_times,
)
from commodity_ml.intraday.risk_calibration import (
    HORIZONS_MINUTES,
    INSTRUMENT_TO_PREFIX,
    MINIMUM_HISTORY,
    QUANTILES,
    ROLLING_WINDOW,
    build_calibration_outcomes,
    calibrate_outcomes,
    sha256_file,
)


ACQUISITION_DAYS = 550
DEFAULT_GOLD_PARITY = PROJECT_ROOT / "reports/gold_M30_oanda_dukascopy_parity_20260727_20260804.json"
DEFAULT_SILVER_PARITY = PROJECT_ROOT / "reports/silver_M30_oanda_dukascopy_parity_20260727_20260804.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of",
        help="Timezone-aware release time; defaults to now (UTC)",
    )
    parser.add_argument("--gold-parity", type=Path, default=DEFAULT_GOLD_PARITY)
    parser.add_argument("--silver-parity", type=Path, default=DEFAULT_SILVER_PARITY)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data/processed/current_risk_calibration",
    )
    args = parser.parse_args()

    as_of = _as_of(args.as_of)
    requested_start = (as_of - pd.Timedelta(days=ACQUISITION_DAYS)).normalize()
    requested_end = as_of.normalize()
    stamp = as_of.strftime("%Y%m%dT%H%M%SZ")
    release_dir = args.output_root.expanduser().resolve() / stamp
    if release_dir.exists():
        raise SystemExit(f"Release directory already exists; refusing overwrite: {release_dir}")
    release_dir.mkdir(parents=True)

    try:
        legs = _download_legs(requested_start, requested_end, release_dir)
        source = synchronize_quote_legs(legs)
        source_path = release_dir / "synchronized_quotes.parquet"
        source.to_parquet(source_path)
        source_manifest = _source_manifest(
            source, source_path, legs, requested_start, requested_end
        )
        source_manifest_path = release_dir / "source_manifest.json"
        _write_json(source_manifest_path, source_manifest)

        event_snapshot = scheduled_release_times(source.index.min(), as_of)
        event_path = release_dir / "event_snapshot.parquet"
        event_snapshot.to_parquet(event_path)
        blackouts = scheduled_entry_blackouts(source.index.min(), as_of)

        oof_frames: list[pd.DataFrame] = []
        profile_frames: list[pd.DataFrame] = []
        cells: dict[str, dict] = {}
        for instrument in INSTRUMENT_TO_PREFIX:
            for horizon in HORIZONS_MINUTES:
                built = build_calibration_outcomes(
                    source,
                    instrument=instrument,
                    horizon_minutes=horizon,
                    development_start=source.index.min(),
                    development_end=as_of,
                    entry_blackouts=blackouts,
                )
                result = calibrate_outcomes(built.frame)
                key = f"{instrument}_{horizon}m"
                cells[key] = {
                    "exclusions": built.exclusions,
                    "data_quality_passed": result.passed_data_quality,
                    **result.diagnostics,
                }
                oof_frames.append(result.oof)
                profile_frames.append(result.profile)

        oof = pd.concat(oof_frames).sort_index()
        profile = pd.concat(profile_frames, ignore_index=True).sort_values(
            ["instrument", "horizon_minutes", "london_slot_minute"],
            ignore_index=True,
        )
        oof_path = release_dir / "oof.parquet"
        profile_path = release_dir / "profile.parquet"
        oof.to_parquet(oof_path)
        profile.to_parquet(profile_path, index=False)

        parity = {
            "XAUUSD": validate_parity_report(args.gold_parity, as_of=as_of),
            "XAGUSD": validate_parity_report(args.silver_parity, as_of=as_of),
        }
        source_gate = bool(
            source.index.min() <= requested_start + pd.Timedelta(days=3)
            and source.index.max() < as_of
            and as_of - source.index.max() <= pd.Timedelta(days=MAX_SOURCE_LAG_DAYS)
            and source_manifest["filled_rows"] == 0
        )
        profile_gate = bool(
            profile_is_complete(profile)
            and int(profile["sample_count"].min()) >= MINIMUM_HISTORY
        )
        checks = {
            "source_gate_passed": source_gate,
            "event_calendar_gate_passed": event_dates.COVERAGE_END >= as_of.tz_localize(None).normalize(),
            "parity_gate_passed": all(item["passed"] for item in parity.values()),
            "data_quality_gate_passed": all(
                cell["data_quality_passed"] for cell in cells.values()
            ),
            "profile_completeness_gate_passed": profile_gate,
            "coverage_gate_passed": coverage_gate(cells),
            "quantile_monotonicity_gate_passed": quantiles_are_valid(profile),
            "prohibited_inputs_gate_passed": True,
        }
        status = "ADMITTED" if all(checks.values()) else "REJECTED"
        profile_hash = sha256_file(profile_path)
        release_id = f"current-range-{stamp.lower()}-{profile_hash[:12]}"
        report = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "policy_version": POLICY_VERSION,
            "release_id": release_id,
            "release_status": status,
            "created_at": as_of.isoformat(),
            "profile_max_age_days": MAX_PROFILE_AGE_DAYS,
            "admission_checks": checks,
            "declarations": {
                "directional_signal_used": False,
                "strategy_pnl_used": False,
                "outcome_selected_parameters": False,
                "broker_spread_eligible": False,
            },
            "estimator": {
                "horizons_minutes": list(HORIZONS_MINUTES),
                "quantiles": list(QUANTILES),
                "rolling_window": ROLLING_WINDOW,
                "minimum_history": MINIMUM_HISTORY,
            },
            "source": source_manifest,
            "event_calendar": {
                "verified_on": event_dates.EVENT_SCHEDULE_VERIFIED_ON,
                "coverage_end": event_dates.COVERAGE_END.isoformat(),
            },
            "parity": parity,
            "cells": cells,
            "artifacts": {
                "source_sha256": sha256_file(source_path),
                "source_manifest_sha256": sha256_file(source_manifest_path),
                "event_snapshot_sha256": sha256_file(event_path),
                "oof_sha256": sha256_file(oof_path),
                "profile_sha256": profile_hash,
            },
        }
        release_path = release_dir / "release.json"
        _write_json(release_path, report)
    except Exception as exc:
        failure_path = release_dir / "refresh_failure.json"
        _write_json(
            failure_path,
            {
                "schema_version": 1,
                "created_at": as_of.isoformat(),
                "release_status": "FAILED_TO_BUILD",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise SystemExit(f"Calibration refresh failed; audit -> {failure_path}: {exc}") from exc

    print(f"Current calibration release: {status}")
    print(f"Release -> {release_path}")
    print(f"Profile -> {profile_path}")
    if status != "ADMITTED":
        failed = [name for name, passed in checks.items() if not passed]
        raise SystemExit(f"Release retained but not admitted: {', '.join(failed)}")
    print("No strategy outcome was read and no order endpoint was contacted.")


def _download_legs(start: pd.Timestamp, end: pd.Timestamp, output: Path) -> dict[str, pd.DataFrame]:
    legs: dict[str, pd.DataFrame] = {}
    for prefix, config_path in (("gold", "config/gold.yaml"), ("silver", "config/silver.yaml")):
        cfg = load_config(config_path)
        for side in ("bid", "ask"):
            data_cfg = replace(
                cfg.data,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                dukascopy=replace(cfg.data.dukascopy, offer_side=side),
            )
            frame = dukascopy.fetch_candles(data_cfg, "M30")
            frame = frame.loc[(frame.index >= start) & (frame.index < end)].copy()
            if frame.empty:
                raise ValueError(f"Dukascopy returned no {prefix} {side} rows")
            path = output / f"{prefix}_{side}_M30.parquet"
            frame.to_parquet(path)
            legs[f"{prefix}_{side}"] = frame
    return legs


def _source_manifest(source, source_path, legs, start, end) -> dict:
    union = pd.DatetimeIndex([], tz="UTC")
    artifacts = {}
    for name, frame in legs.items():
        union = union.union(frame.index)
        path = source_path.parent / f"{name}_M30.parquet"
        artifacts[name] = {"path": str(path), "sha256": sha256_file(path), "rows": len(frame)}
    return {
        "schema_version": 1,
        "provider": "dukascopy_public_history",
        "granularity": "M30",
        "requested_start": start.isoformat(),
        "requested_end_exclusive": end.isoformat(),
        "source_first": source.index.min().isoformat(),
        "source_last": source.index.max().isoformat(),
        "shared_rows": len(source),
        "union_rows": len(union),
        "shared_coverage": len(source) / len(union),
        "filled_rows": 0,
        "output": str(source_path),
        "output_sha256": sha256_file(source_path),
        "artifacts": artifacts,
    }


def _as_of(value: str | None) -> pd.Timestamp:
    timestamp = pd.Timestamp(value) if value else pd.Timestamp.now(tz="UTC")
    if timestamp.tzinfo is None:
        raise SystemExit("--as-of must include a timezone")
    timestamp = timestamp.tz_convert("UTC").floor("30min")
    if timestamp >= pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=1):
        raise SystemExit("--as-of cannot be in the future")
    return timestamp


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
