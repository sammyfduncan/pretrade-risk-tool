"""Build development-only gold/silver pre-trade risk calibration artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.intraday.event_blackouts import (
    scheduled_entry_blackouts,
    scheduled_release_times,
)
from commodity_ml.intraday.risk_calibration import (
    DEVELOPMENT_END,
    DEVELOPMENT_START,
    HORIZONS_MINUTES,
    INSTRUMENT_TO_PREFIX,
    QUANTILES,
    build_calibration_outcomes,
    calibrate_outcomes,
    sha256_file,
    volatility_mapping_diagnostics,
)


DESIGN = "docs/pretrade_risk_calibration.md"
SOURCE = PROJECT_ROOT / "data" / "processed" / "gold_silver_m30_quotes.parquet"
SOURCE_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "gold_silver_m30_quotes_manifest.json"
)
EVENT_SOURCE = (
    PROJECT_ROOT / "data" / "raw" / "pretrade_risk_calibration_events.parquet"
)
VOLATILITY_OOF = (
    PROJECT_ROOT / "data" / "processed" / "gold_intraday_joint_risk_oof.parquet"
)
OOF_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "pretrade_risk_calibration_oof.parquet"
)
PROFILE_OUTPUT = (
    PROJECT_ROOT / "data" / "processed" / "pretrade_risk_calibration_profile.parquet"
)
REPORT_OUTPUT = PROJECT_ROOT / "reports" / "pretrade_risk_calibration.json"


def main() -> None:
    source_manifest = _verified_source_manifest()
    quotes = pd.read_parquet(SOURCE)
    if quotes.index.min() < DEVELOPMENT_START or quotes.index.max() >= DEVELOPMENT_END:
        raise SystemExit("Calibration source crosses the registered development boundary")
    _pin_event_snapshot()
    blackouts = scheduled_entry_blackouts(DEVELOPMENT_START, DEVELOPMENT_END)

    oof_frames: list[pd.DataFrame] = []
    profiles: list[pd.DataFrame] = []
    cells: dict[str, dict] = {}
    overall_pass = True
    for instrument in INSTRUMENT_TO_PREFIX:
        for horizon in HORIZONS_MINUTES:
            built = build_calibration_outcomes(
                quotes,
                instrument=instrument,
                horizon_minutes=horizon,
                entry_blackouts=blackouts,
            )
            result = calibrate_outcomes(built.frame)
            key = f"{instrument}_{horizon}m"
            cells[key] = {
                "exclusions": built.exclusions,
                "data_quality_passed": result.passed_data_quality,
                **result.diagnostics,
                "final_profile_summary": _profile_summary(result.profile),
            }
            overall_pass &= result.passed_data_quality
            oof_frames.append(result.oof)
            profiles.append(result.profile)

    oof = pd.concat(oof_frames).sort_values(
        ["instrument", "horizon_minutes", "london_slot_minute"],
        kind="mergesort",
    )
    profile = pd.concat(profiles, ignore_index=True).sort_values(
        ["instrument", "horizon_minutes", "london_slot_minute"],
        kind="mergesort",
        ignore_index=True,
    )
    OOF_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(OOF_OUTPUT)
    profile.to_parquet(PROFILE_OUTPUT, index=False)

    volatility_oof = pd.read_parquet(VOLATILITY_OOF)
    volatility_mapping = volatility_mapping_diagnostics(oof, volatility_oof)
    report = {
        "schema_version": 1,
        "design": DESIGN,
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end_exclusive": DEVELOPMENT_END.isoformat(),
        "confirmation_opened": False,
        "strategy_pnl_used": False,
        "directional_signal_used": False,
        "broker_spread_eligible": False,
        "paper_sizing_enabled": False,
        "source": str(SOURCE),
        "source_sha256": sha256_file(SOURCE),
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_manifest_output_sha256": source_manifest["output_sha256"],
        "source_rows": len(quotes),
        "source_first": quotes.index.min().isoformat(),
        "source_last": quotes.index.max().isoformat(),
        "event_source": str(EVENT_SOURCE),
        "event_source_sha256": sha256_file(EVENT_SOURCE),
        "horizons_minutes": list(HORIZONS_MINUTES),
        "quantiles": list(QUANTILES),
        "rolling_window": 252,
        "minimum_history": 126,
        "cells": cells,
        "volatility_mapping": volatility_mapping,
        "data_quality_passed": overall_pass,
        "oof_output": str(OOF_OUTPUT),
        "oof_sha256": sha256_file(OOF_OUTPUT),
        "profile_output": str(PROFILE_OUTPUT),
        "profile_sha256": sha256_file(PROFILE_OUTPUT),
    }
    REPORT_OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for key, cell in cells.items():
        range_p95 = cell["coverage"]["range_bps"]["p95"]["actual_coverage"]
        long_p95 = cell["coverage"]["long_adverse_bps"]["p95"][
            "actual_coverage"
        ]
        short_p95 = cell["coverage"]["short_adverse_bps"]["p95"][
            "actual_coverage"
        ]
        print(
            f"{key}: eligible={cell['eligible_rows']:,}, "
            f"calibrated={cell['calibrated_rows']:,} "
            f"({cell['calibrated_fraction']:.1%}), "
            f"p95 coverage range={range_p95:.1%}, "
            f"long adverse={long_p95:.1%}, short adverse={short_p95:.1%}"
        )
    print(f"OOF -> {OOF_OUTPUT}")
    print(f"Profile -> {PROFILE_OUTPUT}")
    print(f"Report -> {REPORT_OUTPUT}")
    print("Confirmation remained unopened; IG spread and paper sizing remain disabled.")


def _verified_source_manifest() -> dict:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    actual_hash = sha256_file(SOURCE)
    if manifest.get("output_sha256") != actual_hash:
        raise SystemExit("Synchronized quote source does not match its pinned manifest")
    if manifest.get("strategy_evaluated") is not False:
        raise SystemExit("Unexpected synchronized quote manifest state")
    return manifest


def _pin_event_snapshot() -> None:
    expected = scheduled_release_times(DEVELOPMENT_START, DEVELOPMENT_END)
    if EVENT_SOURCE.exists():
        observed = pd.read_parquet(EVENT_SOURCE)
        if not observed.equals(expected):
            raise SystemExit("Pinned calibration event snapshot does not match the verified calendar")
        return
    EVENT_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    expected.to_parquet(EVENT_SOURCE)


def _profile_summary(profile: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}
    for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps"):
        result[metric] = {}
        for quantile in QUANTILES:
            column = f"{metric}_p{int(quantile * 100):02d}"
            values = profile[column].to_numpy(dtype=float)
            result[metric][f"p{int(quantile * 100):02d}"] = {
                "minimum_across_clocks": float(np.min(values)),
                "median_across_clocks": float(np.median(values)),
                "maximum_across_clocks": float(np.max(values)),
            }
    return result


if __name__ == "__main__":
    main()
