from __future__ import annotations

import json

import pandas as pd

from commodity_ml.intraday.calibration_refresh import (
    coverage_gate,
    profile_is_complete,
    quantiles_are_valid,
    synchronize_quote_legs,
    validate_parity_report,
)
from commodity_ml.intraday.risk_calibration import HORIZONS_MINUTES, sha256_file


def _leg(index: pd.DatetimeIndex, offset: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": 100 + offset,
            "high": 101 + offset,
            "low": 99 + offset,
            "close": 100.5 + offset,
        },
        index=index,
    )


def test_synchronization_uses_only_shared_rows_and_never_fills():
    index = pd.date_range("2026-08-01", periods=4, freq="30min", tz="UTC")
    legs = {
        "gold_bid": _leg(index, 0),
        "gold_ask": _leg(index.delete(1), 0.2),
        "silver_bid": _leg(index, -75),
        "silver_ask": _leg(index, -74.95),
    }

    result = synchronize_quote_legs(legs)

    assert result.index.equals(index.delete(1))
    assert not result.isna().any().any()
    assert set(result.columns) == {
        f"{asset}_{side}_{field}"
        for asset in ("gold", "silver")
        for side in ("bid", "ask")
        for field in ("open", "high", "low", "close")
    }


def _complete_profile() -> pd.DataFrame:
    rows = []
    for instrument in ("XAUUSD", "XAGUSD"):
        for horizon in HORIZONS_MINUTES:
            for slot in range(7 * 60, 19 * 60 + 30, 30):
                if slot + horizon > 21 * 60 + 30:
                    continue
                row = {
                    "instrument": instrument,
                    "horizon_minutes": horizon,
                    "london_slot_minute": slot,
                    "sample_count": 252,
                }
                for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps"):
                    for label, value in (("p50", 1), ("p80", 2), ("p95", 3), ("p99", 4)):
                        row[f"{metric}_{label}"] = value
                rows.append(row)
    return pd.DataFrame(rows)


def test_profile_completeness_and_monotonicity_are_fail_closed():
    profile = _complete_profile()
    assert profile_is_complete(profile)
    assert quantiles_are_valid(profile)

    assert not profile_is_complete(profile.iloc[:-1])
    broken = profile.copy()
    broken.loc[0, "range_bps_p99"] = 2
    assert not quantiles_are_valid(broken)


def test_coverage_gate_checks_every_side_and_cell():
    coverage = {
        metric: {
            "p95": {"actual_coverage": 0.94},
            "p99": {"actual_coverage": 0.98},
        }
        for metric in ("range_bps", "long_adverse_bps", "short_adverse_bps")
    }
    cells = {"XAUUSD_60m": {"coverage": coverage}}
    assert coverage_gate(cells)
    coverage["short_adverse_bps"]["p95"]["actual_coverage"] = 0.92
    assert not coverage_gate(cells)


def test_parity_report_requires_fresh_hashed_inputs(tmp_path):
    index = pd.date_range("2026-07-30", periods=120, freq="30min", tz="UTC")
    sources = {}
    for provider in ("dukascopy", "oanda"):
        path = tmp_path / f"{provider}.parquet"
        _leg(index, 0).to_parquet(path)
        sources[provider] = {"path": str(path), "sha256": sha256_file(path)}
    report_path = tmp_path / "parity.json"
    report_path.write_text(
        json.dumps(
            {
                "granularity": "M30",
                "verdict": "PASS",
                "matched_bars": 120,
                "broker_coverage": 1.0,
                "dukascopy_coverage": 1.0,
                "return_correlation": 0.99,
                "best_shift_bars": 0,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )

    result = validate_parity_report(
        report_path, as_of=pd.Timestamp("2026-08-01T00:00:00Z")
    )

    assert result["passed"]
    sources["oanda"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps({**json.loads(report_path.read_text()), "sources": sources}))
    result = validate_parity_report(
        report_path, as_of=pd.Timestamp("2026-08-01T00:00:00Z")
    )
    assert not result["passed"]
    assert "parity_oanda_hash_mismatch" in result["reasons"]
