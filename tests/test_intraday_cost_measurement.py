"""Observed broker quote-cost profiles and candidate matching."""

import pandas as pd
import pytest

from commodity_ml.intraday.cost_measurement import (
    measure_candidate_costs,
    measure_quote_costs,
    quote_round_trip_costs,
)


def quotes() -> pd.DataFrame:
    index = pd.date_range("2026-01-05 08:00", periods=8, freq="30min", tz="UTC")
    frame = pd.DataFrame(index=index)
    for field, base in (
        ("open", 100.0),
        ("high", 100.2),
        ("low", 99.8),
        ("close", 100.0),
    ):
        frame[f"bid_{field}"] = base - 0.01
        frame[f"ask_{field}"] = base + 0.01
    return frame


def test_round_trip_proxy_uses_full_spread_plus_two_sided_slippage():
    costs = quote_round_trip_costs(
        quotes(),
        slippage_pct_per_side=0.00005,
    )

    assert costs.iloc[0] == pytest.approx(0.0003)


def test_quote_report_profiles_london_slots_and_stop_risk():
    report = measure_quote_costs(
        quotes(),
        slippage_pct_per_side=0.00005,
        representative_stop_pct=0.01,
    )

    assert report.overall.median_bp == pytest.approx(3.0)
    assert report.median_cost_r == pytest.approx(0.03)
    assert "08:00" in report.by_london_slot
    assert report.by_london_weekday["Monday"].observations == 8


def test_candidate_profile_matches_entry_quotes_and_reports_break_even():
    candles = quotes()
    ledger = pd.DataFrame(
        {
            "strategy": ["P3V2", "P3V2"],
            "status": ["executed", "executed"],
            "entry_time": [candles.index[1], candles.index[3]],
            "gross_return": [0.001, -0.0002],
            "initial_r_pct": [0.01, 0.01],
        }
    )

    profile = measure_candidate_costs(candles, ledger)

    assert profile.coverage == 1.0
    assert profile.median_cost_bp == pytest.approx(2.0)
    assert profile.median_cost_r == pytest.approx(0.02)
    assert profile.gross_break_even_cost_bp == pytest.approx(4.0)
    assert profile.estimated_mean_net_bp == pytest.approx(2.0)


def test_candidate_profile_refuses_non_overlapping_history():
    ledger = pd.DataFrame(
        {
            "strategy": ["P3V2"],
            "status": ["executed"],
            "entry_time": [pd.Timestamp("2025-01-01", tz="UTC")],
            "gross_return": [0.001],
            "initial_r_pct": [0.01],
        }
    )

    with pytest.raises(ValueError, match="no entry timestamps"):
        measure_candidate_costs(quotes(), ledger)
