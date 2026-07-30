"""Tests for complete-bar, freshness, and spread admission gates."""

import pandas as pd
import pytest

from commodity_ml.intraday.market_data import (
    assess_spread_history,
    closed_quote_window,
)


def quote_candles(n: int = 120, spread: float = 0.02) -> pd.DataFrame:
    index = pd.date_range("2026-01-05 07:00", periods=n, freq="30min", tz="UTC")
    frame = pd.DataFrame(index=index)
    for field in ("open", "high", "low", "close"):
        base = 100.0
        if field == "high":
            base = 100.2
        elif field == "low":
            base = 99.8
        frame[f"bid_{field}"] = base - spread / 2
        frame[f"ask_{field}"] = base + spread / 2
    return frame


def test_only_complete_contiguous_quote_bars_are_returned():
    candles = quote_candles()
    now = candles.index[-1] + pd.Timedelta(minutes=45)

    out = closed_quote_window(
        candles,
        bar_duration=pd.Timedelta(minutes=30),
        now=now,
        min_bars=100,
    )

    assert len(out) == 100
    assert out.index[-1] == candles.index[-1]


def test_incomplete_current_bar_is_excluded():
    candles = quote_candles()
    now = candles.index[-1] + pd.Timedelta(minutes=15)

    out = closed_quote_window(
        candles,
        bar_duration=pd.Timedelta(minutes=30),
        now=now,
        min_bars=100,
    )

    assert out.index[-1] == candles.index[-2]


def test_gap_or_staleness_fails_closed():
    candles = quote_candles()
    now = candles.index[-1] + pd.Timedelta(minutes=30)
    gapped = candles.drop(candles.index[-50])

    with pytest.raises(RuntimeError, match="non_contiguous"):
        closed_quote_window(
            gapped,
            bar_duration=pd.Timedelta(minutes=30),
            now=now,
            min_bars=100,
        )

    with pytest.raises(RuntimeError, match="stale_closed"):
        closed_quote_window(
            candles,
            bar_duration=pd.Timedelta(minutes=30),
            now=now + pd.Timedelta(hours=1),
            min_bars=100,
        )


def test_spread_gate_reports_cost_in_stop_r_units():
    assessment = assess_spread_history(
        quote_candles(spread=0.02),
        stop_distance_pct=0.01,
        slippage_pct_per_side=0.00005,
    )

    assert assessment.allowed
    assert assessment.median_cost_r == pytest.approx(0.03, rel=0.02)


def test_wide_spread_fails_median_and_p95_gates():
    assessment = assess_spread_history(
        quote_candles(spread=0.40),
        stop_distance_pct=0.01,
    )

    assert not assessment.allowed
    assert "median_cost_above_0.10R" in assessment.reasons
    assert "p95_cost_above_0.25R" in assessment.reasons


def test_spread_gate_uses_worse_synchronized_open_or_close_not_extrema():
    candles = quote_candles(spread=0.02)
    candles["ask_open"] = candles["bid_open"] + 0.20
    # Deliberately nonsensical high/low cross-differences must not be treated
    # as synchronized spread observations.
    candles["ask_high"] = candles["bid_high"] + 5.0
    candles["ask_low"] = candles["bid_low"] + 4.0

    assessment = assess_spread_history(
        candles,
        stop_distance_pct=0.01,
    )

    assert assessment.median_cost_r == pytest.approx(0.20, rel=0.02)
