from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from commodity_ml.intraday.risk_calibration import (
    build_calibration_outcomes,
    calibrate_outcomes,
    lookup_calibrated_risk,
)


def quote_history(days: int = 300) -> pd.DataFrame:
    business_days = pd.bdate_range("2023-01-02", periods=days)
    timestamps: list[pd.Timestamp] = []
    levels: list[float] = []
    for day_number, day in enumerate(business_days):
        local_start = pd.Timestamp(day.date(), tz="Europe/London") + pd.Timedelta(
            hours=9
        )
        for bar in range(8):
            timestamps.append(
                (local_start + pd.Timedelta(minutes=30 * bar)).tz_convert("UTC")
            )
            levels.append(100 + day_number * 0.02 + bar * 0.1)
    index = pd.DatetimeIndex(timestamps, name="time")
    data: dict[str, np.ndarray] = {}
    base = np.asarray(levels)
    for asset, multiplier, spread in (
        ("gold", 1.0, 0.20),
        ("silver", 0.25, 0.04),
    ):
        bid_open = base * multiplier
        bid_close = bid_open + 0.03 * multiplier
        bid_high = np.maximum(bid_open, bid_close) + 0.25 * multiplier
        bid_low = np.minimum(bid_open, bid_close) - 0.20 * multiplier
        for side, offset in (("bid", 0.0), ("ask", spread)):
            data[f"{asset}_{side}_open"] = bid_open + offset
            data[f"{asset}_{side}_high"] = bid_high + offset
            data[f"{asset}_{side}_low"] = bid_low + offset
            data[f"{asset}_{side}_close"] = bid_close + offset
    return pd.DataFrame(data, index=index)


def bounds(frame: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return frame.index.min() - pd.Timedelta(days=1), frame.index.max() + pd.Timedelta(days=1)


def test_outcomes_are_symmetric_direction_free_and_cost_exclusive():
    quotes = quote_history(5)
    start, end = bounds(quotes)
    result = build_calibration_outcomes(
        quotes,
        instrument="XAUUSD",
        horizon_minutes=60,
        development_start=start,
        development_end=end,
    )

    first = result.frame.iloc[0]
    assert first["range_bps"] > 0
    assert first["long_adverse_bps"] > 0
    assert first["short_adverse_bps"] > 0
    assert first["spread_points"] == pytest.approx(0.20)
    assert first["entry_price"] == pytest.approx(100.10)
    assert result.exclusions["eligible"] == len(result.frame)


def test_missing_bar_is_reason_coded_and_never_bridged():
    quotes = quote_history(5).drop(index=quote_history(5).index[1])
    start, end = bounds(quotes)

    result = build_calibration_outcomes(
        quotes,
        instrument="XAUUSD",
        horizon_minutes=60,
        development_start=start,
        development_end=end,
    )

    assert result.exclusions["non_contiguous_path"] >= 1
    assert quotes.index[0] not in result.frame.index


def test_rolling_quantiles_are_strictly_prior_and_pass_data_quality():
    quotes = quote_history(500)
    start, end = bounds(quotes)
    outcomes = build_calibration_outcomes(
        quotes,
        instrument="XAUUSD",
        horizon_minutes=60,
        development_start=start,
        development_end=end,
    ).frame
    baseline = calibrate_outcomes(outcomes)
    poisoned = outcomes.copy()
    poisoned.iloc[-1, poisoned.columns.get_loc("range_bps")] = 1_000_000
    changed = calibrate_outcomes(poisoned)

    assert baseline.passed_data_quality
    assert baseline.diagnostics["calibrated_fraction"] >= 0.70
    assert baseline.oof.iloc[-1]["range_bps_p95"] == pytest.approx(
        changed.oof.iloc[-1]["range_bps_p95"]
    )
    assert baseline.oof.iloc[-1]["range_bps"] != changed.oof.iloc[-1]["range_bps"]


def test_lookup_converts_bps_to_points_and_rejects_staleness():
    quotes = quote_history()
    start, end = bounds(quotes)
    outcomes = build_calibration_outcomes(
        quotes,
        instrument="XAUUSD",
        horizon_minutes=60,
        development_start=start,
        development_end=end,
    ).frame
    profile = calibrate_outcomes(outcomes).profile
    last_day = quotes.index.max().tz_convert("Europe/London").normalize()
    decision = (last_day + pd.Timedelta(days=1, hours=9)).tz_convert("UTC")

    lookup = lookup_calibrated_risk(
        profile,
        instrument="XAUUSD",
        decision_time=decision,
        horizon_minutes=60,
        entry_price=2_000,
        side="long",
    )

    row = profile.loc[
        (profile["instrument"] == "XAUUSD")
        & (profile["horizon_minutes"] == 60)
        & (profile["london_slot_minute"] == 9 * 60)
    ].iloc[0]
    assert lookup.range_points["p95"] == pytest.approx(
        row["range_bps_p95"] * 2_000 / 10_000
    )
    assert not lookup.broker_spread_eligible
    with pytest.raises(ValueError, match="stale"):
        lookup_calibrated_risk(
            profile,
            instrument="XAUUSD",
            decision_time=decision + pd.Timedelta(days=100),
            horizon_minutes=60,
            entry_price=2_000,
            side="long",
        )
