"""Step 2b vol-layer harness — offline, synthetic.

Same battery shape as the other feature families (causality, lag pins, loud
refusals) plus what a *forecasting* harness adds: the target must be strictly
forward, the R² benchmark must come from training data, and a cheating column
must be caught by construction.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_cot, make_gdelt

from commodity_ml.evaluation.vol_gate import ablation_delta, walk_forward_r2
from commodity_ml.features import event_dates
from commodity_ml.features.vol_layer import (
    FAMILY_COLUMNS,
    GVZ_LAG_DAYS,
    HAR_WINDOWS,
    MIN_BARS_PER_DAY,
    assemble_vol_dataset,
    build_b0,
    build_calendar_features,
    build_iv_features,
    calendar_columns,
    daily_realized_vol,
    forward_log_vol,
)


def make_h4(n: int = 2100, seed: int = 7, start: str = "2021-01-04") -> pd.Series:
    """H4 closes spanning ~1.4 years inside the verified calendar coverage."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="4h", tz="UTC")
    return pd.Series(
        2000 * np.exp(np.cumsum(rng.normal(0, 0.002, n))), index=idx, name="close"
    )


def make_gvz(n_days: int = 700, seed: int = 3, start: str = "2020-06-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days, freq="B", name="date")
    return pd.Series(
        np.abs(16 + np.cumsum(rng.normal(0, 0.4, n_days))), index=idx, name="gvz"
    )


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------

def test_forward_log_vol_matches_independent_recompute():
    """Pins the diagnose_altdata.forward_outcomes arithmetic: std (ddof=0) of
    daily log returns over close[D]..close[D+h], then log."""
    close = make_h4().resample("1D").last().dropna()
    horizon = 5
    series = forward_log_vol(close, horizon)

    day = 40
    window = np.log(close.iloc[day : day + horizon + 1].to_numpy())
    expected = np.log(np.std(np.diff(window), ddof=0))
    assert series.iloc[day] == pytest.approx(expected, rel=1e-12)


def test_forward_log_vol_is_strictly_forward():
    """Changing history BEFORE day D must not move D's target; the tail that
    lacks a full window must be NaN, not a shortened window."""
    close = make_h4().resample("1D").last().dropna()
    poisoned = close.copy()
    poisoned.iloc[:30] *= 3.0  # violent fake history

    original = forward_log_vol(close, 5)
    after = forward_log_vol(poisoned, 5)
    pd.testing.assert_series_equal(original.iloc[31:], after.iloc[31:])
    assert original.iloc[-5:].isna().all()


# --------------------------------------------------------------------------
# B0 — HAR baseline
# --------------------------------------------------------------------------

def test_daily_rv_matches_manual_std_and_masks_stub_days():
    close = make_h4()
    rv = daily_realized_vol(close)

    one_day = close["2021-02-03"]
    manual = np.std(np.diff(np.log(one_day.to_numpy())), ddof=0)
    assert rv.loc["2021-02-03"] == pytest.approx(manual, rel=1e-12)

    # A day trimmed to below MIN_BARS_PER_DAY bars is missing, not calm.
    stub = close[(close.index < "2021-03-10") | (close.index >= "2021-03-11")]
    stub = pd.concat([stub, close["2021-03-10"].iloc[:MIN_BARS_PER_DAY - 1]]).sort_index()
    assert np.isnan(daily_realized_vol(stub).loc["2021-03-10"])


def test_b0_is_causal_and_warms_up_honestly():
    close = make_h4()
    grid = close.resample("1D").last().dropna().index
    full = build_b0(close, grid)
    truncated = build_b0(close[close.index < "2022-01-01"], grid)

    overlap = grid[grid < pd.Timestamp("2021-12-25", tz="UTC")]
    pd.testing.assert_frame_equal(full.loc[overlap], truncated.loc[overlap])
    # First 22d rung needs 22 complete days — before that, NaN not zero.
    assert full[f"har_log_rv_{HAR_WINDOWS[-1]}d"].iloc[:15].isna().all()


# --------------------------------------------------------------------------
# F-IV — publication lag + staleness
# --------------------------------------------------------------------------

def test_iv_respects_publication_lag():
    """Grid day D sees the GVZ observation dated <= D - lag, never fresher."""
    gvz = make_gvz()
    grid = pd.date_range("2021-02-01", "2021-06-01", freq="B", tz="UTC")
    features = build_iv_features(gvz, grid)

    day = pd.Timestamp("2021-03-10", tz="UTC")
    freshest_legal = gvz[gvz.index <= (day - pd.Timedelta(days=GVZ_LAG_DAYS)).tz_localize(None)].iloc[-1]
    assert features.loc[day, "gvz_log"] == pytest.approx(np.log(freshest_legal), rel=1e-12)


def test_iv_outage_becomes_nan_not_fossil():
    gvz = make_gvz()
    gap = gvz[(gvz.index < "2021-03-01") | (gvz.index >= "2021-03-25")]
    grid = pd.date_range("2021-02-01", "2021-04-15", freq="B", tz="UTC")
    features = build_iv_features(gap, grid)

    assert features.loc[pd.Timestamp("2021-03-18", tz="UTC")].isna().all()
    assert features.loc[pd.Timestamp("2021-04-05", tz="UTC")].notna().all()


# --------------------------------------------------------------------------
# F-CAL
# --------------------------------------------------------------------------

def test_calendar_window_is_exclusive_of_today_and_capped_at_horizon():
    grid = pd.date_range("2021-01-04", "2021-12-31", freq="B", tz="UTC")
    features = build_calendar_features(grid, 5)
    column = "fomc_within_5d"

    decision = pd.Timestamp("2021-03-17", tz="UTC")  # a verified FOMC end day
    assert features.loc[decision - pd.tseries.offsets.BDay(3), column] == 1.0
    assert features.loc[decision, column] == 0.0, "day D is (D, D+h] exclusive"
    assert features.loc[decision - pd.tseries.offsets.BDay(9), column] == 0.0


def test_calendar_refuses_a_grid_past_verified_coverage():
    grid = pd.date_range("2025-11-01", "2026-06-30", freq="B", tz="UTC")
    with pytest.raises(ValueError, match="verified tables end"):
        build_calendar_features(grid, 20)


def test_calendar_override_is_honoured_for_placebos():
    grid = pd.date_range("2021-06-01", "2021-06-30", freq="B", tz="UTC")
    fake_event = pd.DatetimeIndex([pd.Timestamp("2021-06-17")])
    features = build_calendar_features(grid, 5, fomc=fake_event, cpi_nfp=fake_event)

    before = pd.Timestamp("2021-06-15", tz="UTC")
    assert features.loc[before, "fomc_within_5d"] == 1.0
    assert features["fomc_within_5d"].sum() == 5.0  # exactly the 5 days ahead of it


def test_event_tables_have_the_verified_shape():
    """Counts pinned to the primary sources: 8 scheduled FOMC/year 2020-2026;
    12 CPI and 12 NFP releases per year except 2025's shutdown (11 each)."""
    fomc = event_dates.fomc_days()
    assert len(fomc) == 8 * 7
    for year, expected in [(2024, 12), (2025, 11)]:
        assert sum(pd.Timestamp(d).year == year for d in event_dates.CPI_RELEASES) == expected
        assert sum(pd.Timestamp(d).year == year for d in event_dates.NFP_RELEASES) == expected
    # End days, not start days: the 2023 Jan meeting ended Feb 1.
    assert pd.Timestamp("2023-02-01") in fomc


# --------------------------------------------------------------------------
# Walk-forward regression + ablation
# --------------------------------------------------------------------------

def synthetic_regression(n: int = 800, informative: bool = True, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = pd.DataFrame({"signal": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = pd.Series(
        (0.8 * x["signal"] if informative else 0.0) + rng.normal(size=n) * 0.5
    )
    return x, y


def test_r2_finds_a_real_relationship_and_not_a_fake_one():
    x, y = synthetic_regression(informative=True)
    assert walk_forward_r2(x, y, n_splits=5, gap=5).r2_pooled > 0.4

    x_noise, y_noise = synthetic_regression(informative=False, seed=1)
    assert abs(walk_forward_r2(x_noise, y_noise, n_splits=5, gap=5).r2_pooled) < 0.05


def test_r2_survives_a_constant_column():
    x, y = synthetic_regression()
    x["flag"] = 0.0  # a rare-event flag can be constant within a fold
    result = walk_forward_r2(x, y, n_splits=5, gap=5)
    assert np.isfinite(result.r2_pooled)


def test_ablation_credits_signal_and_not_noise():
    rng = np.random.default_rng(2)
    n = 800
    base = pd.Series(rng.normal(size=n))
    extra = pd.Series(rng.normal(size=n))
    y = 0.6 * base + 0.6 * extra + pd.Series(rng.normal(size=n)) * 0.5
    ds = pd.DataFrame({"y": y, "base": base, "extra": extra,
                       "junk": rng.normal(size=n)})

    real = ablation_delta(ds, "y", ["base"], ["extra"], n_splits=5, gap=5)
    fake = ablation_delta(ds, "y", ["base"], ["junk"], n_splits=5, gap=5)
    assert real["delta"] > 0.15 and real["folds_improved"] >= 4
    assert abs(fake["delta"]) < 0.05


# --------------------------------------------------------------------------
# Assembly contract
# --------------------------------------------------------------------------

def test_assemble_builds_the_matched_row_contract():
    h4 = make_h4()
    dataset = assemble_vol_dataset(h4, make_cot(), make_gdelt(), make_gvz(), horizon=5)

    expected = (
        ["fwd_log_vol_5d"] + FAMILY_COLUMNS["b0"] + FAMILY_COLUMNS["cot"]
        + FAMILY_COLUMNS["attn"] + calendar_columns(5) + FAMILY_COLUMNS["iv"]
    )
    assert sorted(dataset.columns) == sorted(expected)
    assert not dataset.isna().any().any(), "matched rows are complete rows"
    assert len(dataset) > 200


def test_assemble_without_gvz_omits_iv_columns_rather_than_nulling_them():
    """The silver case (design §2): absence is a design fact, not missing data."""
    dataset = assemble_vol_dataset(make_h4(), make_cot(), make_gdelt(), None, horizon=5)
    assert not any(c in dataset.columns for c in FAMILY_COLUMNS["iv"])
    assert len(dataset) > 200
