"""Synthetic-only tests for the registered intraday volatility forecast."""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.evaluation.placebo import circular_shift
from commodity_ml.intraday.volatility import (
    ALL_FEATURES,
    BASELINE_FEATURES,
    CALENDAR_FEATURES,
    FORECAST_HORIZON_BARS,
    IV_FEATURES,
    TARGET_COLUMN,
    build_volatility_dataset,
    clean_gvz_observations,
    evaluate_volatility_ablation,
    evaluate_joint_volatility,
    walk_forward_volatility,
)


def make_m30(n: int = 5_000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    close = 1_900 * np.exp(np.cumsum(rng.normal(0, 0.0008, n)))
    open_ = np.r_[close[0], close[:-1]]
    width = np.maximum(np.abs(close - open_), close * 0.0002)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + width,
            "low": np.minimum(open_, close) - width,
            "close": close,
            "volume": np.ones(n),
        },
        index=index,
    )


def make_gvz(end: str = "2021-06-30") -> pd.Series:
    index = pd.date_range("2020-10-01", end, freq="B")
    values = 20 + np.sin(np.arange(len(index)) / 11) + np.arange(len(index)) * 0.01
    return pd.Series(values, index=index, name="gvz")


def make_releases() -> pd.DataFrame:
    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2021-01-15 13:30", tz="UTC"),
            pd.Timestamp("2021-01-27 19:00", tz="UTC"),
        ],
        name="release_time",
    )
    return pd.DataFrame({"kind": ["cpi", "fomc"]}, index=index)


def build(candles: pd.DataFrame):
    end = candles.index[-1] + pd.Timedelta(hours=1)
    return build_volatility_dataset(
        candles,
        make_gvz(end=str(end.date() + pd.Timedelta(days=10))),
        development_start=pd.Timestamp("2021-01-01", tz="UTC"),
        development_end=end,
        releases=make_releases(),
    )


def test_dataset_has_one_matched_contract_and_observed_forward_target():
    result = build(make_m30())

    assert tuple(result.frame.columns) == ALL_FEATURES + (TARGET_COLUMN,)
    assert result.matched_rows > 1_000
    assert result.matched_coverage > 0.90
    assert np.isfinite(result.frame.to_numpy(dtype=float)).all()

    decision = result.frame.index[200]
    candles = make_m30()
    signal_position = candles.index.get_loc(decision - pd.Timedelta(minutes=30))
    closes = candles["close"].iloc[
        signal_position : signal_position + FORECAST_HORIZON_BARS + 1
    ]
    expected = np.log(np.sqrt(np.square(np.diff(np.log(closes))).sum()))
    assert result.frame.loc[decision, TARGET_COLUMN] == pytest.approx(expected)


def test_future_price_poison_changes_target_but_not_features():
    candles = make_m30()
    original = build(candles)
    decision = original.frame.index[300]
    first_future_open = decision
    last_future_open = decision + pd.Timedelta(minutes=90)

    poisoned = candles.copy()
    mask = (poisoned.index >= first_future_open) & (
        poisoned.index <= last_future_open
    )
    poisoned.loc[mask, ["open", "high", "low", "close"]] *= 1.08
    changed = build(poisoned)

    pd.testing.assert_series_equal(
        original.frame.loc[decision, list(ALL_FEATURES)],
        changed.frame.loc[decision, list(ALL_FEATURES)],
    )
    assert original.frame.loc[decision, TARGET_COLUMN] != pytest.approx(
        changed.frame.loc[decision, TARGET_COLUMN]
    )


def test_release_proximity_uses_only_the_next_two_hours():
    result = build(make_m30())
    before_cpi = pd.Timestamp("2021-01-15 12:00", tz="UTC")
    after_cpi = pd.Timestamp("2021-01-15 14:00", tz="UTC")

    assert result.frame.loc[before_cpi, CALENDAR_FEATURES[0]] == 1.0
    assert result.frame.loc[before_cpi, CALENDAR_FEATURES[1]] == 0.0
    assert result.frame.loc[after_cpi, CALENDAR_FEATURES[0]] == 0.0


def test_confirmation_candles_are_refused_not_sliced_silently():
    candles = make_m30(500)
    cutoff = candles.index[-10]
    with pytest.raises(ValueError, match="confirmation_candles_loaded"):
        build_volatility_dataset(
            candles,
            make_gvz(),
            development_start=pd.Timestamp("2021-01-01", tz="UTC"),
            development_end=cutoff,
            releases=make_releases(),
        )


def test_walk_forward_forecast_is_oof_and_finds_a_stable_relationship():
    rng = np.random.default_rng(9)
    n = 1_500
    index = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "signal": signal,
            "clock": np.sin(np.arange(n) / 20),
            TARGET_COLUMN: 0.8 * signal + rng.normal(0, 0.2, n),
        },
        index=index,
    )

    result = walk_forward_volatility(frame, ("signal", "clock"))

    assert len(result.folds) == 5
    assert result.pooled_r2_oos > 0.8
    assert result.oof_prediction.index[0] > frame.index[0]
    for fold in result.folds:
        assert fold.train_end < fold.test_start


def test_walk_forward_refuses_a_gap_shorter_than_target():
    index = pd.date_range("2021-01-01", periods=1_000, freq="30min", tz="UTC")
    frame = pd.DataFrame(
        {"x": np.arange(1_000, dtype=float), TARGET_COLUMN: np.arange(1_000)},
        index=index,
    )
    with pytest.raises(ValueError, match="four-bar"):
        walk_forward_volatility(frame, ("x",), gap=3)


def test_ablation_uses_registered_family_and_identical_oof_rows():
    rng = np.random.default_rng(12)
    n = 1_500
    index = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    frame = pd.DataFrame(index=index)
    for feature in ALL_FEATURES:
        frame[feature] = rng.normal(size=n)
    frame[TARGET_COLUMN] = (
        0.8 * frame[IV_FEATURES[0]] + rng.normal(0, 0.3, n)
    )

    result = evaluate_volatility_ablation(frame, IV_FEATURES)

    assert result.delta_r2_oos > 0.5
    assert result.folds_improved == 5
    assert result.baseline.oof_prediction.index.equals(
        result.augmented.oof_prediction.index
    )

    with pytest.raises(ValueError, match="registered"):
        evaluate_volatility_ablation(frame, ("invented_feature",))


def test_registered_feature_families_do_not_overlap():
    assert not set(BASELINE_FEATURES).intersection(IV_FEATURES)
    assert not set(BASELINE_FEATURES).intersection(CALENDAR_FEATURES)
    assert not set(IV_FEATURES).intersection(CALENDAR_FEATURES)


def test_joint_comparison_adds_calendar_to_fixed_iv_reference():
    rng = np.random.default_rng(31)
    n = 1_500
    index = pd.date_range("2021-01-01", periods=n, freq="30min", tz="UTC")
    frame = pd.DataFrame(index=index)
    for feature in ALL_FEATURES:
        frame[feature] = rng.normal(size=n)
    frame[TARGET_COLUMN] = (
        0.7 * frame[IV_FEATURES[0]]
        + 0.5 * frame[CALENDAR_FEATURES[0]]
        + rng.normal(0, 0.2, n)
    )

    result = evaluate_joint_volatility(frame)

    assert result.reference.feature_names == BASELINE_FEATURES + IV_FEATURES
    assert result.joint.feature_names == ALL_FEATURES
    assert result.delta_r2_oos > 0.1
    assert result.folds_improved == 5
    assert result.reference.oof_prediction.index.equals(
        result.joint.oof_prediction.index
    )


def test_gvz_holiday_nans_are_non_observations_and_placebos_keep_fixed_rows():
    raw = make_gvz()
    raw.iloc[[20, 45, 90]] = np.nan
    clean = clean_gvz_observations(raw)
    shifted = circular_shift(clean.to_frame(), 17)["gvz"]
    candles = make_m30()
    end = candles.index[-1] + pd.Timedelta(hours=1)

    observed = build_volatility_dataset(
        candles,
        clean,
        development_start=pd.Timestamp("2021-01-01", tz="UTC"),
        development_end=end,
        releases=make_releases(),
    )
    placebo = build_volatility_dataset(
        candles,
        shifted,
        development_start=pd.Timestamp("2021-01-01", tz="UTC"),
        development_end=end,
        releases=make_releases(),
    )

    assert len(clean) == len(raw) - 3
    assert observed.frame.index.equals(placebo.frame.index)
