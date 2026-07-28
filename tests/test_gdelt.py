"""GDELT tone/attention features — offline, on synthetic daily timelines.

Same battery as test_cot.py (lag pin, causality, warmup, staleness) plus the
failure modes daily news data adds: the tone-0.0-vs-no-articles ambiguity and
the current UTC day being incomplete at fetch time.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_gdelt

from commodity_ml.config import AssetConfig, GDELTConfig
from commodity_ml.features.gdelt import (
    FFILL_LIMIT_DAYS,
    TONE_CHANGE_DAYS,
    VOL_SMOOTH_DAYS,
    Z_WINDOW_DAYS,
    _check_gdelt_sidecar,
    _drop_incomplete_day,
    _mask_zero_article_days,
    _parse_timeline,
    build_gdelt_features,
    gdelt_feature_names,
)


def bars(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="4h", tz="UTC")


def independent_derived(gdelt: pd.DataFrame) -> pd.DataFrame:
    """The §3 formulas recomputed with plain pandas, independent of the module."""
    daily = gdelt.reindex(pd.date_range(gdelt.index.min(), gdelt.index.max(), freq="D"))
    tone = daily["tone"].ffill(limit=FFILL_LIMIT_DAYS)
    vol = daily["vol"].ffill(limit=FFILL_LIMIT_DAYS)

    out = pd.DataFrame(index=daily.index)
    roll = tone.rolling(Z_WINDOW_DAYS, min_periods=Z_WINDOW_DAYS // 2)
    out["gdelt_tone_z"] = (tone - roll.mean()) / roll.std()
    out["gdelt_tone_chg_5d"] = tone.diff(TONE_CHANGE_DAYS)
    smooth = vol.rolling(VOL_SMOOTH_DAYS, min_periods=VOL_SMOOTH_DAYS).mean()
    roll_v = smooth.rolling(Z_WINDOW_DAYS, min_periods=Z_WINDOW_DAYS // 2)
    out["gdelt_vol_smooth7_z"] = (smooth - roll_v.mean()) / roll_v.std()
    return out


@pytest.fixture(scope="module")
def gdelt() -> pd.DataFrame:
    return make_gdelt()


def test_every_declared_feature_is_produced(gdelt):
    index = bars("2021-03-01", "2021-05-01")
    features = build_gdelt_features(gdelt, index)

    assert list(features.columns) == gdelt_feature_names()
    assert features.index.equals(index)


def test_values_respect_publication_lag(gdelt):
    """A bar at noon on day D sees at most day D−1's complete aggregate —
    never day D's own, which is still accumulating articles."""
    index = bars("2021-03-01", "2021-06-15 12:00")
    features = build_gdelt_features(gdelt, index)
    expected = independent_derived(gdelt)

    for feature in gdelt_feature_names():
        yesterday = expected.loc["2021-06-14", feature]
        today = expected.loc["2021-06-15", feature]
        assert yesterday != today, "fixture too flat to catch the bug"
        assert features[feature].iloc[-1] == pytest.approx(yesterday, rel=1e-9)


def test_gdelt_features_are_causal(gdelt):
    index = bars("2021-01-04", "2021-06-30")
    full = build_gdelt_features(gdelt, index)
    truncated = build_gdelt_features(gdelt.loc[:"2021-09-30"], index)

    pd.testing.assert_frame_equal(full, truncated, rtol=1e-9)


def test_warmup_produces_nans_not_silent_zeros():
    young = make_gdelt(n_days=100)  # < the 126-day min_periods
    features = build_gdelt_features(young, bars("2020-03-01", "2020-04-08"))

    assert features["gdelt_tone_z"].isna().all()
    assert features["gdelt_vol_smooth7_z"].isna().all()
    # The 5-day change needs only 6 days of history.
    assert features["gdelt_tone_chg_5d"].notna().any()


def test_zero_article_days_are_nan_not_zero(gdelt):
    """A zero-article day reports tone 0.0 in the raw timeline. If that 0.0
    entered the rolling stats as a real observation it would silently drag the
    z-score; the mask must turn it into a bridged gap instead."""
    poisoned = gdelt.copy()
    day = pd.Timestamp("2021-05-10")
    poisoned.loc[day, ["tone", "n_articles"]] = [45.0, 0.0]  # absurd tone, no articles

    reference = gdelt.copy()
    reference.loc[day, ["tone", "vol"]] = np.nan  # what the mask should produce

    index = bars("2021-05-01", "2021-06-15")
    masked = build_gdelt_features(_mask_zero_article_days(poisoned), index)
    expected = build_gdelt_features(reference, index)
    unmasked = build_gdelt_features(poisoned, index)

    pd.testing.assert_frame_equal(masked, expected, rtol=1e-9)
    assert not masked["gdelt_tone_z"].equals(unmasked["gdelt_tone_z"]), (
        "the poison must be strong enough that forgetting the mask would show"
    )


def test_drought_beyond_staleness_is_nan(gdelt):
    """ffill bridges up to 3 missing days; a longer drought must become NaN
    features, not week-old tone presented as current."""
    drought = gdelt.copy()
    drought.loc[drought.index >= "2022-06-09", ["tone", "vol"]] = np.nan  # last real day 2022-06-08

    features = build_gdelt_features(drought, bars("2022-05-15", "2022-06-18"))

    assert features.loc[: pd.Timestamp("2022-06-07", tz="UTC")].notna().all().all()
    assert features.loc[pd.Timestamp("2022-06-17", tz="UTC") :].isna().all().all()


# --------------------------------------------------------------------------
# Offline parsing, fetch post-processing, provenance
# --------------------------------------------------------------------------

def test_timeline_json_parses_offline():
    payload = {
        "timeline": [
            {
                "series": "Average Tone",
                "data": [
                    {"date": "20210101T000000Z", "value": -1.5},
                    {"date": "20210102000000", "value": -1.0},  # both formats occur
                ],
            }
        ]
    }
    series = _parse_timeline(payload, "tone")

    assert list(series.index) == [pd.Timestamp("2021-01-01"), pd.Timestamp("2021-01-02")]
    assert series.tolist() == [-1.5, -1.0]


def test_malformed_timeline_is_refused():
    with pytest.raises(ValueError, match="no timeline data"):
        _parse_timeline({}, "tone")
    with pytest.raises(ValueError, match="no timeline data"):
        _parse_timeline({"timeline": []}, "tone")


def test_current_partial_day_is_dropped(gdelt):
    now = pd.Timestamp("2022-06-18 14:30")  # fixture's last day, mid-afternoon
    kept = _drop_incomplete_day(gdelt, now=now)

    assert kept.index.max() == pd.Timestamp("2022-06-17")
    assert gdelt.index.max() == pd.Timestamp("2022-06-18"), "fixture should end today"


def test_sidecar_query_mismatch_is_refused():
    cfg = AssetConfig(
        asset="gold", symbol="XAUUSD", granularity="H4",
        gdelt=GDELTConfig(query='"gold price" OR bullion'),
    )

    _check_gdelt_sidecar({"query": '"gold price" OR bullion'}, cfg)  # matching: no raise
    with pytest.raises(ValueError, match="snapshot IS the sample"):
        _check_gdelt_sidecar({"query": "gold"}, cfg)
