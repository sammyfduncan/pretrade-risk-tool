"""GDELT tone/attention features — offline, on synthetic daily timelines.

Same battery as test_cot.py (lag pin, causality, warmup, staleness) plus the
failure modes daily news data adds: the tone-0.0-vs-no-articles ambiguity and
the current UTC day being incomplete at fetch time.
"""

import re

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
    _matched_stories_sql,
    assemble_snapshot,
    build_gdelt_features,
    build_title_regex,
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

def test_title_regex_anchors_to_the_headline_tag():
    """Extras also holds page links, author bylines and AMP URLs. A bare
    substring search would match 'silver' in an author's surname or a related-
    links URL; anchoring inside <page_title> is what makes this an entity
    filter rather than a keyword grep over a blob."""
    regex = build_title_regex(["silver price"])
    assert regex.startswith("<page_title>")

    body = "<page_title>silver price hits high</page_title>"
    byline = "<page_authors>Nate Silver</page_authors><page_title>elections</page_title>"
    assert re.search(regex, body)
    assert not re.search(regex, byline)


def test_title_regex_escapes_phrases_and_refuses_an_empty_list():
    """Phrases are data, not pattern syntax — an unescaped '.' would widen the
    match, and an empty list would match every article ever indexed."""
    assert re.search(build_title_regex(["u.s. gold"]), "<page_title>u.s. gold</page_title>")
    assert not re.search(build_title_regex(["u.s. gold"]), "<page_title>uxsx gold</page_title>")

    with pytest.raises(ValueError, match="match every article"):
        build_title_regex([])


def test_sql_deduplicates_by_title_before_counting():
    """The failure this guards is silent and was measured on real data: raw GKG
    stores one row per outlet that ran a wire story, duplication correlates
    +0.51 with news volume, and it moved a month's mean tone by 55%. The daily
    aggregate must therefore run over distinct titles, never raw rows."""
    sql = _matched_stories_sql({"gold": "<page_title>[^<]*(gold price)"},
                               "2018-01-01", "2026-07-01")

    per_story = sql.split("per_story AS")[1]
    assert "GROUP BY day, title" in per_story, "stories must collapse per title first"
    assert "AVG(tone)" in per_story, "copies of one story average to one tone"
    # And the daily aggregate counts those collapsed rows, not the raw ones.
    assert "COUNT(*) AS n_articles" in sql.split("FROM per_story")[0]


def test_one_scan_covers_every_asset():
    """Cost is bytes scanned, and the 8.5y pull is ~54% of the monthly free
    tier — so a per-asset scan would exhaust it. Both assets must come out of
    a single pass over the table."""
    sql = _matched_stories_sql(
        {"gold": "<page_title>[^<]*(gold price)",
         "silver": "<page_title>[^<]*(silver price)"},
        "2018-01-01", "2026-07-01",
    )

    assert sql.count("FROM `gdelt-bq.gdeltv2.gkg_partitioned`") == 1
    assert "'gold' AS asset" in sql and "'silver' AS asset" in sql


def test_assemble_refuses_a_missing_volume_denominator():
    """vol is stories/total. A day present in the numerator but absent from the
    denominator would divide by NaN and zero out attention on that day."""
    stories = pd.DataFrame({
        "day": pd.to_datetime(["2021-01-01", "2021-01-02"]),
        "asset": ["gold", "gold"], "n_articles": [10, 12], "tone": [1.0, -0.5],
    })
    totals = pd.DataFrame({"day": pd.to_datetime(["2021-01-01"]), "total": [1000]})

    with pytest.raises(ValueError, match="no total-volume denominator"):
        assemble_snapshot(stories, totals, "gold")


def test_assemble_builds_the_snapshot_contract():
    stories = pd.DataFrame({
        "day": pd.to_datetime(["2021-01-01", "2021-01-03"]),  # note the gap
        "asset": ["gold", "gold"], "n_articles": [10, 20], "tone": [1.0, -0.5],
    })
    totals = pd.DataFrame({
        "day": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
        "total": [1000, 1000, 1000],
    })
    frame = assemble_snapshot(stories, totals, "gold")

    assert list(frame.columns) == ["tone", "vol", "n_articles"]
    assert frame["vol"].iloc[0] == pytest.approx(0.01)
    # The absent day becomes an explicit NaN row, not a missing index entry,
    # so the zero-article mask and the staleness tolerance both see it.
    assert len(frame) == 3
    assert frame.loc[pd.Timestamp("2021-01-02")].isna().all()


def test_assemble_refuses_an_asset_with_no_rows():
    stories = pd.DataFrame({
        "day": pd.to_datetime(["2021-01-01"]),
        "asset": ["gold"], "n_articles": [10], "tone": [1.0],
    })
    totals = pd.DataFrame({"day": pd.to_datetime(["2021-01-01"]), "total": [1000]})

    with pytest.raises(ValueError, match="no matching stories"):
        assemble_snapshot(stories, totals, "silver")


def test_current_partial_day_is_dropped(gdelt):
    now = pd.Timestamp("2022-06-18 14:30")  # fixture's last day, mid-afternoon
    kept = _drop_incomplete_day(gdelt, now=now)

    assert kept.index.max() == pd.Timestamp("2022-06-17")
    assert gdelt.index.max() == pd.Timestamp("2022-06-18"), "fixture should end today"


def test_sidecar_phrase_mismatch_is_refused():
    cfg = AssetConfig(
        asset="gold", symbol="XAUUSD", granularity="H4",
        gdelt=GDELTConfig(title_phrases=("gold price", "gold bullion")),
    )

    _check_gdelt_sidecar({"title_phrases": ["gold price", "gold bullion"]}, cfg)
    with pytest.raises(ValueError, match="snapshot IS the sample"):
        _check_gdelt_sidecar({"title_phrases": ["gold price"]}, cfg)
    # Order is part of the identity: a reordered list is a different regex
    # alternation, so it may not silently pass as the same sample.
    with pytest.raises(ValueError, match="snapshot IS the sample"):
        _check_gdelt_sidecar({"title_phrases": ["gold bullion", "gold price"]}, cfg)
