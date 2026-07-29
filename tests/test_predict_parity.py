"""Train/live parity: the live path must compute what training computed.

This is the guard against the worst bug class in a deployed trading system.
Train/live skew is silent — no exception, no warning, just a model receiving
inputs from a distribution it was never fitted on, and a live account that
quietly underperforms its own backtest.

The property under test: features computed from a *trailing window* (what a
live bot has) must equal the same bar's features computed from *full history*
(what training had). Anything that breaks it — a re-anchored rolling window,
an unwitting forward-fill, a partner series joined the wrong way — shows up
here rather than in the P&L.

Runs offline: macro is disabled so no FRED call is needed, and the cross-asset
leg is synthetic.
"""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.config import (
    AssetConfig,
    CrossAssetConfig,
    CVConfig,
    FeatureConfig,
    LabelConfig,
)
from commodity_ml.features.pipeline import (
    MIN_HISTORY_BARS,
    build_feature_frame,
    latest_feature_row,
    partner_symbols,
)

from commodity_ml.config import COTConfig, GDELTConfig
from conftest import make_cot, make_gdelt, make_ohlcv

GOLD, SILVER = "XAUUSD", "XAGUSD"


@pytest.fixture
def cfg() -> AssetConfig:
    return AssetConfig(
        asset="gold",
        symbol=GOLD,
        granularity="H4",
        # Macro off: it needs the network, and its causality is pinned
        # separately in test_macro.py.
        features=FeatureConfig(technical=True, macro=False, crossasset=True),
        crossasset=CrossAssetConfig(numerator_symbol=GOLD, denominator_symbol=SILVER),
        label=LabelConfig(horizon_bars=12),
        cv=CVConfig(gap_bars=12),
    )


@pytest.fixture
def candles():
    n = MIN_HISTORY_BARS * 3
    gold = make_ohlcv(n=n, seed=1)
    silver = make_ohlcv(n=n, seed=2)
    for frame in (gold, silver):
        frame.index = pd.date_range("2020-01-01", periods=n, freq="4h", tz="UTC")
    return gold, {SILVER: silver}


def test_live_row_matches_the_training_row(cfg, candles):
    """The headline property. A live bot holding only a trailing window must
    arrive at the same feature vector training recorded for that bar."""
    gold, partners = candles
    full, _ = build_feature_frame(cfg, gold, partners=partners)

    for cut in (MIN_HISTORY_BARS * 2, len(gold) - 5, len(gold)):
        window = gold.iloc[:cut].iloc[-MIN_HISTORY_BARS * 2:]
        live = latest_feature_row(
            cfg,
            window,
            partners={s: df.loc[: window.index[-1]] for s, df in partners.items()},
        )
        expected = full.loc[window.index[-1]]

        pd.testing.assert_series_equal(
            live, expected, check_exact=False, rtol=1e-6,
            obj=f"live feature row at bar {cut}",
        )


def test_partner_data_from_the_future_is_never_used(cfg, candles):
    """The live path sees only the partner's past. If replacing every future
    partner bar with garbage changed today's features, the cross-asset join
    would be reaching forward."""
    gold, partners = candles
    cut = len(gold) - 200
    window = gold.iloc[:cut]

    honest = {s: df.loc[: window.index[-1]] for s, df in partners.items()}
    corrupted = {}
    for symbol, df in partners.items():
        poisoned = df.copy()
        poisoned.loc[poisoned.index > window.index[-1], ["open", "high", "low", "close"]] = 1e6
        corrupted[symbol] = poisoned

    pd.testing.assert_series_equal(
        latest_feature_row(cfg, window, partners=honest),
        latest_feature_row(cfg, window, partners=corrupted),
    )


def test_crossasset_features_are_causal(cfg, candles):
    """Truncation test, extended to the cross-asset family — the same
    property test_features.py pins for technicals."""
    gold, partners = candles
    full, _ = build_feature_frame(cfg, gold, partners=partners)

    cut = len(gold) - 300
    truncated, _ = build_feature_frame(
        cfg,
        gold.iloc[:cut],
        partners={s: df.iloc[:cut] for s, df in partners.items()},
    )
    pd.testing.assert_frame_equal(
        full.iloc[:cut], truncated, check_exact=False, rtol=1e-9
    )


def test_short_history_is_refused_not_guessed(cfg, candles):
    """Features have warmup. A live bot handed too few bars must refuse to
    predict, not quietly score a half-computed vector."""
    gold, partners = candles
    with pytest.raises(ValueError, match="at least"):
        latest_feature_row(cfg, gold.iloc[: MIN_HISTORY_BARS - 1], partners=partners)


def test_incomplete_feature_row_is_refused(cfg, candles):
    """A NaN reaching the model is a malformed input, not a neutral value."""
    gold, partners = candles
    window = gold.iloc[-MIN_HISTORY_BARS * 2:]

    # A partner that went stale weeks ago: the ratio is genuinely unknown.
    stale = {s: df.loc[: window.index[-1] - pd.Timedelta(days=30)]
             for s, df in partners.items()}
    with pytest.raises(ValueError, match="incomplete"):
        latest_feature_row(cfg, window, partners=stale)


def test_missing_partner_is_reported_clearly(cfg, candles):
    gold, _ = candles
    with pytest.raises(ValueError, match=SILVER):
        build_feature_frame(cfg, gold, partners={})


def test_partner_symbols_excludes_the_asset_itself(cfg):
    assert partner_symbols(cfg) == [SILVER]


def test_ratio_moves_the_right_way(cfg, candles):
    """Sanity: a silver crash must push the gold/silver ratio up, so its
    z-score should rise. Catches an inverted numerator/denominator."""
    gold, partners = candles
    crashed = {SILVER: partners[SILVER].copy()}
    tail = crashed[SILVER].index[-100:]
    crashed[SILVER].loc[tail, ["open", "high", "low", "close"]] *= 0.5

    base, _ = build_feature_frame(cfg, gold, partners=partners)
    shocked, _ = build_feature_frame(cfg, gold, partners=crashed)
    assert shocked["gsr_z"].iloc[-1] > base["gsr_z"].iloc[-1]
    assert np.isfinite(shocked["gsr_z"].iloc[-1])


# --------------------------------------------------------------------------
# Step-2 alt-data families through the same parity gauntlet
# --------------------------------------------------------------------------

@pytest.fixture
def altdata_cfg(cfg) -> AssetConfig:
    return AssetConfig(
        asset="gold",
        symbol=GOLD,
        granularity="H4",
        features=FeatureConfig(
            technical=True, macro=False, crossasset=True, cot=True, gdelt=True
        ),
        crossasset=CrossAssetConfig(numerator_symbol=GOLD, denominator_symbol=SILVER),
        cot=COTConfig(contract_code="088691"),
        gdelt=GDELTConfig(query='"gold price" OR bullion'),
        label=LabelConfig(horizon_bars=12),
        cv=CVConfig(gap_bars=12),
    )


@pytest.fixture
def altdata():
    # Candles run 2020-01 -> 2022-01; both snapshots comfortably span that.
    return make_cot(), make_gdelt()


def test_live_row_matches_training_row_with_altdata(altdata_cfg, candles, altdata):
    """The headline parity property with all families on — and the live call
    only gets alt-data up to the window end, so this pins causality too."""
    gold, partners = candles
    cot, gdelt = altdata
    full, _ = build_feature_frame(altdata_cfg, gold, partners=partners, cot=cot, gdelt=gdelt)

    for cut in (MIN_HISTORY_BARS * 2, len(gold) - 5, len(gold)):
        window = gold.iloc[:cut].iloc[-MIN_HISTORY_BARS * 2:]
        end = window.index[-1].tz_localize(None)
        live = latest_feature_row(
            altdata_cfg,
            window,
            partners={s: df.loc[: window.index[-1]] for s, df in partners.items()},
            cot=cot.loc[:end],
            gdelt=gdelt.loc[:end],
        )
        pd.testing.assert_series_equal(
            live, full.loc[window.index[-1]], check_exact=False, rtol=1e-6,
            obj=f"live feature row at bar {cut}",
        )


def test_future_cot_reports_are_never_used(altdata_cfg, candles, altdata):
    gold, partners = candles
    cot, gdelt = altdata
    window = gold.iloc[: len(gold) - 200]
    end = window.index[-1].tz_localize(None)
    honest_partners = {s: df.loc[: window.index[-1]] for s, df in partners.items()}

    poisoned = cot.copy()
    poisoned.loc[poisoned.index > end, ["mm_long", "mm_short", "oi"]] = 1e9

    pd.testing.assert_series_equal(
        latest_feature_row(altdata_cfg, window, partners=honest_partners,
                           cot=cot, gdelt=gdelt),
        latest_feature_row(altdata_cfg, window, partners=honest_partners,
                           cot=poisoned, gdelt=gdelt),
    )


def test_future_gdelt_days_are_never_used(altdata_cfg, candles, altdata):
    gold, partners = candles
    cot, gdelt = altdata
    window = gold.iloc[: len(gold) - 200]
    end = window.index[-1].tz_localize(None)
    honest_partners = {s: df.loc[: window.index[-1]] for s, df in partners.items()}

    poisoned = gdelt.copy()
    poisoned.loc[poisoned.index > end, "tone"] = 100.0

    pd.testing.assert_series_equal(
        latest_feature_row(altdata_cfg, window, partners=honest_partners,
                           cot=cot, gdelt=gdelt),
        latest_feature_row(altdata_cfg, window, partners=honest_partners,
                           cot=cot, gdelt=poisoned),
    )


def test_missing_altdata_frame_is_reported(altdata_cfg, candles, altdata):
    gold, partners = candles
    cot, gdelt = altdata

    with pytest.raises(ValueError, match="load_cot_snapshot"):
        build_feature_frame(altdata_cfg, gold, partners=partners, cot=None, gdelt=gdelt)
    with pytest.raises(ValueError, match="load_gdelt_snapshot"):
        build_feature_frame(altdata_cfg, gold, partners=partners, cot=cot, gdelt=None)
