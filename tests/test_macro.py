"""Macro feature alignment — where publication-lag lookahead would hide.

Macro series are stamped with the date they *describe*, not the date they
became public. Joining on the observation date leaks the future, and because
macro series are persistent, the leak produces convincing-looking results.
These tests run offline on synthetic FRED-shaped data.
"""

import numpy as np
import pandas as pd

from commodity_ml.features.macro import (
    COMBO_FEATURE,
    MACRO_SERIES,
    Z_WINDOW_DAYS,
    _column_lags,
    build_macro_features,
    macro_feature_names,
)


def synthetic_macro(n_days: int = 900) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {s.name: np.cumsum(rng.normal(0, 0.05, n_days)) + 100 for s in MACRO_SERIES},
        index=dates,
    )


def bar_index(start: str = "2024-06-01", periods: int = 400) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="4h", tz="UTC")


def test_every_declared_feature_is_produced():
    out = build_macro_features(synthetic_macro(), bar_index())
    assert set(out.columns) == set(macro_feature_names())
    assert len(out) == 400


def test_combo_feature_takes_the_slowest_lag():
    """The combined feature mixes real yields and the dollar; it can only be
    known once the slower of the two is published."""
    lags = _column_lags()
    dollar_lag = next(s.publication_lag_days for s in MACRO_SERIES if s.name == "dollar_broad")
    yield_lag = next(s.publication_lag_days for s in MACRO_SERIES if s.name == "real_yield_10y")
    assert lags[COMBO_FEATURE] == max(dollar_lag, yield_lag) > yield_lag


def test_values_respect_publication_lag():
    """A bar must carry the last observation published by then — not the most
    recent one *dated* by then. Those differ by exactly the lag, and using the
    latter is the lookahead bug this whole module exists to prevent."""
    macro = synthetic_macro(n_days=1400)
    bars = bar_index()
    out = build_macro_features(macro, bars)

    # Recompute the raw 5-day change independently and check the alignment.
    for series in MACRO_SERIES:
        column = f"{series.name}_chg_5d"
        truth = macro[series.name].ffill().diff(5)
        cutoff = bars[-1].tz_convert(None) - pd.Timedelta(days=series.publication_lag_days)

        expected = truth.loc[truth.index <= cutoff].iloc[-1]
        assert np.isclose(out[column].iloc[-1], expected), (
            f"{column} at the final bar should be the last value published "
            f"{series.publication_lag_days}d earlier"
        )

        # And it must NOT be the value dated at the bar itself, which is what a
        # naive join would pick up.
        naive = truth.loc[truth.index <= bars[-1].tz_convert(None)].iloc[-1]
        assert not np.isclose(naive, expected), "fixture too flat to detect a lag error"


def test_macro_features_are_causal():
    """Appending future macro observations must not change past bars' features."""
    macro = synthetic_macro()
    bars = bar_index()

    full = build_macro_features(macro, bars)
    # Truncate macro data to what existed at the midpoint, keep the same bars.
    midpoint = bars[len(bars) // 2].tz_convert(None)
    truncated_macro = macro.loc[macro.index <= midpoint]
    partial = build_macro_features(truncated_macro, bars)

    early = bars[: len(bars) // 2 - 20]  # bars comfortably before the cut
    pd.testing.assert_frame_equal(
        full.loc[early], partial.loc[early], check_exact=False, rtol=1e-9
    )


def test_zscore_uses_only_trailing_window():
    """A spike in macro data dated *after* the bars must not alter their
    features. Rolling statistics are trailing-only, so nothing may reach back."""
    macro = synthetic_macro(n_days=1400)  # extends well past the bar window
    bars = bar_index()  # 2024-06-01 onward, 400 x 4h bars

    future_cut = bars[-1].tz_convert(None) + pd.Timedelta(days=30)
    spiked = macro.copy()
    spiked.loc[spiked.index > future_cut] += 50.0
    assert (spiked.index > future_cut).sum() > 0, "fixture must contain future data to spike"

    base = build_macro_features(macro, bars)
    after = build_macro_features(spiked, bars)
    pd.testing.assert_frame_equal(base, after, check_exact=False, rtol=1e-9)


def test_warmup_produces_nans_not_silent_zeros():
    """Before the z-score window is warm, features must be NaN so the dataset
    builder drops those rows rather than training on fabricated values."""
    short = synthetic_macro(n_days=Z_WINDOW_DAYS // 4)
    out = build_macro_features(short, bar_index(start="2022-02-01", periods=50))
    z_columns = [c for c in out.columns if c.endswith("_z")]
    assert out[z_columns].isna().all().all()
