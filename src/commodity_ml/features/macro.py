"""Macro features from FRED — the only plausible source of *directional* signal.

The first gold run established that technical indicators on a single price
series carry essentially no directional information (best feature correlated
0.048 with forward return, 0.024 by rank) while carrying a great deal of
volatility information (0.62). Twenty technical indicators are twenty ways of
re-describing the same recent price path, so no amount of modeling extracts
direction that isn't there.

Macro variables are different in kind. README section 3 specifies them for
Track A: "dollar/real-yield proxies for gold and silver". Gold pays no yield,
so the real interest rate is its opportunity cost — this is the single
best-established fundamental driver of the gold price, and it is genuinely
exogenous to gold's own price path.

FRED needs no API key, consistent with the rest of the data layer.

Lookahead risk lives here
-------------------------
Macro data is *published later than it is dated*. FRED stamps an observation
with the date it describes, not the date it became knowable — the broad dollar
index is routinely a week behind. Naively joining on the observation date
hands the model information from the future, and because macro series are
persistent, this produces impressively wrong backtests.

Every series therefore declares a `publication_lag_days`, and alignment uses
only observations whose date is at least that far before the bar. The lags
below are deliberately *conservative* (longer than reality). Overstating a lag
can only weaken results; understating one manufactures them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroSeries:
    fred_code: str
    name: str
    publication_lag_days: int
    description: str


# Publication lags are conservative: FRED daily rates post next business day,
# but the broad dollar index runs about a week behind.
MACRO_SERIES = (
    MacroSeries("DFII10", "real_yield_10y", 2, "10y TIPS real yield — gold's opportunity cost"),
    MacroSeries("DTWEXBGS", "dollar_broad", 8, "Trade-weighted broad US dollar index"),
    MacroSeries("VIXCLS", "vix", 2, "Equity implied volatility — safe-haven demand proxy"),
    MacroSeries("T10YIE", "breakeven_10y", 2, "10y breakeven inflation expectations"),
)

Z_WINDOW_DAYS = 252  # ~1 trading year, for standardizing levels
CHANGE_WINDOWS_DAYS = (5, 20)
COMBO_FEATURE = "real_yield_dollar_combo"


def fetch_macro(start: str, end: str | None = None, verbose: bool = True) -> pd.DataFrame:
    """Download the raw macro series from FRED, indexed by observation date."""
    import pandas_datareader.data as web

    # Pull extra history so rolling z-scores are warm at the first bar we use.
    fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=2 * Z_WINDOW_DAYS)).to_pydatetime()
    fetch_end = (pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None)).to_pydatetime()

    frames = []
    for series in MACRO_SERIES:
        if verbose:
            print(f"  FRED {series.fred_code:10} {series.description}")
        raw = web.DataReader(series.fred_code, "fred", fetch_start, fetch_end)
        frames.append(raw[series.fred_code].rename(series.name))

    macro = pd.concat(frames, axis=1).sort_index()
    macro.index.name = "date"
    return macro


def build_macro_features(macro: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive stationary features and align them to bars without lookahead.

    Levels are non-stationary (a dollar index of 120 means nothing without
    context), so the model sees rolling z-scores and multi-day changes instead.
    """
    derived = pd.DataFrame(index=macro.index)

    for series in MACRO_SERIES:
        level = macro[series.name].ffill()
        rolling = level.rolling(Z_WINDOW_DAYS, min_periods=Z_WINDOW_DAYS // 2)
        derived[f"{series.name}_z"] = (level - rolling.mean()) / rolling.std()
        for window in CHANGE_WINDOWS_DAYS:
            derived[f"{series.name}_chg_{window}d"] = level.diff(window)

    # Gold's textbook driver combination: falling real yields plus a weakening
    # dollar. Interacting them lets a tree split on "both easing at once"
    # without having to discover the product from scratch.
    derived[COMBO_FEATURE] = (
        derived["real_yield_10y_chg_20d"] * derived["dollar_broad_chg_20d"]
    )

    return _align_to_bars(derived, bar_index)


def _column_lags() -> dict[str, int]:
    """Publication lag for every derived column.

    A feature combining several series is only knowable once the *slowest*
    of its inputs has been published, so it takes the maximum lag — not the
    lag of whichever series it happens to be grouped with.
    """
    lags = {}
    for series in MACRO_SERIES:
        lags[f"{series.name}_z"] = series.publication_lag_days
        for window in CHANGE_WINDOWS_DAYS:
            lags[f"{series.name}_chg_{window}d"] = series.publication_lag_days

    combo_inputs = ("real_yield_10y", "dollar_broad")
    lags[COMBO_FEATURE] = max(
        s.publication_lag_days for s in MACRO_SERIES if s.name in combo_inputs
    )
    return lags


def _align_to_bars(derived: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join macro observations onto bars, respecting publication lag.

    A bar at time t may only see an observation dated on or before
    t - publication_lag_days. Columns are grouped by lag so each is joined
    against exactly the data that was knowable at that bar.
    """
    if derived.index.tz is None:
        derived = derived.copy()
        derived.index = derived.index.tz_localize("UTC")

    bars = pd.DataFrame(index=bar_index).sort_index()
    out = pd.DataFrame(index=bar_index)
    lags = _column_lags()

    for lag in sorted(set(lags.values())):
        columns = [c for c in derived.columns if lags[c] == lag]
        block = derived[columns].dropna(how="all").sort_index()
        # Shift observation dates forward by the lag: an observation dated D is
        # treated as first knowable at D + lag.
        block = block.copy()
        block.index = block.index + pd.Timedelta(days=lag)

        merged = pd.merge_asof(
            bars, block, left_index=True, right_index=True, direction="backward"
        )
        out[columns] = merged[columns]

    return out


def macro_feature_names() -> list[str]:
    return list(_column_lags().keys())
