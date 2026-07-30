"""Step 2b: the volatility-forecast feature set (docs/step2b_vol_design.md §3).

Everything here lives on a DAILY observation grid — the same grid Tier-1 used —
because the target is realized vol over the next 5/20 *trading days* and the
alt-data sources are daily/weekly. The H4 candles supply intraday returns for
the realized-vol computation, then drop out of the picture.

Observation convention (matches diagnose_altdata.forward_outcomes): a feature
row dated day D is knowable at D's close, and the forward window runs from D's
close to D+h's close. Trailing features may use day D itself; forward targets
may not.

Family definitions are fixed by the pre-registration. B0 is HAR-style on
purpose — the standard, linear, three-knob benchmark — and the alt families
must beat B1 (= B0 + implied vol where it exists), not B0, so that a feature
which merely proxies information the market already publishes for free cannot
pass. F-COT and F-ATTN reuse the step-2 builders verbatim: same formulas, same
lags, same staleness rules, aligned to the daily grid instead of H4 bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from commodity_ml.features import event_dates

# Mirrors macro.py's MacroSeries convention for VIXCLS: FRED daily index
# closes post next business day; 2 calendar days is the conservative whole-day
# bound. GVZ deliberately does NOT join MACRO_SERIES — that tuple is part of
# the frozen step-1 template and adding a series would change every dataset.
GVZ_FRED_CODE = "GVZCLS"
GVZ_LAG_DAYS = 2
GVZ_START = "2008-06-03"  # first observation FRED carries; a fetch from
                          # earlier silently returns the same thing, so pin it
GVZ_MAX_STALENESS = pd.Timedelta(days=5)  # long weekends bridge; gaps do not

HAR_WINDOWS = (1, 5, 22)  # trailing trading days — the standard HAR ladder
MIN_BARS_PER_DAY = 3      # a 1–2 bar holiday stub is not a day's worth of
                          # intraday variance; treat it as missing, not as calm

TARGET_HORIZONS = (5, 20)


def fetch_gvz(start: str = GVZ_START, end: str | None = None) -> pd.Series:
    """GVZCLS from FRED, same fetch path as features/macro.py."""
    import pandas_datareader.data as web

    fetch_end = (pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None))
    raw = web.DataReader(GVZ_FRED_CODE, "fred", pd.Timestamp(start), fetch_end)
    series = raw[GVZ_FRED_CODE].rename("gvz").sort_index()
    series.index.name = "date"
    return series


# --------------------------------------------------------------------------
# Target
# --------------------------------------------------------------------------

def forward_log_vol(daily_close: pd.Series, horizon: int) -> pd.Series:
    """log realized vol of the next `horizon` trading days, per observation day.

    Identical arithmetic to diagnose_altdata.forward_outcomes' vol leg —
    std (ddof=0) of daily log returns over close[D] .. close[D+h] — pinned by
    a test, vectorized here because the gate computes it thousands of times.
    Log-transformed: vol is right-skewed, and an OLS on the raw level would
    spend its fit on a handful of crisis weeks.
    """
    logp = np.log(daily_close.to_numpy(dtype=float))
    returns = np.diff(logp)
    n = len(returns)
    out = np.full(len(daily_close), np.nan)
    for i in range(len(daily_close)):
        if i + horizon <= n:
            out[i] = np.std(returns[i : i + horizon])
    with np.errstate(divide="ignore"):  # a zero-vol window is honest -inf -> NaN
        logged = np.where(out > 0, np.log(out), np.nan)
    return pd.Series(logged, index=daily_close.index, name=f"fwd_log_vol_{horizon}d")


# --------------------------------------------------------------------------
# B0 — price-only HAR baseline
# --------------------------------------------------------------------------

def daily_realized_vol(h4_close: pd.Series) -> pd.Series:
    """One number per trading day: std of that day's intraday H4 log returns.

    This is what makes B0 a genuine HAR rather than a squashed proxy — the
    intraday path carries the variance information a single daily close throws
    away. Within-day returns only: each day's first bar spans the boundary
    from the previous session, and including that return would make a day's RV
    depend on which side of midnight a gap happens to fall. Days with fewer
    than MIN_BARS_PER_DAY bars become NaN.
    """
    logp = np.log(h4_close)
    returns = logp.diff()
    day = pd.Series(logp.index.date, index=logp.index)
    returns[day != day.shift()] = np.nan  # cross-day gap, excluded
    by_day = returns.groupby(day)
    rv = by_day.std(ddof=0)
    counts = by_day.count()  # k bars give k-1 within-day returns
    rv[counts < MIN_BARS_PER_DAY - 1] = np.nan
    rv.index = pd.DatetimeIndex(rv.index)
    if h4_close.index.tz is not None:
        rv.index = rv.index.tz_localize(h4_close.index.tz)
    return rv.rename("rv_1d")


def build_b0(h4_close: pd.Series, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """log trailing mean RV over the HAR windows, aligned to the grid.

    The windows roll over COMPLETED trading days — `daily_realized_vol` NaNs
    the 1–2 bar Sunday-evening stubs Dukascopy opens the week with, and
    rolling over the raw series would let one such stub void every 22-day
    window it touches (measured on real gold: the 22d rung had zero valid
    rows). Dropping incomplete days first means "the last 5 days of vol"
    means the last 5 days that HAD a day's worth of trading.

    Trailing windows end on the observation day itself (knowable at its
    close); min_periods equals the window so a half-warm rung is NaN rather
    than quietly averaged over fewer days. The reindex ffill (limit 2) only
    bridges grid rows that have no completed-day RV of their own — a Sunday
    row seeing Friday's trailing vol is exactly what was knowable that
    evening.
    """
    rv = daily_realized_vol(h4_close).dropna()
    out = pd.DataFrame(index=grid)
    for window in HAR_WINDOWS:
        mean_rv = rv.rolling(window, min_periods=window).mean()
        with np.errstate(divide="ignore"):
            logged = np.log(mean_rv)
        out[f"har_log_rv_{window}d"] = logged.reindex(grid).ffill(limit=2).to_numpy()
    return out


# --------------------------------------------------------------------------
# F-IV — implied vol (gold only; see design §2 for the VXSLV exclusion)
# --------------------------------------------------------------------------

def build_iv_features(gvz: pd.Series, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """log GVZ level + its 5-day change, published-lag aligned.

    Same mechanism as macro's aligner: shift observation dates forward by the
    lag, as-of join backward, then a staleness cap so a data outage becomes
    NaN instead of a weeks-old vol level presented as current.
    """
    derived = pd.DataFrame(index=gvz.index)
    with np.errstate(divide="ignore"):
        derived["gvz_log"] = np.log(gvz.to_numpy(dtype=float))
    derived["gvz_chg_5d"] = derived["gvz_log"].diff(5)

    derived = derived.dropna(how="all").copy()
    observed = derived.index  # publication dates, pre-shift, for staleness
    derived.index = derived.index + pd.Timedelta(days=GVZ_LAG_DAYS)
    if derived.index.tz is None and grid.tz is not None:
        derived.index = derived.index.tz_localize(grid.tz)

    frame = pd.DataFrame(index=grid).sort_index()
    merged = pd.merge_asof(
        frame, derived, left_index=True, right_index=True, direction="backward"
    )
    # Staleness: how old is the observation each grid day is actually seeing?
    knowable = derived.index.to_series(index=derived.index)
    age_source = pd.merge_asof(
        frame, knowable.rename("knowable_at").to_frame(),
        left_index=True, right_index=True, direction="backward",
    )["knowable_at"]
    stale = (grid.to_series() - age_source) > GVZ_MAX_STALENESS + pd.Timedelta(days=GVZ_LAG_DAYS)
    merged.loc[stale.to_numpy(), :] = np.nan
    del observed
    return merged


# --------------------------------------------------------------------------
# F-CAL — scheduled events inside the forecast horizon
# --------------------------------------------------------------------------

def build_calendar_features(
    grid: pd.DatetimeIndex,
    horizon: int,
    fomc: pd.DatetimeIndex | None = None,
    cpi_nfp: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Binary: does a scheduled event fall within the next `horizon` grid days?

    "Within the horizon" is measured in TRADING days on the grid itself, so the
    flag describes exactly the window the target covers. The event dates are
    future-dated relative to the observation and that is legal — schedules are
    published in advance (see event_dates.py). The window is (D, D+h]: an
    event on day D itself is this morning's news, already in trailing RV.

    `fomc`/`cpi_nfp` override the verified tables — that exists ONLY for the
    placebo path, which needs the same logic run over time-shifted event days.
    Real features must come from the defaults.
    """
    event_dates.check_coverage(grid.max(), horizon)
    if fomc is None:
        fomc = event_dates.fomc_days()
    if cpi_nfp is None:
        cpi_nfp = event_dates.cpi_nfp_days()
    naive_grid = grid.tz_localize(None) if grid.tz is not None else grid
    grid_days = pd.DatetimeIndex(naive_grid.normalize())

    def within(events: pd.DatetimeIndex) -> np.ndarray:
        is_event = np.isin(grid_days, events.normalize())
        flags = np.zeros(len(grid_days), dtype=float)
        for offset in range(1, horizon + 1):
            shifted = np.roll(is_event, -offset)
            shifted[len(shifted) - offset:] = False
            flags = np.maximum(flags, shifted.astype(float))
        return flags

    return pd.DataFrame(
        {
            f"fomc_within_{horizon}d": within(fomc),
            f"cpi_or_nfp_within_{horizon}d": within(cpi_nfp),
        },
        index=grid,
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

FAMILY_COLUMNS = {
    "b0": [f"har_log_rv_{w}d" for w in HAR_WINDOWS],
    "iv": ["gvz_log", "gvz_chg_5d"],
    "cot": ["cot_mm_net_z", "cot_mm_net_chg_4w", "cot_prod_net_z"],
    "attn": ["gdelt_vol_smooth7_z"],
}
# calendar columns are horizon-suffixed; resolved per horizon in the gate


def calendar_columns(horizon: int) -> list[str]:
    return [f"fomc_within_{horizon}d", f"cpi_or_nfp_within_{horizon}d"]


def assemble_vol_dataset(
    h4_close: pd.Series,
    cot_raw: pd.DataFrame,
    gdelt_raw: pd.DataFrame,
    gvz: pd.Series | None,
    horizon: int,
    calendar_override: dict[str, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    """Matched-row frame: target + every family, on the daily trading grid.

    One frame with everything present, ablated by column in the gate — the
    step-2 §6 argument: dropping a family by rebuilding would move the row set
    and the fold boundaries, and the delta would measure sample composition.
    gvz=None is the silver case (design §2): B1 degenerates to B0 and the IV
    columns are simply absent, not NaN — absence is a documented design fact,
    NaN would look like a data problem.
    """
    from commodity_ml.features.cot import build_cot_features
    from commodity_ml.features.gdelt import build_gdelt_features

    daily_close = h4_close.resample("1D").last().dropna()
    grid = daily_close.index

    override = calendar_override or {}
    parts = [
        forward_log_vol(daily_close, horizon),
        build_b0(h4_close, grid),
        build_cot_features(cot_raw, grid)[FAMILY_COLUMNS["cot"]],
        build_gdelt_features(gdelt_raw, grid)[FAMILY_COLUMNS["attn"]],
        build_calendar_features(
            grid, horizon,
            fomc=override.get("fomc"), cpi_nfp=override.get("cpi_nfp"),
        ),
    ]
    if gvz is not None:
        parts.append(build_iv_features(gvz, grid))

    dataset = pd.concat(parts, axis=1)
    return dataset.dropna()
