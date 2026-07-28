"""Tier-1 diagnostics for the step-2 alt-data features (docs/step2_design.md §5).

Usage:
    python scripts/diagnose_altdata.py --config config/gold.yaml

What each feature actually measures, before the full pipeline runs: Spearman
rank correlation (IC) of every pre-registered feature against forward returns
AND forward volatility at 5 and 20 trading days — the run-1 lesson made both
mandatory (a "direction signal" once turned out to be volatility in costume).

Honesty rules baked in:
  * DEVELOPMENT SLICE ONLY. Everything is truncated at the holdout fence.
  * Native frequency. COT is ~327 weekly observations, GDELT ~2,300 daily
    ones; computing ICs on ffilled H4 bars would fake the sample size.
  * Forward windows start at the observation's KNOWABLE date (7-day COT lag
    with the interruption table applied; next UTC midnight for GDELT), so the
    correlation is with returns a trader could actually have captured.
  * Newey-West standard errors with lag = the forward horizon, because
    overlapping forward windows make neighbouring observations dependent —
    the naive SE would be up to ~sqrt(horizon) times too small.
  * NO GATING. Tier 1 informs interpretation of Tier 2; nothing is retained
    or retired on these numbers (pre-registered in §5).
  * Every run of this script is a look — ledger it in §8.

Output: printed tables + reports/step2/tier1_diagnostics.txt.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.features import cot as cot_module
from commodity_ml.features import gdelt as gdelt_module

HORIZONS_TRADING_DAYS = (5, 20)  # pre-registered in docs/step2_design.md §5


def spearman_ic_newey_west(x: pd.Series, y: pd.Series, lag: int) -> dict:
    """Spearman IC with an autocorrelation-robust (Newey-West) z-score.

    Rank-transform both series, standardize, and take the mean of the product
    — that mean IS the rank correlation. Its variance is then estimated with
    Bartlett-weighted autocovariances up to `lag`, because overlapping forward
    windows make consecutive products dependent; the plain 1/sqrt(n) SE would
    overstate significance by up to ~sqrt(lag).
    """
    both = pd.concat([x, y], axis=1).dropna()
    n = len(both)
    if n < 30:
        return {"ic": np.nan, "se": np.nan, "z": np.nan, "n": n}

    zx = _standardized_ranks(both.iloc[:, 0])
    zy = _standardized_ranks(both.iloc[:, 1])
    u = (zx * zy).to_numpy()
    ic = u.mean()

    centred = u - ic
    variance = centred @ centred / n
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1 - k / (lag + 1)
        variance += 2 * weight * (centred[k:] @ centred[:-k]) / n
    se = np.sqrt(max(variance, 1e-12) / n)
    return {"ic": ic, "se": se, "z": ic / se, "n": n}


def _standardized_ranks(series: pd.Series) -> pd.Series:
    ranks = series.rank()
    return (ranks - ranks.mean()) / ranks.std(ddof=0)


def forward_outcomes(daily_close: pd.Series, t0: pd.Timestamp, horizon: int) -> tuple:
    """(forward log return, forward realized vol) over `horizon` trading days
    strictly after the knowable moment t0."""
    idx = daily_close.index
    start = idx.searchsorted(t0, side="left")
    end = start + horizon
    if end >= len(idx):
        return np.nan, np.nan
    window = np.log(daily_close.iloc[start : end + 1].to_numpy())
    return window[-1] - window[0], float(np.std(np.diff(window), ddof=0))


def ic_table(derived: pd.DataFrame, knowable: pd.DatetimeIndex,
             daily_close: pd.Series, obs_per_horizon_lag) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS_TRADING_DAYS:
        outcomes = [forward_outcomes(daily_close, t, horizon) for t in knowable]
        fwd_ret = pd.Series([o[0] for o in outcomes], index=derived.index)
        fwd_vol = pd.Series([o[1] for o in outcomes], index=derived.index)
        lag = obs_per_horizon_lag(horizon)

        for feature in derived.columns:
            for target, series in (("return", fwd_ret), ("volatility", fwd_vol)):
                stats = spearman_ic_newey_west(derived[feature], series, lag)
                halves = _half_ics(derived[feature], series)
                rows.append({
                    "feature": feature, "horizon_td": horizon, "vs": target,
                    "ic": round(stats["ic"], 4), "se_nw": round(stats["se"], 4),
                    "z": round(stats["z"], 2), "n": stats["n"],
                    "ic_half1": halves[0], "ic_half2": halves[1],
                })
    return pd.DataFrame(rows)


def _half_ics(x: pd.Series, y: pd.Series) -> tuple:
    both = pd.concat([x, y], axis=1).dropna()
    if len(both) < 60:
        return (np.nan, np.nan)
    mid = len(both) // 2
    first = both.iloc[:mid].corr(method="spearman").iloc[0, 1]
    second = both.iloc[mid:].corr(method="spearman").iloc[0, 1]
    return (round(first, 4), round(second, 4))


def refuse_dirty_preregistration(allow: bool) -> None:
    """Tier-1 numbers are only interpretable against a pre-registration that
    was fixed BEFORE they were read — an uncommitted design doc could be
    edited afterwards and nobody could prove otherwise."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "docs/step2_design.md"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    if result.stdout.strip():
        message = (
            "docs/step2_design.md has uncommitted changes. Commit the "
            "pre-registration BEFORE reading any Tier-1 number, or the numbers "
            "cannot be interpreted against it."
        )
        if not allow:
            raise SystemExit(message + "\n(--allow-dirty-preregistration to override.)")
        print(f"WARNING: {message} (override active)\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--allow-dirty-preregistration", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    refuse_dirty_preregistration(args.allow_dirty_preregistration)

    candles = pd.read_parquet(cfg.raw_path())
    candles = cfg.development_slice(candles)  # the holdout stays fenced
    daily_close = candles["close"].resample("1D").last().dropna()
    dev_end = daily_close.index[-1]
    print(f"Development slice only: candles through {dev_end.date()} "
          f"({len(daily_close):,} trading days)\n")

    sections = []

    # --- COT: weekly observations, knowable per the lag + interruption table
    cot_raw = cot_module.load_cot_snapshot(cfg)
    derived = cot_module.derive_weekly(cot_raw).dropna()
    knowable = cot_module.knowable_index(derived.index)
    keep = knowable < dev_end - pd.Timedelta(days=45)  # room for the 20d window
    derived, knowable = derived.loc[keep], knowable[keep]
    # Weekly obs vs a 5-trading-day horizon barely overlap; vs 20 days each
    # observation shares its future with ~3 neighbours.
    table = ic_table(derived, knowable, daily_close,
                     lambda h: max(1, round(h / 5)))
    sections.append(("COT (weekly, n≈%d — EXPLORATORY: detectable |ic|≈0.155)"
                     % len(derived), table))

    # --- GDELT: daily observations, knowable next UTC midnight
    gdelt_raw = gdelt_module.load_gdelt_snapshot(cfg)
    derived = gdelt_module.derive_daily(gdelt_raw).dropna()
    if derived.index.tz is None:
        derived.index = derived.index.tz_localize("UTC")
    knowable = derived.index + pd.Timedelta(days=gdelt_module.PUBLICATION_LAG_DAYS)
    keep = knowable < dev_end - pd.Timedelta(days=45)
    derived, knowable = derived.loc[keep], knowable[keep]
    table = ic_table(derived, knowable, daily_close, lambda h: h)
    sections.append(("GDELT (daily, n≈%d)" % len(derived), table))

    report_lines = []
    for title, table in sections:
        block = f"=== {title} ===\n{table.to_string(index=False)}\n"
        print(block)
        report_lines.append(block)

    print("Reminders (pre-registered): no gating on these numbers; |z| ~ 2 on a\n"
          "single cell is unremarkable across 24 cells; a volatility-only result\n"
          "is a step-4 sizing candidate, not a direction signal. Ledger this run\n"
          "in docs/step2_design.md §8.")

    out = cfg.reports_dir / "step2" / "tier1_diagnostics.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"Tier-1 diagnostics, run {pd.Timestamp.utcnow().isoformat()}\n"
        f"dev slice through {dev_end.date()}\n\n" + "\n".join(report_lines)
    )
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
