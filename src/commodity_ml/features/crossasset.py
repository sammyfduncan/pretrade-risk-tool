"""The gold/silver ratio — one cross-asset feature, then feature work closes.

Why this one and nothing else
-----------------------------
Runs 1-3 established that a single metal's own price path carries no
directional information (best feature correlates ~0.05-0.09 with forward
return) and that the same result reproduces on silver. Adding more
descriptions of the *same* price series cannot fix that.

The gold/silver ratio is different in kind: it is a relationship between two
instruments, and neither leg's own history contains it. Its usual reading is
regime, not direction — the ratio rises when demand is defensive (gold is the
monetary metal, bought in risk-off conditions) and falls when demand is
industrial (roughly half of silver consumption is industrial, so silver
outperforms in expansions). That is exactly the kind of question meta-labeling
asks the model: not "which way", but "is this the sort of environment where
obeying the trend rule pays off".

Honest accounting: every extra feature family tried before the holdout is
spent is another comparison, and enough comparisons manufacture a winner. This
one was on the pre-registered list in `docs/step1_freeze_record.md` *before*
any results were seen, which is what makes it principled rather than a reaction to a bad number.
CFTC positioning and ETF flows stay unbuilt. Feature work is closed after this.

Causality
---------
The ratio at bar t uses each leg's close at t, both known at t. The partner
leg is joined backward-only with a staleness tolerance, so a gap in one feed
produces NaN (dropped) rather than a ratio quietly built from a stale price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RATIO_NAME = "gsr"  # gold/silver ratio

# Windows are in *bars*, sized for H4. The z-window is deliberately long: the
# ratio trends for years, so "high or low versus recent history" needs a
# reference period measured in months, not days.
Z_WINDOW_BARS = 750  # ~6 months at H4
CHANGE_WINDOWS_BARS = (30, 120)  # ~1 week, ~1 month at H4

# How stale a partner quote may be before the ratio is treated as unknown.
# Comfortably covers a weekend plus a missed session, without letting a
# genuinely dead feed carry a fossil price forward for weeks.
MAX_STALENESS = pd.Timedelta(days=3)


def build_crossasset_features(
    numerator: pd.DataFrame,
    denominator: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Derive stationary ratio features aligned to `bar_index`.

    The raw ratio level is not a feature: it has drifted between roughly 30
    and 120 over the past decade, so an absolute value means nothing without
    context. The model sees a rolling z-score and log changes instead — the
    same treatment the macro series get.
    """
    ratio = _aligned_ratio(numerator["close"], denominator["close"], bar_index)

    out = pd.DataFrame(index=bar_index)
    rolling = ratio.rolling(Z_WINDOW_BARS, min_periods=Z_WINDOW_BARS // 2)
    out[f"{RATIO_NAME}_z"] = (ratio - rolling.mean()) / rolling.std()

    log_ratio = np.log(ratio)
    for window in CHANGE_WINDOWS_BARS:
        out[f"{RATIO_NAME}_chg_{window}"] = log_ratio.diff(window)

    return out


def crossasset_feature_names() -> list[str]:
    return [f"{RATIO_NAME}_z"] + [f"{RATIO_NAME}_chg_{w}" for w in CHANGE_WINDOWS_BARS]


def _aligned_ratio(
    numerator_close: pd.Series,
    denominator_close: pd.Series,
    bar_index: pd.DatetimeIndex,
) -> pd.Series:
    num = _asof(numerator_close, bar_index)
    den = _asof(denominator_close, bar_index)
    return (num / den.replace(0.0, np.nan)).rename(RATIO_NAME)


def _asof(series: pd.Series, bar_index: pd.DatetimeIndex) -> pd.Series:
    """Backward-only as-of join onto the bar index.

    `direction="backward"` is the whole point: bar t may see the partner's
    last quote at or before t, never the next one. Forward-filling from a
    future quote would be lookahead, and because the two metals are highly
    correlated it would be a particularly flattering kind.
    """
    frame = series.dropna().sort_index().to_frame("value")
    bars = pd.DataFrame(index=pd.DatetimeIndex(bar_index).sort_values())
    merged = pd.merge_asof(
        bars,
        frame,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=MAX_STALENESS,
    )
    return merged["value"].reindex(bar_index)
