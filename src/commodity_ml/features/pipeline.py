"""The one feature-assembly path, shared by training and live prediction.

Why this module exists
----------------------
Train/live skew is the most dangerous bug class in a deployed trading system,
because it is silent. If the live path computes `ema_50_dist` even slightly
differently from the training path — a different window, a different column
order, a forward-fill the backtest never did — the model receives inputs from
a distribution it was never fitted on. Nothing raises. The backtest still
looks fine. The live account just quietly underperforms it.

The structural fix is to make it impossible to have two paths. Both
`scripts/build_dataset.py` and `scripts/predict.py` call `build_feature_frame`
here; neither assembles features itself. `tests/test_predict_parity.py` pins
the property that matters: the feature row the live path computes from a
trailing window must equal the row the training path computed from full
history.

History requirements
--------------------
Features have warmup. The longest lookback is the cross-asset z-score
(750 bars), ahead of EMA-200 and the 100-bar volatility ratio. A live caller
handing over 300 bars would get NaNs, or worse, subtly wrong EWM values —
exponential moving averages have infinite memory, so a truncated window
converges to the true value rather than equalling it immediately.
MIN_HISTORY_BARS is set well past the point where that discrepancy dies.
"""

from __future__ import annotations

import pandas as pd

from commodity_ml.config import AssetConfig
from commodity_ml.features.crossasset import (
    Z_WINDOW_BARS,
    build_crossasset_features,
    crossasset_feature_names,
)
from commodity_ml.features.macro import build_macro_features, macro_feature_names
from commodity_ml.features.technical import build_features

# Bars of trailing history a caller must supply for every feature to be both
# defined and converged. Generous on purpose: over-supplying costs nothing.
MIN_HISTORY_BARS = 2 * Z_WINDOW_BARS


def partner_symbols(cfg: AssetConfig) -> list[str]:
    """Symbols other than this asset's own whose candles the features need."""
    if not cfg.features.crossasset:
        return []
    legs = (cfg.crossasset.numerator_symbol, cfg.crossasset.denominator_symbol)
    return [s for s in dict.fromkeys(legs) if s != cfg.symbol]


def load_partner_candles(cfg: AssetConfig) -> dict[str, pd.DataFrame]:
    """Read the partner legs' cached candles from data/raw/."""
    frames = {}
    for symbol in partner_symbols(cfg):
        path = cfg.raw_path(symbol=symbol)
        if not path.exists():
            raise SystemExit(
                f"Cross-asset features need {symbol} candles, but {path} is missing.\n"
                f"Fetch them first, e.g.:\n"
                f"    python scripts/fetch_data.py --config config/"
                f"{'silver' if symbol.startswith('XAG') else 'gold'}.yaml"
            )
        frames[symbol] = pd.read_parquet(path)
    return frames


def build_feature_frame(
    cfg: AssetConfig,
    raw: pd.DataFrame,
    *,
    macro: pd.DataFrame | None = None,
    partners: dict[str, pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble every enabled feature family onto `raw`'s bar index.

    `macro` is the raw FRED frame from features.macro.fetch_macro; `partners`
    maps symbol -> candles for cross-asset legs. Both are passed in rather than
    fetched here so the caller controls network access — training pulls fresh
    data, the parity test passes synthetic frames, and the live path can cache.
    """
    frames: list[pd.DataFrame] = []
    names: list[str] = []

    if cfg.features.technical:
        technical, technical_names = build_features(raw)
        frames.append(technical)
        names += technical_names

    if cfg.features.macro:
        if macro is None:
            raise ValueError(
                "features.macro is enabled but no macro frame was passed. "
                "Call features.macro.fetch_macro() and hand the result in."
            )
        frames.append(build_macro_features(macro, raw.index))
        names += macro_feature_names()

    if cfg.features.crossasset:
        available = {cfg.symbol: raw, **(partners or {})}
        numerator = _leg(available, cfg.crossasset.numerator_symbol)
        denominator = _leg(available, cfg.crossasset.denominator_symbol)
        frames.append(build_crossasset_features(numerator, denominator, raw.index))
        names += crossasset_feature_names()

    return pd.concat(frames, axis=1), names


def latest_feature_row(
    cfg: AssetConfig,
    raw: pd.DataFrame,
    *,
    macro: pd.DataFrame | None = None,
    partners: dict[str, pd.DataFrame] | None = None,
) -> pd.Series:
    """The feature vector for the most recent bar — the live prediction entry point.

    Raises rather than returning NaNs if history is too short or the row is
    incomplete. A live bot must refuse to trade on a malformed feature vector,
    not pass it to the model and hope.
    """
    if len(raw) < MIN_HISTORY_BARS:
        raise ValueError(
            f"Need at least {MIN_HISTORY_BARS} bars of history to compute converged "
            f"features, got {len(raw)}. Fetch more history before predicting."
        )

    features, _ = build_feature_frame(cfg, raw, macro=macro, partners=partners)
    row = features.iloc[-1]
    if row.isna().any():
        missing = list(row.index[row.isna()])
        raise ValueError(
            f"Feature vector for {features.index[-1]} is incomplete: {missing}. "
            "Refusing to predict on partial inputs."
        )
    return row


def _leg(available: dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
    if symbol not in available:
        raise ValueError(
            f"Cross-asset features need candles for {symbol}, which were not supplied. "
            f"Have: {sorted(available)}"
        )
    return available[symbol]
