"""Feature engineering on OHLCV candles.

Every feature at bar t must use only information available at the close of
bar t (no centered windows, no forward shifts). tests/test_features.py
enforces this with a truncation/causality test — extend that test whenever
a feature is added here.

Price-level features are expressed as ratios (e.g. close/EMA - 1) rather than
raw prices, so the model can't latch onto the absolute price level, which
drifts over years and won't repeat out of sample.

ATR is hand-rolled (Wilder smoothing) rather than taken from pandas-ta:
it's shared with label construction, and implementing it ourselves keeps the
exact definition visible where correctness matters most.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range with Wilder's smoothing.

    True range at t compares bar t's high/low against the *previous* close,
    so it captures overnight/weekend gaps; the EWM with adjust=False is a
    purely backward-looking recursion.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute the feature matrix from raw OHLCV candles.

    Returns (features_df, feature_names). Warmup rows (longest lookback is
    EMA-200) contain NaNs — the dataset builder drops them after joining
    with labels.
    """
    out = pd.DataFrame(index=df.index)
    close, high, low, open_ = df["close"], df["high"], df["low"], df["open"]

    log_close = np.log(close)

    # Momentum: log returns over several horizons
    for lag in (1, 3, 6, 12, 24):
        out[f"ret_{lag}"] = log_close.diff(lag)

    # Oscillators / trend via pandas-ta-classic
    out["rsi_14"] = ta.rsi(close, length=14)

    macd = ta.macd(close, fast=12, slow=26, signal=9)
    # Normalized by price so the feature is comparable across price regimes
    out["macd_hist"] = macd["MACDh_12_26_9"] / close
    out["macd_line"] = macd["MACD_12_26_9"] / close

    bb = ta.bbands(close, length=20, std=2)
    out["bb_pctb"] = bb["BBP_20_2.0"]  # position within the bands, ~0..1

    for length in (20, 50, 200):
        out[f"ema_{length}_dist"] = close / ta.ema(close, length=length) - 1

    # Volatility level and regime
    out["atr_14_pct"] = atr(df, length=14) / close
    vol_20 = log_close.diff().rolling(20).std()
    vol_100 = log_close.diff().rolling(100).std()
    out["vol_20"] = vol_20
    out["vol_ratio_20_100"] = vol_20 / vol_100

    # Candle shape
    out["range_pct"] = (high - low) / close
    out["body_pct"] = (close - open_) / close

    # Session/time-of-week effects (gold has distinct London/NY session behavior)
    hours = df.index.hour + df.index.minute / 60
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    out["dow"] = df.index.dayofweek.astype("int8")

    return out, list(out.columns)
