"""Causal GC trade-flow features from Databento TBBO records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from commodity_ml.intraday.futures_state import (
    LONDON,
    completed_session_selections,
)

ORDER_FLOW_FEATURES = (
    "gc_side_aggressive_imbalance_5m",
    "gc_side_aggressive_imbalance_30m",
    "gc_side_depth_imbalance_5m",
    "gc_spread_ticks_5m",
    "gc_side_sweep_imbalance_5m",
    "gc_trade_intensity_surprise_5m",
)


@dataclass(frozen=True)
class OrderFlowFeatureResult:
    features: pd.DataFrame
    eligible: pd.Series
    report: dict


def build_order_flow_candidate_features(
    spot_m30: pd.DataFrame,
    candidates: pd.DataFrame,
    tbbo: pd.DataFrame,
    ohlcv: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    degraded_dates: tuple[str, ...] | list[str] = (),
    tick_size: float = 0.1,
) -> OrderFlowFeatureResult:
    """Attach the frozen OF-GC family at each candidate's decision time."""

    if tick_size <= 0 or not np.isfinite(tick_size):
        raise ValueError("tick_size must be finite and positive")
    for frame, label in ((spot_m30, "spot M30"), (candidates, "candidates")):
        _require_utc_index(frame, label)
    if "decision_time" not in spot_m30 or "is_long" not in candidates:
        raise ValueError("Order-flow construction requires decision_time and is_long")
    if len(candidates.index.difference(spot_m30.index)):
        raise ValueError("Candidate timestamps are missing from spot M30")
    market = build_active_order_flow(tbbo, ohlcv, definitions, tick_size=tick_size)
    decisions = pd.DatetimeIndex(
        pd.to_datetime(spot_m30.loc[candidates.index, "decision_time"], utc=True)
    )
    rows = market.reindex(decisions)
    rows.index = candidates.index
    side = np.where(candidates["is_long"].astype(int).to_numpy() == 1, 1.0, -1.0)
    features = pd.DataFrame(index=candidates.index)
    features["gc_side_aggressive_imbalance_5m"] = side * rows[
        "gc_aggressive_imbalance_5m"
    ].to_numpy()
    features["gc_side_aggressive_imbalance_30m"] = side * rows[
        "gc_aggressive_imbalance_30m"
    ].to_numpy()
    features["gc_side_depth_imbalance_5m"] = side * rows[
        "gc_depth_imbalance_5m"
    ].to_numpy()
    features["gc_spread_ticks_5m"] = rows["gc_spread_ticks_5m"].to_numpy()
    features["gc_side_sweep_imbalance_5m"] = side * rows[
        "gc_sweep_imbalance_5m"
    ].to_numpy()
    features["gc_trade_intensity_surprise_5m"] = rows[
        "gc_trade_intensity_surprise_5m"
    ].to_numpy()
    finite = pd.Series(
        np.isfinite(features.to_numpy(dtype=float)).all(axis=1),
        index=features.index,
    )
    quality = _quality_mask(decisions, degraded_dates)
    quality.index = features.index
    eligible = finite & quality
    return OrderFlowFeatureResult(
        features=features,
        eligible=eligible,
        report={
            "candidate_rows": len(candidates),
            "eligible_rows": int(eligible.sum()),
            "coverage": float(eligible.mean()) if len(eligible) else 0.0,
            "incomplete_rows": int((~finite).sum()),
            "quality_excluded_rows": int((finite & ~quality).sum()),
            "degraded_source_dates": list(degraded_dates),
            "feature_names": list(ORDER_FLOW_FEATURES),
            "active_flow_minutes": len(market),
        },
    )


def build_active_order_flow(
    tbbo: pd.DataFrame,
    ohlcv: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    tick_size: float = 0.1,
) -> pd.DataFrame:
    """Aggregate active-contract TBBO records into causal minute windows."""

    _require_utc_index(tbbo, "TBBO")
    required = {
        "raw_symbol",
        "side",
        "size",
        "bid_px_00",
        "ask_px_00",
        "bid_sz_00",
        "ask_sz_00",
    }
    if missing := sorted(required - set(tbbo)):
        raise ValueError(f"Normalized TBBO missing columns: {missing}")
    selections = completed_session_selections(ohlcv, definitions)
    left = tbbo.reset_index().rename(columns={tbbo.index.name: "receive_time"})
    mapped = pd.merge_asof(
        left.sort_values("receive_time"),
        selections.loc[:, ["available_time", "active_symbol"]].sort_values(
            "available_time"
        ),
        left_on="receive_time",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    mapped = mapped.loc[mapped["raw_symbol"] == mapped["active_symbol"]].copy()
    if mapped.empty:
        raise ValueError("No TBBO rows map to the causal active GC contract")
    mapped["side_sign"] = mapped["side"].map({"B": 1.0, "A": -1.0, "N": 0.0})
    if mapped["side_sign"].isna().any():
        raise ValueError("TBBO contains an unsupported aggressor side")
    size = mapped["size"].astype(float)
    total_depth = mapped["bid_sz_00"] + mapped["ask_sz_00"]
    mapped["signed_volume"] = mapped["side_sign"] * size
    mapped["depth_weighted"] = (
        (mapped["bid_sz_00"] - mapped["ask_sz_00"]) / total_depth
    ) * size
    mapped["spread_weighted"] = (
        (mapped["ask_px_00"] - mapped["bid_px_00"]) / tick_size
    ) * size
    opposing_depth = np.where(
        mapped["side"] == "B",
        mapped["ask_sz_00"],
        np.where(mapped["side"] == "A", mapped["bid_sz_00"], np.inf),
    )
    mapped["signed_sweep_volume"] = np.where(
        size >= opposing_depth,
        mapped["signed_volume"],
        0.0,
    )
    mapped["minute"] = mapped["receive_time"].dt.floor("min")
    minute = mapped.groupby("minute", sort=True).agg(
        raw_symbol=("raw_symbol", "first"),
        signed_volume=("signed_volume", "sum"),
        total_volume=("size", "sum"),
        depth_weighted=("depth_weighted", "sum"),
        spread_weighted=("spread_weighted", "sum"),
        signed_sweep_volume=("signed_sweep_volume", "sum"),
        trade_count=("size", "size"),
    )
    if (mapped.groupby("minute")["raw_symbol"].nunique() > 1).any():
        raise ValueError("Active contract changes inside a TBBO minute")

    grid = pd.date_range(minute.index.min(), minute.index.max(), freq="1min", tz="UTC")
    flow = minute.reindex(grid)
    flow.index.name = "minute"
    selection_grid = pd.merge_asof(
        pd.DataFrame({"minute": grid}),
        selections.loc[:, ["available_time", "active_symbol"]].sort_values(
            "available_time"
        ),
        left_on="minute",
        right_on="available_time",
        direction="backward",
        allow_exact_matches=True,
    )
    flow["raw_symbol"] = selection_grid["active_symbol"].to_numpy()
    additive = [
        "signed_volume",
        "total_volume",
        "depth_weighted",
        "spread_weighted",
        "signed_sweep_volume",
        "trade_count",
    ]
    flow.loc[:, additive] = flow.loc[:, additive].fillna(0.0)
    same_5 = flow["raw_symbol"].eq(flow["raw_symbol"].shift(4))
    same_30 = flow["raw_symbol"].eq(flow["raw_symbol"].shift(29))
    volume_5 = flow["total_volume"].rolling(5, min_periods=5).sum()
    volume_30 = flow["total_volume"].rolling(30, min_periods=30).sum()
    result = pd.DataFrame(index=flow.index + pd.Timedelta(minutes=1))
    result.index.name = "decision_time"
    result["gc_aggressive_imbalance_5m"] = (
        flow["signed_volume"].rolling(5, min_periods=5).sum() / volume_5
    ).where(same_5).to_numpy()
    result["gc_aggressive_imbalance_30m"] = (
        flow["signed_volume"].rolling(30, min_periods=30).sum() / volume_30
    ).where(same_30).to_numpy()
    result["gc_depth_imbalance_5m"] = (
        flow["depth_weighted"].rolling(5, min_periods=5).sum() / volume_5
    ).where(same_5).to_numpy()
    result["gc_spread_ticks_5m"] = (
        flow["spread_weighted"].rolling(5, min_periods=5).sum() / volume_5
    ).where(same_5).to_numpy()
    result["gc_sweep_imbalance_5m"] = (
        flow["signed_sweep_volume"].rolling(5, min_periods=5).sum() / volume_5
    ).where(same_5).to_numpy()
    count_5 = flow["trade_count"].rolling(5, min_periods=5).sum().where(same_5)
    london = result.index.tz_convert(LONDON)
    slot = pd.Series(london.hour * 60 + london.minute, index=result.index)
    count_series = pd.Series(count_5.to_numpy(), index=result.index)
    log_count = np.log1p(count_series).where(count_series > 0)
    prior = log_count.groupby(slot).transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).median()
    )
    result["gc_trade_intensity_surprise_5m"] = log_count - prior
    return result


def _quality_mask(
    index: pd.DatetimeIndex,
    degraded_dates: tuple[str, ...] | list[str],
) -> pd.Series:
    dates = index.normalize()
    valid = np.ones(len(index), dtype=bool)
    for value in degraded_dates:
        start = pd.Timestamp(value, tz="UTC")
        valid &= ~((dates >= start) & (dates <= start + pd.Timedelta(days=28)))
    return pd.Series(valid, index=index)


def _require_utc_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{label} requires a timezone-aware index")
    if str(frame.index.tz) != "UTC":
        raise ValueError(f"{label} index must be UTC")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} index must be sorted")
