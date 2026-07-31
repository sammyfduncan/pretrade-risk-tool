"""Observed quote-cost measurement for cached IG bid/ask candles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from commodity_ml.data.schema import QUOTE_COLUMNS
from commodity_ml.evaluation.trading_metrics import profit_factor


@dataclass(frozen=True)
class CostStats:
    observations: int
    mean_bp: float
    median_bp: float
    p75_bp: float
    p90_bp: float
    p95_bp: float
    p99_bp: float
    maximum_bp: float


@dataclass(frozen=True)
class CandidateCostProfile:
    strategy: str
    candidate_trades: int
    matched_trades: int
    coverage: float
    median_cost_bp: float
    p95_cost_bp: float
    median_cost_r: float
    p95_cost_r: float
    gross_break_even_cost_bp: float
    estimated_mean_net_bp: float
    estimated_profit_factor: float


@dataclass(frozen=True)
class QuoteCostReport:
    start: pd.Timestamp
    end: pd.Timestamp
    slippage_bp_per_side: float
    overall: CostStats
    by_london_slot: dict[str, CostStats]
    by_london_weekday: dict[str, CostStats]
    representative_stop_pct: float | None
    median_cost_r: float | None
    p95_cost_r: float | None
    candidate: CandidateCostProfile | None = None

    def report_dict(self) -> dict:
        return _json_safe({
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "slippage_bp_per_side": self.slippage_bp_per_side,
            "overall": asdict(self.overall),
            "by_london_slot": {
                key: asdict(value) for key, value in self.by_london_slot.items()
            },
            "by_london_weekday": {
                key: asdict(value) for key, value in self.by_london_weekday.items()
            },
            "representative_stop_pct": self.representative_stop_pct,
            "median_cost_r": self.median_cost_r,
            "p95_cost_r": self.p95_cost_r,
            "candidate": asdict(self.candidate) if self.candidate else None,
        })


def quote_round_trip_costs(
    candles: pd.DataFrame,
    *,
    slippage_pct_per_side: float = 0.0,
) -> pd.Series:
    """Return a conservative synchronized spread-plus-slippage proxy.

    A round trip crossing an unchanged spread loses one full bid/ask spread.
    Historical OHLC quote extrema are not synchronized, so the proxy uses the
    worse of each bar's synchronized open and close spreads and adds the
    explicit per-side slippage allowance.
    """

    _validate_quotes(candles)
    if not np.isfinite(slippage_pct_per_side) or slippage_pct_per_side < 0:
        raise ValueError("slippage_pct_per_side must be finite and non-negative")
    endpoint_spreads = pd.DataFrame(index=candles.index)
    for endpoint in ("open", "close"):
        bid = candles[f"bid_{endpoint}"].astype(float)
        ask = candles[f"ask_{endpoint}"].astype(float)
        mid = (ask + bid) / 2
        endpoint_spreads[endpoint] = (ask - bid) / mid
    spread = endpoint_spreads.max(axis=1)
    if not np.isfinite(spread.to_numpy()).all():
        raise RuntimeError("non_finite_spread_history")
    if (spread < 0).any():
        raise RuntimeError("crossed_spread_history")
    result = spread + 2 * slippage_pct_per_side
    result.name = "round_trip_cost_fraction"
    return result


def measure_quote_costs(
    candles: pd.DataFrame,
    *,
    slippage_pct_per_side: float = 0.0,
    representative_stop_pct: float | None = None,
    candidate_ledger: pd.DataFrame | None = None,
    session_timezone: str = "Europe/London",
) -> QuoteCostReport:
    """Measure aggregate, time-of-day, and optional candidate-entry costs."""

    costs = quote_round_trip_costs(
        candles,
        slippage_pct_per_side=slippage_pct_per_side,
    )
    if representative_stop_pct is not None and (
        not np.isfinite(representative_stop_pct) or representative_stop_pct <= 0
    ):
        raise ValueError("representative_stop_pct must be finite and positive")
    local = costs.index.tz_convert(session_timezone)
    slots = pd.Series(local.strftime("%H:%M"), index=costs.index)
    weekdays = pd.Series(local.day_name(), index=costs.index)
    by_slot = {
        str(slot): _cost_stats(costs.loc[slots == slot])
        for slot in sorted(slots.unique())
    }
    weekday_order = [
        day
        for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
        if day in set(weekdays)
    ]
    by_weekday = {
        day: _cost_stats(costs.loc[weekdays == day]) for day in weekday_order
    }
    candidate = (
        measure_candidate_costs(
            candles,
            candidate_ledger,
            slippage_pct_per_side=slippage_pct_per_side,
        )
        if candidate_ledger is not None
        else None
    )
    return QuoteCostReport(
        start=costs.index.min(),
        end=costs.index.max(),
        slippage_bp_per_side=slippage_pct_per_side * 10_000,
        overall=_cost_stats(costs),
        by_london_slot=by_slot,
        by_london_weekday=by_weekday,
        representative_stop_pct=representative_stop_pct,
        median_cost_r=(
            float(costs.median() / representative_stop_pct)
            if representative_stop_pct is not None
            else None
        ),
        p95_cost_r=(
            float(costs.quantile(0.95) / representative_stop_pct)
            if representative_stop_pct is not None
            else None
        ),
        candidate=candidate,
    )


def measure_candidate_costs(
    candles: pd.DataFrame,
    ledger: pd.DataFrame,
    *,
    slippage_pct_per_side: float = 0.0,
) -> CandidateCostProfile:
    """Estimate matched strategy costs using the quote bar at each entry.

    The quote-bar cost is a round-trip proxy, not realized slippage. Actual
    entry/exit fills must later replace it in the paper execution ledger.
    """

    required = {
        "strategy",
        "status",
        "entry_time",
        "gross_return",
        "initial_r_pct",
    }
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f"candidate ledger missing columns: {missing}")
    executed = ledger.loc[ledger["status"] == "executed"].copy()
    if executed.empty:
        raise ValueError("candidate ledger has no executed trades")
    strategies = executed["strategy"].dropna().unique()
    if len(strategies) != 1:
        raise ValueError("candidate ledger must contain exactly one strategy")
    entry_time = pd.DatetimeIndex(pd.to_datetime(executed["entry_time"], utc=True))
    if entry_time.has_duplicates:
        raise ValueError("candidate entry times must be unique")
    costs = quote_round_trip_costs(
        candles,
        slippage_pct_per_side=slippage_pct_per_side,
    )
    matched = entry_time.intersection(costs.index)
    if not len(matched):
        raise ValueError("candidate ledger has no entry timestamps in quote history")

    executed.index = entry_time
    trades = executed.loc[matched].sort_index()
    matched_costs = costs.loc[trades.index]
    stop_risk = trades["initial_r_pct"].astype(float)
    if (
        not np.isfinite(stop_risk).all()
        or (stop_risk <= 0).any()
        or not np.isfinite(trades["gross_return"]).all()
    ):
        raise ValueError("candidate ledger contains invalid return/risk values")
    cost_r = matched_costs / stop_risk
    estimated_net = trades["gross_return"].astype(float) - matched_costs
    return CandidateCostProfile(
        strategy=str(strategies[0]),
        candidate_trades=len(executed),
        matched_trades=len(trades),
        coverage=len(trades) / len(executed),
        median_cost_bp=float(matched_costs.median() * 10_000),
        p95_cost_bp=float(matched_costs.quantile(0.95) * 10_000),
        median_cost_r=float(cost_r.median()),
        p95_cost_r=float(cost_r.quantile(0.95)),
        gross_break_even_cost_bp=float(trades["gross_return"].mean() * 10_000),
        estimated_mean_net_bp=float(estimated_net.mean() * 10_000),
        estimated_profit_factor=float(profit_factor(estimated_net)),
    )


def _cost_stats(costs: pd.Series) -> CostStats:
    if costs.empty:
        raise ValueError("cost statistics require at least one observation")
    basis_points = costs.astype(float) * 10_000
    return CostStats(
        observations=len(basis_points),
        mean_bp=float(basis_points.mean()),
        median_bp=float(basis_points.median()),
        p75_bp=float(basis_points.quantile(0.75)),
        p90_bp=float(basis_points.quantile(0.90)),
        p95_bp=float(basis_points.quantile(0.95)),
        p99_bp=float(basis_points.quantile(0.99)),
        maximum_bp=float(basis_points.max()),
    )


def _validate_quotes(candles: pd.DataFrame) -> None:
    missing = [column for column in QUOTE_COLUMNS if column not in candles.columns]
    if missing:
        raise RuntimeError(f"quote_history_incomplete:{missing}")
    if candles.empty:
        raise RuntimeError("empty_quote_history")
    if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.tz is None:
        raise ValueError("quote history requires a timezone-aware DatetimeIndex")
    if not candles.index.is_monotonic_increasing or not candles.index.is_unique:
        raise ValueError("quote history must be sorted and unique")


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
