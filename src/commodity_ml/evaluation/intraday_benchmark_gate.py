"""Frozen economic and time-placebo gate for benchmark strategy B1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_filter_gate import TradeMetrics, trade_metrics


@dataclass(frozen=True)
class BenchmarkGateConfig:
    n_blocks: int = 5
    min_trades: int = 300
    min_profit_factor: float = 1.10
    max_drawdown_floor: float = -0.10
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class BenchmarkGateResult:
    strategy: str
    config: BenchmarkGateConfig
    metrics_by_cost: dict[float, TradeMetrics]
    block_metrics: tuple[TradeMetrics, ...]
    profitable_blocks: int
    placebo_metrics: dict[str, TradeMetrics]
    maximum_placebo_pf: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def report_dict(self) -> dict:
        report = {
            "strategy": self.strategy,
            "config": asdict(self.config),
            "metrics_by_cost": {
                str(cost): asdict(metrics)
                for cost, metrics in self.metrics_by_cost.items()
            },
            "block_metrics": [asdict(metrics) for metrics in self.block_metrics],
            "profitable_blocks": self.profitable_blocks,
            "placebo_metrics": {
                name: asdict(metrics) for name, metrics in self.placebo_metrics.items()
            },
            "maximum_placebo_pf": self.maximum_placebo_pf,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }
        return _json_safe(report)


def evaluate_benchmark_gate(
    ledger: pd.DataFrame,
    placebo_ledgers: dict[str, pd.DataFrame],
    *,
    strategy: str,
    cost_pct_per_side: float,
    config: BenchmarkGateConfig | None = None,
) -> BenchmarkGateResult:
    """Evaluate B1 against its registered non-benchmark clock placebos."""

    config = config or BenchmarkGateConfig()
    frame = _executed(ledger, strategy)
    if not len(frame):
        raise ValueError("benchmark gate requires executed B1 trades")
    metrics_by_cost = {
        multiplier: trade_metrics(
            frame,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    block_metrics = tuple(
        trade_metrics(
            frame.iloc[positions],
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        for positions in np.array_split(np.arange(len(frame)), config.n_blocks)
    )
    profitable_blocks = sum(item.profit_factor > 1.0 for item in block_metrics)
    placebo_metrics = {
        name: trade_metrics(
            _executed(placebo, name),
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        for name, placebo in placebo_ledgers.items()
    }
    maximum_placebo_pf = max(
        (item.profit_factor for item in placebo_metrics.values()),
        default=0.0,
    )
    base = metrics_by_cost[1.0]
    stressed = metrics_by_cost[2.0]
    failures: list[str] = []
    if base.trades < config.min_trades:
        failures.append(f"trades:{base.trades}<{config.min_trades}")
    if base.profit_factor < config.min_profit_factor:
        failures.append(
            f"profit_factor:{base.profit_factor:.4f}<{config.min_profit_factor:.2f}"
        )
    if base.mean_r <= 0:
        failures.append(f"mean_r_not_positive:{base.mean_r:.4f}")
    if profitable_blocks < 4:
        failures.append(f"profitable_blocks:{profitable_blocks}<4")
    if stressed.profit_factor <= 1.0:
        failures.append(
            f"double_cost_profit_factor:{stressed.profit_factor:.4f}<=1.00"
        )
    if base.max_drawdown < config.max_drawdown_floor:
        failures.append(
            f"max_drawdown:{base.max_drawdown:.4f}<{config.max_drawdown_floor:.2f}"
        )
    if base.profit_factor <= maximum_placebo_pf:
        failures.append(
            f"profit_factor_not_above_placebos:{base.profit_factor:.4f}"
            f"<={maximum_placebo_pf:.4f}"
        )
    return BenchmarkGateResult(
        strategy=strategy,
        config=config,
        metrics_by_cost=metrics_by_cost,
        block_metrics=block_metrics,
        profitable_blocks=profitable_blocks,
        placebo_metrics=placebo_metrics,
        maximum_placebo_pf=maximum_placebo_pf,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _executed(ledger: pd.DataFrame, strategy: str) -> pd.DataFrame:
    required = {"strategy", "status", "signal_time", "gross_return", "initial_r_pct"}
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f"{strategy} ledger missing columns: {missing}")
    frame = ledger.loc[ledger["status"] == "executed"].copy()
    if len(frame) and (frame["strategy"] != strategy).any():
        raise ValueError(f"{strategy} ledger contains the wrong strategy")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame["signal_time"], utc=True),
        name="time",
    )
    if not frame.index.is_unique:
        raise ValueError(f"{strategy} signal times must be unique")
    if len(frame) and (
        not np.isfinite(frame["gross_return"]).all()
        or not np.isfinite(frame["initial_r_pct"]).all()
        or (frame["initial_r_pct"] <= 0).any()
    ):
        raise ValueError(f"{strategy} execution metrics are invalid")
    return frame.sort_index()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
