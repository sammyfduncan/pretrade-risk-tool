"""Frozen economic gate for version-2 bare intraday primary rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_filter_gate import TradeMetrics, trade_metrics


@dataclass(frozen=True)
class PrimaryGateConfig:
    n_blocks: int = 5
    min_trades: int = 300
    min_profit_factor: float = 1.10
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class PrimaryGateResult:
    strategy: str
    control_strategy: str
    config: PrimaryGateConfig
    block_metrics: tuple[TradeMetrics, ...]
    metrics_by_cost: dict[float, TradeMetrics]
    control_metrics: TradeMetrics
    profitable_blocks: int
    passed: bool
    failure_reasons: tuple[str, ...]

    def report_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "control_strategy": self.control_strategy,
            "config": asdict(self.config),
            "block_metrics": [asdict(metrics) for metrics in self.block_metrics],
            "metrics_by_cost": {
                str(cost): asdict(metrics)
                for cost, metrics in self.metrics_by_cost.items()
            },
            "control_metrics": asdict(self.control_metrics),
            "profitable_blocks": self.profitable_blocks,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_primary_gate(
    dataset: pd.DataFrame,
    control: pd.DataFrame,
    *,
    strategy: str,
    control_strategy: str,
    cost_pct_per_side: float,
    min_trades: int,
) -> PrimaryGateResult:
    """Evaluate one pre-registered bare V2 rule against its V1 control."""

    frame = _validated(dataset, strategy)
    control_frame = _validated(control, control_strategy)
    config = PrimaryGateConfig(min_trades=min_trades)
    block_metrics = tuple(
        trade_metrics(
            frame.iloc[positions],
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        for positions in np.array_split(np.arange(len(frame)), config.n_blocks)
    )
    metrics_by_cost = {
        multiplier: trade_metrics(
            frame,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    control_metrics = trade_metrics(
        control_frame,
        cost_pct_per_side=cost_pct_per_side,
        cost_multiplier=1.0,
    )
    profitable_blocks = sum(
        metrics.profit_factor > 1.0 for metrics in block_metrics
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
    if base.profit_factor <= control_metrics.profit_factor:
        failures.append(
            f"pf_not_above_v1:{base.profit_factor:.4f}<="
            f"{control_metrics.profit_factor:.4f}"
        )
    if base.max_drawdown <= control_metrics.max_drawdown:
        failures.append(
            f"max_drawdown_not_better:{base.max_drawdown:.4f}<="
            f"{control_metrics.max_drawdown:.4f}"
        )
    return PrimaryGateResult(
        strategy=strategy,
        control_strategy=control_strategy,
        config=config,
        block_metrics=block_metrics,
        metrics_by_cost=metrics_by_cost,
        control_metrics=control_metrics,
        profitable_blocks=profitable_blocks,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _validated(frame: pd.DataFrame, strategy: str) -> pd.DataFrame:
    required = {"strategy", "gross_return", "initial_r_pct"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Primary-gate dataset missing columns: {missing}")
    out = frame.sort_index().copy()
    if not isinstance(out.index, pd.DatetimeIndex) or out.index.tz is None:
        raise ValueError("Primary-gate dataset requires timezone-aware times")
    if not out.index.is_unique:
        raise ValueError("Primary-gate dataset index must be unique")
    if len(out) and (out["strategy"] != strategy).any():
        raise ValueError("Primary-gate dataset contains wrong strategy")
    if not np.isfinite(out["gross_return"]).all():
        raise ValueError("Primary-gate gross_return must be finite")
    if (out["initial_r_pct"] <= 0).any():
        raise ValueError("Primary-gate initial_r_pct must be positive")
    return out
