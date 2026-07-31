"""Matched economic gate for the frozen M15 execution challenger."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_filter_gate import TradeMetrics, trade_metrics


@dataclass(frozen=True)
class M15GateConfig:
    n_blocks: int = 5
    min_coverage: float = 0.95
    min_profit_factor: float = 1.10
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class M15GateResult:
    strategy: str
    config: M15GateConfig
    control_trades: int
    replayed_trades: int
    coverage: float
    m30_by_cost: dict[float, TradeMetrics]
    m15_by_cost: dict[float, TradeMetrics]
    block_mean_r_deltas: tuple[float, ...]
    improved_blocks: int
    exit_transitions: dict[str, int]
    passed: bool
    failure_reasons: tuple[str, ...]

    def report_dict(self) -> dict:
        report = {
            "strategy": self.strategy,
            "config": asdict(self.config),
            "control_trades": self.control_trades,
            "replayed_trades": self.replayed_trades,
            "coverage": self.coverage,
            "m30_by_cost": {
                str(cost): asdict(metrics)
                for cost, metrics in self.m30_by_cost.items()
            },
            "m15_by_cost": {
                str(cost): asdict(metrics)
                for cost, metrics in self.m15_by_cost.items()
            },
            "block_mean_r_deltas": list(self.block_mean_r_deltas),
            "improved_blocks": self.improved_blocks,
            "exit_transitions": self.exit_transitions,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }
        return _json_safe(report)


def evaluate_m15_gate(
    m30_ledger: pd.DataFrame,
    m15_ledger: pd.DataFrame,
    *,
    strategy: str,
    cost_pct_per_side: float,
    config: M15GateConfig | None = None,
) -> M15GateResult:
    """Compare identical successfully replayed candidates at M30 and M15."""

    config = config or M15GateConfig()
    control = _indexed_executions(m30_ledger, strategy, "M30")
    replay = _indexed_replay(m15_ledger, strategy)
    if not len(control):
        raise ValueError("M15 gate requires at least one M30 control trade")
    unknown = replay.index.difference(control.index)
    if len(unknown):
        raise ValueError("M15 replay contains signals outside the M30 control")

    successful = replay.loc[replay["status"] == "executed"].copy()
    matched_index = control.index.intersection(successful.index).sort_values()
    matched_m30 = control.loc[matched_index]
    matched_m15 = successful.loc[matched_index]
    coverage = len(matched_index) / len(control)

    m30_by_cost = {
        multiplier: trade_metrics(
            matched_m30,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    m15_by_cost = {
        multiplier: trade_metrics(
            matched_m15,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    block_deltas: list[float] = []
    for positions in np.array_split(np.arange(len(matched_index)), config.n_blocks):
        if not len(positions):
            block_deltas.append(0.0)
            continue
        m30_metrics = trade_metrics(
            matched_m30.iloc[positions],
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        m15_metrics = trade_metrics(
            matched_m15.iloc[positions],
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        block_deltas.append(m15_metrics.mean_r - m30_metrics.mean_r)
    improved_blocks = sum(delta > 0 for delta in block_deltas)

    transitions = (
        matched_m30["exit_reason"].astype(str)
        + "->"
        + matched_m15["exit_reason"].astype(str)
    ).value_counts()
    exit_transitions = {
        str(name): int(count) for name, count in transitions.sort_index().items()
    }

    base_m30 = m30_by_cost[1.0]
    base_m15 = m15_by_cost[1.0]
    stressed_m15 = m15_by_cost[2.0]
    failures: list[str] = []
    if coverage < config.min_coverage:
        failures.append(f"coverage:{coverage:.4f}<{config.min_coverage:.2f}")
    if base_m15.profit_factor < config.min_profit_factor:
        failures.append(
            f"m15_profit_factor:{base_m15.profit_factor:.4f}"
            f"<{config.min_profit_factor:.2f}"
        )
    if base_m15.mean_r <= 0:
        failures.append(f"m15_mean_r_not_positive:{base_m15.mean_r:.4f}")
    if base_m15.profit_factor <= base_m30.profit_factor:
        failures.append(
            f"profit_factor_not_better:{base_m15.profit_factor:.4f}"
            f"<={base_m30.profit_factor:.4f}"
        )
    if base_m15.mean_r <= base_m30.mean_r:
        failures.append(
            f"mean_r_not_better:{base_m15.mean_r:.4f}<={base_m30.mean_r:.4f}"
        )
    if improved_blocks < 4:
        failures.append(f"mean_r_improved_blocks:{improved_blocks}<4")
    if base_m15.max_drawdown <= base_m30.max_drawdown:
        failures.append(
            f"max_drawdown_not_better:{base_m15.max_drawdown:.4f}"
            f"<={base_m30.max_drawdown:.4f}"
        )
    if stressed_m15.profit_factor <= 1.0:
        failures.append(
            f"double_cost_profit_factor:{stressed_m15.profit_factor:.4f}<=1.00"
        )

    return M15GateResult(
        strategy=strategy,
        config=config,
        control_trades=len(control),
        replayed_trades=len(matched_index),
        coverage=coverage,
        m30_by_cost=m30_by_cost,
        m15_by_cost=m15_by_cost,
        block_mean_r_deltas=tuple(block_deltas),
        improved_blocks=improved_blocks,
        exit_transitions=exit_transitions,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _indexed_executions(
    ledger: pd.DataFrame,
    strategy: str,
    label: str,
) -> pd.DataFrame:
    required = {
        "strategy",
        "status",
        "signal_time",
        "gross_return",
        "initial_r_pct",
        "exit_reason",
    }
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f"{label} ledger missing columns: {missing}")
    frame = ledger.loc[ledger["status"] == "executed"].copy()
    if len(frame) and (frame["strategy"] != strategy).any():
        raise ValueError(f"{label} ledger contains the wrong strategy")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame["signal_time"], utc=True),
        name="time",
    )
    if not frame.index.is_unique:
        raise ValueError(f"{label} executed signal times must be unique")
    if len(frame) and (
        not np.isfinite(frame["gross_return"]).all()
        or not np.isfinite(frame["initial_r_pct"]).all()
        or (frame["initial_r_pct"] <= 0).any()
    ):
        raise ValueError(f"{label} execution metrics are invalid")
    return frame.sort_index()


def _indexed_replay(ledger: pd.DataFrame, strategy: str) -> pd.DataFrame:
    required = {"strategy", "status", "signal_time"}
    missing = sorted(required - set(ledger))
    if missing:
        raise ValueError(f"M15 ledger missing columns: {missing}")
    frame = ledger.copy()
    if len(frame) and (frame["strategy"] != strategy).any():
        raise ValueError("M15 ledger contains the wrong strategy")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame["signal_time"], utc=True),
        name="time",
    )
    if not frame.index.is_unique:
        raise ValueError("M15 replay signal times must be unique")
    executed = frame["status"] == "executed"
    if executed.any():
        required_execution = {"gross_return", "initial_r_pct", "exit_reason"}
        missing_execution = sorted(required_execution - set(frame))
        if missing_execution:
            raise ValueError(
                f"M15 executed ledger missing columns: {missing_execution}"
            )
        if (
            not np.isfinite(frame.loc[executed, "gross_return"]).all()
            or not np.isfinite(frame.loc[executed, "initial_r_pct"]).all()
            or (frame.loc[executed, "initial_r_pct"] <= 0).any()
        ):
            raise ValueError("M15 execution metrics are invalid")
    return frame.sort_index()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
