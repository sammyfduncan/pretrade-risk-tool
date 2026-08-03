"""Frozen performance and placebo gate for RV1 gold–silver relative value."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from commodity_ml.evaluation.trading_metrics import max_drawdown, profit_factor


@dataclass(frozen=True)
class RelativeValueMetrics:
    trades: int
    win_rate: float
    profit_factor: float
    mean_r: float
    total_return: float
    max_drawdown: float


@dataclass(frozen=True)
class RelativeValueGateConfig:
    n_blocks: int = 5
    min_coverage: float = 0.95
    min_trades: int = 200
    min_trades_per_block: int = 30
    min_side_fraction: float = 0.30
    min_profit_factor: float = 1.15
    min_profitable_blocks: int = 4
    max_drawdown_floor: float = -0.10


@dataclass(frozen=True)
class RelativeValueGateResult:
    config: RelativeValueGateConfig
    coverage: float
    metrics_by_stress: dict[str, RelativeValueMetrics]
    block_metrics: tuple[RelativeValueMetrics, ...]
    profitable_blocks: int
    side_fractions: dict[str, float]
    placebo_metrics: dict[str, RelativeValueMetrics]
    maximum_placebo_pf: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def report_dict(self) -> dict:
        return _json_safe(
            {
                "strategy": "RV1",
                "config": asdict(self.config),
                "coverage": self.coverage,
                "metrics_by_stress": {
                    key: asdict(value)
                    for key, value in self.metrics_by_stress.items()
                },
                "block_metrics": [asdict(value) for value in self.block_metrics],
                "profitable_blocks": self.profitable_blocks,
                "side_fractions": self.side_fractions,
                "placebo_metrics": {
                    key: asdict(value) for key, value in self.placebo_metrics.items()
                },
                "maximum_placebo_pf": self.maximum_placebo_pf,
                "passed": self.passed,
                "failure_reasons": list(self.failure_reasons),
            }
        )


def evaluate_relative_value_gate(
    ledger: pd.DataFrame,
    placebo_ledgers: dict[str, pd.DataFrame],
    *,
    coverage: float,
    config: RelativeValueGateConfig | None = None,
) -> RelativeValueGateResult:
    config = config or RelativeValueGateConfig()
    if not 0 <= coverage <= 1:
        raise ValueError("RV1 coverage must be between zero and one")
    if len(placebo_ledgers) != 20:
        raise ValueError("RV1 requires exactly 20 registered placebos")
    frame = _executed(ledger)
    metrics_by_stress = {
        "1.0": _metrics(frame, "net_return_1_0"),
        "1.5": _metrics(frame, "net_return_1_5"),
        "2.0": _metrics(frame, "net_return_2_0"),
    }
    blocks = np.array_split(np.arange(len(frame)), config.n_blocks)
    block_metrics = tuple(_metrics(frame.iloc[positions], "net_return_1_0") for positions in blocks)
    profitable_blocks = sum(item.profit_factor > 1.0 for item in block_metrics)
    side_fractions = {
        side: float((frame["side"] == side).mean()) if len(frame) else 0.0
        for side in ("long_residual", "short_residual")
    }
    placebo_metrics = {
        name: _metrics(_executed(placebo), "net_return_1_0")
        for name, placebo in placebo_ledgers.items()
    }
    maximum_placebo_pf = max(
        (item.profit_factor for item in placebo_metrics.values()),
        default=0.0,
    )

    base = metrics_by_stress["1.0"]
    stressed = metrics_by_stress["2.0"]
    failures: list[str] = []
    if coverage < config.min_coverage:
        failures.append(f"coverage:{coverage:.4f}<{config.min_coverage:.2f}")
    if base.trades < config.min_trades:
        failures.append(f"trades:{base.trades}<{config.min_trades}")
    small_blocks = [item.trades for item in block_metrics if item.trades < config.min_trades_per_block]
    if small_blocks:
        failures.append(f"block_trades_below:{config.min_trades_per_block}:{small_blocks}")
    for side, fraction in side_fractions.items():
        if fraction < config.min_side_fraction:
            failures.append(
                f"side_fraction:{side}:{fraction:.4f}<{config.min_side_fraction:.2f}"
            )
    if base.profit_factor < config.min_profit_factor:
        failures.append(
            f"profit_factor:{base.profit_factor:.4f}<{config.min_profit_factor:.2f}"
        )
    if base.mean_r <= 0:
        failures.append(f"mean_r_not_positive:{base.mean_r:.4f}")
    if profitable_blocks < config.min_profitable_blocks:
        failures.append(
            f"profitable_blocks:{profitable_blocks}<{config.min_profitable_blocks}"
        )
    if stressed.profit_factor <= 1.0:
        failures.append(
            f"double_stress_profit_factor:{stressed.profit_factor:.4f}<=1.00"
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
    return RelativeValueGateResult(
        config=config,
        coverage=coverage,
        metrics_by_stress=metrics_by_stress,
        block_metrics=block_metrics,
        profitable_blocks=profitable_blocks,
        side_fractions=side_fractions,
        placebo_metrics=placebo_metrics,
        maximum_placebo_pf=maximum_placebo_pf,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _executed(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"status", "signal_time"}
    if missing := sorted(required - set(ledger)):
        raise ValueError(f"RV1 ledger missing columns: {missing}")
    frame = ledger.loc[ledger["status"] == "executed"].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "side",
                "initial_r",
                "net_return_1_0",
                "net_return_1_5",
                "net_return_2_0",
            ],
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
        )
    needed = {
        "side",
        "initial_r",
        "net_return_1_0",
        "net_return_1_5",
        "net_return_2_0",
    }
    if missing := sorted(needed - set(frame)):
        raise ValueError(f"RV1 executed ledger missing columns: {missing}")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame["signal_time"], utc=True),
        name="time",
    )
    if not frame.index.is_unique:
        raise ValueError("RV1 signal times must be unique")
    values = frame.loc[:, ["initial_r", "net_return_1_0", "net_return_1_5", "net_return_2_0"]]
    if not np.isfinite(values.to_numpy(dtype=float)).all() or (frame["initial_r"] <= 0).any():
        raise ValueError("RV1 execution metrics are invalid")
    return frame.sort_index()


def _metrics(frame: pd.DataFrame, return_column: str) -> RelativeValueMetrics:
    if frame.empty:
        return RelativeValueMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    returns = frame[return_column].astype(float)
    return RelativeValueMetrics(
        trades=len(frame),
        win_rate=float((returns > 0).mean()),
        profit_factor=profit_factor(returns),
        mean_r=float((returns / frame["initial_r"].astype(float)).mean()),
        total_return=float((1 + returns).prod() - 1),
        max_drawdown=max_drawdown(returns),
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
