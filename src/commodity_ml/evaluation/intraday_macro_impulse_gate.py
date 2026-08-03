"""Frozen performance and time-placebo gate for P4 macro impulse."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_filter_gate import TradeMetrics, trade_metrics


@dataclass(frozen=True)
class MacroImpulseGateConfig:
    n_blocks: int = 5
    min_coverage: float = 0.90
    min_trades: int = 200
    min_trades_per_block: int = 30
    min_profit_factor: float = 1.10
    min_profitable_blocks: int = 4
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)


@dataclass(frozen=True)
class MacroImpulseGateResult:
    config: MacroImpulseGateConfig
    coverage: float
    metrics_by_cost: dict[float, TradeMetrics]
    block_metrics: tuple[TradeMetrics, ...]
    profitable_blocks: int
    placebo_metrics: dict[str, TradeMetrics]
    maximum_placebo_pf: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def report_dict(self) -> dict:
        return _json_safe(
            {
                "strategy": "P4",
                "config": asdict(self.config),
                "coverage": self.coverage,
                "metrics_by_cost": {
                    str(key): asdict(value)
                    for key, value in self.metrics_by_cost.items()
                },
                "block_metrics": [asdict(value) for value in self.block_metrics],
                "profitable_blocks": self.profitable_blocks,
                "placebo_metrics": {
                    key: asdict(value) for key, value in self.placebo_metrics.items()
                },
                "maximum_placebo_pf": self.maximum_placebo_pf,
                "passed": self.passed,
                "failure_reasons": list(self.failure_reasons),
            }
        )


def evaluate_macro_impulse_gate(
    ledger: pd.DataFrame,
    placebo_ledgers: dict[str, pd.DataFrame],
    *,
    coverage: float,
    cost_pct_per_side: float,
    config: MacroImpulseGateConfig | None = None,
) -> MacroImpulseGateResult:
    """Apply every registered P4 gate without selecting a variant."""

    config = config or MacroImpulseGateConfig()
    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be between zero and one")
    if len(placebo_ledgers) != 20:
        raise ValueError("P4 requires exactly 20 registered time-shift placebos")
    frame = _executed(ledger)
    metrics_by_cost = {
        multiplier: trade_metrics(
            frame,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    blocks = np.array_split(np.arange(len(frame)), config.n_blocks)
    block_metrics = tuple(
        trade_metrics(
            frame.iloc[positions],
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        for positions in blocks
    )
    profitable_blocks = sum(value.profit_factor > 1.0 for value in block_metrics)
    placebo_metrics = {
        name: trade_metrics(
            _executed(placebo),
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        for name, placebo in placebo_ledgers.items()
    }
    maximum_placebo_pf = max(
        (value.profit_factor for value in placebo_metrics.values()),
        default=0.0,
    )

    base = metrics_by_cost[1.0]
    stressed = metrics_by_cost[2.0]
    failures: list[str] = []
    if coverage < config.min_coverage:
        failures.append(f"coverage:{coverage:.4f}<{config.min_coverage:.2f}")
    if base.trades < config.min_trades:
        failures.append(f"trades:{base.trades}<{config.min_trades}")
    small_blocks = [value.trades for value in block_metrics if value.trades < config.min_trades_per_block]
    if small_blocks:
        failures.append(
            f"block_trades_below:{config.min_trades_per_block}:{small_blocks}"
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
            f"double_cost_profit_factor:{stressed.profit_factor:.4f}<=1.00"
        )
    if base.profit_factor <= maximum_placebo_pf:
        failures.append(
            f"profit_factor_not_above_placebos:{base.profit_factor:.4f}"
            f"<={maximum_placebo_pf:.4f}"
        )
    return MacroImpulseGateResult(
        config=config,
        coverage=coverage,
        metrics_by_cost=metrics_by_cost,
        block_metrics=block_metrics,
        profitable_blocks=profitable_blocks,
        placebo_metrics=placebo_metrics,
        maximum_placebo_pf=maximum_placebo_pf,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def registered_shift_offsets(row_count: int) -> tuple[int, ...]:
    """Return 20 deterministic, equally spaced, non-zero circular offsets."""

    if row_count < 21:
        raise ValueError("At least 21 driver-state rows are required")
    offsets = tuple(int(np.floor(step * row_count / 21)) for step in range(1, 21))
    if len(set(offsets)) != 20 or any(value <= 0 or value >= row_count for value in offsets):
        raise ValueError("Driver-state history is too short for unique placebo shifts")
    return offsets


def circular_shift_joint_state(state: pd.DataFrame, offset: int) -> pd.DataFrame:
    """Rotate all joint driver columns together while retaining decision clocks."""

    if not 0 < offset < len(state):
        raise ValueError("offset must be strictly inside the state length")
    shifted = pd.concat([state.iloc[-offset:], state.iloc[:-offset]]).copy()
    shifted.index = state.index
    return shifted


def _executed(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"status", "signal_time"}
    if missing := sorted(required - set(ledger)):
        raise ValueError(f"P4 ledger missing columns: {missing}")
    frame = ledger.loc[ledger["status"] == "executed"].copy()
    if frame.empty:
        return pd.DataFrame(
            {
                "gross_return": pd.Series(dtype=float),
                "initial_r_pct": pd.Series(dtype=float),
            },
            index=pd.DatetimeIndex([], tz="UTC", name="time"),
        )
    if missing := sorted({"gross_return", "initial_r_pct"} - set(frame)):
        raise ValueError(f"P4 executed ledger missing columns: {missing}")
    if "strategy" in frame and len(frame) and (frame["strategy"] != "P4").any():
        raise ValueError("P4 ledger contains another strategy")
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame["signal_time"], utc=True),
        name="time",
    )
    if not frame.index.is_unique:
        raise ValueError("P4 signal times must be unique")
    if len(frame) and (
        not np.isfinite(frame["gross_return"].astype(float)).all()
        or not np.isfinite(frame["initial_r_pct"].astype(float)).all()
        or (frame["initial_r_pct"].astype(float) <= 0).any()
    ):
        raise ValueError("P4 execution metrics are invalid")
    return frame.sort_index()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    return value
