"""Matched economic evaluation for the frozen volatility-risk ladder."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from commodity_ml.evaluation.trading_metrics import max_drawdown, profit_factor
from commodity_ml.intraday.risk import assess_volatility_risk

LONDON = "Europe/London"
M30 = pd.Timedelta(minutes=30)


@dataclass(frozen=True)
class RiskPolicyGateConfig:
    risk_per_trade: float = 0.0025
    minimum_history: int = 500
    minimum_coverage: float = 0.90
    minimum_opportunities: int = 200
    n_blocks: int = 5
    minimum_block_opportunities: int = 30
    minimum_return_retention: float = 0.75
    minimum_drawdown_reduction: float = 0.10


@dataclass(frozen=True)
class RiskPolicyMetrics:
    opportunities: int
    compounded_return: float
    profit_factor: float
    mean_contribution: float
    daily_sharpe: float
    max_drawdown: float
    worst_contribution: float


@dataclass(frozen=True)
class RiskPolicyBlock:
    block: int
    start: pd.Timestamp
    end: pd.Timestamp
    constant: RiskPolicyMetrics
    policy: RiskPolicyMetrics


@dataclass(frozen=True)
class RiskPolicyGateResult:
    config: RiskPolicyGateConfig
    eligible_after_warmup: int
    matched_opportunities: int
    match_coverage: float
    multiplier_counts: dict[str, int]
    constant: RiskPolicyMetrics
    policy: RiskPolicyMetrics
    constant_double_cost: RiskPolicyMetrics
    policy_double_cost: RiskPolicyMetrics
    blocks: tuple[RiskPolicyBlock, ...]
    profitable_policy_blocks: int
    passed: bool
    failure_reasons: tuple[str, ...]
    decisions: pd.DataFrame

    def report_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "eligible_after_warmup": self.eligible_after_warmup,
            "matched_opportunities": self.matched_opportunities,
            "match_coverage": self.match_coverage,
            "multiplier_counts": self.multiplier_counts,
            "constant": asdict(self.constant),
            "policy": asdict(self.policy),
            "constant_double_cost": asdict(self.constant_double_cost),
            "policy_double_cost": asdict(self.policy_double_cost),
            "blocks": [
                {
                    "block": block.block,
                    "start": block.start.isoformat(),
                    "end": block.end.isoformat(),
                    "constant": asdict(block.constant),
                    "policy": asdict(block.policy),
                }
                for block in self.blocks
            ],
            "profitable_policy_blocks": self.profitable_policy_blocks,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_risk_policy_gate(
    candidates: pd.DataFrame,
    oof_forecasts: pd.Series,
    *,
    cost_pct_per_side: float,
    config: RiskPolicyGateConfig | None = None,
) -> RiskPolicyGateResult:
    """Compare constant and volatility-scaled risk on one matched opportunity set."""

    config = config or RiskPolicyGateConfig()
    frame = _validate_candidates(candidates)
    forecasts = _validate_forecasts(oof_forecasts)
    if not np.isfinite(cost_pct_per_side) or cost_pct_per_side < 0:
        raise ValueError("cost_pct_per_side must be finite and non-negative")
    if len(forecasts) < config.minimum_history:
        raise ValueError("OOF forecast history is shorter than the frozen warm-up")

    decision_times = frame.index + M30
    warmup_end = forecasts.index[config.minimum_history - 1]
    after_warmup = decision_times > warmup_end
    eligible = frame.loc[after_warmup].copy()
    eligible["decision_time"] = decision_times[after_warmup]
    exact = eligible["decision_time"].isin(forecasts.index)
    matched = eligible.loc[exact].copy()
    coverage = len(matched) / len(eligible) if len(eligible) else 0.0

    rows: list[dict] = []
    for signal_time, row in matched.iterrows():
        decision_time = pd.Timestamp(row["decision_time"])
        position = forecasts.index.searchsorted(decision_time, side="left")
        history = forecasts.iloc[:position]
        forecast = float(forecasts.loc[decision_time])
        decision = assess_volatility_risk(
            forecast=forecast,
            decision_time=decision_time,
            forecast_history=history,
            artifact_approved=True,
            minimum_history=config.minimum_history,
        )
        if decision.threshold_80 is None or decision.threshold_95 is None:
            raise RuntimeError(f"Matched policy decision failed closed: {decision.reason}")
        rows.append(
            {
                "signal_time": signal_time,
                "decision_time": decision_time,
                "gross_return": float(row["gross_return"]),
                "initial_r_pct": float(row["initial_r_pct"]),
                "forecast": forecast,
                "threshold_80": decision.threshold_80,
                "threshold_95": decision.threshold_95,
                "reference_count": decision.reference_count,
                "multiplier": decision.multiplier,
                "reason": decision.reason,
            }
        )
    decisions = pd.DataFrame(rows)
    if decisions.empty:
        decisions = pd.DataFrame(
            columns=(
                "decision_time",
                "gross_return",
                "initial_r_pct",
                "forecast",
                "threshold_80",
                "threshold_95",
                "reference_count",
                "multiplier",
                "reason",
            ),
            index=pd.DatetimeIndex([], tz="UTC", name="signal_time"),
        )
    else:
        decisions = decisions.set_index("signal_time").sort_index()
        decisions.index = pd.DatetimeIndex(decisions.index, name="signal_time")

    base_contributions = _contributions(
        decisions,
        cost_pct_per_side=cost_pct_per_side,
        cost_multiplier=1.0,
        risk_per_trade=config.risk_per_trade,
    )
    stressed_contributions = _contributions(
        decisions,
        cost_pct_per_side=cost_pct_per_side,
        cost_multiplier=2.0,
        risk_per_trade=config.risk_per_trade,
    )
    constant = _metrics(base_contributions["constant"])
    policy = _metrics(base_contributions["policy"])
    constant_stress = _metrics(stressed_contributions["constant"])
    policy_stress = _metrics(stressed_contributions["policy"])

    blocks: list[RiskPolicyBlock] = []
    if len(decisions):
        for block, positions in enumerate(
            np.array_split(np.arange(len(decisions)), config.n_blocks)
        ):
            block_contributions = base_contributions.iloc[positions]
            blocks.append(
                RiskPolicyBlock(
                    block=block,
                    start=decisions.index[positions[0]],
                    end=decisions.index[positions[-1]],
                    constant=_metrics(block_contributions["constant"]),
                    policy=_metrics(block_contributions["policy"]),
                )
            )
    profitable_blocks = sum(block.policy.compounded_return > 0 for block in blocks)
    counts = {
        key: int((decisions["multiplier"] == value).sum()) if len(decisions) else 0
        for key, value in (("1.0", 1.0), ("0.5", 0.5), ("0.0", 0.0))
    }
    failures = _failures(
        config,
        eligible=len(eligible),
        coverage=coverage,
        blocks=blocks,
        constant=constant,
        policy=policy,
        policy_stress=policy_stress,
        profitable_blocks=profitable_blocks,
    )
    decisions = decisions.join(
        base_contributions.rename(
            columns={
                "constant": "constant_contribution",
                "policy": "policy_contribution",
            }
        )
    )
    return RiskPolicyGateResult(
        config=config,
        eligible_after_warmup=len(eligible),
        matched_opportunities=len(decisions),
        match_coverage=coverage,
        multiplier_counts=counts,
        constant=constant,
        policy=policy,
        constant_double_cost=constant_stress,
        policy_double_cost=policy_stress,
        blocks=tuple(blocks),
        profitable_policy_blocks=profitable_blocks,
        passed=not failures,
        failure_reasons=tuple(failures),
        decisions=decisions,
    )


def _validate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"gross_return", "initial_r_pct"}
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Risk-policy candidates missing columns: {missing}")
    out = frame.sort_index().copy()
    if not isinstance(out.index, pd.DatetimeIndex) or out.index.tz is None:
        raise ValueError("Risk-policy candidates require timezone-aware signal times")
    out.index = out.index.tz_convert("UTC")
    if not out.index.is_unique:
        raise ValueError("Risk-policy candidate signal times must be unique")
    values = out.loc[:, ["gross_return", "initial_r_pct"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (out["initial_r_pct"] <= 0).any():
        raise ValueError("Risk-policy candidate returns and initial R must be valid")
    return out


def _validate_forecasts(series: pd.Series) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise ValueError("OOF forecasts must be a pandas Series")
    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
        raise ValueError("OOF forecasts require timezone-aware decision times")
    out = series.astype(float).sort_index().copy()
    out.index = out.index.tz_convert("UTC")
    if not out.index.is_unique or not np.isfinite(out.to_numpy()).all():
        raise ValueError("OOF forecasts must be unique and finite")
    return out


def _contributions(
    decisions: pd.DataFrame,
    *,
    cost_pct_per_side: float,
    cost_multiplier: float,
    risk_per_trade: float,
) -> pd.DataFrame:
    if decisions.empty:
        return pd.DataFrame(index=decisions.index, columns=("constant", "policy"))
    net_r = (
        decisions["gross_return"] - 2 * cost_pct_per_side * cost_multiplier
    ) / decisions["initial_r_pct"]
    constant = risk_per_trade * net_r
    return pd.DataFrame(
        {
            "constant": constant,
            "policy": constant * decisions["multiplier"],
        },
        index=decisions.index,
    )


def _metrics(contributions: pd.Series) -> RiskPolicyMetrics:
    values = pd.Series(contributions, dtype=float)
    if values.empty:
        return RiskPolicyMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    london_date = values.index.tz_convert(LONDON).date
    daily = values.groupby(london_date).apply(lambda x: float((1 + x).prod() - 1))
    daily_std = float(daily.std(ddof=1))
    sharpe = (
        float(np.sqrt(252) * daily.mean() / daily_std)
        if len(daily) > 1 and daily_std > 0
        else 0.0
    )
    return RiskPolicyMetrics(
        opportunities=len(values),
        compounded_return=float((1 + values).prod() - 1),
        profit_factor=profit_factor(values),
        mean_contribution=float(values.mean()),
        daily_sharpe=sharpe,
        max_drawdown=max_drawdown(values),
        worst_contribution=float(values.min()),
    )


def _failures(
    config: RiskPolicyGateConfig,
    *,
    eligible: int,
    coverage: float,
    blocks: list[RiskPolicyBlock],
    constant: RiskPolicyMetrics,
    policy: RiskPolicyMetrics,
    policy_stress: RiskPolicyMetrics,
    profitable_blocks: int,
) -> list[str]:
    failures: list[str] = []
    if coverage < config.minimum_coverage:
        failures.append(f"forecast_match_coverage:{coverage:.4f}<{config.minimum_coverage:.2f}")
    if policy.opportunities < config.minimum_opportunities:
        failures.append(f"opportunities:{policy.opportunities}<{config.minimum_opportunities}")
    if len(blocks) != config.n_blocks or any(
        block.policy.opportunities < config.minimum_block_opportunities for block in blocks
    ):
        failures.append(
            f"block_opportunities_below:{config.minimum_block_opportunities}"
        )
    if constant.compounded_return <= 0:
        failures.append("constant_return_not_positive")
    if policy.compounded_return <= 0:
        failures.append("policy_return_not_positive")
    elif policy.compounded_return < (
        config.minimum_return_retention * constant.compounded_return
    ):
        failures.append(
            f"return_retention:{policy.compounded_return:.6f}"
            f"<{config.minimum_return_retention:.2f}x_constant"
        )
    required_drawdown = abs(constant.max_drawdown) * (
        1 - config.minimum_drawdown_reduction
    )
    if abs(policy.max_drawdown) > required_drawdown:
        failures.append(
            f"drawdown_reduction:{policy.max_drawdown:.6f}>-{required_drawdown:.6f}"
        )
    if policy.daily_sharpe <= constant.daily_sharpe:
        failures.append("daily_sharpe_not_above_constant")
    if policy.profit_factor <= constant.profit_factor:
        failures.append("profit_factor_not_above_constant")
    if profitable_blocks < 4:
        failures.append(f"profitable_policy_blocks:{profitable_blocks}<4")
    if policy_stress.compounded_return <= 0:
        failures.append("double_cost_policy_return_not_positive")
    return failures
