"""Leakage-safe retention gate for intraday candidate ML filters.

Each outer fold contains three chronological blocks:

    model-fit prefix -> score-threshold reference tail -> untouched test block

The threshold is a quantile of scores from the reference tail. Its labels are
not used. The same fitted model and numeric threshold are then applied to the
outer test block, so neither score scale nor selection rate can be repaired
with future information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from commodity_ml.evaluation.trading_metrics import max_drawdown, profit_factor
from commodity_ml.modeling.models import make_model
from commodity_ml.modeling.sample_weights import uniqueness_weights


@dataclass(frozen=True)
class FilterGateConfig:
    n_splits: int = 5
    retained_fraction: float = 0.30
    reference_fraction: float = 0.25
    min_fit_rows: int = 100
    min_reference_rows: int = 40
    min_pooled_filtered: int = 200
    min_fold_filtered: int = 30
    min_base_profit_factor: float = 1.10
    cost_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        for name in ("retained_fraction", "reference_fraction"):
            value = getattr(self, name)
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between zero and one")
        for name in (
            "min_fit_rows",
            "min_reference_rows",
            "min_pooled_filtered",
            "min_fold_filtered",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_base_profit_factor <= 1:
            raise ValueError("min_base_profit_factor must exceed one")
        if 1.0 not in self.cost_multipliers or 2.0 not in self.cost_multipliers:
            raise ValueError("cost_multipliers must include the 1x and 2x gates")
        if any(value <= 0 for value in self.cost_multipliers):
            raise ValueError("cost multipliers must be positive")


@dataclass(frozen=True)
class TradeMetrics:
    trades: int
    win_rate: float
    profit_factor: float
    mean_r: float
    total_return: float
    max_drawdown: float


@dataclass(frozen=True)
class FilterFoldResult:
    fold: int
    fit_rows: int
    reference_rows: int
    test_rows: int
    filtered_rows: int
    retained_fraction: float
    threshold: float
    auc: float
    bare: TradeMetrics
    filtered: TradeMetrics


@dataclass(frozen=True)
class FilterGateResult:
    strategy: str
    model_kind: str
    config: FilterGateConfig
    folds: tuple[FilterFoldResult, ...]
    bare_by_cost: dict[float, TradeMetrics]
    filtered_by_cost: dict[float, TradeMetrics]
    improved_folds: int
    passed: bool
    failure_reasons: tuple[str, ...]
    predictions: pd.DataFrame

    def report_dict(self) -> dict:
        """JSON-safe report without duplicating the predictions table."""

        report = {
            "strategy": self.strategy,
            "model_kind": self.model_kind,
            "config": asdict(self.config),
            "folds": [asdict(fold) for fold in self.folds],
            "bare_by_cost": {
                str(cost): asdict(metrics)
                for cost, metrics in self.bare_by_cost.items()
            },
            "filtered_by_cost": {
                str(cost): asdict(metrics)
                for cost, metrics in self.filtered_by_cost.items()
            },
            "improved_folds": self.improved_folds,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }
        return _json_safe(report)


def evaluate_filter_gate(
    dataset: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    *,
    strategy: str,
    model_kind: str,
    cost_pct_per_side: float,
    config: FilterGateConfig | None = None,
) -> FilterGateResult:
    """Evaluate one registered model/filter without selecting on test outcomes."""

    config = config or FilterGateConfig()
    frame = dataset.sort_index().copy()
    _validate_dataset(frame, feature_names, strategy, cost_pct_per_side)
    X = frame.loc[:, feature_names]
    y = frame["label"].astype("int8")
    exit_time = pd.to_datetime(frame["exit_time"], utc=True)
    splitter = TimeSeriesSplit(n_splits=config.n_splits)
    fold_results: list[FilterFoldResult] = []
    prediction_chunks: list[pd.DataFrame] = []

    for fold, (outer_train, outer_test) in enumerate(splitter.split(X)):
        test_start = X.index[outer_test[0]]
        train_exits = exit_time.iloc[outer_train]
        outer_train = outer_train[(train_exits < test_start).to_numpy()]
        if len(outer_train) < config.min_fit_rows + config.min_reference_rows:
            raise ValueError(
                f"Fold {fold} has only {len(outer_train)} purged training rows; "
                "not enough for the registered fit/reference split"
            )

        reference_rows = max(
            config.min_reference_rows,
            int(np.ceil(len(outer_train) * config.reference_fraction)),
        )
        if len(outer_train) - reference_rows < config.min_fit_rows:
            reference_rows = len(outer_train) - config.min_fit_rows
        fit_rows = outer_train[:-reference_rows]
        reference = outer_train[-reference_rows:]
        reference_start = X.index[reference[0]]
        fit_exits = exit_time.iloc[fit_rows]
        fit_rows = fit_rows[(fit_exits < reference_start).to_numpy()]
        if len(fit_rows) < config.min_fit_rows:
            raise ValueError(
                f"Fold {fold} has only {len(fit_rows)} rows after inner exit-time purge"
            )
        if y.iloc[fit_rows].nunique() != 2:
            raise ValueError(f"Fold {fold} model-fit prefix contains one label class")

        model = make_model(model_kind)
        weights = uniqueness_weights(
            X.index[fit_rows],
            exit_time.iloc[fit_rows],
        ).to_numpy()
        _fit_with_weights(model, X.iloc[fit_rows], y.iloc[fit_rows], weights)
        reference_scores = model.predict_proba(X.iloc[reference])[:, 1]
        threshold = float(
            np.quantile(
                reference_scores,
                1 - config.retained_fraction,
                method="higher",
            )
        )
        test_scores = model.predict_proba(X.iloc[outer_test])[:, 1]
        selected = test_scores >= threshold
        test = frame.iloc[outer_test].copy()
        test["fold"] = fold
        test["score"] = test_scores
        test["threshold"] = threshold
        test["selected"] = selected
        prediction_chunks.append(
            test[
                [
                    "fold",
                    "score",
                    "threshold",
                    "selected",
                    "label",
                    "gross_return",
                    "initial_r_pct",
                ]
            ]
        )

        bare_metrics = trade_metrics(
            test,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        filtered_metrics = trade_metrics(
            test.loc[selected],
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=1.0,
        )
        auc = (
            float(roc_auc_score(test["label"], test_scores))
            if test["label"].nunique() == 2
            else float("nan")
        )
        fold_results.append(
            FilterFoldResult(
                fold=fold,
                fit_rows=len(fit_rows),
                reference_rows=len(reference),
                test_rows=len(test),
                filtered_rows=int(selected.sum()),
                retained_fraction=float(selected.mean()),
                threshold=threshold,
                auc=auc,
                bare=bare_metrics,
                filtered=filtered_metrics,
            )
        )

    predictions = pd.concat(prediction_chunks).sort_index()
    outer_frame = frame.loc[predictions.index].copy()
    filtered_frame = outer_frame.loc[predictions["selected"]]
    bare_by_cost = {
        multiplier: trade_metrics(
            outer_frame,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    filtered_by_cost = {
        multiplier: trade_metrics(
            filtered_frame,
            cost_pct_per_side=cost_pct_per_side,
            cost_multiplier=multiplier,
        )
        for multiplier in config.cost_multipliers
    }
    improved_folds = sum(
        fold.filtered.profit_factor > fold.bare.profit_factor
        for fold in fold_results
    )
    failures = _gate_failures(
        fold_results,
        bare_by_cost,
        filtered_by_cost,
        improved_folds,
        config,
    )
    return FilterGateResult(
        strategy=strategy,
        model_kind=model_kind,
        config=config,
        folds=tuple(fold_results),
        bare_by_cost=bare_by_cost,
        filtered_by_cost=filtered_by_cost,
        improved_folds=improved_folds,
        passed=not failures,
        failure_reasons=tuple(failures),
        predictions=predictions,
    )


def _validate_dataset(
    frame: pd.DataFrame,
    feature_names: list[str] | tuple[str, ...],
    strategy: str,
    cost_pct_per_side: float,
) -> None:
    required = {
        "strategy",
        "label",
        "exit_time",
        "gross_return",
        "initial_r_pct",
        *feature_names,
    }
    missing = sorted(required - set(frame))
    if missing:
        raise ValueError(f"Filter-gate dataset missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("Filter-gate dataset needs a timezone-aware index")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Filter-gate dataset index must be sorted and unique")
    if len(frame) and (frame["strategy"] != strategy).any():
        raise ValueError("Filter-gate dataset contains the wrong strategy")
    if frame["label"].nunique() != 2:
        raise ValueError("Filter-gate labels must contain both classes")
    if not np.isfinite(frame.loc[:, feature_names].astype(float).to_numpy()).all():
        raise ValueError("Filter-gate features contain non-finite values")
    if not np.isfinite(frame["gross_return"]).all():
        raise ValueError("Filter-gate gross returns contain non-finite values")
    if (frame["initial_r_pct"] <= 0).any():
        raise ValueError("Filter-gate initial_r_pct must be positive")
    if not np.isfinite(cost_pct_per_side) or cost_pct_per_side < 0:
        raise ValueError("cost_pct_per_side must be finite and non-negative")


def trade_metrics(
    frame: pd.DataFrame,
    *,
    cost_pct_per_side: float,
    cost_multiplier: float,
) -> TradeMetrics:
    if frame.empty:
        return TradeMetrics(
            trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            mean_r=0.0,
            total_return=0.0,
            max_drawdown=0.0,
        )
    net = frame["gross_return"].astype(float) - (
        2 * cost_pct_per_side * cost_multiplier
    )
    r_multiple = net / frame["initial_r_pct"].astype(float)
    return TradeMetrics(
        trades=len(frame),
        win_rate=float((net > 0).mean()),
        profit_factor=profit_factor(net),
        mean_r=float(r_multiple.mean()),
        total_return=float((1 + net).prod() - 1),
        max_drawdown=max_drawdown(net),
    )


def _gate_failures(
    folds: list[FilterFoldResult],
    bare_by_cost: dict[float, TradeMetrics],
    filtered_by_cost: dict[float, TradeMetrics],
    improved_folds: int,
    config: FilterGateConfig,
) -> list[str]:
    base_bare = bare_by_cost[1.0]
    base_filtered = filtered_by_cost[1.0]
    stress_filtered = filtered_by_cost[2.0]
    failures: list[str] = []
    if base_filtered.trades < config.min_pooled_filtered:
        failures.append(
            f"pooled_filtered_trades:{base_filtered.trades}"
            f"<{config.min_pooled_filtered}"
        )
    if any(fold.filtered_rows < config.min_fold_filtered for fold in folds):
        failures.append(f"fold_filtered_trades_below:{config.min_fold_filtered}")
    if base_filtered.profit_factor < config.min_base_profit_factor:
        failures.append(
            f"base_profit_factor:{base_filtered.profit_factor:.4f}"
            f"<{config.min_base_profit_factor:.2f}"
        )
    if base_filtered.mean_r <= 0:
        failures.append(f"base_mean_r_not_positive:{base_filtered.mean_r:.4f}")
    if improved_folds < 4:
        failures.append(f"pf_improved_folds:{improved_folds}<4")
    if base_filtered.max_drawdown <= base_bare.max_drawdown:
        failures.append(
            "filtered_max_drawdown_not_better:"
            f"{base_filtered.max_drawdown:.4f}<="
            f"{base_bare.max_drawdown:.4f}"
        )
    if stress_filtered.profit_factor <= 1.0:
        failures.append(
            f"double_cost_profit_factor:{stress_filtered.profit_factor:.4f}<=1.00"
        )
    return failures


def _fit_with_weights(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
) -> None:
    if isinstance(model, Pipeline):
        final_step = model.steps[-1][0]
        model.fit(X, y, **{f"{final_step}__sample_weight": weights})
    else:
        model.fit(X, y, sample_weight=weights)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value
