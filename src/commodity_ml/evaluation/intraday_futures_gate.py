"""Matched retention gate for the preregistered COMEX GC feature family."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from commodity_ml.evaluation.intraday_filter_gate import (
    FilterGateResult,
    evaluate_filter_gate,
)


@dataclass(frozen=True)
class FuturesGateResult:
    baseline: FilterGateResult
    challenger: FilterGateResult
    baseline_auc: float
    challenger_auc: float
    delta_auc: float
    auc_improved_folds: int
    placebo_results: tuple[dict, ...]
    coverage: float
    passed: bool
    failure_reasons: tuple[str, ...]

    def report_dict(self) -> dict:
        return {
            "program": "intraday_f_gc_p1_logistic",
            "coverage": self.coverage,
            "baseline_auc": self.baseline_auc,
            "challenger_auc": self.challenger_auc,
            "delta_auc": self.delta_auc,
            "auc_improved_folds": self.auc_improved_folds,
            "placebo_results": list(self.placebo_results),
            "placebo_max_delta_auc": max(
                item["delta_auc"] for item in self.placebo_results
            ),
            "baseline": self.baseline.report_dict(),
            "challenger": self.challenger.report_dict(),
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
        }


def evaluate_futures_gate(
    dataset: pd.DataFrame,
    spot_feature_names: tuple[str, ...] | list[str],
    futures_feature_names: tuple[str, ...] | list[str],
    *,
    cost_pct_per_side: float,
    coverage: float,
    n_placebos: int = 20,
) -> FuturesGateResult:
    """Compare matched spot-only and futures-informed P1 logistic filters."""

    if not 0 <= coverage <= 1:
        raise ValueError("coverage must be between zero and one")
    if n_placebos != 20:
        raise ValueError("The frozen F-GC design requires exactly 20 placebos")
    spot_names = tuple(spot_feature_names)
    futures_names = tuple(futures_feature_names)
    overlap = sorted(set(spot_names) & set(futures_names))
    if overlap:
        raise ValueError(f"Spot and futures features overlap: {overlap}")

    baseline = evaluate_filter_gate(
        dataset,
        spot_names,
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=cost_pct_per_side,
    )
    challenger = evaluate_filter_gate(
        dataset,
        spot_names + futures_names,
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=cost_pct_per_side,
    )
    baseline_auc = _pooled_auc(baseline)
    challenger_auc = _pooled_auc(challenger)
    delta_auc = challenger_auc - baseline_auc
    improved_folds = sum(
        challenge.auc > control.auc
        for control, challenge in zip(baseline.folds, challenger.folds)
    )

    offsets = _placebo_offsets(len(dataset), n_placebos)
    placebo_results: list[dict] = []
    futures_values = dataset.loc[:, futures_names].to_numpy()
    for offset in offsets:
        placebo = dataset.copy()
        placebo.loc[:, futures_names] = np.roll(futures_values, offset, axis=0)
        result = evaluate_filter_gate(
            placebo,
            spot_names + futures_names,
            strategy="P1",
            model_kind="logistic",
            cost_pct_per_side=cost_pct_per_side,
        )
        placebo_auc = _pooled_auc(result)
        placebo_results.append(
            {
                "offset_rows": offset,
                "auc": placebo_auc,
                "delta_auc": placebo_auc - baseline_auc,
                "profit_factor": result.filtered_by_cost[1.0].profit_factor,
            }
        )

    placebo_max = max(item["delta_auc"] for item in placebo_results)
    baseline_pf = baseline.filtered_by_cost[1.0].profit_factor
    challenger_pf = challenger.filtered_by_cost[1.0].profit_factor
    failures: list[str] = []
    if coverage < 0.90:
        failures.append(f"matched_coverage:{coverage:.4f}<0.90")
    if delta_auc < 0.015:
        failures.append(f"delta_auc:{delta_auc:.4f}<0.015")
    if improved_folds < 4:
        failures.append(f"auc_improved_folds:{improved_folds}<4")
    if delta_auc <= placebo_max:
        failures.append(
            f"delta_auc_not_above_placebo:{delta_auc:.4f}<={placebo_max:.4f}"
        )
    if not challenger.passed:
        failures.append("challenger_failed_economic_gate")
    if challenger_pf <= baseline_pf:
        failures.append(
            f"challenger_pf_not_above_control:{challenger_pf:.4f}<={baseline_pf:.4f}"
        )
    return FuturesGateResult(
        baseline=baseline,
        challenger=challenger,
        baseline_auc=baseline_auc,
        challenger_auc=challenger_auc,
        delta_auc=delta_auc,
        auc_improved_folds=improved_folds,
        placebo_results=tuple(placebo_results),
        coverage=coverage,
        passed=not failures,
        failure_reasons=tuple(failures),
    )


def _pooled_auc(result: FilterGateResult) -> float:
    predictions = result.predictions
    return float(roc_auc_score(predictions["label"], predictions["score"]))


def _placebo_offsets(rows: int, count: int) -> tuple[int, ...]:
    if rows < 21:
        raise ValueError("At least 21 rows are required for circular-shift placebos")
    offsets = tuple(int(round(rows * step / (count + 1))) for step in range(1, count + 1))
    if len(set(offsets)) != count or any(offset <= 0 or offset >= rows for offset in offsets):
        raise ValueError("Could not construct 20 distinct non-zero placebo offsets")
    return offsets
