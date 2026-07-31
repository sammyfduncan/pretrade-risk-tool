"""Registered P1 logistic gate for the GC TBBO order-flow family."""

import pandas as pd

from commodity_ml.evaluation.intraday_futures_gate import (
    FuturesGateResult,
    evaluate_futures_gate,
)


def evaluate_order_flow_gate(
    dataset: pd.DataFrame,
    spot_feature_names: tuple[str, ...] | list[str],
    order_flow_feature_names: tuple[str, ...] | list[str],
    *,
    cost_pct_per_side: float,
    coverage: float,
) -> FuturesGateResult:
    """Apply the same matched/placebo gate under the distinct OF-GC ledger entry."""

    return evaluate_futures_gate(
        dataset,
        spot_feature_names,
        order_flow_feature_names,
        cost_pct_per_side=cost_pct_per_side,
        coverage=coverage,
        n_placebos=20,
        program="intraday_of_gc_p1_logistic",
    )
