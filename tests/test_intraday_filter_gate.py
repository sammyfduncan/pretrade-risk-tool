"""Nested intraday filter gate: threshold causality and retention controls."""

import numpy as np
import pandas as pd

from commodity_ml.evaluation.intraday_filter_gate import evaluate_filter_gate


def separable_dataset(n: int = 900) -> pd.DataFrame:
    index = pd.date_range("2019-01-01", periods=n, freq="1D", tz="UTC")
    phase = np.arange(n) % 10
    feature = phase.astype(float)
    label = (phase >= 7).astype("int8")
    return pd.DataFrame(
        {
            "strategy": "P1",
            "label": label,
            "exit_time": index + pd.Timedelta(hours=2),
            "gross_return": np.where(label == 1, 0.015, -0.010),
            "initial_r_pct": 0.01,
            "signal": feature,
        },
        index=index,
    )


def test_separable_filter_passes_registered_economic_gate():
    result = evaluate_filter_gate(
        separable_dataset(),
        ["signal"],
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=0.0001,
    )

    assert result.passed
    assert result.filtered_by_cost[1.0].trades >= 200
    assert result.filtered_by_cost[1.0].profit_factor > 1.10
    assert result.filtered_by_cost[2.0].profit_factor > 1.0
    assert result.improved_folds == 5
    assert all(fold.filtered_rows >= 30 for fold in result.folds)


def test_last_test_labels_cannot_change_scores_thresholds_or_selection():
    original = separable_dataset()
    poisoned = original.copy()
    poisoned.loc[poisoned.index[-150:], "label"] = (
        1 - poisoned.loc[poisoned.index[-150:], "label"]
    )

    first = evaluate_filter_gate(
        original,
        ["signal"],
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=0.0001,
    )
    second = evaluate_filter_gate(
        poisoned,
        ["signal"],
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=0.0001,
    )

    pd.testing.assert_series_equal(
        first.predictions["score"],
        second.predictions["score"],
    )
    pd.testing.assert_series_equal(
        first.predictions["threshold"],
        second.predictions["threshold"],
    )
    pd.testing.assert_series_equal(
        first.predictions["selected"],
        second.predictions["selected"],
    )


def test_outer_test_features_cannot_change_training_derived_thresholds():
    original = separable_dataset()
    poisoned = original.copy()
    poisoned.loc[poisoned.index[-150:], "signal"] *= -100

    first = evaluate_filter_gate(
        original,
        ["signal"],
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=0.0001,
    )
    second = evaluate_filter_gate(
        poisoned,
        ["signal"],
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=0.0001,
    )

    assert [fold.threshold for fold in first.folds] == [
        fold.threshold for fold in second.folds
    ]
    final_fold = first.predictions["fold"] == 4
    assert not first.predictions.loc[final_fold, "score"].equals(
        second.predictions.loc[final_fold, "score"]
    )


def test_constant_ranker_fails_instead_of_forcing_top_thirty_percent():
    dataset = separable_dataset()
    dataset["signal"] = 1.0
    dataset["gross_return"] = -0.01

    result = evaluate_filter_gate(
        dataset,
        ["signal"],
        strategy="P1",
        model_kind="logistic",
        cost_pct_per_side=0.0001,
    )

    assert not result.passed
    assert all(fold.retained_fraction == 1.0 for fold in result.folds)
    assert any("base_profit_factor" in reason for reason in result.failure_reasons)
