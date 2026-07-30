"""Walk-forward OLS volatility forecasting + the step-2b ablation delta.

Deliberately small. The model is plain OLS on standardized features because
HAR — the benchmark this field actually uses — is linear, and this repo has
measured twice that its constraint is information, not capacity. Anything
cleverer would add design surface to a gate whose whole value is that it was
fixed in advance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass
class VolForecastResult:
    """Pooled out-of-sample R² plus its per-fold decomposition.

    R²_OOS is measured against the TRAIN-set mean of the target, per fold —
    the forecast a no-information model would make (Campbell–Thompson
    convention). Using the test-set mean would hand every model the answer to
    "what was the average vol this fold?", which is most of what a vol
    forecast is for.
    """

    r2_pooled: float
    per_fold: list[float] = field(default_factory=list)
    n_rows: int = 0


def walk_forward_r2(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    gap: int,
) -> VolForecastResult:
    """Time-ordered folds, gap >= horizon so overlapping targets cannot leak.

    Standardization statistics come from each fold's train block only —
    scaling with full-sample moments would leak the test period's vol regime
    into training, which for a vol target is leaking the answer's scale.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    sse_total = sst_total = 0.0
    per_fold: list[float] = []

    values = X.to_numpy(dtype=float)
    target = y.to_numpy(dtype=float)

    for train_idx, test_idx in splitter.split(values):
        X_train, X_test = values[train_idx], values[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]

        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0  # a constant column (rare-event flag in a quiet
                             # fold) carries no slope; don't divide by zero
        Z_train = (X_train - mean) / std
        Z_test = (X_test - mean) / std

        design = np.column_stack([np.ones(len(Z_train)), Z_train])
        coef, *_ = np.linalg.lstsq(design, y_train, rcond=None)
        prediction = np.column_stack([np.ones(len(Z_test)), Z_test]) @ coef

        benchmark = y_train.mean()
        sse = float(((y_test - prediction) ** 2).sum())
        sst = float(((y_test - benchmark) ** 2).sum())
        sse_total += sse
        sst_total += sst
        per_fold.append(1.0 - sse / sst if sst > 0 else np.nan)

    return VolForecastResult(
        r2_pooled=1.0 - sse_total / sst_total,
        per_fold=per_fold,
        n_rows=len(y),
    )


def ablation_delta(
    dataset: pd.DataFrame,
    target_column: str,
    baseline_columns: list[str],
    family_columns: list[str],
    n_splits: int,
    gap: int,
) -> dict:
    """ΔR²_OOS = (baseline + family) − baseline, identical rows and folds.

    Both arms are sliced from the one matched-row dataset, so the difference
    is attributable to the family's columns alone — the same construction the
    step-2 gate used for ΔAUC.
    """
    y = dataset[target_column]
    with_family = walk_forward_r2(
        dataset[baseline_columns + family_columns], y, n_splits, gap
    )
    without = walk_forward_r2(dataset[baseline_columns], y, n_splits, gap)

    per_fold_delta = [
        w - b for w, b in zip(with_family.per_fold, without.per_fold)
    ]
    return {
        "delta": with_family.r2_pooled - without.r2_pooled,
        "r2_with": with_family.r2_pooled,
        "r2_without": without.r2_pooled,
        "per_fold_delta": per_fold_delta,
        "folds_improved": int(sum(d > 0 for d in per_fold_delta)),
        "n_rows": len(dataset),
    }
