"""Walk-forward harness: time ordering, gap enforcement, honest OOF coverage."""

import numpy as np
import pandas as pd
import pytest

from commodity_ml.config import AssetConfig, CVConfig, LabelConfig
from commodity_ml.modeling.walkforward import walk_forward


def synthetic_xy(n: int = 500, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=list("abcd"))
    # Learnable signal so the model isn't degenerate single-class
    y = pd.Series((X["a"] + rng.normal(0, 0.5, n) > 0).astype(float), index=idx)
    return X, y


def test_folds_are_time_ordered_with_gap():
    X, y = synthetic_xy()
    gap = 12
    wf = walk_forward(X, y, n_splits=4, gap=gap, model_kind="logistic")
    for f in wf.folds:
        assert f.train_end < f.test_start, "training data must strictly precede test data"
        gap_bars = X.index.get_loc(f.test_start) - X.index.get_loc(f.train_end) - 1
        assert gap_bars >= gap, f"fold {f.fold}: gap {gap_bars} < required {gap}"


def test_oof_predictions_cover_all_test_folds_once():
    X, y = synthetic_xy()
    wf = walk_forward(X, y, n_splits=4, gap=12, model_kind="logistic")
    assert not wf.oof_proba.index.duplicated().any()
    assert wf.oof_proba.index.is_monotonic_increasing
    expected = sum(f.n_test for f in wf.folds)
    assert len(wf.oof_proba) == expected
    # First fold's training block must have no OOF predictions (never tested on)
    assert wf.oof_proba.index.min() > wf.folds[0].train_end


def test_oof_predictions_have_signal():
    """On data with a real learnable pattern, OOF AUC must beat coin flip —
    sanity check that fit/predict plumbing is actually wired up."""
    from sklearn.metrics import roc_auc_score

    X, y = synthetic_xy(n=800)
    wf = walk_forward(X, y, n_splits=4, gap=12, model_kind="logistic")
    auc = roc_auc_score(y.loc[wf.oof_proba.index], wf.oof_proba)
    assert auc > 0.7


def test_unsorted_index_rejected():
    X, y = synthetic_xy()
    X_shuffled = X.sample(frac=1, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        walk_forward(X_shuffled, y, n_splits=4, gap=12, model_kind="logistic")


def test_config_rejects_gap_smaller_than_horizon():
    with pytest.raises(ValueError, match="gap_bars"):
        AssetConfig(
            asset="x",
            symbol="XAUUSD",
            granularity="H1",
            label=LabelConfig(horizon_bars=12),
            cv=CVConfig(n_splits=5, gap_bars=6),
        )
