"""ML-quality metrics for classifier predictions.

These answer "does the model rank up-move bars above others at all?" —
not whether it makes money (that's trading_metrics + the backtest).

Key readings:
- AUC 0.50 = coin flip. On noisy hourly financial data, honest OOF AUCs of
  0.52-0.55 are already interesting; anything above ~0.60 deserves suspicion
  (look for leakage) before celebration.
- Brier score = mean squared error of the probability itself (lower better);
  compare against the naive baseline of always predicting the base rate.
- Always read metrics per fold, not just pooled: a model that only worked in
  one regime shows up as one good fold and four flat ones.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from commodity_ml.modeling.walkforward import FoldResult


def overall_metrics(y: pd.Series, proba: pd.Series) -> dict:
    """Metrics over the pooled OOF period. y is aligned to proba's index."""
    y_aligned = y.loc[proba.index]
    base_rate = float(y_aligned.mean())
    return {
        "n": int(len(y_aligned)),
        "base_rate": base_rate,
        "auc": float(roc_auc_score(y_aligned, proba)),
        "brier": float(brier_score_loss(y_aligned, proba)),
        # Brier of always predicting the base rate — the 'no skill' reference
        "brier_baseline": float(base_rate * (1 - base_rate)),
    }


def fold_metrics_table(y: pd.Series, proba: pd.Series, folds: list[FoldResult]) -> pd.DataFrame:
    rows = []
    for f in folds:
        mask = (proba.index >= f.test_start) & (proba.index <= f.test_end)
        p = proba.loc[mask]
        yt = y.loc[p.index]
        rows.append(
            {
                "fold": f.fold,
                "test_start": f.test_start.date(),
                "test_end": f.test_end.date(),
                "n_train": f.n_train,
                "n_test": len(yt),
                "base_rate": round(float(yt.mean()), 3),
                "auc": round(float(roc_auc_score(yt, p)), 4) if yt.nunique() > 1 else None,
                "brier": round(float(brier_score_loss(yt, p)), 4),
            }
        )
    return pd.DataFrame(rows).set_index("fold")
