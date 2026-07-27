"""Model factories.

Step-1 policy: fixed, conservative defaults — no hyperparameter tuning until
the walk-forward harness itself is trusted (see CLAUDE.md). The LightGBM
settings below lean toward *underfitting*: financial labels are mostly noise,
so the regularization (min_child_samples, subsampling, shallow-ish trees)
matters more than raw capacity.

The LogisticRegression baseline exists to answer one question honestly:
is the boosted model actually earning its complexity? If LightGBM can't beat
a scaled linear model out-of-fold, the features (not the model) need work.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def make_model(kind: str):
    if kind == "lightgbm":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=100,   # each leaf must cover >=100 bars -> resists noise-fitting
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
        )
    if kind == "logistic":
        # Trees don't care about feature scale; a linear model does.
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)),
            ]
        )
    raise ValueError(f"Unknown model kind: {kind!r} (expected 'lightgbm' or 'logistic')")
