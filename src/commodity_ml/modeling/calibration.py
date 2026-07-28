"""Probability calibration — making the number mean what it says.

The problem
-----------
A classifier's raw output is a *score*, not a probability. LightGBM's
0.72 does not mean "this happens 72% of the time"; it means "this ranks high".
Run 2 and run 3 both reported a Brier score worse than the no-skill baseline
of always predicting the base rate — the model's ranking was fine, its
probabilities were overconfident nonsense.

That matters for three concrete reasons here:
  * position sizing off an uncalibrated probability sizes off noise;
  * a bot running unsupervised has nothing but the number to reason with;
  * comparing LightGBM against the logistic baseline on Brier is meaningless
    while one of them is systematically miscalibrated.

What it does NOT do
-------------------
Both calibrators are monotonic, so **AUC is essentially unchanged** and — since
the backtest enters on an expanding *quantile* of the prediction distribution
rather than an absolute probability — trading results barely move either.
Calibration buys interpretability, not edge. Expect the Brier score to improve
and everything else to stay put; if returns jump, something is wrong.

Avoiding the obvious leak
-------------------------
A calibrator fitted on the same rows the model was fitted on learns to correct
in-sample overconfidence that does not exist out-of-sample, and one fitted on
the *test* rows is straightforward leakage. So within each walk-forward fold
the calibrator is fitted on a held-out tail of that fold's training data,
separated from it by the same `gap` the outer split uses.

The model is then refitted on the full training block, so the calibrator maps
the outputs of a *slightly* differently-trained model than the one it grades.
That is the standard trade (sklearn's CalibratedClassifierCV does the same
with ensemble=False); the alternative wastes 20% of every fold's training data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Fraction of each fold's training block held back to fit the calibrator.
CALIBRATION_FRACTION = 0.2
# Below this many rows, a fitted calibrator is noisier than the miscalibration
# it corrects — isotonic especially, which can chase individual points.
MIN_CALIBRATION_ROWS = 250

_EPS = 1e-6  # keeps logit() finite at p = 0 or 1

# Grid used to verify a fitted calibrator is non-decreasing. See _is_monotone.
_MONOTONICITY_PROBE = np.linspace(0.01, 0.99, 99)


class IdentityCalibrator:
    """Pass-through, used when calibration is off or cannot be fitted safely."""

    def fit(self, proba, y):  # noqa: ARG002 - signature parity
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return np.asarray(proba, dtype=float)


class IsotonicCalibrator:
    """Non-parametric monotone fit — flexible, but hungry for data.

    Makes no shape assumption, so it can correct any monotone distortion. The
    cost is variance: with a few hundred calibration rows it will happily fit
    a staircase to noise, which is what MIN_CALIBRATION_ROWS guards against.
    """

    def __init__(self) -> None:
        self._model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self._model.fit(np.asarray(proba, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(proba, dtype=float))


class SigmoidCalibrator:
    """Platt scaling: a one-parameter-ish logistic fit on the score's logit.

    Assumes the distortion is a smooth squash/stretch, which is restrictive but
    exactly the right bias when calibration data is scarce — it cannot overfit
    a staircase because it can only bend one way.
    """

    def __init__(self) -> None:
        self._model = LogisticRegression(C=1e6, solver="lbfgs")

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "SigmoidCalibrator":
        self._model.fit(_logit(proba), np.asarray(y, dtype=float))
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(_logit(proba))[:, 1]


def make_calibrator(kind: str):
    if kind == "none":
        return IdentityCalibrator()
    if kind == "isotonic":
        return IsotonicCalibrator()
    if kind == "sigmoid":
        return SigmoidCalibrator()
    raise ValueError(f"Unknown calibration kind: {kind!r}")


def fit_calibrator(kind: str, proba: np.ndarray, y: np.ndarray):
    """Fit a calibrator, degrading to pass-through when it cannot be trusted.

    Silently returning an identity map is the right failure mode: an unfitted
    calibrator that pretends to work would corrupt every downstream number,
    whereas passing raw scores through leaves them exactly as honest (or not)
    as they were before.

    The monotonicity check is not a formality. Platt scaling is an
    *unconstrained* logistic fit, so when the score/outcome relationship is
    negative on the calibration slice it happily learns a negative slope and
    inverts the model. That is not calibration — it is a sign flip decided by a
    few hundred rows, and it is exactly the trap this dataset sets: several
    folds here score below 0.50 AUC, so an inverting calibrator would turn a
    model with no skill into one with apparent skill. It did, on first run: the
    logistic baseline's pooled AUC "improved" from 0.482 to 0.513 purely
    through inversion.
    """
    proba = np.asarray(proba, dtype=float)
    y = np.asarray(y, dtype=float)
    if kind == "none" or len(proba) < MIN_CALIBRATION_ROWS or len(np.unique(y)) < 2:
        return IdentityCalibrator()

    calibrator = make_calibrator(kind).fit(proba, y)
    if not _is_monotone(calibrator):
        return IdentityCalibrator()
    return calibrator


def _is_monotone(calibrator) -> bool:
    """Does the fitted map preserve the model's ranking?

    Checked on a probe grid rather than by inspecting coefficients, so it
    covers any calibrator shape rather than just Platt's.
    """
    mapped = calibrator.transform(_MONOTONICITY_PROBE)
    return bool(np.all(np.diff(mapped) >= -1e-9))


def calibration_split(n_train: int, gap: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Split a fold's training block into (fit rows, calibration rows).

    Time-ordered and gapped, for the same reason the outer walk-forward split
    is: the calibration rows sit at the end, and the `gap` bars before them are
    dropped so overlapping forward labels cannot straddle the boundary.

    Returns None when the split would leave too little of either side.
    """
    n_cal = int(n_train * CALIBRATION_FRACTION)
    n_fit = n_train - n_cal - gap
    if n_cal < MIN_CALIBRATION_ROWS or n_fit < MIN_CALIBRATION_ROWS:
        return None
    return np.arange(n_fit), np.arange(n_train - n_cal, n_train)


@dataclass
class CalibratedModel:
    """The live-prediction contract: features in, one honest probability out.

    Bundling the estimator, its calibrator and the exact feature order into a
    single object is deliberate. It means the live path cannot accidentally
    apply the model without its calibrator, or hand columns over in a
    different order than training used — two silent failures that would each
    produce plausible-looking but wrong numbers.
    """

    model: object
    calibrator: object
    feature_names: list[str]

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        """The uncalibrated score — what entry decisions rank on.

        The entry rule compares against a quantile of the score distribution,
        which is a statement about ordering, so it must see the same scale the
        threshold was derived from.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Missing features at prediction time: {missing}")
        return self.model.predict_proba(X[self.feature_names])[:, 1]

    def positive_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated P(label = 1) — what the number *means*.

        Use this to report, to reason about, and later to size positions. Do
        not rank on it: calibration is allowed to flatten regions where the
        score does not discriminate, which is informative but destroys the
        fine ordering the quantile entry rule needs.

        Named this rather than `predict_proba` so it can never be mistaken for
        sklearn's two-column convention.
        """
        return self.calibrator.transform(self.raw_score(X))


def _logit(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(proba, dtype=float), _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)
