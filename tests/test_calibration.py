"""Calibration: improves Brier, leaves ranking alone, never touches test data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from commodity_ml.modeling.calibration import (
    CalibratedModel,
    IdentityCalibrator,
    MIN_CALIBRATION_ROWS,
    calibration_split,
    fit_calibrator,
    make_calibrator,
)
from commodity_ml.modeling.walkforward import walk_forward


def overconfident_scores(n: int = 3000, seed: int = 11):
    """Scores that rank well but are pushed toward 0 and 1 — exactly the
    failure mode LightGBM showed on overlapping labels."""
    rng = np.random.default_rng(seed)
    truth = rng.uniform(0.2, 0.8, n)
    y = (rng.uniform(size=n) < truth).astype(float)
    # Squash the true probability outward: monotone, so ranking is preserved.
    scores = np.clip((truth - 0.5) * 3 + 0.5, 0.001, 0.999)
    return scores, y


@pytest.mark.parametrize("kind", ["isotonic", "sigmoid"])
def test_calibration_improves_brier(kind):
    scores, y = overconfident_scores()
    fit, test = slice(0, 2000), slice(2000, None)

    calibrator = fit_calibrator(kind, scores[fit], y[fit])
    calibrated = calibrator.transform(scores[test])

    assert brier_score_loss(y[test], calibrated) < brier_score_loss(y[test], scores[test])


@pytest.mark.parametrize("kind", ["isotonic", "sigmoid"])
def test_calibration_preserves_ranking(kind):
    """Both calibrators are monotone, so AUC must not move materially. If a
    'calibration' change swings AUC, it is not calibration — it is a bug."""
    scores, y = overconfident_scores()
    calibrator = fit_calibrator(kind, scores[:2000], y[:2000])

    before = roc_auc_score(y[2000:], scores[2000:])
    after = roc_auc_score(y[2000:], calibrator.transform(scores[2000:]))
    assert abs(after - before) < 0.01


def test_calibrated_probabilities_stay_in_range():
    scores, y = overconfident_scores()
    for kind in ("isotonic", "sigmoid"):
        out = fit_calibrator(kind, scores[:2000], y[:2000]).transform(scores[2000:])
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_calibration_never_inverts_the_model():
    """The bug this suite exists to prevent.

    Platt scaling is an unconstrained logistic fit, so on scores that are
    *negatively* related to the outcome it learns a negative slope and flips
    the model's ranking. A below-chance model would then report above-chance
    AUC, with the sign decided by a few hundred calibration rows. On real gold
    data this turned the logistic baseline's pooled AUC from 0.482 into 0.513.
    """
    scores, y = overconfident_scores()
    inverted = 1.0 - scores  # ranks backwards: AUC well below 0.5

    calibrator = fit_calibrator("sigmoid", inverted[:2000], y[:2000])
    assert isinstance(calibrator, IdentityCalibrator), "must refuse, not flip"

    before = roc_auc_score(y[2000:], inverted[2000:])
    after = roc_auc_score(y[2000:], calibrator.transform(inverted[2000:]))
    assert before < 0.5
    assert np.isclose(after, before), "a rescaling must not rescue a bad model"


def test_too_little_data_degrades_to_identity():
    """A calibrator fitted on a handful of rows is noise. Passing scores
    through unchanged is the honest failure mode."""
    scores, y = overconfident_scores(n=MIN_CALIBRATION_ROWS - 1)
    calibrator = fit_calibrator("isotonic", scores, y)
    assert isinstance(calibrator, IdentityCalibrator)
    assert np.allclose(calibrator.transform(scores), scores)


def test_single_class_calibration_data_degrades_to_identity():
    scores, _ = overconfident_scores()
    y = np.ones(len(scores))
    assert isinstance(fit_calibrator("isotonic", scores, y), IdentityCalibrator)


def test_calibration_split_is_time_ordered_and_gapped():
    """The calibration rows are the *tail* of training, separated by the same
    gap the outer split uses, so overlapping labels can't straddle it."""
    gap = 12
    split = calibration_split(5000, gap)
    assert split is not None
    fit_rows, cal_rows = split
    assert fit_rows.max() < cal_rows.min(), "calibration data must come after fit data"
    assert cal_rows.min() - fit_rows.max() - 1 >= gap
    assert cal_rows.max() == 4999, "calibration block must end at the training boundary"


def test_calibration_split_refuses_when_too_small():
    assert calibration_split(200, gap=12) is None


def test_unknown_kind_rejected():
    with pytest.raises(ValueError, match="Unknown calibration"):
        make_calibrator("platt-ish")


def synthetic_xy(n: int = 4000, seed: int = 3):
    """Big enough that folds clear MIN_CALIBRATION_ROWS — at n=1200 every fold
    falls back to identity and the calibration tests pass vacuously."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    X = pd.DataFrame(rng.normal(size=(n, 4)), index=idx, columns=list("abcd"))
    y = pd.Series((X["a"] + rng.normal(0, 0.5, n) > 0).astype(float), index=idx)
    return X, y


@pytest.mark.parametrize("kind", ["isotonic", "sigmoid"])
def test_calibration_never_changes_the_ranking_series(kind):
    """The separation that makes 'calibration changes meaning, not behaviour'
    true by construction: oof_proba is the raw score and must be bit-identical
    whether calibration is on or off. Only oof_calibrated may move."""
    X, y = synthetic_xy()
    plain = walk_forward(X, y, n_splits=4, gap=12, model_kind="logistic")
    calibrated = walk_forward(X, y, n_splits=4, gap=12, model_kind="logistic",
                              calibration=kind)

    pd.testing.assert_series_equal(plain.oof_proba, calibrated.oof_proba)
    assert not calibrated.oof_calibrated.equals(calibrated.oof_proba)


def test_uncalibrated_run_reports_raw_scores_in_both_series():
    X, y = synthetic_xy()
    wf = walk_forward(X, y, n_splits=4, gap=12, model_kind="logistic")
    pd.testing.assert_series_equal(
        wf.oof_proba, wf.oof_calibrated, check_names=False
    )


def test_calibrated_model_enforces_feature_order():
    """Columns handed over in a different order than training used would be
    silently mis-scored, so the artifact selects by name and fails loudly."""
    X, y = synthetic_xy(n=300)
    from commodity_ml.modeling.models import make_model

    model = make_model("logistic").fit(X, y)
    artifact = CalibratedModel(model, IdentityCalibrator(), list(X.columns))

    shuffled = X[["d", "c", "b", "a"]]
    assert np.allclose(artifact.positive_proba(shuffled), artifact.positive_proba(X))

    with pytest.raises(ValueError, match="Missing features"):
        artifact.positive_proba(X[["a", "b"]])
