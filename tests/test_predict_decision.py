"""The live trade/no-trade decision, and the artifact it depends on.

tests/test_predict_parity.py covers the *features* the live path builds. This
covers what happens to them afterwards: the model artifact surviving a save and
reload intact, and the decision rule itself. This is the code that will place
real orders, so the failure modes worth pinning are the silent ones — a
threshold quietly recomputed, a calibrated probability used where a raw score
was meant, features reordered between training and prediction.
"""

import joblib
import numpy as np
import pandas as pd
import pytest

from commodity_ml.decision import Decision, decide
from commodity_ml.modeling.walkforward import fit_final_model


def synthetic_xy(n: int = 800, seed: int = 3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)},
        index=idx,
    )
    y = pd.Series((X["a"] + rng.normal(0, 0.5, n) > 0).astype(int), index=idx, name="label")
    return X, y


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    """A saved artifact, reloaded — exactly what predict.py consumes."""
    X, y = synthetic_xy()
    model = fit_final_model(X, y, "logistic", calibration="isotonic")
    payload = {
        "model": model,
        "feature_names": list(X.columns),
        "entry_quantile": 0.90,
        "entry_threshold": 0.70,
    }
    path = tmp_path_factory.mktemp("models") / "test.joblib"
    joblib.dump(payload, path)
    return joblib.load(path), X


def feature_row(X: pd.DataFrame, i: int = -1) -> pd.Series:
    return X.iloc[i]


# --------------------------------------------------------------------------
# The artifact survives the round trip
# --------------------------------------------------------------------------

def test_saved_model_scores_identically_after_reload(artifact):
    reloaded, X = artifact
    fresh = fit_final_model(X, synthetic_xy()[1], "logistic", calibration="isotonic")

    np.testing.assert_allclose(
        reloaded["model"].raw_score(X), fresh.raw_score(X), rtol=0, atol=0
    )


def test_artifact_carries_everything_the_live_path_needs(artifact):
    reloaded, _ = artifact
    for key in ("model", "feature_names", "entry_threshold", "entry_quantile"):
        assert key in reloaded, f"predict.py depends on artifact['{key}']"
    assert reloaded["model"].feature_names == reloaded["feature_names"]


def test_features_are_reordered_to_match_training(artifact):
    """A dict-built feature row can arrive in any order; the model must not
    silently score column 'b' as if it were 'a'."""
    reloaded, X = artifact
    row = feature_row(X)
    shuffled = row[["c", "a", "b"]]

    straight = decide(reloaded, row, primary_long=True)
    reordered = decide(reloaded, shuffled, primary_long=True)
    assert straight.score == reordered.score


def test_a_missing_feature_raises_rather_than_guessing(artifact):
    reloaded, X = artifact
    incomplete = feature_row(X).drop("b")

    with pytest.raises(ValueError, match="Missing features"):
        decide(reloaded, incomplete, primary_long=True)


# --------------------------------------------------------------------------
# The decision rule
# --------------------------------------------------------------------------

def test_both_conditions_are_required_to_trade():
    """Meta-labeling: the rule picks the side, the model only filters. A strong
    score on a bar where the rule is flat is not a trade — there is nothing to
    filter."""
    cases = {
        (True, 0.95): True,    # rule long, model agrees
        (True, 0.10): False,   # rule long, model vetoes
        (False, 0.95): False,  # model likes it but there is no trade to take
        (False, 0.10): False,
    }
    for (primary_long, score), expected in cases.items():
        d = Decision(
            bar_time=pd.Timestamp("2026-01-01", tz="UTC"), score=score,
            calibrated_proba=0.5, primary_long=primary_long, threshold=0.70,
        )
        assert d.take_trade is expected, f"rule={primary_long}, score={score}"


def test_threshold_is_inclusive_at_the_boundary():
    d = Decision(
        bar_time=pd.Timestamp("2026-01-01", tz="UTC"), score=0.70,
        calibrated_proba=0.5, primary_long=True, threshold=0.70,
    )
    assert d.model_agrees, "a score exactly at the threshold enters, matching the backtest's >="


def test_decision_uses_the_raw_score_not_the_calibrated_probability(artifact):
    """The stored threshold is a quantile of the raw OOF distribution. Isotonic
    calibration is a step function that collapses distinct scores, so comparing
    a calibrated value against a raw threshold would change which bars trade for
    reasons unrelated to signal (`docs/step1_freeze_record.md`)."""
    reloaded, X = artifact
    row = feature_row(X)
    d = decide(reloaded, row, primary_long=True)

    assert d.score == pytest.approx(float(reloaded["model"].raw_score(row.to_frame().T)[0]))
    assert d.calibrated_proba == pytest.approx(
        float(reloaded["model"].positive_proba(row.to_frame().T)[0])
    )
    assert d.model_agrees == (d.score >= d.threshold)


def test_threshold_comes_from_the_artifact_and_is_not_recomputed(artifact):
    """A single bar has no distribution to take a quantile of. If predict.py
    ever derived its own threshold it would be answering a different question
    than the backtest measured."""
    reloaded, X = artifact
    d = decide(reloaded, feature_row(X), primary_long=True)
    assert d.threshold == reloaded["entry_threshold"]

    moved = {**reloaded, "entry_threshold": 0.999}
    assert decide(moved, feature_row(X), primary_long=True).threshold == 0.999


def test_decision_reports_the_bar_it_scored(artifact):
    """predict.py prints this to check the bar is not inside training or past
    the holdout fence, so it must be the feature row's own timestamp."""
    reloaded, X = artifact
    row = feature_row(X, i=-5)
    assert decide(reloaded, row, primary_long=True).bar_time == X.index[-5]
