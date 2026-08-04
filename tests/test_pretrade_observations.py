from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from commodity_ml.intraday.pretrade_observations import (
    ObservationLedgerError,
    PaperTradeOutcome,
    PretradeObservationStore,
)


ASSESSMENT_TIME = pd.Timestamp("2026-08-03 10:00", tz="UTC")
ENTRY_TIME = pd.Timestamp("2026-08-03 10:01", tz="UTC")
EXIT_TIME = pd.Timestamp("2026-08-03 11:00", tz="UTC")


def assessment(intent_id: str = "paper-1", decision: str = "ALLOW") -> dict:
    return {
        "schema_version": 1,
        "policy_version": "test-policy",
        "intent_id": intent_id,
        "decision": decision,
        "reasons": [],
        "warnings": [],
        "trade": {
            "instrument": "XAUUSD",
            "side": "long",
            "decision_time": ASSESSMENT_TIME.isoformat(),
        },
        "sizing": {"assessed_size": 10, "recommended_max_size": 10},
        "cash_risk": {
            "proposed": {"planned_stop_loss": 200},
            "at_recommended_max": {"planned_stop_loss": 200},
        },
        "costs": {},
        "market_risk": {},
        "portfolio": {},
        "scenarios": [],
        "limitations": [],
    }


def outcome(intent_id: str = "paper-1", pnl: float = -250) -> PaperTradeOutcome:
    return PaperTradeOutcome(
        intent_id=intent_id,
        entry_time=ENTRY_TIME,
        exit_time=EXIT_TIME,
        filled_size=10,
        expected_entry_price=2400,
        actual_entry_price=2400.2,
        actual_exit_price=2375,
        max_adverse_excursion_points=26,
        max_favorable_excursion_points=4,
        entry_spread_points=0.4,
        exit_spread_points=0.5,
        entry_adverse_slippage_points=0.2,
        exit_adverse_slippage_points=0.1,
        fees=0,
        peak_margin=1200,
        realized_pnl=pnl,
        source="OANDA_PRACTICE",
        source_reference="DEAL-REFERENCE-1",
    )


def test_assessment_is_immutable_and_exact_retry_is_idempotent(tmp_path):
    store = PretradeObservationStore(tmp_path / "observations.sqlite")
    report = assessment()

    assert store.record_assessment(report, inserted_at=ASSESSMENT_TIME) is True
    assert store.record_assessment(report, inserted_at=ASSESSMENT_TIME) is False

    changed = deepcopy(report)
    changed["warnings"] = ["changed_after_outcome_was_possible"]
    with pytest.raises(ObservationLedgerError, match="assessment_collision"):
        store.record_assessment(changed, inserted_at=ASSESSMENT_TIME)

    assert store.assessment_result("paper-1") == ("ALLOW", ())
    assert store.assessment_result("missing") is None


def test_outcome_requires_prior_allow_or_reduce_assessment(tmp_path):
    store = PretradeObservationStore(tmp_path / "observations.sqlite")

    with pytest.raises(ObservationLedgerError, match="prior_assessment"):
        store.record_outcome(outcome(), inserted_at=EXIT_TIME)

    store.record_assessment(
        assessment(decision="BLOCK"), inserted_at=ASSESSMENT_TIME
    )
    with pytest.raises(ObservationLedgerError, match="blocked_assessment"):
        store.record_outcome(outcome(), inserted_at=EXIT_TIME)


def test_outcome_is_immutable_and_summary_is_descriptive(tmp_path):
    store = PretradeObservationStore(tmp_path / "observations.sqlite")
    store.record_assessment(assessment(), inserted_at=ASSESSMENT_TIME)

    assert store.record_outcome(outcome(), inserted_at=EXIT_TIME) is True
    assert store.record_outcome(outcome(), inserted_at=EXIT_TIME) is False
    with pytest.raises(ObservationLedgerError, match="outcome_collision"):
        store.record_outcome(outcome(pnl=-100), inserted_at=EXIT_TIME)

    summary = store.summary()
    assert summary["assessment_counts"] == {"ALLOW": 1}
    assert summary["completed_outcomes"] == 1
    assert summary["planned_stop_comparisons"] == 1
    assert summary["realized_stop_loss_exceedance_rate"] == pytest.approx(1.0)
    assert summary["realized_pnl"]["mean"] == pytest.approx(-250)
    assert "profitability" in summary["limitations"][0]


def test_filled_size_scales_the_registered_planned_loss(tmp_path):
    store = PretradeObservationStore(tmp_path / "observations.sqlite")
    store.record_assessment(assessment(), inserted_at=ASSESSMENT_TIME)
    smaller = PaperTradeOutcome(
        **{**outcome(pnl=-110).__dict__, "filled_size": 5}
    )
    store.record_outcome(smaller, inserted_at=EXIT_TIME)

    summary = store.summary()
    # Registered loss was 200 at size 10, hence 100 at the actual size 5.
    assert summary["realized_stop_loss_exceedance_rate"] == pytest.approx(1.0)


def test_outcome_validation_rejects_naive_time_and_negative_observations():
    value = outcome().to_dict()
    value["entry_time"] = "2026-08-03 10:01"
    with pytest.raises(ValueError, match="timezone-aware"):
        PaperTradeOutcome.from_mapping(value)

    value = outcome().to_dict()
    value["entry_spread_points"] = -0.1
    with pytest.raises(ValueError, match="non-negative"):
        PaperTradeOutcome.from_mapping(value)
