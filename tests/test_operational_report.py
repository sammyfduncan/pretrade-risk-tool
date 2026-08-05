import pandas as pd

from commodity_ml.intraday.operational_report import (
    OperationalReadinessPolicy,
    build_operational_readiness_report,
)
from commodity_ml.intraday.quote_quality import summarize_quote_collection


def checks(status="NO_SIGNAL", decision=None, reasons=()):
    return pd.DataFrame(
        {
            "check_id": ["XAUUSD:P1:20260804T0930Z"],
            "instrument": ["XAUUSD"],
            "strategy": ["P1"],
            "signal_time": [pd.Timestamp("2026-08-04T09:30:00Z")],
            "status": [status],
            "decision": [decision],
            "reasons": [reasons],
        }
    )


def quote_quality(*, crossed=False, invalid_event=False):
    quotes = pd.DataFrame(
        {
            "instrument": ["XAU_USD"],
            "broker_time": ["2026-08-04T10:00:00Z"],
            "received_at": ["2026-08-04T10:00:00.020Z"],
            "bid": [2400.2 if crossed else 2400.0],
            "ask": [2400.1],
        }
    )
    events = pd.DataFrame(
        {"event_type": ["SERVER_ERROR" if invalid_event else "HEARTBEAT"]}
    )
    return summarize_quote_collection(quotes, events, schema_version=2)


def minimal_policy():
    return OperationalReadinessPolicy(
        minimum_london_session_days=1,
        minimum_entry_slots_per_pair=1,
        minimum_entry_slot_coverage=0.01,
        minimum_quotes_per_instrument=1,
    )


def test_complete_minimal_collection_is_operationally_ready():
    report = build_operational_readiness_report(
        checks(),
        quote_quality(),
        instrument_map={"XAUUSD": "XAU_USD"},
        strategies=("P1",),
        policy=minimal_policy(),
    )

    assert report["readiness"] == "OPERATIONALLY_READY"
    assert report["shadow"]["signal_count"] == 0
    assert report["shadow"]["pair_coverage"]["XAUUSD:P1"]["coverage"] == 0.04
    assert not report["integrity_issues"]


def test_integrity_fault_requires_review_and_block_reason_is_counted():
    report = build_operational_readiness_report(
        checks(
            status="ASSESSED",
            decision="BLOCK",
            reasons=("median_cost_above_0.10R",),
        ),
        quote_quality(invalid_event=True),
        instrument_map={"XAUUSD": "XAU_USD"},
        strategies=("P1",),
        policy=minimal_policy(),
    )

    assert report["readiness"] == "REVIEW"
    assert report["shadow"]["decision_counts"] == {"BLOCK": 1}
    assert report["shadow"]["reason_counts"] == {"median_cost_above_0.10R": 1}
    assert "unresolved_stream_invalidating_events" in report["integrity_issues"]


def test_default_policy_keeps_first_cycle_in_collecting_state():
    report = build_operational_readiness_report(
        checks(),
        quote_quality(),
        instrument_map={"XAUUSD": "XAU_USD"},
        strategies=("P1",),
    )

    assert report["readiness"] == "COLLECTING"
    assert "insufficient_london_session_days" in report["collection_issues"]
