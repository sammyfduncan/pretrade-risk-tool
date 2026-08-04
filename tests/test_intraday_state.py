"""Persistent paper state survives restarts and fails closed on ambiguity."""

import json

import pandas as pd
import pytest

from commodity_ml.intraday.paper import PaperOrderIntent
from commodity_ml.intraday.state import PaperStateError, PaperStateStore


NOW = pd.Timestamp("2026-01-09 16:00", tz="UTC")


def intent(intent_id: str = "P1:2026-01-09T16:00Z") -> PaperOrderIntent:
    return PaperOrderIntent(
        intent_id=intent_id,
        strategy="P1",
        broker_instrument="XAU_USD",
        account_id="ABC123",
        account_currency="GBP",
        direction="BUY",
        size=1.2,
        reference_price=2500.0,
        stop_level=2490.0,
        target_level=2515.0,
        created_at=NOW,
    )


def test_state_survives_restart_and_blocks_duplicate_intent(tmp_path):
    path = tmp_path / "paper-state.json"
    first_process = PaperStateStore(path)
    initialized = first_process.initialize(
        account_id="ABC123",
        equity=100_000,
        now=NOW,
    )
    assert initialized.revision == 0
    reserved = first_process.reserve_intent(intent())
    assert reserved.intents[0].status == "prepared"

    restarted = PaperStateStore(path)
    loaded = restarted.load()
    assert loaded.runtime_state().submitted_intent_ids == (intent().intent_id,)
    with pytest.raises(PaperStateError, match="duplicate_intent_id"):
        restarted.reserve_intent(intent())


def test_confirmed_intent_updates_exact_position_state_idempotently(tmp_path):
    store = PaperStateStore(tmp_path / "paper-state.json")
    store.initialize(account_id="ABC123", equity=100_000, now=NOW)
    store.reserve_intent(intent())
    submitted = store.mark_submitted(
        intent_id=intent().intent_id,
        deal_reference="DEAL-1",
        now=NOW + pd.Timedelta(seconds=1),
    )
    assert submitted.intents[0].status == "submitted"
    confirmed = store.mark_confirmed(
        intent_id=intent().intent_id,
        deal_reference="DEAL-1",
        position_id="POSITION-1",
        now=NOW + pd.Timedelta(seconds=2),
    )
    repeated = store.mark_confirmed(
        intent_id=intent().intent_id,
        deal_reference="DEAL-1",
        position_id="POSITION-1",
        now=NOW + pd.Timedelta(seconds=3),
    )

    assert confirmed.entries_today == 1
    assert confirmed.expected_position_ids == ("POSITION-1",)
    assert repeated.revision == confirmed.revision
    store.reconcile(
        account_id="ABC123",
        broker_position_ids=("POSITION-1",),
    )
    with pytest.raises(PaperStateError, match="reconciliation_mismatch"):
        store.reconcile(
            account_id="ABC123",
            broker_position_ids=("OTHER",),
        )
    closed = store.mark_position_closed(
        position_id="POSITION-1",
        now=NOW + pd.Timedelta(seconds=4),
    )
    assert closed.expected_position_ids == ()
    killed = store.set_kill_switch(
        active=True,
        now=NOW + pd.Timedelta(seconds=5),
    )
    assert PaperStateStore(store.path).load().runtime_state().kill_switch_active
    assert killed.kill_switch_active


def test_initialization_refuses_to_adopt_unknown_broker_position(tmp_path):
    store = PaperStateStore(tmp_path / "paper-state.json")
    with pytest.raises(PaperStateError, match="requires_flat_broker"):
        store.initialize(
            account_id="ABC123",
            equity=100_000,
            now=NOW,
            broker_position_ids=("UNKNOWN",),
        )


def test_day_and_week_boundaries_reset_once_in_london_time(tmp_path):
    store = PaperStateStore(tmp_path / "paper-state.json")
    store.initialize(account_id="ABC123", equity=100_000, now=NOW)
    store.reserve_intent(intent())
    store.mark_confirmed(
        intent_id=intent().intent_id,
        deal_reference="DEAL-1",
        position_id="POSITION-1",
        now=NOW,
    )
    monday = pd.Timestamp("2026-01-12 07:00", tz="UTC")
    rolled = store.roll_boundaries(
        account_id="ABC123",
        equity=99_500,
        now=monday,
    )
    repeated = store.roll_boundaries(
        account_id="ABC123",
        equity=99_000,
        now=monday + pd.Timedelta(minutes=1),
    )

    assert rolled.entries_today == 0
    assert rolled.day_start_equity == 99_500
    assert rolled.week_start_equity == 99_500
    assert repeated.day_start_equity == 99_500
    assert repeated.week_start_equity == 99_500


def test_corrupt_or_unknown_schema_is_refused(tmp_path):
    path = tmp_path / "paper-state.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(PaperStateError, match="corrupt"):
        PaperStateStore(path).load()

    payload = {
        "schema_version": 999,
        "account_id": "ABC123",
        "revision": 0,
        "updated_at": NOW.isoformat(),
        "london_session_date": "2026-01-09",
        "london_week_start": "2026-01-05",
        "day_start_equity": 100_000,
        "week_start_equity": 100_000,
        "entries_today": 0,
        "expected_position_ids": [],
        "kill_switch_active": False,
        "intents": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PaperStateError, match="corrupt"):
        PaperStateStore(path).load()
