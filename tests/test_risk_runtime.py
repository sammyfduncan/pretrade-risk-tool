from __future__ import annotations

import pandas as pd
import pytest

from commodity_ml.data.broker import BrokerAccountSnapshot
from commodity_ml.intraday.risk_runtime import roll_and_reconcile_paper_runtime
from commodity_ml.intraday.state import PaperStateError, PaperStateStore


FRIDAY = pd.Timestamp("2026-08-07 16:00", tz="UTC")


def account(account_id: str = "ABC123", equity: float = 99_000) -> BrokerAccountSnapshot:
    return BrokerAccountSnapshot(
        account_id=account_id,
        currency="GBP",
        account_type="SPREADBET",
        status="ENABLED",
        balance=equity,
        open_pnl=0,
        equity=equity,
        available=90_000,
    )


def test_runtime_reconciles_and_rolls_day_week_anchors_once(tmp_path):
    store = PaperStateStore(tmp_path / "paper-state.json")
    store.initialize(account_id="ABC123", equity=100_000, now=FRIDAY)
    monday = pd.Timestamp("2026-08-10 07:00", tz="UTC")

    runtime = roll_and_reconcile_paper_runtime(
        store,
        account=account(equity=99_000),
        broker_position_ids=(),
        now=monday,
    )
    repeated = roll_and_reconcile_paper_runtime(
        store,
        account=account(equity=98_000),
        broker_position_ids=(),
        now=monday + pd.Timedelta(minutes=1),
    )

    assert runtime.day_start_equity == 99_000
    assert runtime.week_start_equity == 99_000
    assert repeated.day_start_equity == 99_000
    assert repeated.week_start_equity == 99_000
    assert repeated.risk_inputs()["kill_switch_active"] is False
    assert repeated.state_revision > runtime.state_revision


def test_account_or_position_mismatch_fails_before_state_mutation(tmp_path):
    store = PaperStateStore(tmp_path / "paper-state.json")
    original = store.initialize(account_id="ABC123", equity=100_000, now=FRIDAY)

    with pytest.raises(PaperStateError, match="account_mismatch"):
        roll_and_reconcile_paper_runtime(
            store,
            account=account(account_id="OTHER"),
            broker_position_ids=(),
            now=FRIDAY + pd.Timedelta(minutes=1),
        )
    with pytest.raises(PaperStateError, match="reconciliation_mismatch"):
        roll_and_reconcile_paper_runtime(
            store,
            account=account(),
            broker_position_ids=("UNKNOWN",),
            now=FRIDAY + pd.Timedelta(minutes=1),
        )

    assert store.load().revision == original.revision


def test_kill_switch_flows_into_risk_inputs(tmp_path):
    store = PaperStateStore(tmp_path / "paper-state.json")
    store.initialize(account_id="ABC123", equity=100_000, now=FRIDAY)
    store.set_kill_switch(active=True, now=FRIDAY + pd.Timedelta(seconds=1))

    runtime = roll_and_reconcile_paper_runtime(
        store,
        account=account(),
        broker_position_ids=(),
        now=FRIDAY + pd.Timedelta(minutes=1),
    )

    assert runtime.kill_switch_active is True
    assert runtime.risk_inputs()["kill_switch_active"] is True
