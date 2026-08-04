from copy import deepcopy

import pandas as pd
import pytest

from commodity_ml.data.broker import BrokerMarketQuote
from commodity_ml.intraday.execution import TradeCandidate
from commodity_ml.intraday.shadow import (
    ShadowCheck,
    ShadowCheckStore,
    ShadowLedgerError,
    latest_closed_candidates,
    shadow_check_id,
    trade_intent_from_candidate,
)


SIGNAL = pd.Timestamp("2026-08-04T09:30:00Z")
DECISION = SIGNAL + pd.Timedelta(minutes=30)


def candidate(strategy="P1"):
    return TradeCandidate(
        signal_time=SIGNAL,
        side="long",
        stop_distance_pct=None,
        target_distance_pct=None,
        max_holding_bars=6,
        strategy=strategy,
        stop_distance_abs=20,
        target_r_multiple=1.5,
    )


def test_shadow_intent_has_stable_id_and_frozen_atr_geometry():
    quote = BrokerMarketQuote(
        instrument="XAU_USD",
        bid=2399.8,
        ask=2400.2,
        timestamp=DECISION,
        market_status="TRADEABLE",
    )

    intent = trade_intent_from_candidate(
        candidate(), symbol="XAUUSD", quote=quote, assessed_at=DECISION
    )

    assert intent["intent_id"] == "shadow-xauusd-p1-20260804T0930Z"
    assert intent["entry_price"] == pytest.approx(2400)
    assert intent["stop_price"] == pytest.approx(2380)
    assert intent["target_price"] == pytest.approx(2430)
    assert intent["requested_size"] is None
    assert intent["holding_period_minutes"] == 180


def test_latest_closed_candidates_only_accepts_newest_signal(monkeypatch):
    index = pd.DatetimeIndex(
        [SIGNAL - pd.Timedelta(minutes=30), SIGNAL], name="time"
    )
    frame = pd.DataFrame(
        {"decision_time": index + pd.Timedelta(minutes=30)}, index=index
    )
    monkeypatch.setattr(
        "commodity_ml.intraday.shadow.generate_candidates",
        lambda history, strategy: [candidate(strategy)],
    )

    signal, output = latest_closed_candidates(
        frame, ("P1", "P2"), now=DECISION + pd.Timedelta(minutes=5)
    )

    assert signal == SIGNAL
    assert output["P1"].strategy == "P1"
    assert output["P2"].strategy == "P2"

    with pytest.raises(RuntimeError, match="stale_latest_completed"):
        latest_closed_candidates(
            frame, ("P1",), now=DECISION + pd.Timedelta(minutes=31)
        )


def test_shadow_check_store_is_idempotent_and_collision_safe(tmp_path):
    store = ShadowCheckStore(tmp_path / "shadow.sqlite")
    check = ShadowCheck(
        check_id=shadow_check_id("XAUUSD", "P1", SIGNAL),
        instrument="XAUUSD",
        strategy="P1",
        signal_time=SIGNAL,
        checked_at=DECISION,
        status="ASSESSED",
        intent_id="shadow-xauusd-p1-20260804T0930Z",
        decision="BLOCK",
        reasons=("median_cost_above_0.10R",),
    )

    assert store.record(check) is True
    assert store.record(check) is False
    assert store.check_result(check.check_id) == (
        "ASSESSED",
        "BLOCK",
        ("median_cost_above_0.10R",),
    )
    assert store.summary()["decision_counts"] == {"BLOCK": 1}

    changed = ShadowCheck(
        **{**check.__dict__, "reasons": ("changed_after_recording",)}
    )
    with pytest.raises(ShadowLedgerError, match="collision"):
        store.record(changed)
