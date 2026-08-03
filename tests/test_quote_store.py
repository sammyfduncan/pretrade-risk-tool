from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from commodity_ml.intraday.ig_stream import (
    IGConnectionRecorderListener,
    IGTickRecorderListener,
    TICK_FIELDS,
    stream_from_authenticated_session,
    tick_subscription,
)
from commodity_ml.intraday.quote_store import QuoteStoreError, StreamingQuoteStore


EPIC = "CS.D.TEST.GOLD.IP"
START = pd.Timestamp("2026-08-03 10:00:00", tz="UTC")


class FakeUpdate:
    def __init__(self, values: dict[str, str], item: str | None = None):
        self.values = values
        self.item = item or f"CHART:{EPIC}:TICK"

    def getItemName(self):
        return self.item

    def getValue(self, field):
        return self.values.get(field)


def connected_store(tmp_path) -> StreamingQuoteStore:
    store = StreamingQuoteStore(tmp_path / "quotes.sqlite")
    store.record_event(
        event_type="CONNECTED:STREAM-SENSING",
        event_time=START - pd.Timedelta(seconds=1),
    )
    return store


def test_store_preserves_raw_quotes_and_returns_unfilled_path(tmp_path):
    with connected_store(tmp_path) as store:
        for offset, bid in ((1, 100), (2, 99), (3, 101)):
            timestamp = START + pd.Timedelta(seconds=offset)
            store.record_quote(
                epic=EPIC,
                broker_time=timestamp,
                received_at=timestamp + pd.Timedelta(milliseconds=20),
                bid=bid,
                ask=bid + 0.2,
            )

        frame = store.quote_path(
            epic=EPIC,
            start=START,
            end=START + pd.Timedelta(seconds=3),
        )

        assert list(frame["bid"]) == [100, 99, 101]
        assert list(frame.index) == list(
            pd.date_range(START + pd.Timedelta(seconds=1), periods=3, freq="1s")
        )
        assert "received_at" in frame
        assert store.counts() == {"quotes": 3, "events": 1}


def test_disconnect_or_lost_update_invalidates_path(tmp_path):
    for event in ("DISCONNECTED", "LOST_UPDATES"):
        path = tmp_path / f"{event}.sqlite"
        with StreamingQuoteStore(path) as store:
            store.record_event(
                event_type="CONNECTED",
                event_time=START - pd.Timedelta(seconds=1),
            )
            store.record_quote(
                epic=EPIC,
                broker_time=START + pd.Timedelta(seconds=1),
                received_at=START + pd.Timedelta(seconds=1),
                bid=100,
                ask=100.2,
            )
            store.record_event(
                event_type=event,
                event_time=START + pd.Timedelta(seconds=2),
                epic=EPIC if event == "LOST_UPDATES" else None,
            )

            with pytest.raises(QuoteStoreError, match="invalidating_event"):
                store.quote_path(
                    epic=EPIC,
                    start=START,
                    end=START + pd.Timedelta(seconds=3),
                )


def test_path_requires_confirmed_connected_state_at_entry(tmp_path):
    with StreamingQuoteStore(tmp_path / "quotes.sqlite") as store:
        store.record_quote(
            epic=EPIC,
            broker_time=START + pd.Timedelta(seconds=1),
            received_at=START + pd.Timedelta(seconds=1),
            bid=100,
            ask=100.2,
        )
        with pytest.raises(QuoteStoreError, match="not_confirmed_connected"):
            store.quote_path(
                epic=EPIC,
                start=START,
                end=START + pd.Timedelta(seconds=2),
            )


def test_duplicate_broker_timestamps_are_preserved_not_repaired(tmp_path):
    with connected_store(tmp_path) as store:
        timestamp = START + pd.Timedelta(seconds=1)
        for bid in (100, 100.1):
            store.record_quote(
                epic=EPIC,
                broker_time=timestamp,
                received_at=timestamp,
                bid=bid,
                ask=bid + 0.2,
            )
        frame = store.quote_path(
            epic=EPIC,
            start=START,
            end=START + pd.Timedelta(seconds=2),
        )

        assert len(frame) == 2
        assert frame.index.has_duplicates


def test_tick_listener_uses_utm_as_explicit_utc_milliseconds(tmp_path):
    received = START + pd.Timedelta(milliseconds=50)
    with StreamingQuoteStore(tmp_path / "quotes.sqlite") as store:
        listener = IGTickRecorderListener(EPIC, store, clock=lambda: received)
        broker_time = START + pd.Timedelta(milliseconds=25)
        listener.onItemUpdate(
            FakeUpdate(
                {
                    "UTM": str(int(broker_time.timestamp() * 1000)),
                    "BID": "2400.1",
                    "OFR": "2400.4",
                }
            )
        )

        store.record_event(
            event_type="CONNECTED",
            event_time=START - pd.Timedelta(seconds=1),
        )
        frame = store.quote_path(
            epic=EPIC,
            start=START,
            end=START + pd.Timedelta(seconds=1),
        )
        assert frame.index[0] == broker_time
        assert frame["bid"].iloc[0] == pytest.approx(2400.1)
        assert frame["ask"].iloc[0] == pytest.approx(2400.4)


def test_invalid_tick_is_logged_instead_of_partially_recorded(tmp_path):
    with StreamingQuoteStore(tmp_path / "quotes.sqlite") as store:
        listener = IGTickRecorderListener(EPIC, store, clock=lambda: START)
        listener.onItemUpdate(FakeUpdate({"BID": "100", "OFR": "100.2"}))

        assert store.counts() == {"quotes": 0, "events": 1}


def test_subscription_uses_raw_tick_fields_and_connection_transitions_are_logged(tmp_path):
    with StreamingQuoteStore(tmp_path / "quotes.sqlite") as store:
        subscription = tick_subscription(EPIC, store)
        assert subscription.getItems() == [f"CHART:{EPIC}:TICK"]
        assert subscription.getFields() == list(TICK_FIELDS)

        listener = IGConnectionRecorderListener(store, clock=lambda: START)
        listener.onStatusChange("CONNECTED:STREAM-SENSING")
        listener.onStatusChange("DISCONNECTED:WILL-RETRY")
        assert store.counts()["events"] == 2


def test_stream_reuses_authenticated_tokens_without_second_session():
    service = SimpleNamespace(
        session=SimpleNamespace(
            headers={"CST": "cst-token", "X-SECURITY-TOKEN": "security-token"}
        )
    )

    stream = stream_from_authenticated_session(
        service,
        lightstreamer_endpoint="https://example.invalid",
        account_id="ACCOUNT1",
    )

    assert stream.acc_number == "ACCOUNT1"
    assert stream.ls_client.connectionDetails.getUser() == "ACCOUNT1"


def test_stream_refuses_missing_authenticated_tokens():
    service = SimpleNamespace(session=SimpleNamespace(headers={}))

    with pytest.raises(RuntimeError, match="missing stream tokens"):
        stream_from_authenticated_session(
            service,
            lightstreamer_endpoint="https://example.invalid",
            account_id="ACCOUNT1",
        )
