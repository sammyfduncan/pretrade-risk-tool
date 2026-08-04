from __future__ import annotations

import json
import threading

import pandas as pd
import pytest

from commodity_ml.intraday.oanda_stream import (
    record_pricing_stream,
    record_pricing_stream_with_reconnect,
)
from commodity_ml.intraday.quote_store import QuoteStoreError, StreamingQuoteStore

INSTRUMENT = "XAU_USD"
START = pd.Timestamp("2026-08-03 10:00:00", tz="UTC")


def connected_store(tmp_path) -> StreamingQuoteStore:
    store = StreamingQuoteStore(tmp_path / "quotes.sqlite")
    store.record_event(event_type="CONNECTED", event_time=START - pd.Timedelta(seconds=1))
    return store


def test_store_preserves_raw_quotes_and_returns_unfilled_path(tmp_path):
    with connected_store(tmp_path) as store:
        for offset, bid in ((1, 100), (2, 99), (3, 101)):
            timestamp = START + pd.Timedelta(seconds=offset)
            store.record_quote(
                instrument=INSTRUMENT,
                broker_time=timestamp,
                received_at=timestamp + pd.Timedelta(milliseconds=20),
                bid=bid,
                ask=bid + 0.2,
            )
        frame = store.quote_path(
            instrument=INSTRUMENT,
            start=START,
            end=START + pd.Timedelta(seconds=3),
        )
        assert list(frame["bid"]) == [100, 99, 101]
        assert frame.index.is_monotonic_increasing
        assert store.counts() == {"quotes": 3, "events": 1}


def test_disconnect_or_invalid_update_invalidates_path(tmp_path):
    for event in ("DISCONNECTED", "INVALID_UPDATE"):
        with connected_store(tmp_path / event) as store:
            store.record_quote(
                instrument=INSTRUMENT,
                broker_time=START + pd.Timedelta(seconds=1),
                received_at=START + pd.Timedelta(seconds=1),
                bid=100,
                ask=100.2,
            )
            store.record_event(
                event_type=event,
                event_time=START + pd.Timedelta(seconds=2),
                instrument=INSTRUMENT if event == "INVALID_UPDATE" else None,
            )
            with pytest.raises(QuoteStoreError, match="invalidating_event"):
                store.quote_path(
                    instrument=INSTRUMENT,
                    start=START,
                    end=START + pd.Timedelta(seconds=3),
                )


def test_path_requires_confirmed_connected_state_at_entry(tmp_path):
    with StreamingQuoteStore(tmp_path / "quotes.sqlite") as store:
        store.record_quote(
            instrument=INSTRUMENT,
            broker_time=START + pd.Timedelta(seconds=1),
            received_at=START + pd.Timedelta(seconds=1),
            bid=100,
            ask=100.2,
        )
        with pytest.raises(QuoteStoreError, match="not_confirmed_connected"):
            store.quote_path(
                instrument=INSTRUMENT,
                start=START,
                end=START + pd.Timedelta(seconds=2),
            )


def test_duplicate_broker_timestamps_are_preserved_not_repaired(tmp_path):
    with connected_store(tmp_path) as store:
        timestamp = START + pd.Timedelta(seconds=1)
        for bid in (100, 100.1):
            store.record_quote(
                instrument=INSTRUMENT,
                broker_time=timestamp,
                received_at=timestamp,
                bid=bid,
                ask=bid + 0.2,
            )
        frame = store.quote_path(
            instrument=INSTRUMENT,
            start=START,
            end=START + pd.Timedelta(seconds=2),
        )
        assert len(frame) == 2
        assert frame.index.has_duplicates


class FakeResponse:
    def __init__(self, messages):
        self.messages = messages
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        return iter(json.dumps(message) for message in self.messages)

    def close(self):
        self.closed = True


class FakeClient:
    environment = "practice"

    def __init__(self, messages):
        self.response = FakeResponse(messages)

    def open_pricing_stream(self, instruments):
        assert tuple(instruments) == (INSTRUMENT,)
        return self.response


def test_oanda_stream_records_price_and_heartbeat_without_order_authority(tmp_path):
    messages = [
        {"type": "HEARTBEAT", "time": START.isoformat()},
        {
            "type": "PRICE",
            "instrument": INSTRUMENT,
            "time": (START + pd.Timedelta(seconds=1)).isoformat(),
            "bids": [{"price": "2400.1"}],
            "asks": [{"price": "2400.4"}],
        },
    ]
    client = FakeClient(messages)
    with StreamingQuoteStore(tmp_path / "stream.sqlite") as store:
        record_pricing_stream(
            client,
            instruments=[INSTRUMENT],
            store=store,
            stop=threading.Event(),
            duration_seconds=5,
        )
        assert store.counts() == {"quotes": 1, "events": 3}
    assert client.response.closed


def test_invalid_oanda_stream_row_is_logged_not_partially_recorded(tmp_path):
    client = FakeClient(
        [{"type": "PRICE", "instrument": INSTRUMENT, "time": START.isoformat()}]
    )
    with StreamingQuoteStore(tmp_path / "stream.sqlite") as store:
        record_pricing_stream(
            client,
            instruments=[INSTRUMENT],
            store=store,
            stop=threading.Event(),
            duration_seconds=5,
        )
        assert store.counts() == {"quotes": 0, "events": 3}


class ReconnectingFakeClient:
    environment = "practice"

    def __init__(self):
        self.calls = 0

    def open_pricing_stream(self, instruments):
        self.calls += 1
        return FakeResponse([])


def test_oanda_stream_supervisor_reconnects_and_fails_boundedly(tmp_path):
    client = ReconnectingFakeClient()
    with StreamingQuoteStore(tmp_path / "reconnect.sqlite") as store:
        with pytest.raises(RuntimeError, match="reconnect budget exhausted"):
            record_pricing_stream_with_reconnect(
                client,
                instruments=[INSTRUMENT],
                store=store,
                stop=threading.Event(),
                duration_seconds=5,
                max_reconnects=1,
                initial_backoff_seconds=0,
                maximum_backoff_seconds=0,
            )
        assert client.calls == 2
        assert store.counts()["events"] == 6
