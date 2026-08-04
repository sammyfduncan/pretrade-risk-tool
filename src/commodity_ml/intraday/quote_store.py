"""Crash-resilient broker quote/event storage with explicit outage refusal."""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 2
INVALIDATING_EVENTS = (
    "DISCONNECTED",
    "SERVER_ERROR",
    "SUBSCRIPTION_ERROR",
    "LOST_UPDATES",
    "INVALID_UPDATE",
)


class QuoteStoreError(RuntimeError):
    """The recorded stream is incomplete or internally inconsistent."""


class StreamingQuoteStore:
    """Append raw top-of-book updates and connection events to SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock:
            existing_version = int(
                self._connection.execute("PRAGMA user_version").fetchone()[0]
            )
            if existing_version not in (0, SCHEMA_VERSION):
                self._connection.close()
                raise QuoteStoreError(
                    f"quote_store_schema_mismatch:{existing_version}->{SCHEMA_VERSION}; "
                    "retain the legacy database as evidence and start a new OANDA database"
                )
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS quotes (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    broker_time TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    bid REAL NOT NULL,
                    ask REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS quotes_instrument_time_idx
                    ON quotes(instrument, broker_time);
                CREATE TABLE IF NOT EXISTS stream_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT,
                    event_time TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    detail TEXT
                );
                CREATE INDEX IF NOT EXISTS stream_events_time_idx
                    ON stream_events(event_time);
                """
            )
            self._connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def record_quote(
        self,
        *,
        instrument: str,
        broker_time: Any,
        received_at: Any,
        bid: Any,
        ask: Any,
    ) -> int:
        """Append one complete raw quote; never merge or replace a prior row."""

        if not instrument:
            raise ValueError("quote instrument must be non-empty")
        broker_timestamp = _utc(broker_time, "broker_time")
        received_timestamp = _utc(received_at, "received_at")
        bid_value = _positive_number(bid, "bid")
        ask_value = _positive_number(ask, "ask")
        if ask_value < bid_value:
            raise ValueError("cannot record a crossed quote")
        with self._lock:
            cursor = self._connection.execute(
                """INSERT INTO quotes
                   (instrument, broker_time, received_at, bid, ask)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    instrument,
                    broker_timestamp.isoformat(),
                    received_timestamp.isoformat(),
                    bid_value,
                    ask_value,
                ),
            )
            return int(cursor.lastrowid)

    def record_event(
        self,
        *,
        event_type: str,
        event_time: Any,
        instrument: str | None = None,
        detail: str | None = None,
    ) -> int:
        """Append a connection/subscription event without secret-bearing data."""

        normalized = str(event_type).strip().upper()
        if not normalized:
            raise ValueError("event_type must be non-empty")
        timestamp = _utc(event_time, "event_time")
        safe_detail = None if detail is None else str(detail)[:500]
        with self._lock:
            cursor = self._connection.execute(
                """INSERT INTO stream_events
                   (instrument, event_time, event_type, detail)
                   VALUES (?, ?, ?, ?)""",
                (instrument, timestamp.isoformat(), normalized, safe_detail),
            )
            return int(cursor.lastrowid)

    def quote_path(
        self,
        *,
        instrument: str,
        start: Any,
        end: Any,
    ) -> pd.DataFrame:
        """Read an untouched path, refusing any known outage/lost-update event."""

        if not instrument:
            raise ValueError("instrument must be non-empty")
        start_time = _utc(start, "start")
        end_time = _utc(end, "end")
        if end_time <= start_time:
            raise ValueError("end must be after start")
        with self._lock:
            prior = self._connection.execute(
                """SELECT event_type, detail FROM stream_events
                   WHERE event_time <= ? AND instrument IS NULL
                     AND (event_type LIKE 'CONNECTED%' OR event_type LIKE 'DISCONNECTED%')
                   ORDER BY event_time DESC, sequence DESC LIMIT 1""",
                (start_time.isoformat(),),
            ).fetchone()
            events = self._connection.execute(
                """SELECT event_time, event_type, detail FROM stream_events
                   WHERE event_time > ? AND event_time <= ?
                     AND (instrument IS NULL OR instrument = ?)
                   ORDER BY event_time, sequence""",
                (start_time.isoformat(), end_time.isoformat(), instrument),
            ).fetchall()
            rows = self._connection.execute(
                """SELECT broker_time, received_at, bid, ask, sequence
                   FROM quotes WHERE instrument = ? AND broker_time >= ? AND broker_time <= ?
                   ORDER BY broker_time, sequence""",
                (instrument, start_time.isoformat(), end_time.isoformat()),
            ).fetchall()

        if prior is None or not str(prior[0]).startswith("CONNECTED"):
            raise QuoteStoreError("stream_not_confirmed_connected_at_path_start")
        invalid = [
            (timestamp, kind, detail)
            for timestamp, kind, detail in events
            if any(str(kind).startswith(prefix) for prefix in INVALIDATING_EVENTS)
        ]
        if invalid:
            kinds = ",".join(str(row[1]) for row in invalid)
            raise QuoteStoreError(f"stream_path_contains_invalidating_event:{kinds}")
        if not rows:
            raise QuoteStoreError("stream_path_contains_no_quotes")
        frame = pd.DataFrame(
            rows,
            columns=("broker_time", "received_at", "bid", "ask", "sequence"),
        )
        frame["received_at"] = pd.to_datetime(frame["received_at"], utc=True)
        frame.index = pd.DatetimeIndex(
            pd.to_datetime(frame.pop("broker_time"), utc=True), name="time"
        )
        return frame

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def counts(self) -> dict[str, int]:
        with self._lock:
            quote_count = self._connection.execute("SELECT COUNT(*) FROM quotes").fetchone()[0]
            event_count = self._connection.execute(
                "SELECT COUNT(*) FROM stream_events"
            ).fetchone()[0]
        return {"quotes": int(quote_count), "events": int(event_count)}

    def __enter__(self) -> "StreamingQuoteStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _positive_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _utc(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
