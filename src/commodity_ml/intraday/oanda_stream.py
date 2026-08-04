"""OANDA pricing-stream parser that only records quotes and stream evidence."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Iterable, Mapping

import pandas as pd

from commodity_ml.data.oanda import OANDAClient
from commodity_ml.intraday.quote_store import StreamingQuoteStore


def record_pricing_stream(
    client: OANDAClient,
    *,
    instruments: Iterable[str],
    store: StreamingQuoteStore,
    stop: threading.Event,
    duration_seconds: int,
) -> str:
    """Record one bounded stream; disconnects and malformed rows remain visible."""

    names = tuple(dict.fromkeys(str(value).strip() for value in instruments))
    if not names or any(not value for value in names):
        raise ValueError("stream instruments must be non-empty")
    if duration_seconds < 1:
        raise ValueError("duration_seconds must be positive")
    allowed = set(names)
    deadline = time.monotonic() + duration_seconds
    response = None
    disconnect_reason = "ERROR"
    try:
        response = client.open_pricing_stream(names)
        store.record_event(
            event_type="CONNECTED",
            event_time=pd.Timestamp.now(tz="UTC"),
            detail=f"environment={client.environment}; instruments={','.join(names)}",
        )
        for raw_line in response.iter_lines(decode_unicode=True):
            received_at = pd.Timestamp.now(tz="UTC")
            if stop.is_set():
                disconnect_reason = "STOP"
                break
            if time.monotonic() >= deadline:
                disconnect_reason = "DURATION"
                break
            if not raw_line:
                continue
            try:
                message = _mapping(json.loads(raw_line))
                kind = str(message.get("type") or "").upper()
                if kind == "HEARTBEAT":
                    store.record_event(
                        event_type="HEARTBEAT",
                        event_time=_timestamp(message.get("time")),
                    )
                    continue
                if kind != "PRICE":
                    raise ValueError(f"unexpected_message_type:{kind or 'missing'}")
                instrument = str(message.get("instrument") or "")
                if instrument not in allowed:
                    raise ValueError(f"unexpected_instrument:{instrument}")
                if str(message.get("status") or "tradeable").lower() != "tradeable":
                    raise ValueError(f"non_tradeable_price:{message.get('status')}")
                bids = message.get("bids") or []
                asks = message.get("asks") or []
                if not bids or not asks:
                    raise ValueError("missing_top_of_book")
                store.record_quote(
                    instrument=instrument,
                    broker_time=_timestamp(message.get("time")),
                    received_at=received_at,
                    bid=_mapping(bids[0]).get("price"),
                    ask=_mapping(asks[0]).get("price"),
                )
            except (json.JSONDecodeError, RuntimeError, TypeError, ValueError) as exc:
                store.record_event(
                    event_type="INVALID_UPDATE",
                    event_time=received_at,
                    detail=str(exc),
                )
        else:
            disconnect_reason = "EOF"
    except Exception as exc:
        store.record_event(
            event_type="SERVER_ERROR",
            event_time=pd.Timestamp.now(tz="UTC"),
            detail=f"{type(exc).__name__}:{str(exc)[:300]}",
        )
        raise
    finally:
        if response is not None:
            response.close()
        store.record_event(
            event_type=(
                "DISCONNECTED_CLEAN"
                if disconnect_reason in {"STOP", "DURATION"}
                else f"DISCONNECTED_{disconnect_reason}"
            ),
            event_time=pd.Timestamp.now(tz="UTC"),
        )
    return disconnect_reason


def record_pricing_stream_with_reconnect(
    client: OANDAClient,
    *,
    instruments: Iterable[str],
    store: StreamingQuoteStore,
    stop: threading.Event,
    duration_seconds: int,
    max_reconnects: int = 100,
    initial_backoff_seconds: float = 2.0,
    maximum_backoff_seconds: float = 60.0,
) -> None:
    """Record a bounded session and preserve every reconnect as evidence."""

    if duration_seconds < 1:
        raise ValueError("duration_seconds must be positive")
    if (
        isinstance(max_reconnects, bool)
        or not isinstance(max_reconnects, int)
        or max_reconnects < 0
    ):
        raise ValueError("max_reconnects must be a non-negative integer")
    if initial_backoff_seconds < 0 or maximum_backoff_seconds < initial_backoff_seconds:
        raise ValueError("invalid reconnect backoff")
    deadline = time.monotonic() + duration_seconds
    reconnects = 0
    while not stop.is_set():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        try:
            reason = record_pricing_stream(
                client,
                instruments=instruments,
                store=store,
                stop=stop,
                duration_seconds=max(1, math.ceil(remaining)),
            )
        except Exception:
            reason = "ERROR"
        if reason in {"STOP", "DURATION"} or stop.is_set():
            return
        if reconnects >= max_reconnects:
            store.record_event(
                event_type="RECONNECT_EXHAUSTED",
                event_time=pd.Timestamp.now(tz="UTC"),
                detail=f"attempts={reconnects}; last_reason={reason}",
            )
            raise RuntimeError("OANDA pricing stream reconnect budget exhausted")
        delay = min(
            maximum_backoff_seconds,
            initial_backoff_seconds * (2**min(reconnects, 10)),
        )
        reconnects += 1
        store.record_event(
            event_type="RECONNECT_SCHEDULED",
            event_time=pd.Timestamp.now(tz="UTC"),
            detail=f"attempt={reconnects}; delay_seconds={delay:g}; reason={reason}",
        )
        stop.wait(min(delay, max(0.0, deadline - time.monotonic())))


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("stream message must be an object")
    return value


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("stream timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC")
