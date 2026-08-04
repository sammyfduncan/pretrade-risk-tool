"""OANDA pricing-stream parser that only records quotes and stream evidence."""

from __future__ import annotations

import json
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
) -> None:
    """Record one bounded stream; disconnects and malformed rows remain visible."""

    names = tuple(dict.fromkeys(str(value).strip() for value in instruments))
    if not names or any(not value for value in names):
        raise ValueError("stream instruments must be non-empty")
    if duration_seconds < 1:
        raise ValueError("duration_seconds must be positive")
    allowed = set(names)
    deadline = time.monotonic() + duration_seconds
    response = None
    try:
        response = client.open_pricing_stream(names)
        store.record_event(
            event_type="CONNECTED",
            event_time=pd.Timestamp.now(tz="UTC"),
            detail=f"environment={client.environment}; instruments={','.join(names)}",
        )
        for raw_line in response.iter_lines(decode_unicode=True):
            received_at = pd.Timestamp.now(tz="UTC")
            if stop.is_set() or time.monotonic() >= deadline:
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
            event_type="DISCONNECTED_CLEAN",
            event_time=pd.Timestamp.now(tz="UTC"),
        )


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("stream message must be an object")
    return value


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("stream timestamp must be timezone-aware")
    return timestamp.tz_convert("UTC")
