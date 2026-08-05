"""Descriptive quality and spread summaries for prospective broker quotes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from commodity_ml.intraday.quote_store import INVALIDATING_EVENTS


@dataclass(frozen=True)
class InstrumentQuoteQuality:
    instrument: str
    first_broker_time: str
    last_broker_time: str
    quotes: int
    unique_broker_times: int
    duplicate_broker_times: int
    crossed_quotes: int
    median_spread_bp: float
    p95_spread_bp: float
    maximum_spread_bp: float
    median_receive_latency_ms: float
    p95_receive_latency_ms: float


@dataclass(frozen=True)
class QuoteCollectionQuality:
    schema_version: int
    total_quotes: int
    total_events: int
    connected_events: int
    clean_disconnect_events: int
    path_invalidating_events: int
    unexpected_invalidating_events: int
    recovered_invalidating_events: int
    unresolved_invalidating_events: int
    instruments: tuple[InstrumentQuoteQuality, ...]

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "instruments": [asdict(item) for item in self.instruments],
        }


def summarize_quote_collection(
    quotes: pd.DataFrame,
    events: pd.DataFrame,
    *,
    schema_version: int,
) -> QuoteCollectionQuality:
    """Summarize raw observations without filling gaps or claiming tradability."""

    required_quotes = {"instrument", "broker_time", "received_at", "bid", "ask"}
    if missing := sorted(required_quotes - set(quotes)):
        raise ValueError(f"quotes missing columns: {missing}")
    if "event_type" not in events:
        raise ValueError("events missing columns: ['event_type']")
    if schema_version < 1:
        raise ValueError("schema_version must be positive")

    normalized_events = events["event_type"].astype(str).str.upper()
    invalidating = normalized_events.map(
        lambda value: any(value.startswith(prefix) for prefix in INVALIDATING_EVENTS)
    )
    clean_disconnect = normalized_events.str.startswith("DISCONNECTED_CLEAN")
    unexpected = invalidating & ~clean_disconnect
    connected_positions = [
        position
        for position, value in enumerate(normalized_events)
        if value.startswith("CONNECTED")
    ]
    last_connected_position = connected_positions[-1] if connected_positions else -1
    unexpected_positions = [
        position for position, value in enumerate(unexpected) if bool(value)
    ]
    recovered = sum(position < last_connected_position for position in unexpected_positions)
    unresolved = len(unexpected_positions) - recovered
    instruments: list[InstrumentQuoteQuality] = []
    for instrument, group in quotes.groupby("instrument", sort=True):
        frame = group.copy()
        broker_time = pd.to_datetime(frame["broker_time"], utc=True, errors="raise")
        received_at = pd.to_datetime(frame["received_at"], utc=True, errors="raise")
        bid = pd.to_numeric(frame["bid"], errors="raise").astype(float)
        ask = pd.to_numeric(frame["ask"], errors="raise").astype(float)
        if (
            not np.isfinite(np.column_stack((bid, ask))).all()
            or (bid <= 0).any()
            or (ask <= 0).any()
        ):
            raise ValueError(f"{instrument} contains invalid quote prices")
        midpoint = (bid + ask) / 2
        spread_bp = (ask - bid) / midpoint * 10_000
        latency_ms = (received_at - broker_time).dt.total_seconds() * 1_000
        instruments.append(
            InstrumentQuoteQuality(
                instrument=str(instrument),
                first_broker_time=broker_time.min().isoformat(),
                last_broker_time=broker_time.max().isoformat(),
                quotes=len(frame),
                unique_broker_times=int(broker_time.nunique()),
                duplicate_broker_times=int(broker_time.duplicated().sum()),
                crossed_quotes=int((ask < bid).sum()),
                median_spread_bp=float(spread_bp.median()),
                p95_spread_bp=float(spread_bp.quantile(0.95)),
                maximum_spread_bp=float(spread_bp.max()),
                median_receive_latency_ms=float(latency_ms.median()),
                p95_receive_latency_ms=float(latency_ms.quantile(0.95)),
            )
        )
    return QuoteCollectionQuality(
        schema_version=schema_version,
        total_quotes=len(quotes),
        total_events=len(events),
        connected_events=int(normalized_events.str.startswith("CONNECTED").sum()),
        clean_disconnect_events=int(clean_disconnect.sum()),
        path_invalidating_events=int(invalidating.sum()),
        unexpected_invalidating_events=int(unexpected.sum()),
        recovered_invalidating_events=int(recovered),
        unresolved_invalidating_events=int(unresolved),
        instruments=tuple(instruments),
    )
