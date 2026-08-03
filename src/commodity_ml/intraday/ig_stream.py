"""Thin IG Lightstreamer listeners that only append quotes and stream events."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from lightstreamer.client import (
    ClientListener,
    LightstreamerClient,
    Subscription,
    SubscriptionListener,
)
from trading_ig import IGStreamService

from commodity_ml.intraday.quote_store import StreamingQuoteStore


TICK_FIELDS = ("BID", "OFR", "UTM")


def stream_from_authenticated_session(
    service: Any,
    *,
    lightstreamer_endpoint: str,
    account_id: str,
) -> IGStreamService:
    """Reuse one REST session instead of creating a second authenticated session."""

    if not lightstreamer_endpoint or not account_id:
        raise ValueError("Lightstreamer endpoint and account id must be non-empty")
    cst = service.session.headers.get("CST")
    security = service.session.headers.get("X-SECURITY-TOKEN")
    if not cst or not security:
        raise RuntimeError("authenticated IG session is missing stream tokens")
    stream = IGStreamService(service)
    stream.lightstreamerEndpoint = lightstreamer_endpoint
    stream.acc_number = account_id
    stream.ls_client = LightstreamerClient(lightstreamer_endpoint, None)
    stream.ls_client.connectionDetails.setUser(account_id)
    stream.ls_client.connectionDetails.setPassword(f"CST-{cst}|XST-{security}")
    return stream


def tick_subscription(epic: str, store: StreamingQuoteStore) -> Subscription:
    """Create a DISTINCT raw-tick subscription for one already-confirmed epic."""

    if not epic:
        raise ValueError("streaming epic must be non-empty")
    subscription = Subscription(
        mode="DISTINCT",
        items=[f"CHART:{epic}:TICK"],
        fields=list(TICK_FIELDS),
    )
    subscription.addListener(IGTickRecorderListener(epic, store))
    return subscription


class IGTickRecorderListener(SubscriptionListener):
    """Persist complete BID/OFR/UTM values without using naive local time."""

    def __init__(
        self,
        epic: str,
        store: StreamingQuoteStore,
        *,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self.epic = epic
        self.store = store
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))

    def onItemUpdate(self, update: Any) -> None:  # Lightstreamer callback name
        received_at = self.clock()
        try:
            item = str(update.getItemName())
            if item != f"CHART:{self.epic}:TICK":
                raise ValueError(f"unexpected_item:{item}")
            raw_utm = update.getValue("UTM")
            if raw_utm in (None, ""):
                raise ValueError("missing_UTM")
            broker_time = pd.Timestamp(int(raw_utm), unit="ms", tz="UTC")
            self.store.record_quote(
                epic=self.epic,
                broker_time=broker_time,
                received_at=received_at,
                bid=update.getValue("BID"),
                ask=update.getValue("OFR"),
            )
        except (TypeError, ValueError) as exc:
            self.store.record_event(
                event_type="INVALID_UPDATE",
                event_time=received_at,
                epic=self.epic,
                detail=str(exc),
            )

    def onSubscription(self) -> None:
        self.store.record_event(
            event_type="SUBSCRIBED",
            event_time=self.clock(),
            epic=self.epic,
        )

    def onSubscriptionError(self, code: int, message: str) -> None:
        self.store.record_event(
            event_type="SUBSCRIPTION_ERROR",
            event_time=self.clock(),
            epic=self.epic,
            detail=f"code={code}; {message}",
        )

    def onItemLostUpdates(self, item_name: str, lost_updates: int) -> None:
        self.store.record_event(
            event_type="LOST_UPDATES",
            event_time=self.clock(),
            epic=self.epic,
            detail=f"item={item_name}; count={lost_updates}",
        )

    def onUnsubscription(self) -> None:
        self.store.record_event(
            event_type="UNSUBSCRIBED",
            event_time=self.clock(),
            epic=self.epic,
        )


class IGConnectionRecorderListener(ClientListener):
    """Record every Lightstreamer state transition; never conceal reconnects."""

    def __init__(
        self,
        store: StreamingQuoteStore,
        *,
        clock: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        self.store = store
        self.clock = clock or (lambda: pd.Timestamp.now(tz="UTC"))

    def onStatusChange(self, status: str) -> None:
        normalized = str(status).upper()
        event_type = "CONNECTED" if normalized.startswith("CONNECTED") else "DISCONNECTED"
        self.store.record_event(
            event_type=event_type,
            event_time=self.clock(),
            detail=normalized,
        )

    def onServerError(self, error_code: int, error_message: str) -> None:
        self.store.record_event(
            event_type="SERVER_ERROR",
            event_time=self.clock(),
            detail=f"code={error_code}; {error_message}",
        )
