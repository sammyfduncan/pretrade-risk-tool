"""Record raw IG BID/OFR/UTM ticks and connection events; never place orders."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import threading
import time

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import ig
from commodity_ml.intraday.ig_stream import (
    IGConnectionRecorderListener,
    stream_from_authenticated_session,
    tick_subscription,
)
from commodity_ml.intraday.quote_store import StreamingQuoteStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="Configured metal YAML; repeat to record both gold and silver",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/runtime/ig_stream_quotes.sqlite"),
    )
    parser.add_argument("--duration-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.duration_seconds < 1:
        raise SystemExit("--duration-seconds must be at least 1")

    load_dotenv()
    configs = [load_config(path) for path in args.config]
    epics: list[str] = []
    for cfg in configs:
        if not cfg.data.ig.epic:
            raise SystemExit(
                f"Streaming refused: {cfg.asset} data.ig.epic is unset. "
                "Discover and confirm the account-specific spot spread-bet epic first."
            )
        if cfg.data.ig.epic in epics:
            raise SystemExit(f"Streaming refused: duplicate epic {cfg.data.ig.epic}")
        epics.append(cfg.data.ig.epic)

    stop = threading.Event()

    def request_stop(signum, frame) -> None:
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    store = StreamingQuoteStore(args.database)
    stream = None
    subscriptions = []
    try:
        service = ig.create_service()
        session_data = service.create_session()
        account_id = str(session_data.get("currentAccountId") or "")
        endpoint = str(session_data.get("lightstreamerEndpoint") or "")
        if not account_id or not endpoint:
            raise RuntimeError("IG session omitted current account or stream endpoint")
        requested_account = os.environ.get("IG_ACC_NUMBER")
        if requested_account and requested_account != account_id:
            raise RuntimeError(
                "IG_ACC_NUMBER is not the session's current account; explicit "
                "account switching must be validated before streaming"
            )
        account = ig.fetch_account_snapshot(service, account_id=account_id)
        if account.status.upper() != "ENABLED":
            raise RuntimeError(f"IG account is not enabled: {account.status}")
        if account.account_type.upper() != "SPREADBET":
            raise RuntimeError(
                f"Version-one streaming requires SPREADBET, got {account.account_type}"
            )

        now = pd.Timestamp.now(tz="UTC")
        store.record_event(
            event_type="SESSION_AUTHENTICATED",
            event_time=now,
            detail=f"account_suffix={account_id[-4:]}; type={account.account_type}",
        )
        for cfg, epic in zip(configs, epics, strict=True):
            details = ig.fetch_market_details(service, epic)
            instrument = details.get("instrument") or {}
            snapshot = details.get("snapshot") or {}
            instrument_type = str(instrument.get("type") or "").upper()
            expiry = str(instrument.get("expiry") or "").upper()
            delay = int(float(snapshot.get("delayTime", 0)))
            streaming = bool(instrument.get("streamingPricesAvailable", False))
            if instrument_type != "COMMODITIES":
                raise RuntimeError(f"{epic} is not a COMMODITIES instrument")
            if expiry not in ("DFB", "-"):
                raise RuntimeError(f"{epic} is an expiring product: {expiry}")
            if delay != 0:
                raise RuntimeError(f"{epic} has delayed pricing: {delay}s")
            if not streaming:
                raise RuntimeError(f"{epic} does not advertise streaming prices")
            store.record_event(
                event_type="MARKET_VALIDATED",
                event_time=pd.Timestamp.now(tz="UTC"),
                epic=epic,
                detail=f"symbol={cfg.symbol}; type={instrument_type}; expiry={expiry}",
            )

        stream = stream_from_authenticated_session(
            service,
            lightstreamer_endpoint=endpoint,
            account_id=account_id,
        )
        stream.add_client_listener(IGConnectionRecorderListener(store))
        stream.ls_client.connect()
        for epic in epics:
            subscription = tick_subscription(epic, store)
            stream.subscribe(subscription)
            subscriptions.append(subscription)

        deadline = time.monotonic() + args.duration_seconds
        while not stop.is_set() and time.monotonic() < deadline:
            stop.wait(timeout=min(1.0, max(0.0, deadline - time.monotonic())))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"IG quote recording refused: {exc}") from exc
    finally:
        if stream is not None and stream.ls_client is not None:
            try:
                stream.disconnect()
            except Exception as exc:  # preserve recorder evidence on client shutdown faults
                store.record_event(
                    event_type="SERVER_ERROR",
                    event_time=pd.Timestamp.now(tz="UTC"),
                    detail=f"shutdown={type(exc).__name__}",
                )
        store.record_event(
            event_type="RECORDER_STOPPED",
            event_time=pd.Timestamp.now(tz="UTC"),
        )
        counts = store.counts()
        store.close()

    print(f"Recorded {counts['quotes']:,} quotes and {counts['events']:,} events.")
    print(f"Raw stream database -> {args.database.resolve()}")
    print("No order was created, amended, closed or submitted.")


if __name__ == "__main__":
    main()
