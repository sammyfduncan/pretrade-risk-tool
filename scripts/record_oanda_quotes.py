"""Record raw OANDA v20 bid/ask updates and stream events; never place orders."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import threading

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.intraday.oanda_stream import record_pricing_stream
from commodity_ml.intraday.quote_store import StreamingQuoteStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/runtime/oanda_stream_quotes.sqlite"),
    )
    parser.add_argument("--duration-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.duration_seconds < 1:
        raise SystemExit("--duration-seconds must be at least 1")

    load_dotenv()
    configs = [load_config(path) for path in args.config]
    instruments = [cfg.data.oanda.instrument for cfg in configs]
    if any(not value for value in instruments):
        raise SystemExit("Streaming refused: every config must set data.oanda.instrument")
    if len(instruments) != len(set(instruments)):
        raise SystemExit("Streaming refused: duplicate OANDA instrument")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop.set())
    store = StreamingQuoteStore(args.database)
    try:
        client = oanda.get_client()
        account = oanda.fetch_account_snapshot(
            client, account_type=configs[0].data.oanda.account_type
        )
        pricing = client.pricing(instruments)
        quotes = oanda.quotes_from_pricing(pricing)
        metadata_rows = client.account_instruments(instruments)
        if set(quotes) != set(instruments) or {row.get("name") for row in metadata_rows} != set(instruments):
            raise RuntimeError("OANDA omitted a configured stream instrument")
        store.record_event(
            event_type="SESSION_AUTHENTICATED",
            event_time=pd.Timestamp.now(tz="UTC"),
            detail=(
                f"account_suffix={account.account_id[-4:]}; "
                f"type={account.account_type}; environment={client.environment}"
            ),
        )
        for row in metadata_rows:
            name = str(row.get("name"))
            if str(row.get("type") or "").upper() != "METAL":
                raise RuntimeError(f"{name} is not an OANDA METAL")
            quotes[name].validate_for_entry(now=pd.Timestamp.now(tz="UTC"))
            store.record_event(
                event_type="MARKET_VALIDATED",
                event_time=pd.Timestamp.now(tz="UTC"),
                instrument=name,
                detail=f"display={row.get('displayName')}; type=METAL",
            )
        record_pricing_stream(
            client,
            instruments=instruments,
            store=store,
            stop=stop,
            duration_seconds=args.duration_seconds,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"OANDA quote recording refused: {exc}") from exc
    finally:
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
