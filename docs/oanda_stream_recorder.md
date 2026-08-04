# OANDA executable-quote stream recorder

**Status:** implemented; bounded practice-account smoke validation is required
before its data may calibrate paper execution costs.

`scripts/record_oanda_quotes.py` connects to OANDA's v20 pricing stream and
stores every complete top-of-book update in append-only SQLite. It also records
connection, heartbeat, validation, invalid-update and shutdown events. The
recorder contains no order endpoint.

```bash
python3 scripts/record_oanda_quotes.py \
  --config config/gold.yaml \
  --config config/silver.yaml \
  --duration-seconds 300
```

Default database:

```text
data/runtime/oanda_stream_quotes.sqlite
```

The database uses schema version 2 and names instruments directly. A legacy IG
schema is never rewritten in place; retain it as historical evidence and start
a new OANDA database.

Export an uninterrupted path for one completed paper trade:

```bash
python3 scripts/export_oanda_quote_path.py \
  --database data/runtime/oanda_stream_quotes.sqlite \
  --instrument XAU_USD \
  --start 2026-08-04T10:00:00Z \
  --end 2026-08-04T12:00:00Z \
  --output data/processed/xau_trade_quote_path.parquet
```

An export is refused if connection state was not known at entry or if the
interval contains a disconnect, server error, invalid update or lost-update
evidence. Missing periods are never forward-filled.

OANDA's pricing stream samples at most four prices per second per instrument.
That is appropriate for M15/M30/H1 spread and execution-path measurement, but
it is not exchange order-book or tick-complete microstructure data. Databento
remains the research source for CME futures microstructure.
