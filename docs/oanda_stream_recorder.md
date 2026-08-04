# OANDA executable-quote stream recorder

**Status:** implemented and smoke-validated against the practice account on
2026-08-04. The smoke run proves connectivity and storage only; a multi-week
prospective sample is still required before its data may calibrate paper
execution costs.

`scripts/record_oanda_quotes.py` connects to OANDA's v20 pricing stream and
stores every complete top-of-book update in append-only SQLite. It also records
connection, heartbeat, validation, invalid-update and shutdown events. The
recorder contains no order endpoint. Unexpected EOFs and network errors are
recorded and retried with bounded exponential backoff; every reconnect remains
visible in the event ledger rather than being treated as continuous coverage.

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

Inspect accumulated coverage and raw spread quality without making a network
request:

```bash
python3 scripts/summarize_oanda_quotes.py \
  --database data/runtime/oanda_stream_quotes.sqlite \
  --output reports/oanda_quote_collection_quality.json
```

This summary is descriptive. It reports duplicates, crossed quotes, receipt
latency and invalidating stream events, but it cannot pass the cost/admission
gate by itself. Do not delete or forward-fill outages.

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

## Bounded source parity

Never overwrite the multi-year Dukascopy research artifact with a short parity
window. Fetch both sides to explicit snapshot paths, then pass those paths to
`scripts/compare_sources.py`. The fetcher now refuses to replace any existing
artifact unless `--overwrite` is explicitly supplied.

The first bounded M30 comparison on 2026-08-04 used 275 matched bars from
2026-07-27 through 2026-08-03. Gold passed with return correlation `0.9985`;
silver passed with `0.9926`. Both had 100% timestamp coverage and a best shift
of zero bars. This validates feed transfer for that bounded window only, not a
trading edge or future parity.
