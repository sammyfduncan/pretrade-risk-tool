# IG executable-quote stream recorder

**Status:** recorder, outage ledger and exporter implemented and synthetic-tested;
first authenticated demo run pending  
**Authority:** authentication, market reads and price subscriptions only; no
order endpoint

## Recorded data

`scripts/record_ig_quotes.py` subscribes to `CHART:{epic}:TICK` in `DISTINCT`
mode and requests only:

- `BID`;
- `OFR`; and
- `UTM`, IG's millisecond update timestamp.

The listener parses `UTM` directly as UTC. It does not use `trading-ig`'s
higher-level ticker timestamp, which converts through the machine's local time
and returns a naive datetime in the installed library version.

Every complete update is appended to SQLite with both broker UTC time and local
UTC receipt time. Duplicate broker timestamps are preserved for later refusal;
quotes are never merged, resampled or replaced.

## Connection evidence

The same database records connection state, subscription state, server errors,
invalid updates and Lightstreamer lost-update callbacks. A trade-path export is
refused unless the stream was confirmed connected at entry and remained free
of disconnect, server-error, subscription-error, lost-update and invalid-update
events throughout the requested interval.

Reconnects remain visible. The recorder never fills the period between
disconnect and recovery.

## Account and product gates

Before connecting, the CLI:

1. authenticates one IG session;
2. requires the session's current account to match `IG_ACC_NUMBER` when set;
3. requires an enabled `SPREADBET` account;
4. reads every configured epic;
5. requires `COMMODITIES`, non-expiring `DFB`/`-`, zero quote delay and
   advertised streaming availability; and
6. refuses missing or duplicate epics.

It never searches for, guesses or automatically selects an epic.

## Commands

Once both account-specific configs are complete:

```bash
python3 scripts/record_ig_quotes.py \
  --config config/gold.yaml \
  --config config/silver.yaml \
  --duration-seconds 3600
```

The default raw database is `data/runtime/ig_stream_quotes.sqlite`, inside the
ignored runtime-data area.

After a paper trade closes, export its exact fill-to-exit interval:

```bash
python3 scripts/export_ig_quote_path.py \
  --database data/runtime/ig_stream_quotes.sqlite \
  --epic YOUR_CONFIRMED_EPIC \
  --start 2026-08-03T10:00:00Z \
  --end 2026-08-03T11:00:00Z \
  --output data/runtime/intent_quote_path.parquet
```

The exporter checks connection evidence and writes a sidecar with the exact
boundaries, row count, duplicates, maximum gap and output SHA-256. The outcome
builder then applies the stricter boundary/internal-gap and executable-side
rules in `docs/pretrade_quote_path.md`.

## Live validation still required

The synthetic implementation is not evidence that the user's actual products
emit these fields correctly. The first demo run must verify:

- each epic is the intended spot/daily-funded spread bet;
- the stream supplies complete `BID`, `OFR` and `UTM` values;
- broker timestamps are monotonic enough for unique path construction;
- receipt latency and update gaps are reasonable during the entry session;
- graceful shutdown and forced reconnect events are recorded; and
- no credentials or account identifiers beyond the redacted suffix enter
  reports or logs.

Until that run passes, the recorder is implemented but not operationally
approved for prospective evidence.
