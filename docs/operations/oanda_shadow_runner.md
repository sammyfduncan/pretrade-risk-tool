# OANDA operational shadow runner

**Status:** implemented and one-cycle practice-account smoke validated on
2026-08-04. The first cycle evaluated six asset/strategy checks and correctly
recorded six `NO_SIGNAL` observations. It created no broker order.

**Product role:** optional strategy-integration and operational diagnostics.
This runner is not the direction-balanced prospective validation design and is
not required to complete the standalone pre-trade risk evaluator.

The shadow runner connects the frozen P1/P2/P3 mathematical rules to the
read-only OANDA risk adapter. It exists to collect prospective operational
evidence before any order adapter is considered.

It does the following on each cycle:

1. reconciles the persistent account and open-trade state exactly;
2. downloads recent OANDA H1 and M30 candles;
3. removes incomplete candles and builds causal completed-H1 context;
4. checks P1, P2 and P3 only on the newest completed M30 signal bar;
5. converts a signal's frozen ATR stop, target multiple and timeout into a
   hypothetical market intent;
6. evaluates account, session, event, spread, sizing, margin, loss and
   portfolio gates; and
7. stores both the comprehensive assessment and the no-signal/assessment
   lifecycle record immutably.

It does **not** call ML, change a signal's direction, manufacture a trade when
there is no signal, simulate a broker fill, or create/amend/close an order.

## State initialization

Initialization is allowed only when OANDA confirms that the account is flat:

```bash
python3 scripts/initialize_oanda_paper_state.py \
  --config config/gold.yaml \
  --state data/runtime/oanda_paper_state.json
```

The state holds day/week equity anchors, entry count, expected position IDs and
the kill switch. Unknown broker positions fail reconciliation and are never
adopted automatically.

## Run modes

Run one assessment cycle:

```bash
python3 scripts/run_oanda_shadow.py
```

Run a bounded eight-hour collection session, polling every five minutes:

```bash
python3 scripts/run_oanda_shadow.py \
  --duration-seconds 28800 \
  --poll-seconds 300
```

Inspect the local lifecycle summary without contacting OANDA:

```bash
python3 scripts/run_oanda_shadow.py --summary-only
```

Combine the lifecycle ledger with the raw executable-quote database:

```bash
python3 scripts/report_oanda_operations.py
```

The combined report preregisters an operational collection target of ten
observed London sessions, at least 200 entry-window checks and 80% observed-date
slot coverage for every asset/strategy pair, at least 10,000 quotes per broker
instrument, and no unresolved integrity fault. A result can be `COLLECTING`,
`REVIEW`, or `OPERATIONALLY_READY`. None of those states measures expectancy or
authorizes a trade.

Default ignored runtime artifacts:

- `data/runtime/oanda_paper_state.json` — atomic risk anchors;
- `data/runtime/oanda_shadow_checks.sqlite` — one immutable asset/strategy
  check per completed M30 bar; and
- `data/runtime/pretrade_observations.sqlite` — full immutable risk reports for
  actual hypothetical signals.

Stable bar-derived IDs make polling idempotent. A repeated process cannot
replace an earlier decision. If more than one candidate could occupy the
single portfolio slot, config order followed by P1/P2/P3 order is the declared
shadow priority; after an `ALLOW` or `REDUCE`, later candidates are recorded as
`SUPPRESSED` with reason `one_position_shadow_priority`.

`ALLOW` remains a hypothetical risk result, not permission to trade. Missing
spread history now blocks admission explicitly. Actual order submission still
requires a separate design, authorization and operational release gate.
