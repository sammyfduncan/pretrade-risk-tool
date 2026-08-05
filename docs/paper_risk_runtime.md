# Restart-safe paper risk runtime

**Status:** state bridge and local controls implemented; authenticated flat
practice-account initialization completed on 2026-08-04

## Purpose

The pre-trade engine must not accept freshly typed day/week equity anchors or
entry counts on every invocation. Those facts determine loss gates and must
survive restarts. `PaperStateStore` already persists them atomically; the risk
runtime bridge now makes that state the preferred source for:

- London day-start equity;
- London week-start equity;
- entries confirmed today;
- expected broker position IDs; and
- the global paper kill switch.

Before any risk report uses the state, the bridge requires the stored account
ID and exact expected trade-ID set to match the current OANDA response. It does
not adopt unknown broker positions. It then rolls day/week boundaries once in
Europe/London time using current equity.

The state file is written atomically with mode `0600` and lives by default at
`data/runtime/oanda_paper_state.json`, inside the ignored runtime area.

## One-time initialization

Initialization is permitted only when the selected demo account is verified
flat:

```bash
python3 scripts/initialize_oanda_paper_state.py --config config/gold.yaml
```

An existing state file is never overwritten. A non-flat account, ambiguous
account selection or missing broker response is refused.

## Risk assessment

Use the persisted state instead of manual anchors:

```bash
python3 scripts/evaluate_oanda_pretrade_risk.py \
  examples/oanda_pretrade_intent.json \
  --config config/gold.yaml \
  --portfolio-config config/silver.yaml \
  --paper-state data/runtime/oanda_paper_state.json
```

Manual `--day-start-equity`, `--week-start-equity` or `--entries-today`
overrides cannot be combined with `--paper-state`. The output provenance
records the state revision, update time and London boundaries, but not the full
account ID.
Persistent operation also requires the automatic verified event calendar; a
manual `clear` assertion is refused.

## Kill switch and inspection

```bash
python3 scripts/manage_paper_state.py status
python3 scripts/manage_paper_state.py kill-on
python3 scripts/manage_paper_state.py kill-off
```

An active switch is now a hard `BLOCK` reason in the quantitative pre-trade
engine. These commands change only the local state; they do not contact OANDA,
submit an order or close a position. Therefore `kill-on` prevents new project
intents but is not itself an emergency broker flatten command.

## Optional lifecycle integration

Entry count and expected position IDs already update atomically when a future
submit-once adapter marks an intent confirmed, and positions are removed only
after authoritative broker closure. That order adapter remains disabled. It is
not required by the standalone pre-trade risk evaluator; it would be required
only for a separately authorized automated paper-trade lifecycle.
