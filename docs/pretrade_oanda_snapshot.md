# Read-only OANDA v20 pre-trade snapshot

**Status:** implemented and authenticated against the practice account on
2026-08-04. Order authority remains disabled.

`scripts/evaluate_oanda_pretrade_risk.py` reads the selected v20 account,
account-specific instrument metadata, current top-of-book prices, bounded M30
bid/ask candles and open trades. It translates those facts into the generic
pre-trade risk contract and never creates, changes or closes an order.

## Local credentials

The ignored local `.env` must contain:

```text
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_ENV=practice
```

`OANDA_API_TOKEN` is also accepted as an alias for the key. Credentials must
never be committed, copied into reports or passed as command-line arguments.

## Account and instrument facts

The YAML fixes only the intended OANDA instrument and declared account
product. Everything that can be read authoritatively stays live:

- `XAU_USD` and `XAG_USD` instrument identifiers;
- minimum trade size and trade-unit precision;
- maximum order units and instrument margin rate;
- executable bid/ask and RFC3339 broker timestamp;
- adverse `accountLoss` conversion for stop risk;
- `positionValue` conversion for notional and margin;
- NAV, balance, unrealized P&L and available margin;
- exact open-trade IDs and linked stop-loss prices.

OANDA v20 does not identify `SPREADBET` versus `CFD` in the account summary.
That product selection is therefore explicit in `config/*.yaml` and remains an
audited limitation. The API currently reports `hedgingEnabled=false` for the
configured practice account; the adapter records this instead of assuming the
portal selection took effect. The current one-position policy does not require
hedging.

## Run a read-only assessment

Copy the template outside Git, fill in only the proposed trade characteristics,
then run:

```bash
python3 scripts/evaluate_oanda_pretrade_risk.py \
  examples/oanda_pretrade_intent.json \
  --config config/gold.yaml \
  --portfolio-config config/silver.yaml \
  --paper-state data/runtime/oanda_paper_state.json
```

The persistent state must first be initialized from an authenticated, flat
practice account:

```bash
python3 scripts/initialize_oanda_paper_state.py --config config/gold.yaml
```

Manual runtime anchors are permitted only for isolated diagnostics. Persistent
paper operation requires the state file and automatic verified event calendar.

## Failure rules

The adapter refuses or leaves the report incomplete when an instrument is
missing, a quote is stale/non-tradeable, conversion data is absent, NAV does
not reconcile, an open trade lacks a stop, or broker trade IDs differ from the
restart-safe state. It does not infer missing values from Dukascopy, Databento,
another OANDA division or a different account.
