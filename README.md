# Quantitative Precious-Metals Pre-Trade Risk Framework

**Status:** active risk-evaluation product; directional strategy discovery is
closed and retained as historical research.

**Active roadmap:** [`docs/risk_framework_roadmap.md`](docs/risk_framework_roadmap.md)

**Documentation map:** [`docs/README.md`](docs/README.md)

## Purpose

This project evaluates the technical and mathematical risk of an intended
intraday gold or silver trade before an order exists.

The user—or a separately validated strategy—supplies the proposed direction,
entry, stop, target, holding period and optional size. The framework combines
that geometry with account state, current executable OANDA prices, broker
contract facts, existing metals exposure, scheduled-event controls and
historically calibrated range/adverse-excursion evidence.

It returns one of four auditable decisions:

- `ALLOW`: the complete proposal fits every versioned risk limit;
- `REDUCE`: the proposal is assessable but size must be reduced;
- `BLOCK`: a hard policy, market, cost or portfolio rule prevents entry; or
- `INCOMPLETE`: a required fact is absent and the framework refuses to guess.

`ALLOW` does not predict profit, endorse the trade thesis or guarantee that a
normal stop is the maximum possible loss.

## Product boundary

The active product:

- covers spot gold (`XAUUSD`) and silver (`XAGUSD`);
- uses quantitative data only;
- accepts market, limit and stop trade geometry;
- calculates cash-at-stop risk, cost-adjusted reward/risk, margin, notional and
  portfolio concentration;
- applies London-session, daily/weekly loss, entry-count, scheduled-event and
  kill-switch controls;
- reports deterministic spread and gap stresses; and
- fails closed on stale, missing, delayed, inconsistent or unreconciled data.

It does not:

- decide whether either metal should rise or fall;
- interpret news, geopolitics or language-model sentiment;
- require a profitable P1/P2/P3 trading strategy;
- treat historical coverage as a guaranteed future probability; or
- create, amend or close an OANDA order.

Order execution, automated strategy integration and profitability research are
optional future extensions, not completion requirements for the risk product.

## Current implementation

### Complete

- Versioned trade, account, instrument, market and portfolio input contracts.
- Cost-inclusive stop risk and broker-increment position sizing.
- Margin, gross/signed exposure and cross-metal portfolio aggregation.
- Current, median and p95 spread-to-risk gates.
- Daily/weekly loss limits, maximum entry count and persistent kill switch.
- Automatic CPI, NFP and FOMC blackout windows from a verified schedule.
- Planned-stop, 1.5x/2x spread and optional calibrated range/MAE scenarios.
- JSON output plus a dependency-free static HTML report.
- Read-only OANDA v20 adapter for account, prices, instrument metadata,
  currency conversions, margin and exact open trades.
- Restart-safe local risk anchors with exact broker-position reconciliation.
- Immutable prospective assessment/outcome and executable quote-path ledgers.
- OANDA quote recording, reconnect evidence, source parity and operational
  shadow diagnostics.
- Manual, non-overwriting Dukascopy refresh and fail-closed admission pipeline
  for current range/adverse-excursion calibration.

### Validated evidence

- 243,598 causal walk-forward gold/silver calibration rows across 60, 120, 180
  and 240-minute horizons.
- Approximately nominal 95% historical coverage for p95 total-range and
  side-specific adverse-excursion estimates in all eight asset/horizon cells.
- A bounded 275-bar M30 Dukascopy/OANDA comparison passed for both metals:
  return correlation `0.9985` for gold and `0.9926` for silver, with no
  preferred timestamp shift.
- Authenticated OANDA practice-account validation of NAV, available funds,
  prices, margin, minimum sizes, home-currency conversion and open trades.
- Broker-aware cost evidence showed that tight-stop silver trades are much more
  cost-sensitive than gold; the framework preserves those failures rather than
  relaxing its limits.
- The complete automated test suite covers deterministic calculations,
  leakage/causality, broker parsing, state reconciliation, immutable ledgers,
  stream recovery and reporting.

### Important limitations

- The validated historical calibration profile ends on 2025-06-30 and is
  intentionally rejected for current live decisions after its 90-day freshness
  limit. Deterministic live risk calculations still work, but current calibrated
  range/MAE scenarios require a separately admitted refresh.
- No standardized prospective trade-intent sample has yet tested whether the
  historical range/MAE coverage persists in current conditions.
- The first current-calibration refresh completed but was correctly rejected:
  its causal p95 scenarios under-covered recent movement, especially in silver.
  No post-development profile is currently admitted for risk reports.
- The primary interface is currently CLI/JSON. A simpler user-facing trade
  form and report workflow remains to be built.
- OANDA v20 does not expose spread-bet versus CFD product type in its account
  summary. The selected product is therefore an explicit audited config fact.
- The practice API reports hedging disabled; version one permits only one open
  precious-metals position and does not depend on hedging.

## Quick start

Create an environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Offline assessment

Copy and complete `examples/pretrade_risk_input.template.json`, then run:

```bash
.venv/bin/python scripts/evaluate_pretrade_risk.py \
  examples/pretrade_risk_input.json \
  --output reports/pretrade_risk.json
```

Render the JSON as a standalone report:

```bash
.venv/bin/python scripts/render_pretrade_risk_report.py \
  reports/pretrade_risk.json \
  --output reports/pretrade_risk.html
```

### Read-only OANDA assessment

The ignored local `.env` supplies:

```text
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_ENV=practice
```

Initialize the local risk anchors only when the authenticated account is flat:

```bash
.venv/bin/python scripts/initialize_oanda_paper_state.py \
  --config config/gold.yaml
```

Copy and complete `examples/oanda_pretrade_intent.template.json`, then run:

```bash
.venv/bin/python scripts/evaluate_oanda_pretrade_risk.py \
  examples/oanda_pretrade_intent.json \
  --config config/gold.yaml \
  --portfolio-config config/silver.yaml \
  --paper-state data/runtime/oanda_paper_state.json \
  --output reports/oanda_pretrade_risk.json
```

This command reads broker facts and writes a report. It cannot submit an order.

### Manual calibration refresh

Refresh the locally stored range/adverse-excursion profile explicitly:

```bash
.venv/bin/python scripts/refresh_current_risk_calibration.py
```

This is the only step that downloads Dukascopy calibration history. It creates
a new timestamped release and never runs implicitly during an assessment. A
rejected refresh remains an audit record and cannot be loaded as current.

## Data boundaries

- **Dukascopy:** deep spot-metals research history and historical calibration.
- **OANDA:** account-specific executable prices, spreads, contract facts,
  margin, positions and operational observations.
- **Databento/CME:** preserved historical futures-market research; not required
  by the active risk engine.

Research, broker and runtime artifacts are namespaced and Git-ignored. A short
broker window cannot silently overwrite the multi-year research archive.

## Project structure

```text
config/                     gold/silver identities and research configuration
docs/                       active product docs and historical research records
examples/                   versioned input and observation templates
scripts/                    CLIs for assessment, calibration and diagnostics
src/commodity_ml/intraday/  risk, state, broker and observation implementation
tests/                      deterministic and integration tests
data/, models/, reports/    ignored local artifacts
```

## Current roadmap

The remaining core work is:

1. define and implement a legitimate current calibration-refresh policy;
2. prospectively evaluate standardized trade intents against later 1–4 hour
   range and adverse-excursion paths;
3. build a clear user-facing trade-input and report workflow; and
4. produce a final validation/audit report describing what the framework can
   and cannot claim.

Continuous streaming, P1/P2/P3 shadow checks and future paper execution remain
useful integration demonstrations, but they do not block completion of the
risk-evaluation framework.

## Historical research

The repository retains the complete negative directional-research ledger:
technical/macro models, COT and GDELT, candidate ML filters, price/session
rules, M15 execution, futures volume/order flow, relative value and macro
impulse tests. Those experiments rejected the sampled hypotheses honestly;
they do not imply that commodities or quantitative risk analysis are invalid.

Start with [`docs/README.md`](docs/README.md) to distinguish active product
documentation from historical experimental records.
