# Final validation and closure report

**Closure date:** 2026-08-10

**Product classification:** completed research prototype

**Assets:** spot gold (`XAUUSD`) and silver (`XAGUSD`)

**Order authority:** none

## 1. Closure decision

The declared product scope is complete. The repository implements a local,
read-only quantitative pre-trade risk framework; it is not a trading strategy,
profitability claim or production execution system.

The earlier 40-London-date prospective checkpoint is not being completed
inside this time-box. Its immutable ledger is preserved with 944 pending
forecasts across five dates and zero outcomes. Nothing was resolved early,
backfilled or interpreted as a pass. That study is optional future validation,
not evidence used to declare this software complete.

## 2. What passed

### Engineering and policy behavior

The reproducible closure matrix contains 18 synthetic, non-predictive cases:

- three `ALLOW` cases covering gold, silver and short geometry;
- two `REDUCE` cases covering risk/size and margin constraints;
- twelve `BLOCK` cases covering invalid geometry, market/quote integrity,
  scheduled events, daily/weekly drawdown, entry count, kill switch, execution
  costs, portfolio count and extreme volatility; and
- one `INCOMPLETE` case proving that a missing contract value is not guessed.

All 18 cases passed their frozen expected decision, reason and pressure-score
checks. The matrix uses no later outcomes, directional signal or strategy P&L.

The complete Python test suite and React test/build workflow pass. Tests cover
the deterministic engine, causal calibration, broker parsing, state
reconciliation, immutable ledgers, reporting, local UI boundary and the final
scenario matrix.

### Historical quantitative evidence

The frozen historical calibration contains 243,598 causal walk-forward rows
across eight asset/horizon cells from 64,955 synchronized M30 source rows. Its
data-quality gate passed. Approximately nominal historical p95 range and
side-specific adverse-excursion coverage is documented in
[`pretrade_risk_calibration.md`](pretrade_risk_calibration.md).

This evidence is historical and deliberately uses neither direction nor
strategy P&L. Its profile ends before July 2025 and is not represented as a
current market calibration.

### Live-data integration

The project has authenticated read-only OANDA practice integration for current
quotes, spreads, account state, instrument facts and positions. The UI also
supports manually supplied GBP spread-bet account facts while retaining OANDA
only as the read-only market source. Neither path has an order endpoint.

## 3. What did not pass

The latest complete current-calibration release is `REJECTED`; its frozen p95
coverage gate failed. It remains an auditable negative result and cannot add
current range/adverse-excursion scenarios to an assessment.

The prospective calibration checkpoint was not reached. At closure its state
is `COLLECTING`, with 944/944 forecasts pending, zero resolved forecasts, five
scheduled London dates and a 40-date primary checkpoint.

These facts mean:

- deterministic account, sizing, cost and policy decisions are usable as a
  research demonstration;
- current calibrated range evidence remains unavailable;
- the 1–7 score measures pressure against policy limits, not loss probability;
- no directional edge, profitability or present-market coverage is claimed;
  and
- the tool is not approved for unattended execution or real-money reliance.

## 4. Reproduce the closure checks

Run the deterministic closure record:

```bash
.venv/bin/python scripts/validate_completed_framework.py
```

Run the full software verification:

```bash
.venv/bin/python -m pytest -q
npm test --prefix web
npm run build --prefix web
```

The first command writes the detailed local JSON audit to
`reports/final_framework_validation.json`. Reports, broker data, credentials
and runtime ledgers remain Git-ignored; this document records the portable
conclusions.

## 5. Final supported claim

This repository demonstrates an evidence-disciplined, offline-first commodity
risk system with causal historical calibration, conservative fail-closed
controls, authenticated read-only broker integration, explicit negative-result
retention and a usable React interface.

It is complete for that research and portfolio-demonstration scope. Prospective
current-regime calibration, production deployment, automated execution and
profitability are explicitly outside the completed claim.
