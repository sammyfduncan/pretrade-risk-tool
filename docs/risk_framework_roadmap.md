# Quantitative pre-trade risk framework roadmap

**Active since:** 2026-08-03

**Assets:** `XAUUSD`, `XAGUSD`

**Cadence:** intended intraday holding periods of 60–240 minutes

**Execution boundary:** OANDA v20 read-only practice data

**Order authority:** none

## 1. Product objective

Build a technically rigorous decision-support tool that evaluates the risk of
an intended gold or silver trade. The tool is conditional on a supplied trade
thesis; it neither produces direction nor needs a profitable strategy.

For each proposal it must answer:

- Is every required fact present, current and internally consistent?
- What is the cost-inclusive cash loss at the proposed stop?
- What is the largest broker-valid size within the risk policy?
- What margin, notional and correlated metals exposure would result?
- Does the trade violate session, event, loss, entry-count or portfolio gates?
- How does its geometry compare with conservative historical and current
  range/adverse-excursion evidence?
- Which assumptions and limitations prevent a stronger claim?

## 2. Completion boundary

The core product is complete when:

1. offline and read-only OANDA assessments implement the same versioned policy;
2. every calculation and fail-closed branch has deterministic tests;
3. a current, provenance-bearing calibration source is admitted or explicitly
   unavailable under a documented policy;
4. standardized prospective intents have measured 60/120/180/240-minute paths
   and coverage diagnostics without directional selection;
5. a user can enter a trade and understand the resulting report without editing
   internal code; and
6. a final audit states the supported and unsupported claims.

The following are not required for core completion:

- profitable P1/P2/P3 rules;
- directional ML or sentiment;
- broker order submission;
- live capital; or
- a strategy P&L threshold.

## 3. Current implementation state

### Complete

- Versioned input schema and `ALLOW`/`REDUCE`/`BLOCK`/`INCOMPLETE` decisions.
- Stop-risk, reward/risk, sizing, notional, margin and portfolio calculations.
- Session, event, daily/weekly loss, entry-count and kill-switch gates.
- Deterministic stress scenarios and HTML/JSON output.
- Causal historical range and side-specific adverse-excursion calibration.
- Read-only OANDA account/market/position adapter.
- Exact restart-safe account and position reconciliation.
- Immutable prospective assessment, outcome and quote-path contracts.
- Source parity, quote-cost and operational integrity diagnostics.
- Basic localhost React workflow for read-only OANDA trade assessment and
  downloadable JSON/HTML reports.

### Partially complete

- **Current calibration:** the first post-development release was rejected for
  p95 under-coverage and is explicitly unavailable to the risk evaluator.
- **Prospective validation:** 944 direction-balanced forecasts covering five
  London dates are registered and immutable; resolution awaits later quote
  paths and the primary checkpoint remains 40 London dates.

### Optional extensions

- continuous OANDA quote recording;
- P1/P2/P3 shadow-signal integration;
- an approved volatility forecast as a size-reduction input;
- practice order execution under separate authority; and
- renewed strategy research under a new preregistration.

## 4. Next implementation phases

### Phase R1 — current calibration policy

**Complete with an honest unavailable result:** the version-one source,
estimator, freshness, parity and release gates are defined in
[`pretrade_calibration_refresh_policy.md`](pretrade_calibration_refresh_policy.md).
The first refresh built successfully but was rejected for p95 under-coverage.
The risk evaluator therefore continues without current calibrated scenarios.
The acquisition span was corrected from 550 to 900 days to satisfy the
unchanged OOF sample-quality design; a read-only diagnostic confirmed that this
does not rescue the coverage result.

### Phase R2 — standardized prospective validation

**Design and implementation complete; collection active:** the weekly
offline-first schedule, immutable ledger, direction-balanced outcomes and
40-London-date checkpoint are defined in
[`prospective_calibration_validation.md`](prospective_calibration_validation.md).
The manual register, delayed resolve and summary commands preserve rejected
releases as research-only and calculate no strategy P&L. The first five-date
schedule contains 944 forecasts from a complete rejected release. A later
manual refresh can resolve matured paths with `--latest-source`; invalid or
partial refresh directories are skipped explicitly.

### Phase R3 — user-facing assessment workflow

**Version-one workflow complete:** a concise trade form accepts instrument,
side, order type, entry, stop,
target, holding period and optional size. Broker/account facts should populate
automatically from read-only OANDA. It presents decision, reason codes,
recommended size, cash risk, margin, costs, portfolio impact, scenarios and
limitations in plain language, with JSON and standalone HTML downloads. It is
localhost-only and intentionally provides no order route.

### Phase R4 — final audit

Freeze the product version, rerun the complete test and calibration suite, and
publish an audit covering data provenance, causal boundaries, historical and
prospective coverage, broker limitations, failure modes and non-claims.

## 5. Evidence discipline

- Historical calibration evaluates risk coverage, not direction or profit.
- Prospective assessments are immutable before their future path is measured.
- Missing or stale inputs remain visible and cannot be replaced by a different
  provider without an explicit policy version.
- Negative coverage, cost or operational results are retained.
- Continuous collection counts are not treated as independent statistical
  observations merely because the broker emits many quotes.
