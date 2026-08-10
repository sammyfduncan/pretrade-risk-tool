# Quantitative pre-trade risk framework closure record

**Started:** 2026-08-03

**Closed:** 2026-08-10

**Final status:** completed research prototype

**Assets:** `XAUUSD`, `XAGUSD`

**Cadence:** intended intraday holding periods of 60–240 minutes

**Order authority:** none

## 1. Product objective

Build a technically rigorous decision-support tool that evaluates the risk of
an intended gold or silver trade. The tool is conditional on a supplied trade
thesis; it neither produces direction nor needs a profitable strategy.

For each proposal it answers:

- Is every required fact present, current and internally consistent?
- What is the cost-inclusive cash loss at the proposed stop?
- What is the largest broker-valid size within the risk policy?
- What margin, notional and metals exposure would result?
- Does the trade violate session, event, loss, entry-count or portfolio gates?
- What deterministic spread and available calibrated-movement stresses apply?
- Which assumptions and limitations prevent a stronger claim?

## 2. Final completion boundary

The research prototype is complete because:

1. offline, OANDA-account and manual-account assessments use the same versioned
   Python policy;
2. deterministic calculations and fail-closed branches have automated tests;
3. historical calibration is causal and current calibration is explicitly
   unavailable when its frozen admission gate fails;
4. prospective schedule and resolution machinery preserve a valid future-data
   boundary, although the optional 40-date sample was not collected;
5. a user can enter a trade through the local React interface and download an
   auditable report; and
6. [`final_validation_report.md`](final_validation_report.md) states every
   supported and unsupported claim.

The prospective checkpoint was an earlier validation ambition, not something
that can be completed honestly inside the final two-day development time-box.
It was moved outside the completion boundary explicitly. Its forecasts remain
pending and provide no evidence for closure.

The following were never required for this completed scope:

- profitable P1/P2/P3 rules;
- directional ML or sentiment;
- broker order submission;
- live capital; or
- a strategy P&L threshold.

## 3. Completed implementation

- Versioned trade, account, instrument, market and portfolio contracts.
- `ALLOW`, `REDUCE`, `BLOCK` and `INCOMPLETE` decisions with stable reason codes.
- Stop-risk, reward/risk, broker-step sizing, notional, margin and portfolio
  calculations.
- Session, scheduled-event, daily/weekly loss, entry-count and kill-switch
  controls.
- Current, median and p95 execution-cost gates plus deterministic stress cases.
- Causal historical range and side-specific adverse-excursion calibration.
- Fail-closed current-calibration refresh and admission machinery.
- Authenticated read-only OANDA account, quote, instrument and position input.
- Manual GBP spread-bet account mode using read-only OANDA market data.
- Restart-safe account state, quote recording, reconciliation and immutable
  prospective ledgers.
- JSON and standalone HTML reports.
- Localhost-only React workflow with a versioned 1–7 policy-pressure score.
- Reproducible 18-case final scenario matrix and closure audit.

## 4. Phase outcomes

### R1 — current calibration policy: complete, unavailable result

The source, estimator, freshness, parity and release gates were frozen. The
latest complete release failed p95 coverage and was correctly marked
`REJECTED`. The active evaluator cannot load it.

### R2 — prospective validation infrastructure: complete, study unfinished

Registration, immutable storage, delayed resolution and coverage reporting are
implemented and tested. The preserved ledger has 944 pending forecasts across
five dates and zero outcomes. The 40-date checkpoint was not reached and is not
claimed.

### R3 — user-facing workflow: complete

The local interface supports OANDA demo facts or a manually described external
GBP spread-bet account. It displays the Python-generated policy score and core
risk numbers, with details and downloadable JSON/HTML. There is no order route.

### R4 — final audit: complete

The closure scenario matrix passes 18/18 cases and covers both metals, both
trade directions, all four decision outcomes, costs, margin, drawdowns,
portfolio state and data-integrity refusals. Historical evidence and unresolved
validation limitations are recorded in
[`final_validation_report.md`](final_validation_report.md).

## 5. Final evidence classification

| Claim | Status |
|---|---|
| Engineering implementation | Complete |
| Deterministic scenario validation | Passed |
| Causal historical range calibration | Passed on frozen development data |
| Authenticated read-only market integration | Demonstrated |
| Current calibration admission | Failed; unavailable |
| 40-date prospective checkpoint | Not reached |
| Directional or profitability edge | Not claimed |
| Production or real-money approval | Not claimed |

## 6. Optional future projects

There is no remaining work for the declared scope. Continuing the 40-date
study, hosting the UI, supporting a reconciled external portfolio, submitting
orders or starting new directional research would each be a separately scoped
project rather than completion work for this repository.
