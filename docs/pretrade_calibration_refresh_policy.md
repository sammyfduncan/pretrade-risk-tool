# Current range-risk calibration refresh policy — version 1

**Frozen:** 2026-08-05, before any post-development calibration outcome was
read

**Purpose:** admit current range and adverse-excursion scenarios into the
standalone pre-trade risk evaluator

**Direction, strategy signals and P&L:** prohibited

## 1. Decision boundary

The historical profile ending 2025-06-30 proved that the estimator was causal
and approximately calibrated on its development period. It is too old for a
current assessment. A refresh is not a new strategy experiment: every eligible
clock contributes both hypothetical long and short risk outcomes, whether or
not anyone wanted to trade.

A release may enrich deterministic risk scenarios only. It may never increase
position size, populate OANDA spread fields, generate direction or authorize an
order. If any admission check fails, the risk assessment continues without
calibrated range scenarios and reports them as unavailable.

## 2. Frozen source contract

- Provider: Dukascopy spot `XAU/USD` and `XAG/USD` M30 bid and ask candles.
- Source shape: one synchronized, timezone-aware wide table containing all four
  streams; no resampling, interpolation or forward filling.
- Minimum acquisition span: 550 calendar days before the release time. The
  implementation requests 900 calendar days so the existing 70% causal-OOF
  sample-quality gate can also be evaluated after its 126-observation burn-in.
  The final profile still uses only the frozen last 252 observations.
- The source must end no more than seven calendar days before release and must
  contain no timestamp at or after the release time.
- Both instruments use the identical timestamp opportunity set. Missing bars
  invalidate affected future paths rather than being repaired.
- The verified CPI, NFP and FOMC schedule must cover the full source period.
- Dukascopy spreads remain research diagnostics only. Current and stressed
  transaction costs come from the account-specific OANDA quote path.

Feed transfer remains conditional on a bounded M30 Dukascopy/OANDA parity check
for each metal no older than 90 days. Each check must contain at least 100
matched bars, at least 95% timestamp coverage on both feeds, return correlation
of at least 0.98, best shift zero and a `PASS` verdict. Parity validates market
and clock transfer; it does not make the two price levels identical.

## 3. Frozen estimator

The refresh reuses historical calibration version 1 without tuning:

- London entry clocks from 07:00 inclusive to 19:30 exclusive;
- 60, 120, 180 and 240-minute horizons, subject to the 21:30 cutoff;
- total midpoint range plus hypothetical long and short adverse excursion;
- same asset, horizon and London half-hour slot only;
- the last 252 eligible, fully matured observations;
- at least 126 observations per final profile row; and
- higher-interpolation p50, p80, p95 and p99 quantiles.

The current entry and every later observation are excluded. The latest retained
entry must have its entire four-hour path available before the release time.
Weekday bins, volatility bins, decay, alternate windows and multiplier tuning
remain outside version 1.

## 4. Release gates

Every release writes immutable hashes for the source, source manifest, profile,
OOF artifact, event snapshot and both parity reports. It is `ADMITTED` only if:

1. all structural, bid/ask, timestamp and exact-path checks pass;
2. all eight asset/horizon cells pass the existing sample-quality gate;
3. every expected asset/horizon/clock profile row exists exactly once and has
   at least 126 observations;
4. p95 OOF coverage is at least 93% for total range, long adverse excursion and
   short adverse excursion in every asset/horizon cell;
5. p99 OOF coverage is at least 97% for those same metrics in every cell;
6. every quantile is finite, non-negative and monotone within its metric;
7. all source, event and parity requirements in section 2 pass; and
8. the release explicitly records that direction, strategy P&L, outcome-based
   parameter selection and broker-spread substitution were not used.

Over-coverage is allowed because this is a conservative risk layer. A failed
release is retained with reason codes; thresholds are not relaxed afterward.

## 5. Freshness and operation

- Scheduled refresh: once per week, after a completed London session.
- Maximum profile age at assessment: ten calendar days, measured from the exact
  selected clock row's last matured observation.
- Maximum parity age: 90 calendar days, measured from the latest timestamp in
  the bounded comparison sources.
- Event-calendar coverage must be extended before its published end; an
  unknown future schedule fails closed.
- A new policy version is required for any provider, cadence, horizon, session,
  event window, rolling window, minimum history, quantile or gate change.

The previous admitted release remains an audit artifact but is not used after
it exceeds the ten-day profile limit. There is no automatic fallback to the
2025 historical profile.

## 6. Prospective validation remains separate

Refreshing the estimator and validating it are different jobs. A refresh may
use matured market paths to keep its rolling quantiles current. The separately
preregistered direction-balanced observations evaluate whether pre-trade p95
and p99 scenarios actually cover later paths. They cannot be filtered by
strategy signals and do not determine the refresh parameters.

Coverage failure in that prospective sample triggers review or a newly frozen
policy version. It is never repaired by deleting difficult observations.

## 7. Manual implementation

The policy is implemented as one explicit command:

```bash
.venv/bin/python scripts/refresh_current_risk_calibration.py
```

It does not run during an assessment and is not scheduled by the repository.
Each invocation creates a new timestamped directory under
`data/processed/current_risk_calibration/` containing the four downloaded
legs, synchronized source, event snapshot, OOF diagnostics, final profile,
source manifest and `release.json`. An existing release directory is never
overwritten.

An admitted profile is used only when both artifacts are supplied:

```bash
.venv/bin/python scripts/evaluate_oanda_pretrade_risk.py \
  examples/oanda_pretrade_intent.json \
  --config config/gold.yaml \
  --portfolio-config config/silver.yaml \
  --paper-state data/runtime/oanda_paper_state.json \
  --calibration-profile data/processed/current_risk_calibration/<release>/profile.parquet \
  --calibration-release data/processed/current_risk_calibration/<release>/release.json
```

If the refresh cannot download, build or pass every gate, it writes a failure
or rejected audit artifact and exits non-zero. It does not replace, delete or
silently fall back to another profile.

## 8. Refresh ledger

| Look | Date | Result |
|---:|---|---|
| 1 | 2026-08-05 | Build completed but release rejected. The 550-day request produced about 67% causal OOF coverage, exposing an arithmetic incompatibility with the unchanged 70% data-quality gate. The p95 coverage gate also failed materially, especially for silver. |
| Amendment | 2026-08-05 | Acquisition increased to 900 days and detailed gate failures added. The 252-observation estimator, quantiles and 93%/97% coverage thresholds were not changed. A read-only 900-day diagnostic passed sample quality but still failed p95 coverage, so this amendment cannot rescue the known result. |
