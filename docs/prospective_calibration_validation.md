# Prospective range-risk validation — version 1

**Frozen:** 2026-08-05, after current calibration version 1 was rejected and
before any prospective path outcome was available

**Purpose:** test whether frozen range and adverse-excursion forecasts cover
later unseen gold/silver paths

**Trading, direction and P&L:** prohibited

**Closure status (2026-08-10):** optional study preserved but not continued.
The immutable ledger contains 944 pending forecasts across five London dates
and zero resolved outcomes. They were not scored early or used to support the
completed research-prototype claim.

## 1. Sampling unit

One forecast row is an asset, London entry timestamp and horizon. It contains
total-range plus hypothetical long- and short-adverse p50/p80/p95/p99 forecasts
in basis points. Both sides are always registered together, so no directional
selection is possible.

Each schedule includes every profile-supported half-hour from 07:00
inclusive to 19:30 exclusive, subject to the 21:30 London cutoff, for 60, 120,
180 and 240-minute horizons. Known CPI, NFP and FOMC blackout entries are
excluded by the frozen event calendar. A batch contains one to five consecutive
future weekdays; the routine batch is Monday through Friday. Allowing a shorter
first or holiday-week batch does not permit selecting clocks, assets, sides or
horizons within a retained date.

## 2. Prospective boundary

- Registration must complete before the earliest scheduled entry.
- The exact profile and release hashes, release status and failed admission
  checks are stored with every forecast.
- An `ADMITTED` or `REJECTED` release may be observed here; a rejected release
  remains research-only and cannot enter a trade assessment.
- Forecast rows are append-only and exact retries are idempotent. A changed
  retry using the same identity is refused.
- Outcomes can be inserted only after their full horizon has matured.
- Resolution uses a later synchronized Dukascopy M30 bid/ask source with no
  interpolation, resampling or filling.

## 3. Outcomes

For each entry, the resolver measures the same three percentage outcomes used
by calibration version 1: total midpoint range, hypothetical long adverse
excursion and hypothetical short adverse excursion. Every required M30 bar
must exist exactly. A missing or incomplete path stays pending and visible; it
is not silently counted as covered.

No stop, target, size, trade return or profit is constructed.

## 4. Evaluation

Coverage is reported for every asset, horizon, metric and quantile. The report
also records pending forecasts, distinct London dates and schedule coverage.
Because half-hour paths and horizons overlap, raw forecast rows are not treated
as independent observations. London dates are the conservative reporting
blocks.

The first primary checkpoint requires at least 40 distinct resolved London
dates and 80% of scheduled rows resolved. Earlier reports are `COLLECTING` and
descriptive only. Reaching the checkpoint does not automatically admit a
calibration release or change policy. Any promotion requires a separately
frozen decision using these results.

## 5. Optional future workflow

Register up to five future weekdays before the first entry:

```bash
.venv/bin/python scripts/prospective_calibration.py register \
  --profile data/processed/current_risk_calibration/<release>/profile.parquet \
  --release data/processed/current_risk_calibration/<release>/release.json \
  --start-date YYYY-MM-DD \
  --weekdays 5
```

After the paths have matured, resolve them using a later manual refresh source:

```bash
.venv/bin/python scripts/prospective_calibration.py resolve --latest-source
```

Inspect coverage without a network request:

```bash
.venv/bin/python scripts/prospective_calibration.py summary
```

The default append-only ledger is
`data/runtime/prospective_calibration.sqlite`.
