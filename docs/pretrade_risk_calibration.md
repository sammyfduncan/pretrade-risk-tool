# Historical pre-trade risk calibration — version 1

**Status:** development calibration complete; data-quality gate passed
**Decision date:** 2026-08-03
**Development end exclusive:** `2025-07-01T00:00:00Z`
**Confirmation data:** unopened
**Direction and strategy P&L:** not used

## 1. Question

This program calibrates the deterministic scenarios in the pre-trade risk
engine. It asks:

> At each eligible London entry clock and intended holding period, what ranges
> and adverse excursions did gold and silver actually experience, and how well
> did estimates formed only from prior observations cover later outcomes?

It does not ask whether either metal should be bought or sold. Long and short
adverse excursions are both measured for every eligible timestamp, regardless
of any strategy signal.

## 2. Frozen source and sample

- Source: the pinned, synchronized Dukascopy gold/silver M30 bid/ask artifact
  built for RV1, without filling missing timestamps.
- Assets: `XAUUSD` and `XAGUSD` only.
- Sample: `2020-01-01T00:00:00Z` through `2025-07-01T00:00:00Z`, exclusive.
- Entry timestamps: M30 bar opens from 07:00 inclusive to 19:30 exclusive in
  `Europe/London`.
- Horizons: 60, 120, 180 and 240 minutes.
- A path is eligible only when every required M30 bar is present and its final
  bar closes no later than the 21:30 London cutoff.
- CPI/NFP entries from 30 minutes before through 30 minutes after release and
  FOMC entries from 30 minutes before through 60 minutes after are excluded,
  using the already-verified event calendar.
- No row on or after the confirmation boundary may be loaded by the builder.

The synchronized artifact is used so both metals have the same timestamp
opportunity set. This costs at most the three unmatched timestamps recorded in
its manifest and avoids asset-specific sample-composition ambiguity.

## 3. Frozen outcomes

For entry midpoint `P`, maximum future midpoint high `H`, minimum future
midpoint low `L`, and the bars from entry through the end of the horizon:

- total range: `(H - L) / P * 10,000` basis points;
- long adverse excursion: `max(P - L, 0) / P * 10,000`;
- short adverse excursion: `max(H - P, 0) / P * 10,000`;
- opening spread in points: `ask_open - bid_open`; and
- opening spread in basis points: `spread / P * 10,000`.

Midpoint highs and lows are the averages of the same bar's bid and ask highs
and lows. The range/adverse outcomes therefore exclude transaction cost. The
pre-trade engine adds broker spread and slippage separately, preventing double
counting.

Every metric is non-negative. The path begins with the entry M30 bar because
the hypothetical assessment is made immediately before its open.

## 4. Causal estimator

Clock seasonality is handled without adding a model search. For each asset,
horizon and London half-hour slot independently:

- keep the previous 252 eligible observations;
- require at least 126 strictly prior observations;
- calculate higher-interpolation p50, p80, p95 and p99 quantiles; and
- never use the current row or a later row in its estimate.

Weekday conditioning, volatility bins, exponential decay, alternate windows,
and optimized quantile levels are not part of version 1. They would reduce
effective sample size or create a tuning search.

The final prospective profile for each asset/horizon/slot uses the last 252
development observations before the cutoff. Values remain in basis points and
are converted to price points only at lookup time using the proposed entry
price.

## 5. Diagnostics and release boundary

For every asset and horizon the report records:

- eligible and calibrated row counts;
- reason-coded path exclusions;
- actual coverage of each causal p50/p80/p95/p99 estimate;
- mean and median forecast-to-outcome ratios;
- final-profile sample counts; and
- source and output SHA-256 hashes.

The artifact is valid only if:

1. source timestamps are sorted, unique, timezone-aware and M30-aligned;
2. every bid/ask field is finite, positive and non-crossed;
3. every retained future path is exactly contiguous;
4. each asset/horizon has at least 1,000 causal calibrated rows;
5. at least 70% of eligible rows survive the fixed 126-observation burn-in; and
6. every final asset/horizon/slot used by the engine has at least 126 observations.

Coverage is reported rather than turned into a post-outcome pass threshold.
Heavy-tailed regime changes can make an honest historical p95 under-cover; the
result must remain visible and may motivate an explicitly new calibration
version, not an after-the-fact multiplier.

## 6. Spread and volatility boundaries

Dukascopy spread distributions are retained for research diagnostics only.
They are not eligible to populate OANDA's `median_spread_points` or
`p95_spread_points`, because broker/product/account spreads must be measured
from the intended OANDA instrument.

The retained gold two-hour volatility model is a separate point-in-time
information source. Version 1 records its OOF relationship to two-hour range
and adverse excursion when exact timestamps match, but it does not mark the
model `volatility_artifact_approved` for automatic paper sizing. Silver and
the other horizons continue to use the transparent same-clock profile.

## 7. Output contract

- `data/processed/pretrade_risk_calibration_oof.parquet`: row-level causal
  estimates and realised outcomes;
- `data/processed/pretrade_risk_calibration_profile.parquet`: final lookup
  table by asset, horizon and London slot;
- `reports/pretrade_risk_calibration.json`: provenance, exclusions,
  diagnostics, hashes and release flags.

All outputs set `confirmation_opened=false`, `strategy_pnl_used=false`,
`broker_spread_eligible=false`, and `paper_sizing_enabled=false`.

The lookup API must fail closed for an unsupported instrument, horizon, clock,
stale profile, insufficient sample or price that is not finite and positive.

## 8. Look ledger

| Look | Date | Item | Outcome available? |
|---:|---|---|---|
| 0 | 2026-08-03 | Source/schema audit and this design frozen | no calibration metrics |
| 1 | 2026-08-03 | Single frozen development build | PASS; 243,598 causal OOF rows; all eight cells passed; confirmation unopened |

## 9. Development result

All eight asset/horizon cells passed the frozen data-quality gate:

| Asset | Horizon | Eligible | Causal OOF | OOF coverage availability | Range p95 realised coverage | Long-adverse p95 | Short-adverse p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| XAUUSD | 60m | 34,997 | 31,847 | 91.0% | 95.0% | 95.0% | 95.0% |
| XAUUSD | 120m | 34,907 | 31,757 | 91.0% | 95.1% | 95.1% | 95.0% |
| XAUUSD | 180m | 33,388 | 30,364 | 90.9% | 95.1% | 95.1% | 95.1% |
| XAUUSD | 240m | 30,603 | 27,831 | 90.9% | 95.1% | 95.1% | 95.2% |
| XAGUSD | 60m | 34,997 | 31,847 | 91.0% | 94.9% | 95.1% | 95.0% |
| XAGUSD | 120m | 34,907 | 31,757 | 91.0% | 94.9% | 95.0% | 95.1% |
| XAGUSD | 180m | 33,388 | 30,364 | 90.9% | 95.0% | 95.1% | 95.0% |
| XAGUSD | 240m | 30,603 | 27,831 | 90.9% | 95.0% | 95.1% | 95.0% |

The final profile contains 192 exact asset/horizon/London-clock rows, each
using 252 observations. Silver has materially wider percentage ranges and
adverse excursions than gold, so it must not inherit gold's point or percentage
stresses.

On 23,760 exact OOF gold two-hour matches, the retained volatility forecast's
log correlation with log total range was `0.5643`. Correlation with log
side-specific adverse excursion was only `0.1031` for hypothetical longs and
`0.1141` for hypothetical shorts. This supports using the model as range-risk
information, not as a directional or stop-hit predictor.

The profile's last observations are 2025-06-30. The lookup refuses profiles
more than 90 days old, so this artifact is a validated historical baseline but
is intentionally stale for a 2026 live decision. Refreshing the range profile
requires a separately admitted post-development source. OANDA spread measurement
remains mandatory and cannot be replaced by the Dukascopy reference spreads.
