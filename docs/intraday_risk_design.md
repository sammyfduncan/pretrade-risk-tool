# Intraday volatility and event-risk experiment

**Design locked:** 2026-07-31, before generating any intraday-risk forecast or
reading any intraday-risk outcome  
**Scope:** gold development data from `2021-01-01` through `2025-06-30`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Status:** development evaluation complete; B0, F-IV, and F-CAL passed their
registered information gates, with B0 retained by the frozen selection rule
**Deployment:** impossible from this experiment alone; no primary strategy is
currently retained

## 1. Question and interpretation boundary

This experiment asks one narrow question:

> At an eligible M30 decision time, do lagged GVZ or the known schedule of a
> regular CPI, Employment Situation, or FOMC release improve a causal forecast
> of gold volatility over the next two hours beyond a price-and-session-only
> baseline?

“Forecast” does not mean inventing or filling market data. Features contain
only observations available by the decision time. The target is calculated
later from prices that actually occurred and is used only to score the earlier
forecast. Missing prices are not imputed, synthetic candles are not created,
and future prices never enter a feature.

The experiment forecasts magnitude, not direction. It cannot choose long or
short, create a candidate, alter a stop or target, or establish profitability.
A pass only qualifies a feature family for a separately pre-registered
downside-only sizing or restriction policy.

The experiment will not run until this document is committed. The future CLI
must refuse a dirty version of this file or its evaluation harness. Diagnostic
code may be exercised only on synthetic data and cannot produce a registered
result.

## 2. Known evidence before this registration

The following results were already known and motivated this experiment:

- the completed daily step-2b program found that gold GVZ and scheduled-event
  dates added information about five-day volatility;
- that result made no intraday claim and cannot be transferred automatically;
- all registered gold P1/P2/P3, V2, ML-filter, matched-M15, and LBMA-PM
  development programs failed their economic gates; and
- OANDA spread and fill measurements require prospective practice collection
  and calibration.

These facts select the two registered feature families below. No intraday
risk-forecast outcome, feature bin, threshold, or candidate-P&L subgroup has
been inspected. COT, GDELT, news tone, LLM sentiment, and additional technical
indicator families remain excluded.

## 3. Data contract

### 3.1 Price observations and evaluation boundary

- Instrument: Dukascopy `XAU/USD`, bid-side M30 candles.
- Development span: `2021-01-01 00:00 UTC` through
  `2025-06-30 23:59:59 UTC`.
- The later confirmation window must not be loaded by the experiment.
- Candle timestamps denote bar-open time. A row becomes knowable only at
  `timestamp + 30 minutes`.
- Only decision times in `[07:00, 19:30)` Europe/London are eligible.
- Rows inside the already-frozen CPI/NFP/FOMC entry-blackout window are
  excluded from every arm before folds are formed.
- The complete two-hour target window must end by the `21:30` London cutoff.
- Weekend gaps, missing bars, duplicates, non-UTC timestamps, crossed OHLC,
  and non-contiguous target windows are rejected or reason-coded. They are
  never repaired by interpolation.

The existing canonical candle validator remains the minimum integrity gate.
The experiment manifest must additionally record source path, SHA-256,
first/last timestamp, row count, duplicate count, missing-interval count,
excluded-row counts by reason, and the development cutoff.

### 3.2 GVZ implied volatility

The registered implied-volatility source is FRED series `GVZCLS`, whose source
is Cboe and whose observation is a daily close. The fetched raw series must be
saved with retrieval time, source URL, unmodified values, and SHA-256 before it
is transformed.

Primary series page: <https://fred.stlouisfed.org/series/GVZCLS>

An observation dated `D` becomes usable only at `D + 2 calendar days 00:00
UTC`, preserving the existing conservative publication lag. It may be carried
forward for at most five calendar days. A longer gap makes the feature missing;
it is not filled from a later observation.

FRED holiday rows whose value is missing are retained in the pinned raw
snapshot but removed before transformation: they are dated placeholders, not
GVZ closes. The five-day change therefore means five actual observations.
Observed and placebo construction use the same cleaned observation index, so
shifting values cannot change the matched price-row sample.

Exactly two implied-volatility columns are registered:

- `gvz_log`: log of the lagged GVZ level; and
- `gvz_chg_5d`: five-observation change in log GVZ.

### 3.3 Scheduled events

The registered events are regular:

- US Consumer Price Index releases;
- US Employment Situation releases; and
- scheduled FOMC statement/rate decisions.

Dates and stated release clocks must come from the official BLS and Federal
Reserve calendars. BLS `08:30` and FOMC `14:00` clocks are localized in
`America/New_York` and then converted to UTC, so daylight-saving changes are
not approximated with a fixed UTC offset.

Primary schedule roots:

- <https://www.bls.gov/schedule/>
- <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm>

The experiment starts in 2021 to avoid treating the exceptional 2020 emergency
and cancelled-meeting sequence as an ordinary point-in-time calendar. Emergency
Fed actions, speeches, minutes, press conferences, revised release outcomes,
and event surprises are excluded. The event snapshot must record event kind,
official date, official local clock, timezone, UTC timestamp, primary-source
URL, verification date, and SHA-256. Any ambiguous or unverified event is
excluded with a reason rather than assigned a guessed time.

Exactly two calendar columns are registered at each eligible decision time:

- `bls_release_next_120m`: one if a CPI or Employment Situation release falls
  strictly after the decision and no later than 120 minutes after it; and
- `fomc_release_next_120m`: the equivalent flag for a scheduled FOMC statement.

The columns describe schedule proximity only. Actual CPI, payroll, or FOMC
outcomes are forbidden.

### 3.4 Matched rows

One master dataset is built with the target, price/session baseline, GVZ, and
calendar columns present. Both ablation arms use the identical row index and
the identical fold boundaries. A family may not appear to improve merely
because its missing values changed the sample.

Report master coverage and every exclusion reason. If fewer than 90% of
otherwise eligible price rows survive the matched-row contract, the program
stops as a data-quality failure and no family verdict is issued.

## 4. Target: observed two-hour realized volatility

For a signal candle with close `C_t`, define the next four contiguous M30 close
returns:

```text
r_(t+i) = log(C_(t+i) / C_(t+i-1)),  i = 1..4
RV_2h(t) = sqrt(sum(r_(t+i)^2, i=1..4))
y_t = log(RV_2h(t))
```

All four later candles must exist at exact 30-minute intervals. `C_t` and every
future close are observed Dukascopy values. A zero or non-finite target is
reason-coded and excluded. No high/low estimator, direction, trade return,
stop result, or candidate label is used.

Two hours is the only horizon in this program. It is the full P3 timeout and a
common near-term risk window for P1/P2; adding a four-hour alternative after
seeing results would be a new registered experiment.

## 5. Price-and-session baseline B0

B0 is a linear intraday HAR-style benchmark. Its inputs, all known at the
signal close, are fixed as:

- log root-sum-squared M30 close returns over the trailing 4 valid intervals;
- the same quantity over the trailing 16 valid intervals;
- the same quantity over the trailing 48 valid intervals;
- sine and cosine of the Europe/London decision minute over a 24-hour cycle;
  and
- fixed weekday indicators for Tuesday through Friday, with Monday as the
  reference category.

Here a valid interval means two adjacent observed candles exactly 30 minutes
apart. Each trailing window uses the most recent 4, 16, or 48 valid intervals,
includes the return ending at the signal close, and includes no later return.
An invalid cross-gap return is omitted rather than treated as a market move;
no candle or return is synthesized to replace it. A row is missing until the
required number of valid historical intervals exists. No ATR variant, price
direction, strategy identity, candidate feature, or learned feature selection
is added.

The model is ordinary least squares. Within each fold, standardization means
and standard deviations are fitted on training rows only. No LightGBM,
hyperparameter search, regularization sweep, or alternate feature set is part
of this program.

## 6. Registered comparisons

There are exactly two primary ablations:

1. **F-IV:** `B0 + gvz_log + gvz_chg_5d` versus B0.
2. **F-CAL:** `B0 + bls_release_next_120m + fomc_release_next_120m` versus B0.

There is no combined model in this program. If both families pass, combining
them requires another pre-registration before its outcome is read. If neither
passes, no new volatility feature is selected from these results.

## 7. Walk-forward evaluation

- Five expanding, chronological folds.
- At least four M30 rows are purged between each training and test block.
- Standardization, coefficients, and the training-target mean benchmark are
  fitted separately inside each fold.
- The primary metric is pooled Campbell-Thompson out-of-sample `R²`, measured
  against that fold's training-target mean.
- Report absolute B0 and augmented `R²`, their difference, per-fold
  differences, row counts, date boundaries, and effective sample size after
  overlapping-target uniqueness.
- No random split, shuffle, test-fold calibration, or full-sample scaling.

Predictions are out-of-fold audit values only. No artifact receives a sizing
threshold or paper-trading permission.

## 8. Placebos and retention gate

Each family receives 20 circular time-shift placebos drawn without replacement
with NumPy seed `0` using the existing shift rules. GVZ shifts preserve its observation sequence. Event
flags shift in whole-week increments so weekday structure is preserved. No
shift may land on the original alignment or inside the exclusion buffer around
it.

A family qualifies as an intraday risk-input candidate only if every condition
holds:

1. augmented pooled out-of-sample `R²` is above zero;
2. matched-row `delta R²` is above zero;
3. `delta R²` is positive in at least four of five folds;
4. observed `delta R²` exceeds the maximum of its 20 placebos; and
5. the data-quality and 90% matched-coverage gates pass.

No minimum absolute `delta R²` is substituted for the placebo bar. There are
two registered family comparisons and 40 placebo runs in the look ledger.
Passing is a development result, not confirmation.

B0 itself qualifies as the price/session risk forecast only if its pooled
out-of-sample `R²` is above zero and its per-fold `R²` is above zero in at least
four of five folds. If B0 fails, no augmented family can activate a risk policy
even if its relative delta passes; improving a forecast that remains unusable
in absolute terms is insufficient.

## 9. Causality and accuracy tests required before evaluation

The implementation must pass synthetic tests establishing that:

1. a feature row built from truncated history equals the same row built from
   the full history;
2. changing any price after decision time cannot alter that decision's
   features;
3. changing prices inside the next four bars changes the target but not the
   features;
4. every target window contains exactly four contiguous future bars;
5. London and New York DST transitions produce the intended decision and
   release timestamps;
6. unverified calendar ranges and ambiguous events fail closed;
7. GVZ is unavailable before its two-day lag and after its staleness limit;
8. both ablation arms use identical rows and folds;
9. train/test purging prevents overlapping future windows from crossing the
   boundary;
10. confirmation timestamps are refused by default; and
11. the placebo generator cannot reproduce a near-original alignment.

The implementation may be debugged with synthetic data. No real-data metric,
prediction file, family delta, feature bin, or chart may be generated until
these tests pass and this document is committed.

## 10. What happens after the result

- **Failure:** retire that family for the first intraday risk layer. Do not try
  neighboring horizons, lags, release windows, models, or volatility features
  in this program.
- **Pass:** retain the family only as an input candidate. The already-registered
  mapping in `docs/intraday_risk_policy.md` remains disabled until the B0 and
  family gates determine which forecast is eligible. Risk may never exceed the
  frozen `0.25%` per-trade cap.
- Because no primary gold strategy is currently retained, this experiment will
  not evaluate whether sizing rescues P1/P2/P3 returns. Applying a risk model to
  failed strategies and selecting the most flattering P&L would be an
  unregistered alpha search.
- The historical confirmation window remains closed until a complete trading
  strategy and risk policy are frozen together.

## 11. Look ledger

| Look | Date | Item | Outcome available? |
|---:|---|---|---|
| 0 | 2026-07-31 | Design written; no implementation run | no |
| 1 | 2026-07-31 | Causal builder, OLS walk-forward, ablation, guarded CLI, and synthetic tests implemented | synthetic only |
| 2 | 2026-07-31 | First real invocation aborted before writing a report because shifted FRED holiday NaNs changed placebo row availability; cleaned non-observations and added a fixed-row regression test | no result written |
| 3 | 2026-07-31 | Corrected retry stopped before evaluation when FRED DNS was unavailable; changed the harness to verify and reuse the already-pinned raw GVZ snapshot before attempting a fetch | no result written |
| 4 | 2026-07-31 | Registered development evaluation on 28,513 matched rows and 40 placebos | B0 R2 OOS 0.2032 passed; F-IV delta R2 0.0210 passed 5/5 folds; F-CAL delta R2 0.0052 passed 4/5 folds; frozen both-pass rule selects B0 only; confirmation unopened |

## 12. Registered execution command

After both this document and `docs/intraday_risk_policy.md` are committed and
clean, the one registered real-data command is:

```bash
python scripts/evaluate_intraday_risk.py
```

It has no model, horizon, feature, fold, placebo-count, or seed arguments. The
registered values are constants so the command cannot become a parameter
sweep. It writes research artifacts with `paper_enabled=False`; a forecast
result cannot silently activate risk sizing.
