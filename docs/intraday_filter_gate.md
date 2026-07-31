# Intraday ML filter retention gate

**Frozen:** 2026-07-31, before reading any nested-filter return result  
**Scope:** gold development data ending before `2025-07-01`  
**Confirmation:** remains unopened  
**Status:** complete — both registered filters failed; no ML retained

## 1. What was already known

This gate is not being designed from a blank slate. Before it was frozen, the
following development results had already been read:

- six pooled OOF model results: logistic and LightGBM for each of P1/P2/P3;
- bare-rule OOF PF and mean-R diagnostics;
- within-outer-test top-quartile label-lift diagnostics, which are not a
  deployable threshold test because each test fold set its own quantile; and
- fold AUCs.

Those observations selected two narrow leads:

- P1 with logistic regression; and
- P2 with LightGBM.

P1 LightGBM, P2 logistic, and both P3 models are retired from this gate. The
ledger therefore contains six initial model comparisons plus the two nested
filter configurations below. Results from this gate are still development
results; only a later frozen confirmation can be called confirmation.

## 2. Meaning of the filter

The primary rule creates the complete candidate: side, next-bar entry, stop,
target, timeout, and session restrictions. ML supplies only a raw ranking
score. A candidate at or above the registered threshold is taken; one below it
is skipped. ML cannot reverse direction, invent a trade, or alter risk.

## 3. Frozen temporal procedure

For each of the same five expanding outer folds:

1. Purge outer-training candidates whose trades had not exited before the outer
   test block.
2. Reserve the final 25% of that outer-training block as a chronological
   threshold-reference tail, with at least 40 rows.
3. Purge model-fit candidates whose trades had not exited before the reference
   tail.
4. Require at least 100 model-fit rows and both label classes.
5. Fit the registered model on the earlier prefix using uniqueness weights.
6. Score the reference tail and set the threshold to its 70th percentile using
   the `higher` order statistic. Labels in the reference tail do not set the
   threshold.
7. Apply that unchanged model and numeric threshold to the later outer test
   block.

This targets roughly the top 30% of candidates. Ties are all retained: a model
that emits a constant score is not allowed to manufacture a top 30% by using
future rank order.

The 30% fraction was selected for power: it should leave roughly 250–330 pooled
P1/P2 development trades and at least 30 per outer fold. No alternative
retained fractions will be inspected in this program.

## 4. Costs and controls

The identical outer-test candidate universe is the bare-rule control. Gross
trade returns are re-costed at:

- 1.0× the configured per-side historical cost;
- 1.5× cost; and
- 2.0× cost.

For bare and filtered trades report:

- trade count and retained fraction;
- win rate;
- profit factor;
- mean net `R`;
- sequential compounded trade return; and
- sequential trade-level maximum drawdown.

Also report raw-score AUC, the numeric threshold, and bare/filtered PF for each
outer fold. Calibration is excluded: all three trained probability artifacts
had worse Brier loss than the fold base-rate control.

## 5. Retention gate

The candidate filter passes development only if every condition holds:

1. at least 200 filtered trades pooled and at least 30 in every fold;
2. pooled base-cost PF is at least `1.10`;
3. pooled base-cost mean net `R` is positive;
4. base-cost filtered PF exceeds the same-fold bare PF in at least four of five
   folds;
5. filtered maximum drawdown is less severe than the bare control;
6. pooled PF remains above `1.00` at 2× cost.

Failure retires that ML filter for intraday version 1. Passing does not deploy
it: it only permits freezing a final training/reference construction before
the confirmation window is opened.

## 6. Correctness correction

The first run of this gate exposed a pre-existing wiring omission: the
candidate CLI had not passed the already-frozen CPI/NFP/FOMC entry blackouts to
the execution engine. It also included 2019, although the exact event tables
are verified only from 2020. This was a correctness defect, not a performance
choice.

The CLI now builds exact, DST-aware release windows, excludes pre-2020 rows,
and records event rejections in the negative ledger. All six candidate bundles,
all six screening models, and both registered gates were regenerated. The
original 2019-inclusive numbers are superseded. The registered filter choices
were **not** changed after the corrected screening: P1 remains logistic even
though corrected pooled LightGBM AUC was slightly higher.

Corrected pooled OOF screening AUCs were:

| Strategy | Logistic | LightGBM |
|---|---:|---:|
| P1 | 0.5201 | 0.5265 |
| P2 | 0.5056 | 0.5445 |
| P3 | 0.4719 | 0.4461 |

These AUCs do not select a trading threshold and are not evidence of economic
value on their own.

## 7. Corrected development outcomes

The implementation was completed and its synthetic causality tests passed
before either registered gold evaluation was run. Both filters then failed the
frozen gate.

### P1 + logistic regression — FAIL

| Metric | Bare rule | Filter |
|---|---:|---:|
| trades | 750 | 239 |
| profit factor | 0.701 | 0.970 |
| mean net R | -0.203 | -0.044 |
| trade-level max drawdown | -22.1% | -5.4% |
| PF at 2x cost | — | 0.786 |

The filter improved PF in all five folds and materially reduced drawdown, so
the ranker was not completely inert. It nevertheless failed three mandatory
conditions: PF remained below `1.10`, mean R remained negative, and PF at 2x
cost remained below `1.00`. No alternative retained fraction will be tried.

### P2 + LightGBM — FAIL

| Metric | Bare rule | Filter |
|---|---:|---:|
| trades | 925 | 391 |
| profit factor | 0.827 | 0.852 |
| mean net R | -0.133 | -0.099 |
| trade-level max drawdown | -19.4% | -8.4% |
| PF at 2x cost | — | 0.697 |

PF improved in four of five folds. The earliest small-sample LightGBM emitted a
constant score, so the numeric threshold correctly retained the whole fold
rather than manufacturing a top-30% ranking. The filter failed PF, mean-R,
and double-cost conditions.

### Decision

P1 logistic, P2 LightGBM, and the four models retired before this gate are all
retired for intraday version 1. The saved model files remain audit artifacts
with no deployment threshold. The confirmation window remains unopened.
