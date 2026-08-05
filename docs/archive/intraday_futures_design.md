# Intraday GC futures-information experiment

**Frozen:** 2026-07-31 before generating any futures-informed filter outcome  
**Development:** gold candidates and CME data strictly before `2025-07-01`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Scope:** one feature family, one strategy, one model

## 1. Hypothesis and control

The registered hypothesis is that real COMEX GC participation and price
leadership can distinguish genuine P1 London-session breakouts from weak spot
breakouts. The only tested pair is **P1 + logistic regression**. This is the
strongest existing mechanistic lead: its spot-only filter improved PF in all
five development folds but did not become profitable.

The matched control is the identical P1 candidate universe, labels, costs,
folds, uniqueness weights, model, chronological reference tails, and 30%
retention rule using only the frozen spot feature set. The challenger adds the
entire futures family below. No P2, P3, V2, B1, model, threshold, or subgroup
alternative is opened in this experiment.

## 2. Causal active-contract rule

GC sessions are labelled by their New York trading date: activity from 18:00
through the following 17:00 belongs to the following date. After each session
has completed, sum its outright-contract one-minute volume. From the next
session onward, use the contract with the greatest completed-session volume;
ties resolve by symbol. The selected contract remains fixed until another
session completes.

This deliberately lags the usual volume-roll observation. Same-session final
volume, future definitions, back-adjusted prices, and post-cutoff records are
forbidden. A contract identity is joined through the definition available at
or before each source record.

## 3. Registered feature family F-GC

At each completed M30 spot signal bar, aggregate only one-minute records from
the causally selected contract. Add exactly six candidate inputs:

1. `gc_side_return_30m`: candidate-side signed GC return over that M30 bar;
2. `gc_side_return_2h`: candidate-side signed sum of the latest four contiguous
   same-contract M30 log returns;
3. `gc_volume_surprise_30m`: log active-contract volume minus the median for
   the same London half-hour over the preceding 20 observations;
4. `gc_side_spot_gap_30m`: candidate-side signed GC return minus spot return;
5. `gc_open_interest_log_change`: latest published active-contract open
   interest versus its preceding distinct reference date; and
6. `gc_basis_deviation`: active GC/spot log basis minus its preceding-20 M30,
   same-contract median.

All rolling references exclude the current observation. Any missing input
rejects that candidate from both the spot control and challenger. Coverage
must be at least 90% of otherwise executed P1 candidates.

## 4. Vendor-quality exclusion

`missing` or `pending` Databento dates make the source inadmissible. A
`degraded` UTC source date and the following 28 calendar days are excluded from
both arms. This conservatively removes the affected observation plus every
20-session rolling dependency. The excluded dates come only from the pinned
source manifest and may not be changed after outcomes are read.

## 5. Temporal evaluation and placebos

Both arms use the already frozen nested filter procedure in
`docs/archive/intraday_filter_gate.md`: five expanding outer folds; exit-time purges;
an earlier model-fit prefix; a final 25% chronological reference tail; and the
reference-score 70th percentile applied unchanged to the later test block.

The complete six-column futures family is circularly shifted jointly through
the candidate sequence at 20 equally spaced non-zero offsets. Each placebo is
retrained and thresholded identically. Joint shifts preserve correlations
inside the family while destroying its correct timestamp association.

## 6. Retention gate

F-GC passes only if every condition holds:

1. matched usable coverage is at least 90%;
2. challenger pooled OOF AUC exceeds the spot control by at least `0.015`;
3. challenger AUC exceeds the control in at least four of five folds;
4. observed AUC improvement strictly exceeds the maximum of 20 placebos;
5. the challenger passes every economic gate in
   `docs/archive/intraday_filter_gate.md` (including PF `>=1.10`, positive mean R,
   at least 200 trades, at least 30 per fold, improved PF in four folds,
   better drawdown, and PF above one at double cost); and
6. challenger pooled PF strictly exceeds the matched spot-only filter PF.

A failure retires this F-GC/P1 configuration. No individual feature is rescued
from a failed family and no nearby roll, lookback, threshold, strategy, or
model is tried. A pass permits a separately frozen confirmation run; it does
not deploy a model.

## 7. Pre-outcome feasibility correction

The first execution stopped before fitting any model: fold zero had fewer than
the frozen minimum rows. A feature-availability audit showed that five inputs
covered 893–901 of 902 candidates, while the original basis reference covered
only 536. Requiring 20 observations at the same clock slot inside each expiring
contract was structurally too sparse.

Before any label metric or return result was generated, item 6 was corrected
to use the preceding 20 M30 observations in the same contract. This remains a
shifted, causal local basis reference and changes no outcome-driven choice.
The volume-surprise feature retains its same-London-slot reference because its
coverage was already adequate. The 90% gate and every other rule remain fixed.

## 8. Development outcome

F-GC failed. The causal feature build retained 844 of 902 candidates (93.6%),
so data coverage passed. The matched spot-only filter had pooled OOF AUC
`0.5294`; adding the full futures family reduced it to `0.5187`, a delta of
`-0.0107`. AUC improved in only two of five folds, while the best of 20 joint
time-shift placebos achieved a `+0.0195` delta.

The economic result was also negative:

| Metric | matched spot filter | F-GC challenger |
|---|---:|---:|
| filtered trades | 234 | 218 |
| profit factor | 0.818 | 0.862 |
| mean net R | — | -0.100 |
| PF at double cost | — | 0.694 |

The small PF improvement does not rescue the family: it remained below one,
mean R stayed negative, one fold fell below the minimum trade count, and the
ranking evidence was worse than both the matched control and placebo noise.
No individual F-GC feature, alternate roll rule, P2/P3 transfer, nonlinear
model, or threshold will be inspected under this program. Confirmation remains
unopened and no deployment model was created.
