# Intraday GC trade-flow experiment

**Frozen:** 2026-07-31 before obtaining or inspecting GC TBBO data  
**Development:** P1 candidates and exchange records strictly before `2025-07-01`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Status:** complete — honest negative; OF-GC retired

## 1. Why this is a different mechanism

The F-GC experiment showed that M30 returns, aggregate volume, open interest,
basis, and contract state did not improve P1. This program does not split or
retune that failed family. It asks a different question:

> Immediately before a spot breakout decision, are aggressive GC traders
> consuming liquidity in the candidate direction, and is the visible top of
> book supportive or resistant?

Databento TBBO is selected instead of plain `trades`, MBP-1, or MBO. TBBO
contains every trade, aggressor side, and the bid/ask prices and sizes
immediately before that trade. It therefore supports signed flow and displayed
liquidity measurements. It is sampled in trade space and omits the much larger
stream of non-trade book updates carried by MBP-1/MBO.

## 2. Data contract and feasibility stage

The original validation query was:

- dataset: `GLBX.MDP3`;
- parent: `GC.FUT`;
- schema: `tbbo`;
- output symbology: `instrument_id`;
- encoding: DBN;
- compression: Zstandard; and
- monthly file splitting for the validation history.

First obtain only `2025-05-01` through `2025-06-01` as a non-performance
validation delivery. It will test schema fields, aggressor-side coverage,
timestamp/definition mapping, active-contract coverage, compressed size, and
processing time. No candidate labels or returns may be joined in that stage.
The sample had to reach at least 95% explicit `A`/`B` aggressor sides and pass
the normalized pipeline before any development acquisition.

### 2.1 Pre-outcome cost-feasibility amendment (2026-07-31)

The vendor quoted approximately $400 for continuous parent-symbol TBBO history.
That cost is not proportionate to one unconfirmed family. No label, return,
model score, or OF-GC performance result had been read when this amendment was
made.

Development acquisition is therefore restricted to the exact 30 minutes before
each already-frozen P1 decision and to the raw GC contract selected by the
already-frozen prior-completed-session rule. Candidate times and prior-session
volume determine the request; outcomes do not. Overlapping windows for the same
contract are merged. `scripts/plan_cme_gc_tbbo_windows.py` writes the auditable,
checksummed request plan. Exact cost must be checked with Databento metadata
before any download.

This sparse design cannot honestly construct a preceding-20-session
same-clock trade-count baseline without acquiring many additional disjoint
windows. The registered intensity-surprise input is therefore withdrawn before
performance evaluation. The five direct measurements below remain frozen.
Coverage and retention thresholds are unchanged; this is not a relaxed gate.

`scripts/import_cme_gc_tbbo.py` verifies every source checksum and query field,
maps reused instrument IDs through definitions as of receive time, retains
outright futures only, applies the historical confirmation fence, and writes
monthly checksummed Parquet partitions. It creates no feature or result.

## 3. Active contract and timing

Use the already frozen causal active-contract rule: after each 17:00 New York
GC session completes, select its outright volume leader for the next session.
Same-session final volume and future definitions are forbidden.

TBBO is indexed by capture-server receive time. A feature at decision time
`t` uses only records with receive timestamps strictly before `t`. One-minute
buckets become available at their ending boundary; five- and 30-minute windows
are trailing and include no post-decision event.

## 4. Registered OF-GC family

Add exactly five inputs to the matched P1 logistic filter:

1. `gc_side_aggressive_imbalance_5m`: candidate-side signed aggressive volume
   imbalance over the preceding five minutes;
2. `gc_side_aggressive_imbalance_30m`: the same measure over 30 minutes;
3. `gc_side_depth_imbalance_5m`: candidate-side signed, trade-size-weighted
   pre-trade top-of-book size imbalance over five minutes;
4. `gc_spread_ticks_5m`: trade-size-weighted pre-trade spread in GC ticks over
   five minutes;
5. `gc_side_sweep_imbalance_5m`: candidate-side signed volume from trades at
   least as large as displayed opposing best size, divided by total volume.

`B` means buy aggressor and `A` sell aggressor. `N` contributes to total
activity but zero signed flow. Current observations are excluded from every
historical reference. A missing feature removes the candidate from both arms.

## 5. Evaluation and retention gate

The matched control remains P1 + logistic with the frozen spot-only features.
The challenger adds all five OF-GC inputs. Both use identical candidates,
labels, costs, uniqueness weights, five outer folds, exit-time purges,
chronological reference tails, and the frozen 30% threshold rule.

The family is jointly circular-shifted at 20 equally spaced non-zero candidate
offsets. OF-GC passes only if all conditions hold:

1. usable matched coverage is at least 90%;
2. challenger OOF AUC improvement is at least `0.015`;
3. AUC improves in at least four of five folds;
4. observed AUC improvement exceeds all 20 placebos;
5. the challenger passes the complete economic filter gate, including PF
   `>=1.10`, positive mean R, sample minima, improved fold PF/drawdown, and PF
   above one at double cost; and
6. challenger pooled PF exceeds the matched spot-only filter PF.

This is the second new-data family after F-GC and is recorded as such in the
multiple-comparisons ledger. Failure retires the entire family: no single
window, feature, model, threshold, P2/P3 transfer, or nearby definition is
rescued. A pass permits a separately frozen confirmation construction; it does
not create a live or paper model.

## 6. Non-performance validation outcome

The pinned sample job `GLBX-20260731-YS9X3DAYXC` covers `2025-05-01` through
`2025-06-02` exclusive (the UI included 1 June). It passed every manifest hash
and reported no degraded dates.

| Structural measurement | Result |
|---|---:|
| compressed DBN size | 68 MB |
| raw TBBO trades | 3,211,804 |
| outright trades | 2,981,510 |
| valid outright pre-trade books | 99.9737% |
| causal active-contract trades | 2,697,591 |
| explicit active `A`/`B` sides | 97.2927% |
| valid active pre-trade books | 99.9992% |
| invalid active book records | 21 |

The 21 invalid active records were identifiable session-opening `N` events
with crossed startup snapshots. They are retained with `book_valid=false` in
the normalized source and excluded before aggregation; nothing is imputed.

The unlabeled streaming builder produced 46,080 causal minute rows. This stage
caught and fixed two correctness defects before performance work: unsigned
book-size subtraction could wrap negative depth imbalance, and a calendar grid
could make the 20-session intensity reference count closed-session gaps. Tests
now pin signed floating-point depth arithmetic, prior observed sessions,
partition/single-frame equivalence, and post-decision poisoning.

The feasibility stage therefore passes, but the quoted full-history cost was
rejected under §2.1. Only the frozen sparse candidate-window plan may be priced
or acquired. No label, candidate return, model score, or strategy result was
read.

## 7. Sparse acquisition quote

On 2026-08-03, `Historical.metadata.get_cost` quoted all 901 planned windows at
**$7.2312 total**. The plan contains 450.5 requested hours across 29 raw
contracts, covers 901/902 frozen P1 candidates (99.89%), and has a maximum
single-window quote of $0.0775. One initial candidate is excluded because it
precedes the first completed CME session and therefore has no causal active
contract selection. The quote used metadata only: no paid endpoint was called,
no data was transmitted, and no strategy outcome was read.

Acquisition remained blocked until the user explicitly authorized the $7.24
ceiling. The request plan, per-window quotes, and resumable quote/download
ledgers are gitignored research artifacts.

## 8. Development outcome (2026-08-03)

The user authorized the $7.24 ceiling and the API downloaded all 901 planned
windows for the frozen $7.2312 quote. The checksummed delivery contains
3,466,268 outright TBBO trades: 98.57% have an explicit aggressor side and
99.99994% have a valid pre-trade book. The six dataset-degraded dates already
present in the aggregate CME condition ledger were carried into the conservative
quality exclusion. No confirmation record at or after `2025-07-01` was read.

OF-GC **failed** its registered gate:

| Measurement | Matched control | OF-GC challenger |
|---|---:|---:|
| matched coverage | 94.5% | 94.5% |
| pooled OOF AUC | 0.5511 | 0.5301 |
| filtered trades | 236 | 233 |
| filtered profit factor | 0.955 | 0.931 |

The AUC delta was `-0.0210`; zero of five folds improved, while the best of 20
joint circular-shift placebos achieved `+0.0068`. Challenger mean R was `-0.068`
and profit factor at double cost was `0.755`. It therefore failed the AUC,
fold-consistency, placebo, economic, and matched-control conditions.

This is stronger than a marginal miss: the direct order-flow family worsened
both discrimination and trading economics on the identical candidate sample.
Under the frozen failure rule, no individual feature, nearby window, alternate
model, P2/P3 transfer, or confirmation test may be used to rescue OF-GC. The
family is retired and no deployment artifact was created.
