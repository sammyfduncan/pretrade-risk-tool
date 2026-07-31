# Intraday GC trade-flow experiment

**Frozen:** 2026-07-31 before obtaining or inspecting GC TBBO data  
**Development:** P1 candidates and exchange records strictly before `2025-07-01`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Status:** importer, causal feature core, and gate complete; source delivery not yet present

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

The required Databento batch query is:

- dataset: `GLBX.MDP3`;
- parent: `GC.FUT`;
- schema: `tbbo`;
- output symbology: `instrument_id`;
- encoding: DBN;
- compression: Zstandard; and
- monthly file splitting for the eventual full history.

First obtain only `2025-05-01` through `2025-06-01` as a non-performance
validation delivery. It will test schema fields, aggressor-side coverage,
timestamp/definition mapping, active-contract coverage, compressed size, and
processing time. No candidate labels or returns may be joined in that stage.
The full `2020-01-01` through `2025-07-01` delivery is justified only if the
sample has at least 95% explicit `A`/`B` aggressor sides and the normalized
pipeline passes.

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

Add exactly six inputs to the matched P1 logistic filter:

1. `gc_side_aggressive_imbalance_5m`: candidate-side signed aggressive volume
   imbalance over the preceding five minutes;
2. `gc_side_aggressive_imbalance_30m`: the same measure over 30 minutes;
3. `gc_side_depth_imbalance_5m`: candidate-side signed, trade-size-weighted
   pre-trade top-of-book size imbalance over five minutes;
4. `gc_spread_ticks_5m`: trade-size-weighted pre-trade spread in GC ticks over
   five minutes;
5. `gc_side_sweep_imbalance_5m`: candidate-side signed volume from trades at
   least as large as displayed opposing best size, divided by total volume;
6. `gc_trade_intensity_surprise_5m`: log five-minute trade count minus the
   preceding-20-session median at the same London clock minute.

`B` means buy aggressor and `A` sell aggressor. `N` contributes to total
activity but zero signed flow. Current observations are excluded from every
historical reference. A missing feature removes the candidate from both arms.

## 5. Evaluation and retention gate

The matched control remains P1 + logistic with the frozen spot-only features.
The challenger adds all six OF-GC inputs. Both use identical candidates,
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
