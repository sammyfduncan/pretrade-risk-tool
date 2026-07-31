# Matched M15 execution challenger

**Frozen:** 2026-07-31 before reading any M15 replay outcome  
**Scope:** corrected gold development candidates ending before `2025-07-01`  
**Candidate rules:** unchanged P1V2 and P3V2 M30 signals only  
**Confirmation:** remains unopened

## 1. Question

P1V2 and P3V2 showed small positive gross effects that did not clear the fixed
two-basis-point round-trip allowance. This experiment asks one narrow question:

> Does resolving the exact same trades on M15 candles improve executable path
> fidelity and net economics relative to M30 resolution?

It does not test an M15 strategy, create extra opportunities, change a side, or
alter any V2 signal threshold.

## 2. Timing and trade contract

M30 candle timestamps are bar-open times. A candidate stamped `09:30` is known
only when that candle closes at `10:00`. The M15 replay therefore:

- anchors availability to the M15 candle opening at `09:45`, which closes at
  `10:00`;
- enters at the next contiguous M15 open, exactly `10:00`, matching the M30
  entry timestamp;
- retains the original side and signal-close absolute ATR stop distance;
- retains the original target in R;
- doubles the maximum bar count so the wall-clock timeout is unchanged;
- retains the same London session, cutoff, event-blackout, and cost policy; and
- uses the same adverse ordering if a single M15 candle touches both barriers.

The candidate universe is fixed to trades executed by the corrected M30
ledger. Each is replayed independently on M15. This prevents a different exit
time from changing which later candidates enter the comparison.

## 3. Matching and diagnostics

The M15 result is matched to its M30 control by original strategy and M30
signal timestamp. Missing/non-contiguous M15 paths are reason-coded, never
silently dropped. The report includes:

- replay coverage;
- base, 1.5x, and 2x cost metrics;
- M30 and M15 PF, mean R, and trade-level maximum drawdown on identical
  successfully replayed trades;
- five equal chronological blocks of paired mean-R differences; and
- exit-reason transition counts, which reveal cases where finer candles alter
  the conservative M30 barrier ordering.

## 4. Retention gate

M15 is retained for a strategy only if every condition holds:

1. at least 95% of the corrected M30 executed candidates replay successfully;
2. M15 base-cost PF is at least `1.10` and mean net R is positive;
3. M15 PF and mean R both exceed the matched M30 control;
4. the paired M15-minus-M30 mean-R difference is positive in at least four of
   five chronological blocks;
5. M15 trade-level maximum drawdown is less severe than M30; and
6. M15 PF remains above `1.00` at double cost.

P1V2 and P3V2 are the only registered comparisons. P2V2 had no positive gross
effect and is excluded. There is one M15 construction and no alternative
confirmation delay, timeout, barrier, or entry rule. A failure retires M15
execution for that V2 rule; it does not authorize parameter tuning.

## 5. Interpretation boundary

Dukascopy M15 is still a bid-only historical feed with a fixed cost allowance.
A passing historical replay would be a research lead only. Promotion requires
recent IG M15 bid/ask parity and observed spread/slippage to pass the existing
instrument-admission gates.

## 6. Development outcomes

Both registered replays achieved 100% coverage and failed the frozen gate:

| Rule | Trades | PF M30 -> M15 | Mean R M30 -> M15 | Improved blocks | M15 PF at 2x cost |
|---|---:|---:|---:|---:|---:|
| P1V2 | 323 | 0.928 -> 0.937 | -0.036 -> -0.029 | 1/5 | 0.773 |
| P3V2 | 185 | 1.000 -> 1.050 | -0.015 -> +0.006 | 2/5 | 0.843 |

M15 changed only three coarse stop outcomes into targets: one P1V2 trade and
two P3V2 trades. All other stop, target, timeout, and cutoff classifications
were unchanged. The finer path therefore modestly improved PF and drawdown,
but the effect was sparse, temporally unstable, below the `1.10` PF gate, and
not robust to double costs.

M15 execution is not retained for either V2 rule. No M15 signal model was
trained and confirmation remains unopened.
