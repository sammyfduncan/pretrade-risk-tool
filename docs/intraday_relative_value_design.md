# RV1 intraday gold–silver residual-reversal experiment

**Frozen:** 2026-08-03 before acquiring XAGUSD M30 quotes or calculating any
gold–silver intraday spread outcome  
**Development:** synchronized quote records strictly before `2025-07-01`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Status:** complete — honest negative; RV1 retired

## 1. New trading problem

Version-one outright-gold directional discovery is closed. RV1 is not another
gold direction feature: it trades a temporary relative move between gold and
silver while normalizing total gross exposure to one. The economic premise is
limited and falsifiable:

> Gold and silver share a precious-metals factor, but silver's additional
> industrial exposure and higher volatility can create short-lived relative
> dislocations. After hedging their recent return relationship, does an
> unusually large M30 residual partially reverse over the next three hours?

This is a convergence hypothesis, not a claim that the gold/silver price ratio
has a fixed fair value or is permanently stationary. No level-ratio threshold,
macro input, futures feature, news feature, or outright trend rule is included.

## 2. Quote-data contract

Use Dukascopy `XAU/USD` and `XAG/USD` M30 bid and ask candles from
`2020-01-01` through `2025-07-01` exclusive. Fetch each offer side separately
and pin all four Parquet artifacts with hashes. Signal calculations use the
bid/ask midpoint. Executions use the quoted sides:

- buy at ask open and sell at bid close;
- sell at bid open and buy back at ask close.

Only exact shared timestamps are retained. A row is ineligible when either leg
is absent, the next entry bar is not contiguous in both legs, ask is below bid
at an open or close, either price is non-positive/non-finite, or a source bar
violates OHLC integrity. No candle, quote, or timestamp is filled.

The outcome-blind stage may inspect only schema, timestamp bounds, hashes,
duplicates, OHLC/quote integrity, shared-row coverage, and gap conditions. It
may not calculate residuals, signals, forward returns, candidates, or P&L.

## 3. Causal hedge and residual state

For synchronized signal bar `t`, define midpoint open-to-close log returns
`g_t` and `s_t` for gold and silver. Using the previous 1,000 valid synchronized
M30 observations, excluding `t`, estimate a through-origin hedge ratio:

`beta_t = sum(g_i * s_i) / sum(s_i^2)`.

The row is ineligible unless the full 1,000-observation history exists,
`0.10 <= beta_t <= 1.50`, and the same reference window's gold/silver Pearson
correlation is at least `0.50`. These are pair-validity conditions, not
candidate variants.

Calculate the current residual using only the completed signal bar:

`e_t = g_t - beta_t * s_t`.

For the same Europe/London decision clock, use the previous 60 observed causal
residuals, excluding the current residual, to calculate:

`z_t = (e_t - prior_median_60) / (1.4826 * prior_MAD_60)`.

The robust residual scale is `q_t = 1.4826 * prior_MAD_60`. Incomplete
references and zero MAD are ineligible; values are never inferred.

## 4. Exact RV1 bare rule

Consider decisions whose next M30 open is in `[08:00, 18:00)` Europe/London.
Existing CPI, Employment Situation, and FOMC entry blackouts apply.

- `z_t >= +2.00`: short the residual—short gold and long silver.
- `z_t <= -2.00`: long the residual—long gold and short silver.
- Otherwise there is no candidate.
- Take the first eligible RV1 candidate per London day.
- Entry is the next contiguous synchronized M30 open.

Freeze gross-normalized signed notionals at the signal:

- long residual: `w_gold = +1/(1+beta)`,
  `w_silver = -beta/(1+beta)`;
- short residual: negate both weights.

Thus `abs(w_gold) + abs(w_silver) = 1`. No weight is updated while a trade is
open. Define one initial risk unit in portfolio-return space as
`R = q_t / (1 + beta_t)`.

At each subsequent synchronized M30 close, calculate both legs' executable
bid/ask returns from their actual entry opens and sum them by frozen signed
notional. Exit at the first close satisfying:

- gross portfolio return `>= +1.00R`: target;
- gross portfolio return `<= -1.00R`: stop;
- six completed M30 bars: timeout; or
- the existing `21:30` London forced-flat cutoff.

Target and stop are close-observed barriers. Intrabar gold and silver extremes
cannot be combined because they need not occur simultaneously.

## 5. Costs and execution stresses

Base P&L embeds the historical Dukascopy bid/ask spread on both legs and
subtracts an additional `0.5 bp` per side on normalized gross notional for
commission/slippage. Report the same trades under:

1. observed spread plus `0.5 bp` per side;
2. `1.5x` every observed half-spread plus `0.75 bp` per side; and
3. `2.0x` every observed half-spread plus `1.0 bp` per side.

Spread stress moves bids and asks symmetrically away from their midpoint before
entry/exit. It does not change signals, candidates, weights, or exit clocks;
only executable P&L is remeasured. Actual IG quote/fill validation remains
mandatory before paper trading even if this historical gate passes.

## 6. Evaluation and empirical null

Use five equal chronological blocks of executed trades. Report count, side
balance, win rate, PF, mean net R, compounded trade return, maximum drawdown,
exit reasons, holding time, beta distribution, and all cost stresses.

Construct 20 equally spaced non-zero circular shifts of the complete joint RV1
state rows—eligibility, z, beta, and scale—relative to the synchronized quote
path. Re-run first-per-day selection and the complete two-leg execution for
every shift. This preserves the state distribution and gold/silver state
relationship while destroying its timestamp alignment with future convergence.

RV1 passes development only if every condition holds:

1. exact synchronized quote coverage is at least 95% after warm-up;
2. at least 200 executed trades remain, with at least 30 per block;
3. long-residual and short-residual trades each comprise at least 30%;
4. base PF is at least `1.15` and mean net R is positive;
5. at least four of five blocks have PF above `1.00`;
6. PF remains above `1.00` under the `2x` spread/cost stress;
7. maximum drawdown is no worse than `-10%`; and
8. observed base PF exceeds the maximum PF of all 20 time-shift placebos.

A pass permits one frozen confirmation construction. It does not enable paper
trading, authorize live execution, or activate ML. ML may be considered only
after the bare rule passes and under a new preregistration.

## 7. Failure and research stop

Failure retires RV1 in full. No beta window, correlation floor, z threshold,
single-leg variant, ratio-level variant, stop, target, timeout, session, weight,
or ML model may be used as a rescue in this program. Confirmation remains
unopened.

RV1 is the only initial intraday precious-metals relative-value family. A
failure closes this family and requires a separately scoped mechanism—such as
multi-commodity cross-sectional trend—not another gold/silver convergence
variant.

## 8. Multiple-comparisons ledger

RV1 adds one primary strategy and 20 registered null alignments. The nulls are
one placebo family, not 20 strategies. All earlier outright, ML, benchmark,
futures-state, order-flow, and macro-impulse failures remain in the project's
broader research history.

## 9. Development result

The four free Dukascopy quote streams were pinned with hashes. Exact
synchronization retained 64,955 M30 rows from 64,958 timestamps in the union:
`99.995%` coverage, no filling, no duplicate timestamps, and no crossed
open/close quotes. The causal state contained 61,075 eligible rows after the
fixed 1,000-observation hedge and same-clock warm-ups.

RV1 **failed**:

| Measurement | Result | Gate |
|---|---:|---:|
| Synchronized coverage | 99.995% | at least 95% — pass |
| Executed trades | 1,014 | at least 200 and 30/block — pass |
| Long/short residual share | 50.0% / 50.0% | each at least 30% — pass |
| Base PF | 0.303 | at least 1.15 — fail |
| Mean net R | -1.256 | above 0 — fail |
| Win rate | 28.4% | reported, not gated |
| Profitable blocks | 0/5 | at least 4/5 — fail |
| Double-stress PF | 0.083 | above 1.00 — fail |
| Maximum drawdown | -43.98% | no worse than -10% — fail |
| Best of 20 placebo PFs | 0.274 | observed must exceed — pass |

The result reproduced across all five chronological blocks: PFs were `0.373`,
`0.236`, `0.268`, `0.311`, and `0.326`. There were 702 stop exits, 272 targets,
and 40 timeouts. No candidate was silently removed and the two sides had nearly
identical mean losses.

### 9.1 Execution audit

Because the executable result was unusually poor, a no-selection audit
remeasured the same trades without changing candidate or exit clocks:

- midpoint PF at the frozen exits was `1.005`, with mean return about `0.022 bp`;
- the immediate next-M30 midpoint reversal had PF `1.042`, mean return about
  `0.126 bp`, and 53.4% wins;
- historical two-leg bid/ask spread drag averaged `4.739 bp` round trip before
  the registered additional `1 bp` allowance.

This rules out an execution-sign defect and locates the failure economically:
the residual contains at most a negligible short-horizon reversal tendency,
far smaller than executable two-leg friction. Its slight advantage over the
placebos does not make it tradeable.

Per §7, RV1 is retired without beta-window, threshold, single-leg, ratio-level,
exit, session, cost-exclusion, or ML rescue. Confirmation remains unopened and
paper trading remains disabled. Further commodity research requires a new
mechanism, with multi-commodity cross-sectional trend the leading candidate.
