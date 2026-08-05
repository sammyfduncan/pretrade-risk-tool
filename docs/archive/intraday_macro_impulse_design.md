# P4 intraday cross-market macro-impulse experiment

**Frozen:** 2026-08-03 before acquiring or inspecting ZN/6E development data  
**Development:** gold and driver records strictly before `2025-07-01`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Status:** complete — honest negative; P4 retired and version-one directional discovery closed

## 1. Final version-one directional question

All price-only gold primaries, their ML filters, M15 execution challenger,
LBMA-window rule, aggregate GC futures family, and GC TBBO order-flow family
failed. P4 is not a rescue variant of any of them. It tests an exogenous
mechanism:

> When US Treasury prices and non-dollar currency futures make an aligned,
> unusually large intraday move, does a contemporaneously confirming but not
> overextended gold move continue after the next executable M30 open?

Gold is non-yielding and USD-denominated. Higher Treasury futures prices proxy
falling nominal yields; higher Euro FX futures prices proxy a weaker dollar.
Both are directionally supportive of gold, while the inverse alignment is
directionally adverse. These relationships are economic motivation, not a
claim that a 30-minute trading edge exists.

Reference context:

- [World Gold Council research](https://www.gold.org/goldhub/research/the-impact-of-monetary-policy-on-gold)
  treats interest rates and the US dollar as important gold drivers;
- [CME's Treasury-futures delivery guide](https://www.cmegroup.com/content/dam/cmegroup/trading/interest-rates/files/us-treasury-futures-delivery-process.pdf)
  identifies `ZN` as 10-Year T-Note futures; and
- [CME's FX-futures product page](https://www.cmegroup.com/markets/fx/fx-futures.html)
  identifies `6E` as Euro FX futures.

## 2. Data contract

Use Databento `GLBX.MDP3` for both driver legs:

| Driver | Parent | Schema | Purpose |
|---|---|---|---|
| 10-Year T-Note futures | `ZN.FUT` | `ohlcv-1m`, `definition` | nominal-yield impulse proxy |
| Euro FX futures | `6E.FUT` | `ohlcv-1m`, `definition` | inverse-USD impulse proxy |

Request DBN with Zstandard compression and `instrument_id` output. Preserve
individual outright contracts; remove calendar spreads only after temporal
definition mapping. For each leg, the active contract for the next session is
the prior completed Globex session's outright volume leader. Same-session
final volume, continuous-contract back-adjustment, and future definitions are
forbidden.

The first stage may inspect only availability, query cost, schema, timestamps,
OHLC integrity, contract mapping, coverage, and degraded-date conditions. It
may not join gold candidates, labels, returns, or outcomes.

The metadata-only check completed without requesting market data. Databento
quoted `$18.939715769142` for all four development requests: `$8.763952776790`
for ZN OHLCV, `$10.095817372203` for 6E OHLCV, and about `$0.08` for both
definition histories. Dataset conditions contain no missing or pending dates;
six dates are marked degraded and are pre-registered as ineligible:
`2020-02-27`, `2020-02-28`, `2020-06-30`, `2020-07-01`, `2021-12-05`, and
`2022-01-02`. No paid request is authorized by this quote.

## 3. Causal driver state

Aggregate the selected outright contract into half-open M30 buckets. At gold
decision time `t`, driver return uses only trades received in `[t-30m, t)`:

`r_driver(t) = log(close(t-30m,t) / open(t-30m,t))`.

For each driver and London clock slot, calculate a robust surprise using the
previous 60 observed values for that same slot, excluding the current value:

`z = (r - prior_median_60) / (1.4826 * prior_MAD_60)`.

The row is ineligible when the 60-observation reference is incomplete, MAD is
zero, fewer than 25 of 30 underlying minutes traded, the active contract
changes inside the bucket, or either source date is degraded. No value is
filled or inferred.

## 4. Exact P4 bare rule

Evaluate every eligible gold M30 signal bar whose next-bar entry falls in the
existing `[07:00, 19:30)` Europe/London entry window.

- **Long macro impulse:** `ZN z >= +1.00` and `6E z >= +1.00`.
- **Short macro impulse:** `ZN z <= -1.00` and `6E z <= -1.00`.
- **Gold confirmation:** signal-bar gold close-to-open return has the same sign
  as the macro impulse.
- **No-chase restriction:** absolute gold signal-bar body is no greater than
  `1.00 * M30 ATR(14)`.
- Take the first eligible P4 signal per London trading day.
- Existing CPI, Employment Situation, and FOMC entry blackouts apply unchanged.
- Entry: next contiguous M30 bar open.
- Stop: `1.00 * M30 ATR(14)` through the signal close.
- Target: `1.50R`.
- Timeout: six M30 bars or the existing forced-flat cutoff, whichever is first.
- Both long and short trades are permitted.

There is no H1 trend filter: the aligned macro shock supplies direction. There
is no neighboring z threshold, alternative lookback, single-driver variant,
gold-divergence variant, or event-window exception in this program.

## 5. Evaluation and placebo gate

Use the existing next-bar-open OHLC resolver, scheduled blackouts, reason-coded
ledger, configured one-basis-point-per-side historical friction, and five equal
chronological trade blocks. Report trade count, PF, mean net R, win rate,
compounded trade return, maximum drawdown, and PF at `1.5x` and `2.0x` cost.

Construct 20 equally spaced non-zero circular shifts of the **joint ZN/6E
driver-state rows** relative to gold. Re-run the complete P4 candidate and
execution path for every shift. This preserves the drivers' internal
relationship and marginal distributions while destroying their timestamp
alignment with gold.

P4 passes development only if every condition holds:

1. causal driver coverage is at least 90% after the fixed 60-slot warm-up;
2. at least 200 executed trades remain, with at least 30 in every block;
3. pooled base-cost PF is at least `1.10` and mean net R is positive;
4. at least four of five chronological blocks have PF above `1.00`;
5. pooled PF remains above `1.00` at double cost; and
6. observed PF exceeds the maximum PF of all 20 time-shift placebos.

A pass permits freezing a confirmation construction; it does not deploy P4,
activate volatility sizing, or authorize paper trading. ML may be considered
only after the bare rule passes, under a new pre-registration.

## 6. Failure rule and research stop

Failure retires P4 in full. No individual driver, alternative standardization,
threshold, lookback, target, timeout, model, or nearby definition may be read as
a rescue. Confirmation remains unopened.

P4 is the final new outright-gold directional family in intraday version one.
If it fails, version-one directional discovery closes honestly. Subsequent work
must be operational engineering or a separately scoped change in trading
problem, such as relative-value metals—not another gold feature search.

## 7. Multiple-comparisons ledger

P4 adds one primary-rule configuration and 20 registered null alignments. The
nulls are a single placebo family, not 20 candidate strategies. All previously
failed H4, intraday primary, ML, benchmark, futures-state, and order-flow looks
remain part of the broader project history when interpreting any result.

## 8. Development result

The user authorized a `$18.94` ceiling and the four requests completed at the
pre-quoted `$18.939715769142`. Checksummed normalization retained 2,193,612
outright ZN bars and 2,482,887 outright 6E bars. Fifty-five ZN rows arrived
before their definitions were causally available and were excluded; no future
definition was used. All six degraded dates were excluded as registered.

P4 **failed**:

| Measurement | Result | Gate |
|---|---:|---:|
| Causal driver coverage | 97.85% | at least 90% — pass |
| Executed trades | 857 | at least 200 and 30/block — pass |
| Base-cost PF | 0.851 | at least 1.10 — fail |
| Mean net R | -0.096 | above 0 — fail |
| Profitable blocks | 1/5 | at least 4/5 — fail |
| Double-cost PF | 0.695 | above 1.00 — fail |
| Best of 20 placebo PFs | 0.913 | observed must exceed — fail |
| Compounded trade return | -12.83% | reported, not gated |
| Maximum drawdown | -15.74% | reported, not gated |

The first chronological block was profitable (`PF 1.176`); the next four were
all below one (`0.658`, `0.763`, `0.910`, `0.783`). This is not a promising
edge hidden by one regime: performance was negative in four of five sequential
samples and weaker than a destroyed-time-alignment placebo.

Per §6, P4 is retired in full. No single-driver, neighboring threshold,
standardization, target, timeout, ML filter, or confirmation test is allowed as
a rescue. Confirmation remains unopened, no ML artifact was trained, and paper
trading remains disabled. Version-one outright-gold directional discovery is
closed; the next project phase must be operational engineering or a separately
pre-registered change in trading problem, such as commodity relative value.
