# Intraday primary strategies — version-2 development program

**Frozen:** 2026-07-31 before generating any V2 candidate outcome  
**Scope:** gold development data ending before `2025-07-01`  
**Confirmation:** remains unopened  
**Model policy:** evaluate bare rules first; ML is disabled unless a bare V2 rule passes

## 1. Known development evidence

Version 1 is an honest negative. At the time this design was frozen, the
then-current bare OOF PF was `0.710` for P1, `0.771` for P2, and `0.841` for
P3. Candidate ML did not make any rule profitable. Section 5 records the later
event-blackout correctness correction and supersedes those numeric artifacts;
the V2 rules themselves were not changed.

The following exploratory development diagnostics were read only to form V2
hypotheses; they are not confirmatory evidence:

- P1 was least poor from 09:00–12:00 London and did not improve monotonically
  as its opening range became more compressed.
- P2 improved with higher M30 ATR and stronger H1 EMA separation, although
  every individual bin remained negative.
- P3 was especially poor in its lowest ATR tercile; its high-ATR tercile was
  positive in this exploratory split, but contained only 124 trades.
- Side and time-of-day subgroup results were also inspected. No side is removed
  in V2, avoiding a further fitted choice.

The literature is suggestive rather than transferable. Intraday commodity
futures studies report both momentum and liquidity-driven reversal depending
on venue and session. Jin et al. find opening-session momentum strongest in
high-volume or high-volatility metals sessions; Baltussen et al. document
intraday momentum across futures; Zheng and Luo find reversal in Chinese
commodity futures; Chaves and Viswanathan distinguish spot mean reversion from
futures momentum. These mechanisms motivate tests but do not validate them for
Dukascopy/OANDA spot gold.

Sources:

- https://doi.org/10.1002/fut.22084
- https://doi.org/10.1016/j.jfineco.2021.05.011
- https://doi.org/10.2139/ssrn.4688712
- https://doi.org/10.1016/j.jcomm.2016.07.001

## 2. Shared causal state

Two new values are computed without future data:

- `m30_atr_regime_ratio`: current M30 ATR% divided by the median ATR% of the
  preceding 960 M30 observations; the current observation is excluded.
- `opening_impulse_atr`: the 07:00–09:00 London opening close minus opening
  open, divided by M30 ATR known at the decision. It is hidden before 09:00.

No global percentile, future session, volume proxy, news sentiment, or M15
column is allowed into these rules.

## 3. Frozen V2 rules

All execution, cost, session, stop, target, timeout, overlap, and daily-limit
semantics remain identical to the corresponding V1 rule.

### P1V2 — high-energy directional opening breakout

Replace the compression hypothesis with an opening-energy hypothesis:

- completed opening range is between `1.00x` and `2.00x` its prior-20-session
  median;
- absolute opening impulse is at least `0.50 M30 ATR`;
- opening impulse direction agrees with the completed-H1 EMA50 direction gate;
- take the first post-09:00 close at least `0.10 ATR` beyond the opening range
  in that direction;
- retain the V1 `1 ATR` stop, `1.5R` target, six-bar timeout, and one/day cap.

### P2V2 — strong, active-regime trend pullback

Retain the exact P2 touch/reclaim candle, but require both:

- absolute completed-H1 EMA20–EMA50 separation is at least `1.00 H1 ATR`; and
- `m30_atr_regime_ratio >= 1.00`.

Retain the V1 `1 ATR` stop, `1.5R` target, eight-bar timeout, and two/day cap.

### P3V2 — active-regime failed excursion

Retain the exact P3 non-trending H1 gate and Bollinger excursion/re-entry, but
require `m30_atr_regime_ratio >= 1.00`. Retain the V1 `1 ATR` stop, `1R`
target, four-bar timeout, and one/day cap.

## 4. Bare-rule retention gate

No ML is trained before this gate. P1V2/P2V2 require at least 300 executed
trades; P3V2 requires at least 100. Each rule must satisfy every condition:

1. pooled PF at base cost is at least `1.10`;
2. pooled mean net R is positive;
3. PF is above `1.00` in at least four of five equal chronological blocks;
4. PF remains above `1.00` at double cost;
5. PF exceeds its V1 counterpart and trade-level maximum drawdown is less
   severe.

One configuration per family is permitted. A failed rule is retired; no
neighboring volatility, impulse, trend-strength, stop, target, or time-window
threshold is inspected. Passing only permits candidate-only ML to be evaluated
under a separately frozen gate. It does not open confirmation.

## 5. Correctness correction and outcomes

Before these outcomes were finalized, the candidate CLI was found not to have
wired the frozen CPI/NFP/FOMC blackouts into execution. It also included 2019,
outside the verified event-calendar coverage. The pipeline was corrected to
use exact DST-aware release times, exclude pre-2020 rows, and regenerate every
V1/V2 artifact. No V2 rule or threshold was changed.

All three corrected V2 rules failed their frozen gate:

| Rule | Trades | PF | Mean R | Profitable blocks | PF at 2x cost | V1 PF | MaxDD (V2 / V1) |
|---|---:|---:|---:|---:|---:|---:|---:|
| P1V2 | 323 | 0.928 | -0.036 | 2/5 | 0.766 | 0.741 | -4.6% / -26.2% |
| P2V2 | 305 | 0.817 | -0.103 | 1/5 | 0.688 | 0.779 | -9.8% / -27.9% |
| P3V2 | 185 | 1.000 | -0.015 | 3/5 | 0.803 | 0.856 | -3.0% / -5.7% |

P1V2 and P3V2 did improve their V1 controls and drawdowns, but neither met the
minimum PF, positive-mean-R, temporal-stability, or double-cost requirements.
P2V2 did not improve its control PF. No V2 model was trained and confirmation
remains unopened.

## 6. What the result says about costs

The fixed historical allowance is one basis point per side, or two basis
points round trip. Looking at gross returns is diagnostic only; it does not
waive costs:

| Rule | Gross PF | Gross mean/trade | Net PF |
|---|---:|---:|---:|
| P1 | 0.913 | -0.85 bp | 0.741 |
| P2 | 0.951 | -0.50 bp | 0.779 |
| P3 | 1.094 | +0.74 bp | 0.856 |
| P1V2 | 1.129 | +1.23 bp | 0.928 |
| P2V2 | 0.972 | -0.33 bp | 0.817 |
| P3V2 | 1.243 | +2.00 bp | 1.000 |

P1V2 and especially P3V2 contain a small gross effect, but it is no larger
than the assumed execution friction. That makes another classifier or a nearby
threshold sweep the wrong next step. It also makes accurate OANDA spread and
slippage measurement important: the project must measure costs, not lower the
assumption because a lower number would rescue a backtest.

## 7. Next research direction

The next development experiment is an **execution-only M15 challenger** for the
unchanged P1V2 and P3V2 M30 signals. Because an M30 signal is not known until
its candle closes, honest M15 replay cannot enter earlier. It uses the same
entry timestamp and asks whether finer path and barrier-order resolution
improves the matched trades. It must not create extra M15 signals or tune V2
thresholds. The exact pre-registration is in
`docs/archive/intraday_m15_execution_design.md`.

Actual OANDA quotes are required before any apparent cost-sensitive improvement
can be promoted. Until then, the M15 work is research infrastructure, not a
paper-trading strategy. Candidate ML stays disabled because no corrected bare
rule passed.

## 8. M15 follow-up outcome

The exact matched experiment in `docs/archive/intraday_m15_execution_design.md` was
implemented and completed. P1V2 improved only from PF `0.928` to `0.937` and
P3V2 from `1.000` to `1.050`. Only one and two trades, respectively, changed
from the conservative M30 stop outcome to an M15 target; improvement occurred
in only one of five P1V2 blocks and two of five P3V2 blocks. Both failed the
double-cost gate. M15 is not retained for these rules.
