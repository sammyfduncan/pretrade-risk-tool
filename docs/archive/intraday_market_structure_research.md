# Gold benchmark-window market-structure research

**Frozen:** 2026-07-31 before generating any B1 outcome  
**Development:** Dukascopy M15, `2020-01-01` through `2025-06-30`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Model policy:** no ML unless the bare benchmark rule passes

## 1. Why investigate benchmark windows

The first three generic strategy families and their V2 variants did not
produce a robust net edge. The next program therefore changes mechanism rather
than adding indicators.

Gold has observable market-structure clocks:

- LBMA sets the internationally recognised gold benchmark twice daily at
  10:30 and 15:00 London time.
- COMEX GC settlement uses trading activity from 13:29 to 13:30 New York time.
- Research finds that gold price discovery alternates between London and New
  York rather than remaining permanently dominant in one venue.
- Caminschi and Heaney report elevated volume/volatility after the London PM
  fixing begins and directionally informative trading early in that process.

Primary and research sources:

- https://www.lbma.org.uk/prices-and-data/about-lbma-daily-auction-prices
- https://www.cmegroup.com/content/dam/cmegroup/notices/ser/2025/12/ser-9637.pdf
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2161905
- https://doi.org/10.1002/fut.21636

The fixing paper predates the current electronic auction and its strongest
effect occurs at a finer interval than M15. It motivates a hypothesis; it does
not establish that the effect survives in 2020–2025 spot gold.

## 2. Registered hypothesis B1

Only the **15:00 London PM auction** is tested. The 10:30 AM auction and COMEX
settlement are excluded from this program to prevent selecting the best of
several clocks after reading outcomes.

For each complete London trading day:

1. At 15:00, measure the completed price move from the 14:00 M15 open through
   the 14:45 M15 close.
2. Require the absolute move to be at least `1.00` causal M15 ATR(14), measured
   at the 14:45 close.
3. Trade in the direction of that move at the 15:00 M15 open.
4. Use a `1.00 ATR` absolute stop, `1.50R` target, and four-M15-bar (one-hour)
   timeout.
5. Retain the existing London cutoff, scheduled-event blackouts, adverse
   same-bar barrier ordering, and one-basis-point-per-side historical cost.

There is at most one B1 candidate per day. No H1 trend filter, side filter,
weekday filter, news outcome, or benchmark price is used.

## 3. Registered time placebos

Run the identical construction as if the benchmark occurred at 13:00, 14:00,
16:00, and 17:00 London time. These are mechanism placebos, not alternative
strategies. They answer whether any apparent B1 result is specific to the
official auction or merely an ordinary intraday momentum pattern.

No other clock, move threshold, stop, target, or timeout will be inspected in
this program.

## 4. Retention gate

B1 passes development only if every condition holds:

1. at least 300 executed B1 trades;
2. base-cost PF is at least `1.10` and mean net R is positive;
3. PF exceeds `1.00` in at least four of five equal chronological blocks;
4. PF remains above `1.00` at double cost;
5. maximum drawdown is no worse than `-10%`; and
6. B1 PF is strictly greater than the maximum PF of the four registered time
   placebos.

A pass would permit a separately frozen candidate-feature/ML gate. Failure
retires B1 without trying the AM auction, COMEX settlement, or neighbouring
parameters. Confirmation remains unopened either way.

## 5. Cost boundary

The historical test still uses bid-only Dukascopy plus a fixed two-basis-point
round-trip allowance. `scripts/measure_oanda_costs.py` profiles cached OANDA bid/ask
candles by London slot and can match future candidate/paper ledgers. A B1 pass
cannot be promoted until its 15:00 entry window clears observed OANDA median/p95
cost gates and demo fills establish a defensible slippage allowance.

## 6. Development outcome

B1 failed every economic/stability condition except sample count:

| Metric | B1 result |
|---|---:|
| executed trades | 776 |
| gross PF / gross mean | 1.045 / +0.42 bp |
| net PF / mean R | 0.851 / -0.112R |
| profitable chronological blocks | 1/5 |
| maximum drawdown | -14.5% |
| PF at double cost | 0.695 |

Registered placebo PFs were `0.700` at 13:00, `0.833` at 14:00, `0.872` at
16:00, and `0.793` at 17:00. The 16:00 placebo exceeded B1, so the weak gross
effect was not specific to the official PM auction.

B1 is retired. The AM auction, COMEX settlement, and neighbouring thresholds
remain unread in this program, as registered. No ML model was trained and
confirmation remains unopened.

## 7. Research implication

The benchmark is economically real, but bid-only spot OHLC does not expose the
auction imbalance or exchange order flow that would embody the mechanism. A
future benchmark/price-discovery program should first add a point-in-time
futures-market source with real GC volume, explicit contract rolls, and ideally
top-of-book or trade data. Useful candidate inputs would be synchronized GC
returns/activity, spot-futures basis, time to roll, and London-versus-New-York
price leadership. They must enter as one pre-registered data family and beat a
matched spot-only control plus time-shift placebos.

Dukascopy candle `volume` is not COMEX traded volume. Dukascopy documents it as
the sum of best-price volumes for ticks in the bar, so it may describe feed
activity but cannot be represented as centralized gold trading volume. It is
not promoted as a strategy input here.
