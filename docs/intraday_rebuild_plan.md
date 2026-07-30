# Intraday commodity-system rebuild plan

**Status:** ACTIVE DESIGN — documentation decision only; implementation has not started

**Decision date:** 2026-07-30

**Active cadence:** H1 context, M30 default entries/management, M15 optional challenger

**Asset class:** commodities; no FX pivot

**Historical baseline:** the completed H4 program remains frozen and reproducible

## 0. Why this is a new program

The completed H4 work asked whether technical, basic macro, COT, and GDELT
features could support per-asset directional classification or meta-labeling.
It produced an honest negative for direction and two limited volatility-sizing
candidates.

The active program asks a different question:

> Can explicit intraday commodity setups produce net-positive candidate trades,
> and can ML, volatility forecasts, and portfolio controls improve those
> candidates without inventing an edge?

This is not a parameter change to the H4 template. It changes cadence, entry
semantics, holding period, cost sensitivity, primary strategies, and the role
of ML. It therefore needs a new design ledger and new evaluation boundary.

No H4 result is erased. The historical records remain:

- `docs/step1_freeze_record.md`
- `docs/step2_design.md`
- `docs/step2b_vol_design.md`

## 1. Fixed decisions

| Decision | Active choice |
|---|---|
| asset class | commodities |
| initial assets | gold, silver; then platinum, palladium |
| development asset | gold |
| transfer control | silver |
| context timeframe | H1 |
| default signal/execution timeframe | M30 |
| optional tighter timeframe | M15, only after a cost/execution gate |
| initial holding style | intraday; no routine overnight exposure |
| direction source | explicit primary strategy rules |
| ML role | candidate filter/ranker, not unconstrained direction finder |
| generic news sentiment | retired |
| event data role | volatility, sizing, restrictions, execution risk |
| model families | logistic baseline and LightGBM challenger |
| paper-to-live path | operational shadow → frozen performance stage → minimum-size live |

M5 and faster scalping are outside version 1.

## 2. Decisions still required before code changes

The following must be resolved explicitly rather than tuned from results:

1. Exact daily position cutoff and the session/timezone it follows.
2. Whether the first version can take both long and short trades.
3. Maximum simultaneous positions and correlated-metals exposure.
4. Risk per trade plus daily and weekly loss limits.
5. Which exact specifications represent the three permitted primary families.
6. Ownership of the old H4 holdout and the clean prospective boundary for the
   intraday program.
7. IG epic, minimum size, typical spread, and trading hours for each asset.

Until these are fixed, the repository's H4 configs remain the executable
historical configuration.

## 3. Target architecture

### 3.1 Primary strategy layer

Primary rules create candidate trades and choose the side. They must be
complete enough to trade without ML.

The first program is capped at three economically distinct families:

1. **Session breakout / range expansion**
   - prior and opening-session ranges;
   - range compression;
   - breakout confirmation;
   - explicit session and event restrictions.
2. **Trend continuation after pullback/consolidation**
   - H1 trend/regime;
   - M30 pullback or consolidation;
   - causal entry trigger;
   - invalidation and maximum holding time.
3. **Regime-conditioned mean reversion**
   - enabled only in a non-trending regime;
   - session anchor or range reference;
   - disabled during volatility expansion and event windows.

Each rule must specify:

- candidate timestamp and tradable entry timestamp;
- long/short eligibility;
- stop, target, timeout, and daily cutoff;
- no-trade windows;
- position overlap behavior; and
- gross and net control metrics.

### 3.2 ML layer

ML receives only primary-rule candidates. Its first target is:

> Given this valid setup and executable entry, did the trade achieve a
> pre-defined net-positive outcome before stop, timeout, or session cutoff?

Requirements:

- logistic regression is the baseline;
- LightGBM is the only initial nonlinear challenger;
- no hyperparameter search;
- no training on non-candidate bars;
- walk-forward, gapped OOF evaluation;
- uniqueness/effective-n corrections where candidate windows overlap;
- raw scores for ranking; calibration evaluated separately;
- bare-rule comparison on identical entries and costs; and
- a time-shift/label-null gate for every new feature family.

An ML filter is retained only if it improves the underlying rule rather than
merely producing attractive standalone metrics.

### 3.3 Volatility and event-risk layer

This layer never chooses direction in version 1. It may:

- predict near-term realized volatility;
- scale trade and portfolio risk;
- block or reduce entries near scheduled releases;
- estimate spread/slippage regimes;
- adjust maximum holding time under a fixed rule; and
- detect when correlated metals represent one risk event.

GVZ and the scheduled-event calendar are historical five-day candidates, not
intraday components. They must pass a new matched-row intraday ablation and
placebo gate before use.

COT and GDELT tone/attention remain retired.

### 3.4 Allocation layer

The allocator consumes only validated candidates and outputs positions or cash.
The first version is transparent and rule-based.

Inputs:

- candidate side and primary-rule identity;
- ML score if retained;
- predicted volatility;
- expected spread/slippage;
- current positions and remaining risk budget; and
- rolling correlation/exposure by metal.

Outputs:

- selected candidate(s);
- risk-sized quantity;
- explicit cash/no-trade;
- rejected-candidate reason codes; and
- complete decision log.

A learned allocator is outside the initial program.

## 4. Timeframe design

### H1: context

Use H1 for:

- higher-level trend and range regime;
- multi-hour realized volatility;
- prior-session structure;
- cross-metal context; and
- slow state transitions.

### M30: default decision clock

Use M30 for:

- primary candidate generation;
- entry/stop/target definitions;
- trade management;
- event proximity; and
- default backtest execution.

### M15: optional execution challenger

M15 may be used to refine an already valid H1/M30 candidate. It is not assumed
to be better.

It must pass all of:

1. actual bid/ask or defensible time-varying spread assumptions;
2. latency and next-tradable-price modeling;
3. OHLC barrier-order handling;
4. higher net performance than M30 on the same candidate family;
5. no material deterioration under worsened-cost stress; and
6. enough independent trades to distinguish improvement from timing noise.

M15 may be retained for entry execution even if no independent M15 ML model is
retained.

## 5. Feature policy

Do not add another indicator merely because it has a different name.

The existing features already cover common price transforms:

- returns and momentum;
- RSI and MACD;
- Bollinger position;
- EMA distance;
- ATR and realized-volatility regime;
- candle shape; and
- periodic time features.

New version-1 families must be mechanism-based:

1. **Session structure**
   - session label and time since open;
   - previous-session OHLC/range/return;
   - opening-range high/low and position;
   - remaining time before cutoff.
2. **Price-path state**
   - range compression/expansion;
   - upside/downside realized variation;
   - trend persistence and pullback depth;
   - gap and post-gap state.
3. **Cross-commodity state**
   - synchronized gold/silver returns;
   - relative strength defined without future alignment;
   - later, platinum/palladium/copper context.
4. **Execution state**
   - current and recent spread;
   - quote staleness;
   - missing-bar/session integrity;
   - real volume/tick activity only where source semantics match.
5. **Scheduled-event state**
   - time to/from exact release;
   - event family;
   - pre/post-event volatility regime.

Each family enters under a pre-registration, causality test, matched-row
ablation, and empirical-null check.

## 6. News and language-model policy

No generic headline-keyword score and no paid LLM API are planned.

The negative GDELT result means a more complex text representation has no
automatic claim on development time. Language-model cost is secondary to the
larger problems of point-in-time availability, timestamp fidelity, entity
linkage, and demonstrated incremental information.

A future text experiment is permitted only when:

- it states a narrow economic mechanism;
- the historical source is reproducible and point-in-time enough;
- the event timestamp precedes the decision;
- a cheap baseline and placebo are defined first; and
- it is a new pre-registered program.

Structured release calendars and, later, point-in-time economic surprises are
preferred to generic sentiment.

## 7. Asset sequence

### Gold

Development asset for rule construction and execution-contract repair.

### Silver

Untouched transfer control. It must not receive asset-specific parameter tuning
after gold results are known.

### Platinum

First new asset after gold/silver correctness and transfer evaluation. Confirm
Dukascopy history, IG epic, spread, gaps, and minimum size before modeling.

### Palladium

Add only if executable liquidity and data quality are acceptable. A wide or
unstable spread can disqualify it even if a frictionless backtest looks useful.

### Later universe

Copper is the leading later candidate because it adds a distinct industrial
demand mechanism. A genuine broad commodity-futures system with carry, curve,
open-interest, and inventory features is a possible future program, not part of
the first IG spot-CFD version.

## 8. Correctness work before alpha work

The following are release blockers:

1. Define the tradable entry timestamp and price.
2. Make labels and backtests use that same entry.
3. Pass actual open/high/low/close into stop/target evaluation.
4. Resolve same-bar stop/target ambiguity conservatively.
5. Add trade-by-trade label/backtest parity tests.
6. Model M30/M15 spread, slippage, and latency.
7. Compare Dukascopy and IG timestamps and returns.
8. Define session, holiday, early-close, and missing-bar behavior.
9. Enforce the daily position cutoff.
10. Add idempotent orders, reconciliation, restart recovery, and a kill switch
    before paper execution.

This phase must not be judged by whether historical returns improve.

## 9. Evaluation design

Every research program must state before outcomes:

- exact features, labels, entries, exits, and costs;
- train/test windows and gaps;
- rule and model controls;
- number of planned configurations;
- retention and retirement gates;
- minimum independent trade count;
- cost-stress scenarios; and
- what constitutes replication on another asset.

Required reporting:

- gross and net return;
- Sharpe and Sortino;
- profit factor and win rate;
- maximum drawdown;
- trade count and effective trade count;
- average win/loss in risk units;
- turnover, spread, and slippage;
- time in market and time in cash;
- long/short and session decomposition;
- event/non-event decomposition; and
- portfolio correlation and concentration.

The old H4 holdout remains unspent and attached to the frozen H4 protocol until
the user explicitly decides otherwise. The safest intraday confirmation is a
new prospective window beginning after the intraday design is frozen.

## 10. Paper-trading protocol

### Stage A: operational shadow

Purpose: prove that the software sees, decides, and executes what the research
assumed.

Several weeks can be informative for:

- data freshness and missing bars;
- feature parity;
- signal timing;
- actual spreads and slippage;
- order rejection and duplication;
- position reconciliation;
- stops, targets, and cutoff exits;
- alerting and restart recovery.

Operational bugs may be fixed immediately, with versioned records.

### Stage B: frozen performance evaluation

Purpose: estimate whether the strategy transfers prospectively.

- Freeze strategy logic before the stage starts.
- Pre-register minimum trades and minimum elapsed time.
- Run bare rule and ML-filtered variants in parallel shadow mode.
- Do not tune from a short run of winners or losers.
- A prediction-logic change creates a new version and restarts its evaluation
  clock.

### Stage C: minimum-size live

Start only after both operational and performance gates pass.

Required controls:

- minimum viable position size;
- per-trade and aggregate risk caps;
- daily and weekly loss limits;
- per-asset and correlated-metals caps;
- stale-data and broker-disconnect fail-closed behavior;
- global kill switch; and
- paper/shadow instance running beside live.

Size increases occur only at pre-defined review points.

## 11. Ordered implementation plan

1. **Mandate freeze**
   - resolve §2 decisions;
   - append exact rule specifications and evaluation boundaries here;
   - commit before reading new outcomes.
2. **Execution-contract repair**
   - implement §8 items;
   - add parity and cost-model tests.
3. **Intraday data path**
   - fetch/build H1 and M30;
   - support synchronized M15 inputs without making M15 mandatory;
   - validate Dukascopy/IG parity.
4. **Primary rules**
   - implement the three pre-registered families;
   - evaluate bare controls on gold.
5. **ML challenger**
   - candidate-only labels;
   - logistic and LightGBM;
   - OOF, effective-n, calibration, and placebo gates.
6. **Risk layer**
   - intraday volatility baseline;
   - event-calendar ablation;
   - spread/slippage regimes and position sizing.
7. **Transfer**
   - frozen gold design on silver;
   - then platinum and palladium one at a time.
8. **Allocator**
   - correlation-aware sizing;
   - explicit cash;
   - reason-coded decisions and turnover controls.
9. **Paper execution**
   - operational shadow;
   - frozen performance stage.
10. **Live progression**
    - minimum size;
    - staged reviews;
    - no automatic scale-up.

## 12. Definition of success

Version 1 succeeds as a project if it produces:

- a reproducible, causally correct intraday research pipeline;
- a realistic broker-aligned execution simulation;
- explicit strategies and controls rather than opaque indicator mining;
- an honest answer about whether ML adds value;
- a multi-asset allocator with cash and correlation control;
- reliable paper execution and observability; and
- a defensible live-capital decision, even if that decision is “not yet.”

Profit is required before meaningful live scale, but an honest negative remains
a valid research result.
