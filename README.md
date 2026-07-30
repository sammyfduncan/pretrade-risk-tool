# Intraday Multi-Commodity Algorithmic Trading System

**Status:** Active intraday redesign after a completed H4 research program

**Canonical implementation plan:** [`docs/intraday_rebuild_plan.md`](docs/intraday_rebuild_plan.md)

## 1. Purpose

This is a learning-first, portfolio-quality algorithmic commodity trading
project. Its goals, in order, are to:

1. build an honestly evaluated and operationally reliable commodity trading
   system;
2. paper trade it long enough to validate both execution and performance;
3. progress to tightly risk-controlled live trading only after pre-registered
   gates are met; and
4. use the process to learn commodity-market structure, machine learning,
   backtesting, execution, and portfolio construction.

Profitability is not the immediate research objective, but an eventual live
system must have positive expectancy after realistic costs. Negative results
are retained as evidence rather than tuned away.

The project remains focused on **commodities, not FX**.

## 2. What has already been learned

The first research program used H4 gold and silver data. It is complete and is
preserved in the historical design records:

- Generic technical, basic macro, CFTC COT, and GDELT tone/attention features
  produced no confirmed directional edge.
- Gold meta-labeling remains an unvalidated lead; it did not transfer cleanly
  to silver and was underpowered after overlap and multiple-comparison
  corrections.
- COT and GDELT failed the pre-registered directional retention gate on all
  four asset × family combinations.
- Implied volatility (GVZ for gold) and the scheduled-event calendar qualified
  as volatility-sizing candidates at a five-day horizon. They are not direction
  signals and must be re-tested for the new intraday target.
- The original gold holdout remains unspent.

These results reject the tested **H4 public-feature direction hypothesis**.
They do not establish that commodities, price-based rules, or ML are unusable.

Historical records:

- [`docs/step1_freeze_record.md`](docs/step1_freeze_record.md)
- [`docs/step2_design.md`](docs/step2_design.md)
- [`docs/step2b_vol_design.md`](docs/step2b_vol_design.md)

## 3. Active system design

The intraday redesign separates four jobs that the original architecture mixed
together:

1. **Primary strategies create direction.** A small set of explicit,
   economically motivated price/session rules produces candidate trades.
2. **ML is an optional meta-filter or ranker.** It estimates whether a valid
   candidate is worth taking; it does not search every candle for direction.
3. **Volatility and scheduled events control risk.** They inform sizing,
   entry restrictions, and execution expectations—not trade direction.
4. **The allocator manages the portfolio.** It compares valid opportunities,
   controls correlated exposure, accounts for costs, and may choose cash.

In compact form:

`primary setup → optional ML filter → volatility/event sizing → allocator → execution`

Every learned component must beat the transparent rule it modifies.

## 4. Trading cadence

The agreed cadence is:

- **H1:** higher-level trend, regime, and market context;
- **M30:** default candidate-entry and trade-management clock;
- **M15:** optional execution/refinement challenger, retained only if its extra
  opportunities survive realistic spread, slippage, and latency assumptions.

This is an intraday project. The working assumption is that positions are not
held overnight routinely, although the exact daily cutoff and exceptional
handling must be fixed before implementation.

M5 and faster scalping are out of scope for the first intraday version. At
those frequencies the available OHLC data and CFD execution model are not
adequate to make realistic claims.

The currently committed configs still describe the frozen H4 research
pipeline. Documentation has moved first; changing data, labels, configs, and
backtests belongs to the new pre-registered implementation program.

## 5. Primary strategy research

The first intraday program will select no more than three distinct strategy
families before reading results:

1. **Session breakout / range expansion**
2. **Trend continuation after a pullback or consolidation**
3. **Regime-conditioned mean reversion**

Each family must define its entry, side, exit, stop, maximum holding time,
session restrictions, event restrictions, and cost assumptions before
evaluation.

The existing RSI, MACD, Bollinger, EMA, return, ATR, volatility, candle-shape,
and calendar features already cover the common technical-indicator families.
Adding more variants of the same indicators is not the plan. New features
should describe market structure or execution, such as:

- session and time since session open;
- previous-session high, low, range, and return;
- opening-range position and range compression;
- upside/downside realized variation;
- synchronized cross-metal returns;
- current spread and recent spread regime; and
- volume or tick activity only where it is real and comparable across sources.

## 6. Role of machine learning

ML remains part of the project, but its role changes:

- Logistic regression remains the simple baseline.
- LightGBM remains the nonlinear challenger.
- The training sample consists of actual primary-strategy candidates.
- The target describes executable trade success or net outcome, not generic
  next-bar direction.
- Raw scores may rank candidates; calibrated probabilities may support sizing.
- Sample-overlap, walk-forward, placebo, and bare-rule controls remain
  mandatory.

No neural-network, LLM, hyperparameter-search, or additional boosting work is
planned until a simpler model demonstrates that the new inputs contain useful
information.

A pooled multi-asset model may later be evaluated as a challenger once at
least three assets exist, but it must be tested against independent per-asset
models and with leave-one-asset-out transfer checks.

## 7. News and event data

Generic news sentiment is retired from the active direction design:

- COT and GDELT remain disabled.
- No paid LLM API sentiment layer is planned.
- A better language model would not repair point-in-time archives, publication
  timing, or the lack of demonstrated incremental information.

The useful successor is a **structured event-risk layer**:

- exact FOMC, CPI, NFP, and other relevant release timestamps;
- time-to-event and time-since-event;
- event-window spread and volatility expectations;
- configurable entry blackout or reduced-risk windows; and
- later, actual-minus-consensus surprises only if an honest point-in-time
  consensus source is available.

Event data is initially a risk and execution input. It becomes directional only
under a separate pre-registered hypothesis.

## 8. Asset universe

The planned sequence is:

1. **Gold** — development asset
2. **Silver** — untouched transferability control
3. **Platinum** — first new asset
4. **Palladium** — conditional on broker availability, spread, gaps, and
   executable liquidity

Copper is the leading candidate for a later expansion beyond precious metals
because it adds a more distinct industrial-demand mechanism. Energy and
agriculture remain later possibilities, not current scope.

Assets are added to test transferability and improve portfolio diversification,
not to find a convenient replication after another asset fails.

## 9. Data and execution

Training history comes from **Dukascopy**. **IG** is the intended paper/live
broker and the source for recent parity checks and executable prices. Both
feeds use the canonical candle schema.

Before any new performance experiment:

- label entry and backtest entry must describe the same trade;
- stops must use actual OHLC information;
- bid/ask spread, slippage, and signal latency must be modeled;
- Dukascopy-versus-IG returns and timestamps must be compared;
- session, holiday, and daily-cutoff behavior must be explicit; and
- trade-by-trade label/backtest parity must be tested.

The old H4 backtest omitted some of this realism. Its ML and feature-ablation
conclusions remain useful, but its PF and Sharpe are not estimates of live
intraday profitability.

## 10. Evaluation principles

- Walk-forward, time-ordered validation only.
- No lookahead; features at time *t* use only information available at *t*.
- Out-of-fold predictions only in backtests.
- Realistic, time-varying costs where data permits.
- Effective sample size rather than raw overlapping row count.
- A multiple-comparisons ledger for every research program.
- Placebo/null testing for every new feature family.
- ML filters are always compared with their bare primary rule.
- Portfolio metrics include return, Sharpe/Sortino, profit factor, maximum
  drawdown, turnover, time in cash, exposure concentration, and slippage.
- A result is not promoted on a few favorable weeks.

The H4 holdout belongs to the frozen H4 question and must not be casually
re-used. Holdout ownership and the prospective evaluation boundary for the
intraday program must be fixed before its first outcome is read.

## 11. Build order

1. Freeze this intraday design and resolve the remaining mandate decisions.
2. Repair label/backtest/execution correctness without searching for alpha.
3. Build H1/M30 datasets and M15-compatible execution inputs.
4. Evaluate the pre-registered primary rule families on gold.
5. Add ML as a challenger only to surviving rules.
6. Re-test volatility and scheduled-event risk controls at intraday horizons.
7. Transfer the frozen pattern to silver, then platinum and palladium.
8. Build correlation-aware allocation with an explicit cash outcome.
9. Paper trade in separate operational and performance-evaluation stages.
10. Begin minimum-size live trading only after both gates pass.

The detailed deliverables and decision gates are in
[`docs/intraday_rebuild_plan.md`](docs/intraday_rebuild_plan.md).

## 12. Paper trading and live progression

Paper trading has two different purposes:

1. **Operational shadow stage:** several weeks may be enough to find feed,
   scheduling, reconciliation, spread, order, and restart failures. Operational
   bugs may be corrected immediately and logged.
2. **Frozen performance stage:** strategy logic is versioned and left unchanged
   until a pre-registered trade count and time window are reached. A logic
   change starts a new evaluation version.

Live trading begins only at minimum viable size, with per-trade, daily,
weekly, per-asset, and portfolio risk limits plus a global kill switch. A
paper/shadow instance continues alongside live.

## 13. Open decisions

Before implementation begins, fix:

- the daily flat-position cutoff and timezone/session definition;
- whether version 1 may trade short as well as long;
- the exact primary strategy specifications;
- the new historical/prospective evaluation boundary;
- risk per trade and daily/weekly loss limits; and
- IG epics and minimum executable sizes for every asset.

These are mandate decisions, not parameters to infer after reading a backtest.
