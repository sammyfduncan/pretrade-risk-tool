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
- The original gold H4 holdout was never spent on the retired H4 strategy. Its
  unread `2025-07-01` onward window is now reserved as the intraday program's
  single-use historical confirmation window; it cannot be used for tuning.

These results reject the tested **H4 public-feature direction hypothesis**.
They do not establish that commodities, price-based rules, or ML are unusable.

Historical records:

- [`docs/step1_freeze_record.md`](docs/step1_freeze_record.md)
- [`docs/step2_design.md`](docs/step2_design.md)
- [`docs/step2b_vol_design.md`](docs/step2b_vol_design.md)
- [`docs/intraday_ml_retraining.md`](docs/intraday_ml_retraining.md)
- [`docs/intraday_filter_gate.md`](docs/intraday_filter_gate.md)
- [`docs/intraday_v2_design.md`](docs/intraday_v2_design.md)
- [`docs/intraday_m15_execution_design.md`](docs/intraday_m15_execution_design.md)
- [`docs/intraday_market_structure_research.md`](docs/intraday_market_structure_research.md)
- [`docs/intraday_cost_measurement.md`](docs/intraday_cost_measurement.md)
- [`docs/intraday_risk_design.md`](docs/intraday_risk_design.md)
- [`docs/intraday_risk_policy.md`](docs/intraday_risk_policy.md)
- [`docs/intraday_joint_risk_design.md`](docs/intraday_joint_risk_design.md)
- [`docs/intraday_risk_economic_design.md`](docs/intraday_risk_economic_design.md)
- [`docs/cme_futures_data.md`](docs/cme_futures_data.md)
- [`docs/intraday_futures_design.md`](docs/intraday_futures_design.md)
- [`docs/intraday_order_flow_design.md`](docs/intraday_order_flow_design.md)

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
- **M15:** optional execution/refinement challenger, retained only if matched
  M30 candidates improve under realistic spread, slippage, and latency.

This is an intraday project. Positions are forced flat by 21:30 London time,
with earlier close overrides; entries are allowed from 07:00 to before 19:30.

M5 and faster scalping are out of scope for the first intraday version. At
those frequencies the available OHLC data and CFD execution model are not
adequate to make realistic claims.

The committed yaml configs still reproduce the frozen H4 research pipeline.
Intraday fetching uses explicit H1/M30 granularity artifacts, and the new
builders write separate intraday base/candidate/model paths.

## 5. Primary strategy research

The first intraday program will select no more than three distinct strategy
families before reading results:

1. **Session breakout / range expansion**
2. **Trend continuation after a pullback or consolidation**
3. **Regime-conditioned mean reversion**

Each family must define its entry, side, exit, stop, maximum holding time,
session restrictions, event restrictions, and cost assumptions before
evaluation.

The exact P1/P2/P3 rules and candidate generators are implemented and have run
on gold development artifacts. The original rules, all three frozen V2 rules,
and both registered M15 execution replays were honest negatives after costs.

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

Initial retraining is now specified: deep Dukascopy H1/M30 development history
creates executable candidates; each rule gets its own logistic baseline and
LightGBM challenger; folds purge trades still open at the test boundary. The
training command saves no deployment threshold, so a research artifact cannot
silently become a paper model. See
[`docs/intraday_ml_retraining.md`](docs/intraday_ml_retraining.md).

Gold development training is now complete. Six strategy/model combinations
were screened; the two registered leads, P1-logistic and P2-LightGBM, both
failed the frozen economic filter gate after the candidate history was
corrected to enforce scheduled-event blackouts. P1 consistently reduced losses
but remained negative; P2 remained negative. No ML filter or deployment
threshold is retained for intraday version 1. See
[`docs/intraday_filter_gate.md`](docs/intraday_filter_gate.md).

A separately frozen V2 primary-rule program also completed. Active-volatility
and opening-impulse conditions improved P1/P3 gross behavior and drawdown, but
all three V2 rules failed after the fixed two-basis-point round-trip cost; ML
was therefore not trained on them. A subsequent matched M15 execution-only
challenger also completed and failed: finer paths changed only three stop outcomes across
508 matched trades, with P1V2 PF `0.937` and P3V2 PF `1.050`. See
[`docs/intraday_v2_design.md`](docs/intraday_v2_design.md).

A first commodity-specific market-structure test then targeted continuation
through the 15:00 London LBMA gold auction. B1 failed with PF `0.851`, only one
of five profitable blocks, and a better 16:00 non-auction placebo. It is
retired without testing neighbouring clocks or thresholds. Further work on
benchmark price discovery requires actual futures volume/order-flow or basis
data rather than another spot-OHLC proxy.

That source was subsequently added and tested. A pre-registered causal GC
family used prior-session volume to select the active contract and added real
GC returns, volume surprise, open-interest change, spot/futures dislocation,
and basis state to the matched P1 logistic filter. It failed: OOF AUC fell
`0.5294 -> 0.5187`, only two folds improved, a shifted placebo performed much
better, and filtered PF remained below one (`0.862`). The family is retired;
individual-feature rescue variants were not opened.

The next and only registered new-data challenger is GC trade flow from
Databento TBBO. Unlike aggregate bars, TBBO exposes aggressor side and the
pre-trade best bid/offer sizes. The checksum-verifying importer, active-contract
mapping, causal flow aggregation, placebo gate, and synthetic anti-lookahead
tests are implemented. Performance evaluation is blocked until a validation
delivery establishes field and aggressor-side coverage; see
[`docs/intraday_order_flow_design.md`](docs/intraday_order_flow_design.md).

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

The historical intraday builder now enforces exact DST-aware CPI/NFP/FOMC entry
blackouts. Because the verified joint calendar begins in 2020, candidate
research excludes earlier rows rather than silently treating them as
event-free.

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

Spot training history comes from **Dukascopy**. **IG** is the intended
paper/live broker and the source for recent parity checks and executable
prices. **Databento CME/COMEX GC** definitions, one-minute OHLCV, and
statistics are the futures-market research source. The pinned development
artifact remains as separate outright contracts. The F-GC experiment created
a causal research view using the prior completed session's volume leader; it
did not back-adjust prices or overwrite the source contracts.

Before any new performance experiment:

- label entry and backtest entry must describe the same trade;
- stops must use actual OHLC information;
- bid/ask spread, slippage, and signal latency must be modeled;
- Dukascopy-versus-IG returns and timestamps must be compared;
- session, holiday, and daily-cutoff behavior must be explicit; and
- trade-by-trade label/backtest parity must be tested.

The isolated intraday engine now enforces the timing/OHLC contract, accepts
complete bid/ask candles for executable long and short fills, and exposes the
frozen stop-sizing and portfolio loss gates. The IG adapter now retains
time-varying quotes, reads account/market metadata, compares exact timestamps
and returns against Dukascopy, and can build a fully checked **dry** paper
intent. Real order submission remains deliberately disabled; account-specific
epic, size increment, and point value must be confirmed before the dry harness
will pass.

Paper runtime facts now persist atomically under `data/runtime/`: exact expected
broker position IDs, day/week equity anchors, entry counts, kill-switch state,
and every reserved intent ID survive a restart. Candidate datasets are likewise
paired with checksummed manifests and exact execution-ledger reconciliation
before the ML trainer will read them.

`scripts/import_cme_gc.py` verifies Databento batch manifests and matching
ranges, maps instrument IDs, removes calendar spreads, rejects malformed
records, and converts split files to checksummed monthly Parquet partitions.
It creates no active contract, feature, or strategy outcome. See
[`docs/cme_futures_data.md`](docs/cme_futures_data.md).

`scripts/import_cme_gc_tbbo.py` provides the separate trade-space boundary. It
requires Databento `tbbo`, validates every pre-trade book and aggressor-side
record, maps reused IDs temporally, and creates no strategy result.

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

Historical development ends 2025-06-30. The unread `2025-07-01` onward window
is the intraday program's single-use historical confirmation period, opened
only after the full version-1 contract and gates are committed. Data from
2026-07-31 onward is the prospective paper boundary.

## 11. Build order

1. **Complete:** freeze the intraday mandate in plan §2.
2. **In progress:** repair label/backtest/execution correctness without
   searching for alpha. Broker/quote/risk dry-run controls, restart-safe paper
   state, exact position reconciliation, and artifact integrity are implemented;
   submit-once adapter wiring and live validation remain.
3. **Complete on development artifacts:** causal H1/M30 base, candidate
   datasets, confirmation fence, scheduled-event blackouts, and optional
   completed-M15 diagnostic context.
4. **Complete, honest negative:** exact pre-registered P1/P2/P3 controls and
   the frozen P1V2/P2V2/P3V2 primary-rule gate ran on gold. No bare rule passed.
5. **Complete, honest negative:** candidate-only logistic/LightGBM training and
   the frozen nested filter gate ran on gold development data. Neither filter
   passed; confirmation remains unopened.
6. **Risk forecast retained; economic use still disabled:** the two-hour B0
   baseline passed at OOS R2 `0.2032`; GVZ and the event calendar passed their
   separate placebo gates. A subsequently pre-registered joint model passed at
   R2 `0.2285`, so B0+GVZ+calendar is the retained forecast. The downside-only
   1.0x/0.5x/0.0x ladder and a matched economic evaluator are implemented, but
   the evaluator refuses to read strategy P&L because no primary rule has
   passed. Confirmation remains unopened and `paper_enabled=false`.
7. **Complete, honest negative:** LBMA-PM benchmark continuation B1 failed its
   economic, temporal, double-cost, and time-placebo gates.
8. **Complete, honest negative:** the point-in-time, roll-aware F-GC family was
   evaluated against matched spot-only P1 and 20 placebos. It failed. IG
   M15/M30 bid/ask cost collection still awaits account setup.
9. Transfer a genuinely retained gold pattern to silver, then platinum and
   palladium.
10. Build correlation-aware allocation with an explicit cash outcome.
11. Paper trade in separate operational and performance-evaluation stages.
12. Begin minimum-size live trading only after both gates pass.

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

## 13. Frozen version-1 mandate

The mandate is fixed in
[`docs/intraday_rebuild_plan.md` §2](docs/intraday_rebuild_plan.md):

- entries from 07:00 to before 19:30 Europe/London;
- forced flat by 21:30, with earlier holiday handling;
- both long and short trades;
- one open precious-metals position portfolio-wide;
- 0.25% initial stop risk, −0.75% daily and −1.50% weekly loss gates;
- exact P1 opening-breakout, P2 trend-pullback, and P3 mean-reversion rules;
- next-contiguous-bar-open execution with actual OHLC barriers; and
- account-specific IG admission gates rather than guessed epics or sizes.

These are pre-outcome choices. A later change creates a new version and a new
ledger entry.
