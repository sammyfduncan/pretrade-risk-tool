# Intraday commodity-system rebuild plan

**Status:** VERSION-ONE DIRECTIONAL DISCOVERY CLOSED — QUANTITATIVE PRE-TRADE
RISK ENGINE ACTIVE

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

## 2. Frozen version-1 mandate

These decisions were fixed on 2026-07-30 before any intraday strategy outcome
was read. They are starting hypotheses, not numbers selected from a sweep.
Changing one after results creates a new version and a new comparison in the
ledger.

### 2.1 Session, cutoff, and event restrictions

- Session timezone: `Europe/London`, so the broker rollover rule follows UK
  daylight-saving time.
- Eligible entry window: **07:00 inclusive to 19:30 exclusive** London time.
  Signals may be computed outside the window for logging, but cannot open a
  position.
- Forced-flat cutoff: **21:30 London time**, 30 minutes before IG's usual
  22:00 funding boundary. The position exits at the close of the final complete
  M30/M15 bar ending at or before the cutoff.
- Holidays and broker early closes override the normal cutoff: flatten no later
  than 30 minutes before the published close.
- Version 1 does not enter during the Asian/overnight session and does not
  carry positions through the daily funding boundary.
- CPI and NFP: no new entry from 30 minutes before until 30 minutes after the
  exact release timestamp.
- FOMC rate decision/statement: no new entry from 30 minutes before until
  60 minutes after.
- An existing position is not automatically closed solely because a blackout
  begins; its pre-defined stop, target, timeout, and daily cutoff still apply.
  A future event-exit rule would be a separate experiment.

### 2.2 Direction and position overlap

- Version 1 may trade **both long and short**. Intraday metals have no
  defensible long-only drift strong enough to justify discarding half the
  primary hypotheses.
- The primary rule chooses the side; ML never reverses it.
- Before the allocator exists, assets are researched separately and only one
  position may be open.
- The first portfolio allocator also starts with **one open precious-metals
  position portfolio-wide**. This avoids pretending that simultaneous
  gold/silver/PGM positions are independent. Raising the limit requires a
  separate correlation/exposure gate.
- A candidate arriving while a position is open is logged and rejected; it is
  not queued for a later, different price.

### 2.3 Risk limits

- Initial stop risk: **0.25% of current equity per trade** (`1R`).
- Maximum gross notional exposure: **100% of current equity**. If the broker's
  minimum size cannot satisfy both the stop-risk and notional caps, the asset
  is paper-only.
- Maximum new entries: **three per London trading day** portfolio-wide.
- Daily loss gate: after realized plus open P&L reaches **−0.75%** from the
  day's starting equity (`−3R`), flatten any open position and allow no new
  entries until the next London trading day.
- Weekly loss gate: after realized plus open P&L reaches **−1.50%** from the
  week's starting equity (`−6R`), flatten and pause until the next London
  Monday.
- The daily and weekly gates may stop trading; they may never increase size to
  recover losses.
- Minimum-size live trading may use less than 0.25% risk. It may not use more
  without a pre-defined scale review.

### 2.4 Common executable trade contract

All three primary families use the same contract:

- Candle timestamps denote bar **open** time.
- A candidate is known only after its signal bar closes.
- Entry is the **next contiguous M30 bar open**. M15 may later refine that
  entry only under the §4 cost gate.
- No candidate may fill across a missing bar, session boundary, or market
  closure.
- Stop/target distances use information fixed at the signal close and are
  anchored to the actual next-bar entry price.
- Stop and target checks use the entry/future bars' real high and low.
- If one OHLC bar touches both stop and target, resolve to the stop
  (conservative/adverse ordering).
- Timeouts exit at the relevant bar close. The daily cutoff overrides a longer
  timeout.
- With complete bid/ask candles, longs enter at ask and exit/check barriers at
  bid; shorts enter at bid and exit/check barriers at ask. Partial quote candles
  are invalid. Mid OHLC plus the configured flat cost remains only a research
  fallback where quote history is unavailable.
- When quotes are present, the configured per-side cost is an additional
  commission/slippage allowance because spread is already embedded in the
  executable prices. Promotion requires observed/time-varying IG spread plus
  slippage stress.
- The ML label is `1` only when the **net** trade return after costs is
  positive; otherwise it is `0`.

Here `ATR` means M30 Wilder ATR(14) computed through the signal close, and `R`
means the fixed entry-to-stop distance.

### 2.5 Exact primary strategy hypotheses

#### P1 — London opening-range breakout

- Opening range: M30 bars opening from **07:00 through 08:30** London time
  (the two-hour `[07:00, 09:00)` window).
- Compression gate: that range is no larger than `1.25 ×` the median opening
  range of the previous 20 valid London sessions.
- H1 direction gate, using the last completed H1 bar:
  - long: close above EMA50 and EMA50 above its value three H1 bars earlier;
  - short: close below EMA50 and EMA50 below its value three H1 bars earlier.
- Signal: the first eligible M30 close after 09:00 that closes at least
  `0.10 × ATR` beyond the opening-range high (long) or low (short), aligned
  with the H1 gate.
- Maximum one P1 trade per asset per London day.
- Stop: `1.00 × ATR`.
- Target: `1.50R`.
- Timeout: six M30 bars (three hours) or the daily cutoff, whichever comes
  first.

#### P2 — H1 trend, M30 pullback continuation

- H1 long regime: EMA20 above EMA50, close above EMA20, and EMA50 above its
  value three H1 bars earlier. The short regime is the exact inverse.
- Long M30 signal: the bar low touches or crosses EMA20, then closes above
  EMA20 with a positive candle body, and its close is no farther than
  `0.50 × ATR` above EMA20.
- Short M30 signal: the bar high touches or crosses EMA20, then closes below
  EMA20 with a negative candle body, and its close is no farther than
  `0.50 × ATR` below EMA20.
- Maximum two P2 trades per asset per London day, still subject to the
  portfolio-wide three-entry limit.
- Stop: `1.00 × ATR`.
- Target: `1.50R`.
- Timeout: eight M30 bars (four hours) or the daily cutoff, whichever comes
  first.

#### P3 — non-trending-regime mean reversion

- H1 non-trending gate, using the last completed H1 bar:
  - `abs(EMA20 − EMA50) / close <= 0.25 × ATR14 / close`; and
  - `abs(EMA50 − EMA50.shift(3)) / close <= 0.25 × ATR14 / close`.
- M30 excursion/re-entry signal:
  - long: previous close below its lower Bollinger Band(20, 2), followed by a
    close back above the current lower band;
  - short: previous close above its upper band, followed by a close back below
    the current upper band.
- Maximum one P3 trade per asset per London day.
- Stop: `1.00 × ATR`.
- Target: `1.00R`.
- Timeout: four M30 bars (two hours) or the daily cutoff, whichever comes
  first.

No neighboring thresholds, alternative EMA lengths, or additional indicator
confirmations are part of the first registered look.

### 2.6 Evaluation boundary and holdout ownership

- Historical intraday development data ends **2025-06-30 23:59 UTC**.
- The old H4 strategy is retired without spending its underpowered holdout.
  Its `2025-07-01` onward window is reassigned, still unread for strategy
  outcomes, as the intraday program's **single-use historical confirmation
  window**.
- The confirmation window is not opened until the execution contract, three
  primary strategies, cost assumptions, and retention gates are implemented
  and committed.
- Confirmation outcomes may confirm or retire version 1; they may not tune it.
  A changed version waits for prospective data.
- **2026-07-31 00:00 UTC onward** is the prospective boundary. Data after that
  time may be used for feed/operational diagnostics, but its strategy P&L is
  reserved for the frozen paper-performance stage.

### 2.7 IG instrument admission gate

IG epics, minimum sizes, and account-specific spreads must be discovered from
the user's demo account; they must never be guessed or hardcoded.

An asset is admitted only if:

1. `scripts/find_epic.py` identifies the regional epic and the user confirms
   the intended spot-metal market;
2. published/dealt hours cover the §2.1 entry window and cutoff;
3. recent IG and Dukascopy M30/H1 returns pass the source-comparison gate;
4. minimum size can keep stop risk at or below 0.25% and gross notional at or
   below 100% of equity;
5. median observed round-trip spread plus slippage is at most `0.10R`; and
6. the 95th-percentile stressed round-trip cost is at most `0.25R`.

Gold and silver are checked first. Platinum and palladium remain excluded until
their own admission records pass.

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

Implementation detail: each primary family receives its own candidate dataset
and model because its mechanism and setup-distance features differ. Initial
training uses deep Dukascopy H1/M30 development history; recent IG data is the
execution/feed-parity control, not a substitute bulk training source. Candidate
folds are purged using actual trade exit times rather than a fixed number of
irregular candidate rows. The complete retraining lifecycle is frozen in
`docs/intraday_ml_retraining.md`.

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

1. **Mandate freeze — COMPLETE 2026-07-30**
   - §2 fixes session/cutoff, long/short, risk, three exact primary rules,
     evaluation boundaries, holdout disposition, and IG admission gates;
   - no intraday strategy outcome was read.
2. **Execution-contract repair — IN PROGRESS**
   - standalone next-bar-open OHLC resolver and reason-coded ledger implemented;
   - quote-aware long/short fills, explicit-cost separation, frozen stop sizing,
     and portfolio loss/entry gates implemented;
   - synthetic parity, quote, cost, cutoff, blackout, early-close, sizing,
     threshold, non-overlap, and exact candidate-label tests added;
   - IG quote ingestion, UTC bar handling, account/market adapters, shifted-feed
     source parity, and a fail-closed dry paper-intent harness implemented;
   - checksummed candidate/ledger manifests, exact persisted reconciliation,
     atomic paper restart state, intent lifecycle/idempotency, exact broker
     position-ID reconciliation, London day/week resets, and a persistent kill
     switch implemented;
   - wiring state to a future submit-once order adapter, live streaming, and
     actual demo-account validation remain before this phase is complete.
3. **Intraday data path**
   - **implemented:** granularity-specific fetch artifacts and causal completed
     H1 -> M30 joins;
   - **implemented:** development-only base builder and confirmation fence;
   - **implemented:** optional synchronized completed-M15 diagnostic context,
     kept outside the frozen M30 rules and initial ML contract;
   - live-validate Dukascopy/IG parity with the user's account.
4. **Primary rules**
   - **implemented and evaluated:** exact P1/P2/P3 candidate generators and
     ATR-distance candidate contract;
   - **complete, honest negative:** the frozen opening-impulse/active-regime
     P1V2/P2V2/P3V2 program also failed its bare economic gate. P1V2 and P3V2
     showed small gross effects that did not clear fixed costs; see
     `docs/intraday_v2_design.md`.
   - **complete, honest negative:** matched M15 replay improved only three of
     508 P1V2/P3V2 trade outcomes and failed its stability/cost gate; see
     `docs/intraday_m15_execution_design.md`.
5. **ML challenger**
   - **implemented:** candidate-only dataset schema and exact execution labels;
   - **implemented:** logistic/LightGBM research trainer with exit-time-purged
     OOF folds, uniqueness weighting, and no deployment threshold;
   - **complete 2026-07-31 — neither filter retained:** after the event-blackout
     correctness rebuild, P1-logistic improved PF `0.701 -> 0.970` in 5/5 folds
     but remained negative (`-0.044R`); P2-LightGBM improved PF `0.827 ->
     0.852` in 4/5 folds and remained negative (`-0.099R`). See
     `docs/intraday_filter_gate.md`.
6. **Risk layer**
   - **implemented and passed on development:** the causal two-hour B0 forecast
     achieved OOS R2 `0.2032`; GVZ and the event calendar each passed their
     matched placebo gates;
   - **joint model retained:** after a separate preregistration, adding the
     calendar to B0+GVZ raised OOS R2 `0.2242 -> 0.2285`, improved 4/5 folds,
     and beat the maximum registered placebo delta (`0.00429 > 0.00082`);
   - **implemented but disabled:** expanding-history 1.0x/0.5x/0.0x volatility
     risk policy, sizing-budget propagation, and dry paper-intent enforcement;
   - **implemented and fail-closed:** a matched constant-risk versus
     volatility-risk economic gate. It cannot read real strategy P&L until one
     specifically named M30 primary strategy has an independently passing
     primary-gate report;
   - **implemented:** offline IG quote-cost profiling by London slot, weekday,
     representative stop, and overlapping candidate ledger;
   - pending credentials/data: measured IG M15/M30 spreads and demo-fill
     slippage.
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

## 13. Implementation ledger

### 2026-07-30 — mandate freeze and execution-contract foundation

- Fixed every former §2 open decision using the user's authorization to choose
  conservative commodity-specific defaults.
- Reassigned the unread old H4 holdout once as the frozen intraday historical
  confirmation window; no outcomes were read.
- Added `commodity_ml.intraday.execution` without modifying the historical H4
  label or vectorbt paths.
- The new resolver uses next-contiguous-bar open, real OHLC barriers, adverse
  same-bar ordering, London/DST-aware cutoff, early-close overrides, scheduled
  entry blackouts, round-trip costs, long/short symmetry, and a one-position
  reason-coded ledger.
- Labels are derived from the exact net executed outcomes.
- This entry records correctness work only. No P1/P2/P3 signal was generated,
  backtested, or inspected.

### 2026-07-30 — quote-aware fills and executable risk policy

- Extended the same resolver to accept complete mid + bid/ask OHLC candles.
  Long fills and barriers now use ask-to-bid economics, short fills and barriers
  use bid-to-ask economics, crossed or partial quotes fail closed, and each
  ledger row records whether it used quotes or the mid-price fallback.
- Kept observed spread and explicit friction separate: quote spread is embedded
  in gross return, while `cost_pct_per_side` represents additional
  commission/slippage.
- Added deterministic stop-based sizing using broker minimum size, size
  increment, account-currency point value, a 0.25% stop-risk budget, and the
  100% gross-notional cap. An instrument is rejected when its minimum size
  exceeds either cap.
- Added portfolio gates for one open position, three entries per London day,
  and exact daily/weekly equity thresholds. Current equity explicitly includes
  open P&L; a loss-gate breach requests an immediate flatten.
- These are policy primitives for the future allocator/paper loop. They are not
  yet wired to broker contract metadata, quote collection, or order placement.
- This entry is correctness work only. No P1/P2/P3 outcome or confirmation data
  was read.

### 2026-07-30 — IG admission and dry paper boundary

- The canonical candle schema now preserves a complete bid/ask OHLC set and
  fails closed on partial, crossed, non-finite, or internally invalid quotes.
  IG historical conversion requests `snapshotTimeUTC` through trading-ig's
  flat formatter, avoiding local-time/DST ambiguity.
- Added bounded recent IG quote-window fetching and M15/M30/H1 CLI granularity
  overrides without changing the frozen H4 configs.
- Added account snapshots where equity is balance plus open P&L, current market
  quote freshness/status/delay checks, market metadata parsing, and broker
  position-ID reconciliation.
- IG minimum deal size is read from the account-specific market. Size increment
  and account-currency P&L per price point must be confirmed from that demo
  order ticket and configured explicitly; the system refuses to infer them
  from a different regional epic or product.
- Replaced the print-only source comparison with a reusable gate covering exact
  timestamp coverage, contiguous returns, tracking error, and ±2-bar shift
  detection. A materially better shifted alignment is a failure, not a pass.
- Added closed-bar freshness/contiguity checks and the frozen median `0.10R` /
  p95 `0.25R` spread-plus-slippage admission gates.
- Added a dry paper-intent path that combines broker reconciliation, duplicate
  intent IDs, kill switch, account/product checks, quote freshness, spread
  history, daily/weekly loss gates, and stop/notional/minimum-size sizing.
  It produces an auditable order-shaped object but contains no submission call.
- Added `scripts/audit_ig.py` for the admission checks and
  `scripts/paper_dry_run.py` for an explicit candidate plumbing check. Both are
  non-dealing commands.
- No IG credentials were available during implementation, so live account facts
  remain unverified. No primary strategy, confirmation outcome, or new P&L was
  generated or read.

### 2026-07-30 — causal intraday data, primary rules, and ML training contract

- Added a development-only H1/M30 builder. H1 state is keyed by bar-close
  availability and joined backward to the M30 decision time with a bounded
  staleness tolerance; a forming H1 candle cannot enter a feature row.
- Added causal previous-session and completed opening-range state plus the
  normalized H1/M30 foundation features.
- Implemented the exact frozen P1 opening breakout, P2 trend pullback, and P3
  non-trending Bollinger re-entry conditions. These generate both long and
  short candidates without ML.
- Corrected the candidate/execution representation so a signal-close ATR stop
  remains an absolute price distance across a next-bar entry gap. Targets are
  exact `R` multiples of that distance.
- The simulator now applies the frozen one/two/one per-family daily entry caps
  and the three-entry portfolio cap to executed trades, while retaining
  reason-coded rejected signals.
- Added a strict candidate-only ML dataset: features are enumerated by strategy,
  exit/label metadata cannot become inputs, and incomplete warmup rows fail
  loudly rather than disappearing.
- Extended walk-forward training with actual-exit-time purging for irregular
  candidates. Added a research trainer that always supports the logistic
  baseline and LightGBM challenger, saves OOF scores, and deliberately saves no
  deployment threshold.
- Added `docs/intraday_ml_retraining.md`. Initial data is Dukascopy H1/M30;
  IG supplies parity, spread, and eventual paper/live truth. Frozen performance
  paper trading never retrains.
- No intraday raw artifacts were fetched and no P1/P2/P3 outcome or model score
  was generated or read during this implementation work.

### 2026-07-31 — restart safety, artifact integrity, and optional M15 context

- Added an atomic, locked JSON paper-state store under `data/runtime/`. It
  persists day/week equity anchors, entries, exact expected broker position
  IDs, the kill switch, and prepared/submitted/confirmed/rejected intent
  records across process restarts.
- A new state can only be initialized against a flat broker account. Unknown
  open positions are never silently adopted. Intent IDs are reserved before a
  future submit call, and all known IDs remain duplicate-blocked after restart.
- Reconciliation now compares exact broker position IDs, not only counts.
  London day and Monday-week boundaries reset once; a backward clock or corrupt
  or unknown state schema fails closed.
- Candidate datasets and their full reason-coded ledgers are now published with
  a manifest containing source, strategy, feature order, row counts, and SHA-256
  hashes. The trainer verifies hashes and exact row-level execution parity
  before fitting anything.
- Added optional causal M15 attachment. Only M15 bars whose closes are available
  by the M30 decision may join; these diagnostic columns cannot enter P1/P2/P3
  or the initial candidate-model feature contract.
- No broker credentials, network data, strategy outcomes, confirmation rows, or
  model scores were used.

### 2026-07-31 — candidate ML training and frozen filter gate

- The initial candidate run exposed that scheduled-event blackouts were not
  wired into the CLI and included 2019 outside verified calendar coverage.
  The implementation was corrected before finalizing the research record; all
  bundles, models, and gates were regenerated over 2020–2025. The initial
  numbers are superseded.
- Corrected gold development produced 902 P1, 1,111 P2, and 312 P3 executable
  candidates. All three model pairs were trained with purged expanding OOF
  evaluation; no threshold was saved into any model.
- Pooled model screening left two weak leads: P1 logistic and P2 LightGBM.
  Before inspecting deployable filtered returns, the single top-30% temporal
  train/reference/test gate and its economic retention conditions were frozen
  in `docs/intraday_filter_gate.md`.
- P1 logistic reduced the OOF universe from 750 to 239 trades and improved PF
  `0.701 -> 0.970` in all five folds, but mean R remained `-0.044` and PF fell
  to `0.786` at double costs.
- P2 LightGBM reduced 925 to 391 trades, with PF `0.827 -> 0.852`, mean R
  `-0.099`, improvement in four folds, and double-cost PF `0.697`.
- Both failed the pre-registered gate. No alternate threshold was tried, no ML
  filter is retained for intraday version 1, and confirmation remains unopened.

### 2026-07-31 — frozen V2 primary-rule research

- Pre-registered one opening-impulse/active-regime variant for each existing
  family before generating outcomes. Added only two causal state variables:
  prior-median-relative M30 ATR and the completed London opening impulse.
- Corrected executed samples were 323 P1V2, 305 P2V2, and 185 P3V2 trades.
  All three failed their bare-rule economic/stability gate. Net PF was `0.928`,
  `0.817`, and `1.000`; double-cost PF was `0.766`, `0.688`, and `0.803`.
- P1V2 and P3V2 had positive gross means of `1.23 bp` and `2.00 bp` per trade,
  respectively, but the fixed historical allowance is `2 bp` round trip. This
  is a cost-sensitive lead, not a net edge. No V2 ML was trained.
- A matched M15 execution-only challenger then replayed all 323 P1V2 and 185
  P3V2 trades at the same entry timestamps. PF improved only `0.928 -> 0.937`
  and `1.000 -> 1.050`; only three total stop outcomes became targets and the
  improvement appeared in one/two of five blocks. Both failed the `1.10` and
  double-cost gates, so M15 is not retained for these rules.
- IG quote-based cost validation and the risk/sizing layer are next. No
  neighboring V2 threshold will be inspected; confirmation remains unopened.

### 2026-07-31 — quote-cost profiler and benchmark-window research

- Added an offline cost profiler for canonical IG bid/ask candles. It reports
  basis-point quantiles, London time slots, weekdays, representative stop-R,
  and an optional matched candidate-ledger estimate. It makes no network call;
  actual values await the user's confirmed demo epic and cached quote sample.
- Pre-registered one distinct gold market-structure hypothesis before outcomes:
  continuation through the official 15:00 London LBMA PM auction after a
  one-ATR preceding-hour move. Four fixed non-auction clocks were registered as
  mechanism placebos; the AM auction and COMEX settlement were excluded.
- B1 produced 776 trades, gross PF `1.045` and net PF `0.851`; only one of five
  blocks was profitable, max drawdown was `-14.5%`, and double-cost PF was
  `0.695`. The 16:00 placebo PF (`0.872`) exceeded B1 (`0.851`). B1 failed and
  no ML model was trained.
- The next market-structure program requires point-in-time COMEX futures volume,
  roll-aware prices/basis, or order-flow data. Dukascopy candle volume is a
  best-quote activity measure, not centralized traded gold volume, and is not
  relabelled as such. Confirmation remains unopened.

### 2026-07-31 — intraday risk experiment pre-registration

- Locked one two-hour volatility-forecast target on eligible M30 decision rows,
  using only observed contiguous Dukascopy closes.
- Registered a causal price/session OLS baseline and exactly two matched-row
  additions: lagged GVZ and precise scheduled-release proximity.
- Fixed five purged expanding folds, 20 time-shift placebos per family, a
  five-condition retention gate, a 90% matched-coverage requirement, and
  synthetic causality/data-integrity tests before any real-data run.
- No real-data forecast or outcome was generated. A pass would name an input candidate,
  not a strategy or sizing policy; `docs/intraday_risk_design.md` must be
  committed before implementation may read real-data results.

### 2026-07-31 — walk-forward volatility and conservative controls

- Implemented the registered observed two-hour realized-volatility target from
  exactly four contiguous future M30 closes. Missing target paths are
  reason-coded; prices are never interpolated or synthesized.
- Added the fixed price/session B0 inputs, lagged GVZ and exact event-proximity
  families, matched-row construction, five-fold train-only-standardized OLS
  forecasts, and registered family ablations.
- Added a guarded real-data CLI with 20 seed-0 placebos per family, pinned GVZ
  and event snapshots, hashes/provenance, a 90% coverage refusal, OOF output,
  and a disabled final artifact. It refuses to run until both registrations are
  tracked, committed, and clean.
- Implemented the frozen expanding-rank policy: below prior p80 keeps the
  existing cap, p80-p95 halves it, and p95+ blocks. At least 500 strictly prior
  OOF/prospective forecasts are required. The multiplier now reaches both
  stop-based sizing and dry paper-intent construction, but remains optional and
  disabled until a forecast gate passes.
- The full offline suite passes. Only synthetic forecasts were generated;
  gold development outcomes and confirmation remain unread by this program.

### 2026-07-31 — volatility forecast gates and matched economic boundary

- The registered development run used 28,513 matched rows with 99.48% coverage.
  B0 passed with pooled OOS R2 `0.2032`; F-IV added `0.0210` in 5/5 folds and
  F-CAL added `0.0052` in 4/5 folds, both above their placebo maxima.
- Because both families passed, the frozen rule initially selected B0 rather
  than choosing the stronger family after seeing results. A new joint design
  was then committed before its first outcome.
- The joint B0+GVZ+calendar model passed: R2 `0.2285`, incremental R2 `0.00429`
  over B0+GVZ, 4/5 improved folds, and placebo maximum `0.00082`. The joint
  forecast is retained with `paper_enabled=false`; confirmation was not read.
- Added a synthetic-tested matched economic evaluator for the frozen
  1.0x/0.5x/0.0x ladder. It preserves blocked opportunities as zero-return
  matched rows, uses strictly prior OOF forecast ranks, and refuses low exact
  timestamp coverage.
- The real economic command is deliberately blocked before loading trade P&L:
  P1/P2/P3, all V2 variants, their ML filters, the M15 challenger, and B1 all
  failed. A future retained primary strategy must be named and committed before
  this risk overlay can be judged or enter shadow paper trading.

### 2026-07-31 — causal GC futures-information gate

- Imported and checksummed the full pre-confirmation Databento GC definition,
  OHLCV-1m, and statistics jobs. The importer handles time-varying instrument
  IDs, removes spreads, records degraded dates, and excludes the 2025-07-01
  files before parsing their records.
- Pre-registered one P1+logistic F-GC family with a prior-completed-session
  volume roll, six causal futures inputs, a matched spot-only control, the
  existing nested threshold procedure, and 20 joint time-shift placebos.
- The first execution stopped before model fitting because the original
  same-clock-slot basis reference was structurally sparse. A documented
  pre-outcome feasibility correction changed it to the prior 20 same-contract
  M30 observations; coverage then passed at 844/902 candidates (93.6%).
- F-GC failed. OOF AUC moved `0.5294 -> 0.5187` (delta `-0.0107`), only 2/5
  folds improved, and the placebo maximum was `+0.0195`. Filtered PF improved
  only `0.818 -> 0.862`, mean R was `-0.100`, and double-cost PF was `0.694`.
- No individual feature rescue, alternate roll, other strategy/model, or
  confirmation run is permitted under this program. No deployment artifact
  was created.

### 2026-07-31 — GC trade-flow successor

- Selected Databento TBBO as the next genuinely distinct source. It carries
  every trade with aggressor side and the best bid/offer state immediately
  before that trade, while avoiding MBP-1's non-trade quote-update volume.
- Pre-registered one OF-GC/P1-logistic family with exact 5/30-minute signed
  flow, depth, spread, and sweep inputs; the same matched control,
  nested threshold, economic gate, and 20 joint placebos remain mandatory.
- Implemented a checksummed, confirmation-fenced TBBO importer; temporal
  instrument mapping; prior-session active-contract filtering; causal minute
  aggregation; and post-decision poisoning tests. No TBBO source data or
  strategy outcome has been read.
- A one-month `2025-05-01` through `2025-06-01` TBBO delivery was requested only
  for schema/coverage/size validation; it was not a performance sample.
- The delivered sample included 1 June and ended 2 June exclusive. It passed:
  2.70 million causally active trades, 97.29% explicit aggressor sides, and
  99.9992% valid active pre-trade books. Twenty-one invalid active startup
  records are reason-coded and excluded without imputation.
- Added partition-streamed minute aggregation and proved it matches a single
  in-memory frame. The sample produced 46,080 causal minute rows without
  reading labels or returns.
- Rejected the approximately $400 continuous-history quote before opening the
  strategy gate. Acquisition is now limited to the active raw contract in the
  30 minutes preceding each frozen P1 decision. The acquisition-heavy
  20-session intensity feature was withdrawn pre-outcome; all original pass
  thresholds remain unchanged.
- Databento's free metadata endpoint subsequently quoted the complete sparse
  plan at $7.2312: 901 windows, 450.5 requested hours, 29 raw contracts, and
  99.89% candidate coverage. No paid endpoint has been called; acquisition
  still requires explicit approval.
- The user authorized the $7.24 ceiling; all 901 windows were downloaded and
  checksum-verified. Normalization retained 3.47 million outright trades with
  98.57% explicit sides and 99.99994% valid pre-trade books.
- OF-GC failed honestly. Matched coverage passed at 94.5%, but AUC fell
  `0.5511 -> 0.5301` (delta `-0.0210`), zero of five folds improved, and the
  best placebo delta was `+0.0068`. Filtered PF fell `0.955 -> 0.931`, mean R
  was `-0.068`, and double-cost PF was `0.755`. The family is retired without
  feature rescue, strategy/model transfer, or confirmation access.

### 2026-08-03 — final P4 macro-impulse family frozen

- Frozen one final outright-gold directional mechanism before acquiring or
  inspecting its development inputs: aligned M30 `ZN` Treasury-price and `6E`
  Euro-FX impulses, with contemporaneous non-extended gold confirmation.
- Fixed prior-session active-contract selection, a previous-60 same-London-slot
  median/MAD standardization, `|z| >= 1` agreement, first signal per day,
  1 ATR stop, 1.5R target, six-bar timeout, existing event blackouts, and all
  execution costs before outcomes.
- Fixed a hard gate: at least 90% causal coverage, 200 trades and 30 per block,
  PF at least 1.10, positive mean R, four profitable blocks, PF above one at
  double cost, and observed PF above all 20 joint time-shift placebos.
- Implemented causal driver construction, exact P4 candidates, placebo and
  performance gates, outcome-blind normalization, a metadata-only acquisition
  planner, and a separate downloader that refuses any quote above an explicit
  user-authorized dollar ceiling or any automatic paid retry.
- No ZN/6E data has been authorized by this design entry. P4 failure closes
  version-one outright-gold directional discovery without threshold, driver,
  model, target, or timeout rescue variants.
- The metadata-only quote subsequently completed at `$18.939715769142` for all
  four requests. No dates are missing or pending. Six dataset-degraded dates
  are frozen as ineligible. The quote did not call a paid endpoint and does not
  authorize acquisition.
- The user authorized the `$18.94` ceiling. All four files downloaded and
  checksum-verified. Normalization retained 2.19 million outright ZN and 2.48
  million outright 6E bars; 55 pre-definition ZN rows were excluded rather
  than mapped with future identity.
- P4 failed its frozen gate despite passing coverage (97.85%) and sample-size
  requirements (857 executed trades). PF was `0.851`, mean R `-0.096`, only
  one of five blocks was profitable, double-cost PF was `0.695`, and the best
  placebo PF was higher at `0.913`. Confirmation remains unopened, no ML was
  trained, and version-one outright-gold directional discovery is closed.

### 2026-08-03 — RV1 gold–silver relative value

- Opened one separately scoped mechanism after closing outright-gold direction:
  prior-1,000-observation gold/silver hedge beta, same-clock residual surprise,
  `|z| >= 2`, gross-normalized two-leg reversal, and first signal per day.
- Fetched and pinned free Dukascopy M30 bid/ask histories for both legs. Exact
  synchronization retained 64,955 rows at 99.995% coverage, with no filling,
  duplicates, or crossed open/close quotes.
- Implemented bid/ask next-open entries, close-observed portfolio barriers,
  frozen weights, six-bar timeout, event blackouts, historical spread execution,
  explicit cost allowance, two spread/cost stresses, and 20 joint-state shifts.
- RV1 failed: 1,014 trades, balanced 50/50 by side, PF `0.303`, mean R `-1.256`,
  zero profitable blocks, double-stress PF `0.083`, and max drawdown `-43.98%`.
  Observed PF exceeded the placebo maximum `0.274`, but remained untradeable.
- A fixed-clock midpoint audit found next-M30 PF `1.042` and only `0.126 bp`
  mean return, against `4.739 bp` average historical two-leg spread drag before
  the additional `1 bp` round-trip allowance. The tiny effect cannot pay costs.
- RV1 is retired without cost deletion, threshold/beta/exit variants, or ML.
  Confirmation remains unopened and paper trading remains disabled.

### 2026-08-03 — quantitative pre-trade risk engine foundation

- Reframed the next product stage around point-in-time risk assessment for the
  current gold and silver spread-betting instruments, without claiming a new
  directional edge or interpreting geopolitical/news content.
- Froze a version-one input contract covering trade geometry, account state,
  verified broker instrument facts, executable quote/cost state, optional
  approved volatility state, deterministic event blackouts, and existing
  precious-metals positions.
- Implemented the offline calculation core: cost-inclusive stop risk, net
  target reward, broker-step sizing, gross exposure, initial margin, daily and
  weekly loss gates, portfolio stop risk, signed one-percent exposure, and
  spread/range/gap stresses with no assigned probabilities.
- Implemented fail-closed `ALLOW`, `REDUCE`, `BLOCK`, and `INCOMPLETE`
  decisions with stable reason codes, JSON serialization, a network-free CLI,
  an intentionally incomplete template, and synthetic tests.
- The engine never guesses the user's IG epic, point value, minimum size,
  increment, margin rate or account values. It creates and submits no order.
  Historical scenario calibration and a read-only IG adapter are the next
  stages; paper execution remains separately disabled.

### 2026-08-03 — development-only historical risk calibration

- Froze four holding horizons (60/120/180/240 minutes), exact London-clock
  conditioning, a prior-252/minimum-126 causal window, and p50/p80/p95/p99
  range/adverse/spread measurements before reading calibration outcomes.
- Built 243,598 strictly-prior OOF rows across gold and silver. All eight
  asset/horizon cells passed with 90.9%-91.0% calibrated-row availability and
  approximately nominal 94.9%-95.2% realised coverage for p95 range and
  long/short adverse excursion.
- The retained gold two-hour volatility forecast had meaningful OOF
  association with total range (log correlation `0.5643`) but weak association
  with side-specific adverse excursion (`0.1031` long, `0.1141` short). It
  remains range-risk information, not directional evidence.
- Added a fail-closed exact asset/horizon/London-clock lookup and historical
  quantile scenarios to the offline evaluator. The profile expires after 90
  days; because its last observation is 2025-06-30, it cannot be presented as
  current in 2026.
- Dukascopy spread quantiles remain diagnostics only. IG spread measurement,
  current account/product facts, paper sizing and order submission are still
  disabled. Confirmation and strategy P&L were not read.

### 2026-08-03 — read-only IG risk snapshot adapter

- Added a pure adapter from current IG account, market, quote, bounded M30
  bid/ask history and open-position responses into the frozen pre-trade risk
  contract. The adapter and CLI contain no create, amend or close call and set
  `order_authority=false` in their audit output.
- Added explicit broker gates for enabled spread-bet accounts, non-expiring
  commodity instruments, exact epic agreement, real-time quotes and scoped
  percentage margin. Tier-one margin is never extended beyond its published
  size band.
- Added causal closed-bar, same-London-session IG median/p95 spread measurement.
  Insufficient observations remain unavailable rather than falling back to
  Dukascopy or future bars.
- Added conservative open-position parsing. Unknown stops, unmapped epics or
  point values fail incomplete; aggregate used margin is assigned to a lone
  position, while multiple-position allocation is refused.
- Synthetic integration tests pass. A live report remains blocked until the
  exact regional epic, demo-ticket size increment and point value are entered,
  runtime day/week anchors and entry count are available, and the current
  mechanical event-calendar state is supplied. No order submission was
  enabled and the stale 2025 development calibration was not presented as
  current.

### 2026-08-03 — cross-metal aggregation and static risk report

- Extended the read-only CLI with explicit additional portfolio configs, so a
  gold assessment can value an existing silver position (and vice versa) only
  when that position's epic and point value are confirmed. Conflicting epic
  mappings are refused.
- Retained the frozen one-open-position policy. Existing cross-metal stop risk,
  gross exposure, signed one-percent exposure and margin flow through the same
  portfolio calculations rather than being assessed as independent trades.
- Added a self-contained, dependency-free HTML renderer for offline and IG
  wrapper reports. It presents decisions, reasons, sizing, cost-inclusive stop
  risk, margin, portfolio exposure, stresses, limitations and provenance; it
  performs no broker action or additional calculation.
- Added escaping, incomplete-report, IG-provenance and no-external-asset tests.
  Live visual validation awaits the first account-specific report, but the
  offline report/portfolio implementation stage is complete.

### 2026-08-03 — prospective paper-risk observation ledger

- Added an append-only SQLite evidence boundary: a risk report is registered
  before an outcome exists, then exactly one separately timestamped realised
  observation may be attached to its intent id after exit.
- Exact retries are idempotent. Changed content under an existing assessment or
  outcome id, an outcome without a prior assessment, an outcome for a blocked
  or incomplete assessment, and entry before assessment are all refused.
- The outcome schema contains only measured paper facts: fill/exit prices and
  times, filled size, MAE/MFE, spread, adverse slippage, fees, peak margin, P&L
  and source. No missing value is predicted or imputed.
- Added descriptive summaries for execution/risk calibration, including
  size-adjusted planned-stop exceedance. These statistics make no expectancy,
  forecast-skill or causal claim.
- Synthetic immutability, ordering, validation, idempotency and summary tests
  pass. Live population and a valid executable-quote MAE/MFE collector still
  await the IG demo runtime.
