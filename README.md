# Multi-Commodity ML + Sentiment Rotation System — Project Scope

**Status:** Design reference, pre-engineering
**Purpose of this document:** A stable, high-level reference to work against during engineering. Contains scope, architecture, and methodology decisions only — no code, folder structures, or library-level specifics.

---

## 1. Purpose & Goals

A learning-first, portfolio-quality algorithmic trading project combining **price-based machine learning** with an **alternative-data sentiment layer**, applied across a small basket of **physical/precious-metal commodities** rather than FX. Profitability is a secondary goal; the primary goal is a technically sound, honestly evaluated system. Live capital is only to be considered after the paper-trading phase meets explicitly defined success criteria (not yet specified).

This project builds on, and shares standards with, an earlier rule-based (non-ML) XAUUSD day-trading strategy project — the methodology and risk discipline below are consistent with that earlier work, not a separate standard.

---

## 2. Asset Universe

- **Core assets:** Gold (XAUUSD), Silver (XAGUSD)
- **Recommended extension:** Platinum (XPTUSD), Palladium (XPDUSD)

**Rationale for extending beyond gold/silver:** gold and silver are highly correlated (both driven primarily by real yields, dollar strength, and safe-haven flow), so a rotation system limited to just these two mostly makes a sizing decision between near-twins rather than a genuine cross-asset choice. Platinum and palladium are still physical/precious metals (consistent with the "no FX" preference) but are driven substantially by industrial demand (automotive catalytic converters) and geopolitically concentrated supply (South Africa, Russia) — giving the system genuinely different signal sources to work with, and giving the sentiment layer real, distinct events to react to.

**Open item:** confirm broker and historical-data availability/quality for platinum and palladium specifically before finalizing the universe — coverage for these is less certain than for gold and silver.

**Design rationale for a shared universe:** rather than running four independent, uncoordinated single-asset bots, the system treats all assets as one competing pool and dynamically favors whichever looks most attractive at a given time — including the option of favoring none of them.

---

## 3. High-Level Architecture

Two parallel signal-generation tracks run per asset, converging into one cross-asset decision layer.

### Track A — ML Price Prediction (per asset)

- **One independently trained model per commodity**, not a single pooled model across all four. Chosen for debuggability (issues in one asset's model don't get tangled with another's), ease of adding or dropping an asset later, and because each commodity's driver set and price dynamics differ enough that a shared model risks one asset's regime dominating the others.
- Feature inputs: technical/statistical features per asset, plus relevant macro variables (e.g. dollar/real-yield proxies for gold and silver; industrial-demand proxies for platinum and palladium).
- Model family: tabular gradient-boosted trees, consistent with the ML approach chosen elsewhere in the broader project.
- Validation discipline: strict time-ordered walk-forward validation only. No random train/test splits, no future information leaking into training at any point.

### Track B — Sentiment / Alternative Data (per asset)

- Sentiment must be **entity- and theme-filtered per asset** — a single blended "commodities sentiment" score is explicitly rejected as too diffuse to be useful. Each asset needs its own filtered query (e.g. monetary-policy/central-bank themes for gold; export-policy and automotive-demand themes for platinum and palladium).
- Data sources: free and fully backtestable sources only. Primary candidates are GDELT and/or published academic geopolitical risk indices. **Twitter/X is explicitly excluded** — as of 2026 there is no meaningful free tier, and the full historical archive access required for honest backtesting is prohibitively expensive.
- Scoring approach (to be decided during engineering): simple lexicon-based scoring as a fast baseline, versus a domain-tuned transformer classifier for higher-nuance scoring.

### Composite Scoring & Cross-Asset Ranking

- The per-asset ML signal and sentiment score are combined into one composite score per commodity, recalculated at each decision point.
- All commodities are ranked by composite score.
- The ranking must include an explicit **"stay in cash / no position"** outcome — the system must never be forced into a position simply because it's the least-bad option among four weak signals.
- **Hysteresis is required**: a meaningful score gap must be required before rotating from one asset to another, to prevent excessive switching driven by ordinary signal noise (the same principle used for regime-switching in the earlier rule-based project).

### Allocation & Execution

- **Correlation-aware position sizing**: if two correlated assets (e.g. gold and silver) both rank highly at once, they must not be treated as two independent bets — combined exposure needs explicit capping or inverse-correlation-weighted sizing.
- **Turnover/switching cost is tracked as its own evaluation dimension**, since every rotation between commodities incurs real spread cost.
- Execution runs on paper trading accounts in the first phase; live capital is only considered after paper trading meets pre-defined success criteria (criteria not yet specified — an open item).

---

## 4. Methodology & Evaluation Principles

These carry over directly from the broader project's standards and apply to every component below (ML models, sentiment layer, and the combined system):

- **Minimum sample size before drawing conclusions**: a pre-registered minimum trade count (suggested: 100+ trades, or a fixed multi-month window, whichever is later) before any component is judged as working or not.
- **Walk-forward, time-ordered validation is mandatory** for both the ML models and any weighting/calibration applied within the sentiment layer.
- **Multiple-comparisons awareness**: testing several assets and components in parallel increases the odds that one looks good purely by chance. No component is promoted toward live consideration without confirmation on a further out-of-sample period.
- **Regime-aware evaluation**: results must be checked across different volatility/trend regimes, and where feasible across different sentiment/news-flow conditions, rather than judged on one arbitrary backtest window.
- **Portfolio-level metrics, not just per-asset ones**: combined Sharpe/Sortino ratio, max drawdown, turnover, and time spent in cash, in addition to each individual asset's signal quality.

---

## 5. Recommended Build Order

1. Build and validate the **price-only ML pipeline end-to-end for one asset (gold)** first, reusing and extending the existing architecture.
2. Add the **sentiment/alt-data layer for that single asset only**, and validate the composite scoring logic in isolation before generalizing to other assets.
3. Generalize the proven pattern to the remaining assets — silver, then platinum, then palladium — one at a time.
4. Only once every individual asset pipeline is validated, build the **cross-asset ranking, allocation, and correlation-aware sizing layer** on top.
5. Paper-trade the full multi-asset system for a sufficient period (per the methodology section above) before any live-capital discussion begins.

---

## 6. Open Items to Resolve Before/During Engineering

- Confirm final asset universe: gold + silver only, versus the full four-asset precious-metals basket — pending data/broker availability checks for platinum and palladium.
- Confirm trading cadence/timeframe for this specific system (day-trading cadence has been assumed for consistency with the rest of the project, but has not been explicitly re-confirmed for this multi-asset rotation system).
- Finalize the primary sentiment/alt-data source(s): GDELT, a geopolitical risk index, or a combination.
- Decide the composite-score combination method (fixed weighting vs. a learned/optimized weighting) and the specific hysteresis threshold approach.
- Define explicit, written success criteria for the paper-trading phase that must be met before any live-capital go/no-go decision.