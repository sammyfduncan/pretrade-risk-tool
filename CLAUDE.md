# CLAUDE.md — ML Commodity Trading

## What this project is

A learning-first, portfolio-quality algorithmic trading system combining per-asset
**price-based ML** (Track A) with an **entity-filtered sentiment layer** (Track B),
across precious metals (gold, silver, later platinum/palladium), converging into one
cross-asset rotation/allocation layer. **Profitability is secondary to honest
evaluation.** Read `README.md` for the full scope — it is the stable design
reference and should not be casually edited.

The user is fairly new to ML: explain ML concepts when they come up, walk through
execution steps interactively, and prefer readable code over clever code.

## Build order & current status

1. **[PIPELINE VALIDATED — no directional signal found] Price-only ML pipeline for gold**
2. Sentiment/alt-data layer for gold only + composite scoring
3. Generalize to silver → platinum → palladium (one at a time)
4. Cross-asset ranking, allocation, correlation-aware sizing
5. Paper-trade full system; live capital only after pre-registered success criteria are met

## Where training data comes from (important — two sources, on purpose)

**Broker ≠ data vendor here.** IG is the execution broker, but its REST API
allows only **10,000 historical price points per week**. Step 1 alone wants
~40,000 hourly bars, so pulling training history from IG would take a month.

- **Dukascopy** (`data/dukascopy.py`) — free, deep, no account. Supplies model
  training history. Default `data.source`.
- **IG** (`data/ig.py`) — epic lookup, recent bars, live prediction, and paper
  execution later. `fetch_candles` estimates the allowance cost *before*
  spending it and refuses oversized pulls unless explicitly overridden.

Both funnel through `data/schema.py::validate_candles`, which pins one canonical
shape (UTC index named `time`, float OHLC, sane high/low). Train-on-one,
trade-on-the-other is only legitimate if the feeds agree — `scripts/compare_sources.py`
measures that (returns correlation is what matters, not price level; Dukascopy
is bid, IG is mid, so a small constant offset is expected).

IG identifies markets by **epic** (`CS.D.USCGC.TODAY.IP`-style), not ticker, and
epics vary by account region — always look them up with `scripts/find_epic.py`,
never hardcode.

## Repo layout

```
config/gold.yaml          # per-asset config; adding an asset later = new yaml
src/commodity_ml/         # installed package (pip install -e .)
  config.py               # yaml -> dataclasses
  data/schema.py          # canonical candle contract + per-source granularity maps
  data/dukascopy.py       # bulk free history -> training data
  data/ig.py              # broker API: epics, recent bars, allowance guard, execution later
  features/technical.py   # pandas-ta-classic indicators + hand-rolled features
  features/macro.py       # FRED real yields/dollar/VIX; publication lags = lookahead risk
  labels.py               # label construction — the module most prone to lookahead bugs
  modeling/models.py      # LightGBM factory + LogisticRegression baseline
  modeling/walkforward.py # TimeSeriesSplit walk-forward harness, OOF predictions
  evaluation/ml_metrics.py       # AUC, Brier, per-fold tables
  evaluation/trading_metrics.py  # Sharpe, profit factor, max drawdown (pure pandas, no vbt dep)
  backtest/engine.py      # vectorbt: OOF probabilities -> simulated long-only trades
scripts/                  # CLI entry points, run in order:
  fetch_data.py -> build_dataset.py -> train.py -> backtest.py
  find_epic.py, compare_sources.py   # IG setup + cross-source validation
data/raw, data/processed, models/, reports/   # all gitignored artifacts
tests/                    # synthetic-data tests, no network or credentials needed
```

## Commands

```bash
source .venv/bin/activate            # Python 3.13 venv (3.14 too new for numba)
pip install -r requirements.txt && pip install -e .
pytest                               # all tests run offline on synthetic data

python scripts/fetch_data.py    --config config/gold.yaml   # Dukascopy, no credentials
python scripts/build_dataset.py --config config/gold.yaml
python scripts/train.py         --config config/gold.yaml [--baseline]
python scripts/backtest.py      --config config/gold.yaml

# IG side (needs .env credentials)
python scripts/find_epic.py --search "spot gold"
python scripts/fetch_data.py --config config/gold.yaml --source ig --start 2026-05-01
python scripts/compare_sources.py --config config/gold.yaml
```

## Hard rules (from README methodology — apply to every component)

- **Walk-forward, time-ordered validation only.** Never random splits, never shuffle.
  `TimeSeriesSplit(gap >= label horizon)` so overlapping forward-return labels can't
  leak across the train/test boundary.
- **No lookahead anywhere.** Features at bar *t* may use only data through *t*;
  labels look strictly forward. `tests/test_features.py` has a causality test
  (features on truncated vs full history must match) — keep it passing, extend it
  for every new feature.
- **External data is published later than it is dated.** Every macro series
  declares a conservative `publication_lag_days`; joining on the observation date
  leaks the future. Overstating a lag only weakens results, understating one
  manufactures them. `tests/test_macro.py` pins this.
- **Entry thresholds must be computed on an expanding window.** A quantile taken
  over the whole backtest lets future predictions set today's entry bar.
- **Backtests use out-of-fold predictions only** — never predictions from a model
  that saw that bar in training.
- **Min sample before conclusions:** 100+ trades or a multi-month window, whichever
  is later. Confirm on a further out-of-sample period before promoting anything.
- Report portfolio-quality metrics (Sharpe, PF, maxDD, turnover, time-in-cash), not
  just ML metrics, and check across volatility regimes.

## Engineering decisions made so far (provisional — confirm with user before changing)

- **Timeframe: H4 candles.** Moved from H1 after the first run. Macro data updates
  daily, so at H1 (24 bars per update) or M15 (96) the model is asked to time
  entries from a feature that is a step function. H4 gives ~12k bars — enough for
  a GBM, unlike daily's ~2.3k — while keeping trade frequency low enough that an
  edge could clear costs. Note the README's "day-trading cadence" was inherited
  from the earlier rule-based project and never re-confirmed for a macro/sentiment
  rotation system, which is structurally a swing-horizon design.
- **Label: symmetric triple barrier** (`label.mode: triple_barrier`). Take profit at
  +k·ATR, stop at −k·ATR, give up after `horizon_bars`. Symmetry is the point: it
  removes the volatility channel that made the old `long_threshold` label look
  predictive. `updown` and the legacy `long_threshold` remain available.
- **Barriers define the exit too**, so the backtest trades the same thing the label
  describes. Entries are debounced by the horizon so trades never overlap and each
  one is a genuinely independent bet.
- **Entry rule: expanding-window quantile**, not an absolute probability. A fixed
  0.60 is unreachable for a calibrated model on a ~50% base rate, and a
  full-sample quantile would let future predictions set today's threshold.
- **Model target = classification** (direction probability), not regression.
- **`pandas-ta-classic`** replaces `pandas-ta` (original was pulled from PyPI in 2024;
  fork has the same API, import name `pandas_ta_classic`).
- **No hyperparameter tuning in step 1** (sensible LightGBM defaults; optuna later,
  only after the walk-forward harness is trusted).
- joblib for artifacts now; MLflow deferred to step 3+ (multi-asset retraining).

## Data quality (audited 2026-07-27, Dukascopy XAUUSD)

Verified good, so price-data quality can be ruled out when explaining results:
weekend/session boundaries correct (401 breaks over 7.6y ≈ 52.8/yr; partial Friday
and Sunday sessions as expected), 1 frozen bar in 44,734 (0.002%), zero OHLC
integrity violations, and yearly ranges match documented gold history exactly
(2020 peak $2,070, Sept 2022 low $1,617, Oct 2024 high $2,788). Extreme moves are
real events, not bad ticks — the largest include the 8 Aug 2021 thin-liquidity
flash crash and the March 2020 COVID liquidation. An independent feed (Yahoo
COMEX futures `GC=F`) agrees on price level to within 0.06%.

Caveats: ~1,556 intraday gaps of 1–6h (holidays/thin liquidity), and the series is
bid, not mid — trading costs are modeled separately via `backtest.fee_pct`.

## Results so far — READ BEFORE PROPOSING "IMPROVEMENTS"

### Run 1 (H1, long_threshold label, technicals only) — apparent signal was an artifact

OOF AUC 0.5243 (LightGBM) vs 0.5382 (logistic baseline), backtest −27.8% against
buy-and-hold +139.9%. Diagnosis: **the model was predicting volatility, not
direction.** Its probabilities correlated −0.001 with forward return and +0.044
with forward volatility; sorting bars by confidence, the *highest*-confidence
decile had the worst forward return. The `long_threshold` label sets its bar from
ATR at time *t*, so it is easier to clear when volatility subsequently rises — and
volatility is forecastable while direction is not.

The underlying asymmetry, measured on 44k H1 bars:

| Best single feature correlates with... | |
|---|---|
| forward **direction** (return) | 0.048 (0.024 by rank) |
| forward **volatility** | **0.623** |

### Run 2 (H4, symmetric triple barrier, technicals + macro) — honest answer: no signal

Label rework and FRED macro features landed together. Class balance moved to 52.2%
(symmetric, as intended — the vol channel is gone).

| | LightGBM | Logistic baseline |
|---|---|---|
| Pooled OOF AUC | 0.4927 | 0.4849 |
| Backtest | −10.4%, PF 0.82, 111 trades | — |

Macro features dominated the importance table (`dollar_broad_z` first,
`real_yield_10y_z` fourth) but produced no predictive power. A horizon sweep on
daily bars — testing whether macro simply operates over longer horizons — found
nothing anywhere:

| horizon (days) | technical only | macro only | both |
|---|---|---|---|
| 3 | 0.4893 | 0.4813 | 0.4710 |
| 5 | 0.5007 | 0.4896 | 0.4751 |
| 10 | 0.4935 | 0.4735 | 0.4597 |
| 20 | 0.4764 | 0.4689 | 0.4588 |
| 40 | 0.4662 | 0.4455 | 0.4526 |

(SE on AUC at ~2,100 daily bars is roughly 0.013; treat 0.48–0.52 as no signal.)

**Conclusion: no directional edge for gold from price technicals or basic macro,
at any horizon from 3 to 40 days.** The pipeline is working correctly — it is
reporting an honest negative rather than manufacturing a signal. Consistently
sub-0.50 AUC also hints the fitted relationships invert out-of-sample, i.e. they
are regime-specific rather than stable.

### What this implies

- **Do not** respond to this by adding model capacity (XGBoost, neural nets) or
  hyperparameter tuning. The logistic baseline matched LightGBM in both runs;
  the constraint is information, not capacity.
- **Volatility is the thing that is genuinely predictable here** (0.62). It is
  useless as a long-only direction signal but valuable for position sizing,
  regime filtering, and the correlation-aware allocation layer in step 4.
- **The rotation design may work even though single-asset direction does not.**
  Ranking assets against each other is a *relative* (cross-sectional) problem,
  and cross-sectional momentum is far better documented than time-series
  direction prediction. Steps 3–4 test a different hypothesis than step 1 did.
- **The sentiment layer (step 2) is now the main open hypothesis** for
  differentiated information, which is what the README always argued.

## Implementation ideas for future steps (not yet built)

- **Step 1 refinements (in rough priority order given the negative result):**
  a held-out final period never looked at until the template is frozen (the OOF
  window has now informed design decisions, so it is no longer virgin);
  a transferability check running the identical config on silver;
  richer macro (CFTC positioning, ETF flows, gold/silver and gold/copper ratios);
  meta-labeling — use the predictable volatility to decide *whether* to trade
  rather than *which way*; probability calibration (isotonic) before thresholding;
  sample weighting by label uniqueness; hand-rolled Kalman-filter trend feature
  (user wants to build this one personally for the learning value).
  Optuna stays deferred — tuning cannot create information that isn't present.
- **Step 2 (sentiment):** GDELT GKG filtered per asset (gold: central-bank/monetary
  themes; PGMs: South Africa/Russia supply + auto demand). Start lexicon-based
  scoring, upgrade to transformer only if the lexicon baseline shows signal.
  Composite = weighted sum of ML prob + sentiment z-score, weights validated
  walk-forward too.
- **Step 3:** `config/<asset>.yaml` per asset + a thin loop; per-asset feature
  additions (e.g. gold/silver ratio for silver, auto-sector proxy for PGMs).
- **Step 4:** rank by composite score with hysteresis band (rotate only if
  challenger beats incumbent by δ for k consecutive bars); explicit cash outcome
  when no score clears the entry bar; correlation-aware sizing via rolling
  correlation matrix (cap combined gold+silver exposure); turnover as a
  first-class evaluation metric.

## Conventions

- All timestamps UTC, DataFrames indexed by candle open time.
- Config-driven: no magic numbers in modules; everything tunable lives in
  `config/gold.yaml` and flows through `config.py` dataclasses.
- Artifacts (`data/`, `models/`, `reports/`) never committed.
- `.env` holds `OANDA_API_KEY` / `OANDA_ENV` (practice|live); never committed,
  `.env.example` documents the shape.
