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

1. **[IN PROGRESS] Price-only ML pipeline for gold (XAU_USD)** — this repo's current focus
2. Sentiment/alt-data layer for gold only + composite scoring
3. Generalize to silver → platinum → palladium (one at a time)
4. Cross-asset ranking, allocation, correlation-aware sizing
5. Paper-trade full system; live capital only after pre-registered success criteria are met

## Repo layout

```
config/gold.yaml          # per-asset config; adding an asset later = new yaml
src/commodity_ml/         # installed package (pip install -e .)
  config.py               # yaml -> dataclasses
  data/oanda.py           # paginated OANDA candle download
  features/technical.py   # pandas-ta-classic indicators + hand-rolled features
  labels.py               # label construction — the module most prone to lookahead bugs
  modeling/models.py      # LightGBM factory + LogisticRegression baseline
  modeling/walkforward.py # TimeSeriesSplit walk-forward harness, OOF predictions
  evaluation/ml_metrics.py       # AUC, Brier, per-fold tables
  evaluation/trading_metrics.py  # Sharpe, profit factor, max drawdown (pure pandas, no vbt dep)
  backtest/engine.py      # vectorbt: OOF probabilities -> simulated long-only trades
scripts/                  # CLI entry points, run in order:
  fetch_data.py -> build_dataset.py -> train.py -> backtest.py
data/raw, data/processed, models/, reports/   # all gitignored artifacts
tests/                    # synthetic-data tests, no network needed
```

## Commands

```bash
source .venv/bin/activate            # Python 3.13 venv (3.14 too new for numba)
pip install -r requirements.txt && pip install -e .
pytest                               # all tests run offline on synthetic data

python scripts/fetch_data.py    --config config/gold.yaml   # needs OANDA_API_KEY in .env
python scripts/build_dataset.py --config config/gold.yaml
python scripts/train.py         --config config/gold.yaml [--baseline]
python scripts/backtest.py      --config config/gold.yaml
```

## Hard rules (from README methodology — apply to every component)

- **Walk-forward, time-ordered validation only.** Never random splits, never shuffle.
  `TimeSeriesSplit(gap >= label horizon)` so overlapping forward-return labels can't
  leak across the train/test boundary.
- **No lookahead anywhere.** Features at bar *t* may use only data through *t*;
  labels look strictly forward. `tests/test_features.py` has a causality test
  (features on truncated vs full history must match) — keep it passing, extend it
  for every new feature.
- **Backtests use out-of-fold predictions only** — never predictions from a model
  that saw that bar in training.
- **Min sample before conclusions:** 100+ trades or a multi-month window, whichever
  is later. Confirm on a further out-of-sample period before promoting anything.
- Report portfolio-quality metrics (Sharpe, PF, maxDD, turnover, time-in-cash), not
  just ML metrics, and check across volatility regimes.

## Engineering decisions made so far (provisional — confirm with user before changing)

- **Timeframe: H1 candles** (day-trading cadence per README; ~40k bars since 2019 is
  a healthy tabular-ML sample). Open item in README — user hasn't re-confirmed.
- **Label: long-only threshold** — did close rise more than `atr_mult × ATR(14)` over
  the next `horizon_bars` (default 12)? Symmetric up/down mode also implemented
  (`label.mode: updown`). Short side deferred.
- **Model target = classification** (direction probability), not regression.
- **`pandas-ta-classic`** replaces `pandas-ta` (original was pulled from PyPI in 2024;
  fork has the same API, import name `pandas_ta_classic`).
- **No hyperparameter tuning in step 1** (sensible LightGBM defaults; optuna later,
  only after the walk-forward harness is trusted).
- joblib for artifacts now; MLflow deferred to step 3+ (multi-asset retraining).
- OANDA instrument names: `XAU_USD`, `XAG_USD`, `XPT_USD`, `XPD_USD` — OANDA does
  list platinum/palladium CFDs, but data quality/history for XPT/XPD still needs
  checking before finalizing the universe (README open item).

## Implementation ideas for future steps (not yet built)

- **Step 1 refinements:** triple-barrier labels (TP/SL/time race) instead of fixed
  horizon; hand-rolled Kalman-filter trend feature (user wants to build this one
  personally for learning value); optuna walk-forward-aware tuning; probability
  calibration (isotonic) before thresholding.
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
