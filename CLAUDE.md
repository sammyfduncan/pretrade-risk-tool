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

1. **[TEMPLATE FROZEN, HOLDOUT UNSPENT — no confirmed edge] Price-only ML pipeline for gold**
   Pipeline validated, no directional signal found. Meta-labeling is a lead, not
   a result (run 5). Frozen at `69f0527`; see `docs/step1_freeze_record.md`.
2. **[IN PROGRESS since 2026-07-28]** Sentiment/alt-data layer for gold only +
   composite scoring — CFTC COT positioning + GDELT news tone, pre-registered
   in `docs/step2_design.md` (read it before running or proposing anything)
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
config/silver.yaml        # transferability control — differs in 3 lines, do not tune
src/commodity_ml/         # installed package (pip install -e .)
  config.py               # yaml -> dataclasses; also the development/holdout fence
  dataset.py              # the build_dataset -> train column contract; refuses a stale
                          # dataset rather than letting exit_bars/exit_time become features
  decision.py             # live trade/no-trade rule (rule long AND score >= threshold)
  primary.py              # direction rule for meta-labeling (model only filters it)
  data/schema.py          # canonical candle contract + per-source granularity maps
  data/dukascopy.py       # bulk free history -> training data
  data/ig.py              # broker API: epics, recent bars, allowance guard, execution later
  features/pipeline.py    # THE shared feature path — training and live both call it
  features/technical.py   # pandas-ta-classic indicators + hand-rolled features
  features/macro.py       # FRED real yields/dollar/VIX; publication lags = lookahead risk
  features/crossasset.py  # gold/silver ratio; needs the partner asset's candles on disk
  labels.py               # label construction — the module most prone to lookahead bugs
  modeling/models.py      # LightGBM factory + LogisticRegression baseline
  modeling/walkforward.py # TimeSeriesSplit walk-forward harness, OOF predictions
  modeling/sample_weights.py  # label-uniqueness weights (overlapping labels aren't independent)
  modeling/calibration.py     # per-fold isotonic/Platt + the CalibratedModel artifact
  evaluation/ml_metrics.py       # AUC, Brier, per-fold tables
  evaluation/trading_metrics.py  # Sharpe, profit factor, max drawdown (pure pandas, no vbt dep)
  backtest/engine.py      # vectorbt: OOF probabilities -> simulated long-only trades
scripts/                  # CLI entry points, run in order:
  fetch_data.py -> build_dataset.py -> train.py -> backtest.py
  predict.py              # live decision path; groundwork, order placement not wired up
  find_epic.py, compare_sources.py   # IG setup + cross-source validation
docs/step1_freeze_record.md   # frozen template + holdout pre-registration (COMMITTED —
                          # the ledger of what was tried is what makes the holdout
                          # interpretable; read before spending it)
data/raw, data/processed, models/, reports/   # all gitignored artifacts
tests/                    # synthetic-data tests, no network or credentials needed
  test_pipeline_integration.py  # the seams between the four scripts — everything else
                          # is unit-level, so this is what catches a contract drifting
```

## Commands

```bash
source .venv/bin/activate            # Python 3.13 venv (3.14 too new for numba)
pip install -r requirements.txt && pip install -e .
pytest                               # all tests run offline on synthetic data

python scripts/fetch_data.py    --config config/gold.yaml   # Dukascopy, no credentials
python scripts/fetch_data.py    --config config/silver.yaml # both legs: the ratio needs them
python scripts/build_dataset.py --config config/gold.yaml
python scripts/train.py         --config config/gold.yaml [--baseline]
python scripts/backtest.py      --config config/gold.yaml   # prints the bare-rule control too
python scripts/predict.py       --config config/gold.yaml   # latest bar -> trade/no-trade

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
- **The holdout (`data.holdout_start`) is single-use.** Development results are
  not out-of-sample with respect to design decisions, because those decisions
  were made after reading them. Do not spend the holdout until the template is
  frozen, and do not tune anything after spending it.
- **Any strategy that filters a rule must be compared against the bare rule.**
  A filter that doesn't beat its own primary signal is decoration.
- **Backtests use out-of-fold predictions only** — never predictions from a model
  that saw that bar in training.
- **Min sample before conclusions:** 100+ trades or a multi-month window, whichever
  is later. Confirm on a further out-of-sample period before promoting anything.
- **Significance is computed on effective sample size, never row count.**
  Overlapping labels are near-duplicates and confident predictions cluster in
  time, so deflate by mean average-uniqueness (`modeling/sample_weights.py`)
  before any p-value or standard error. Run 5: 485 top-decile bars were 86
  independent observations and the naive p-value was 1,800× too small.
- **Do a power calculation before spending a scarce evaluation window.** Ask how
  many trades the observed effect would need, and how many the window supplies,
  *before* running it. Gold's holdout supplies ~24 against ~401 needed.
- **State the multiple-comparisons ledger with any p-value.** The count of
  configurations tried lives in `docs/step1_freeze_record.md` §2; a result quoted
  without it is not interpretable.
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

### Run 3 (28 Jul 2026): holdout frozen, silver transfer test, meta-labeling

**Holdout.** `data.holdout_start: 2025-07-01` is now fenced off in both configs.
Every script defaults to the development slice; `--use-holdout` is required to
touch it and prints a warning. It is a single-use resource — spend it once, when
the template is frozen. Runs below are development-only, so they are NOT
out-of-sample with respect to design choices.

**Silver transfers.** `config/silver.yaml` differs from gold in exactly three
lines (asset, symbol, instrument). Run untouched, it reproduces gold's pattern:

| | gold | silver |
|---|---|---|
| direction AUC (pre-meta) | 0.4965 | 0.4903 |
| best feature vs direction | 0.0585 | 0.0888 |
| best feature vs volatility | 0.5428 | 0.5070 |

So the negative result is **structural across precious metals**, not a gold
quirk, and the pipeline is a genuine template. Dukascopy also carries platinum
and palladium (`XPT.CMD/USD`, `XPD.CMD/USD`) — resolves a README open item.

**Meta-labeling** (`label.mode: meta`, `primary.rule: trend`). A trend rule picks
the side; the model only decides whether to obey it. First result since run 1
that isn't explicable as an artifact — and the first time LightGBM beats the
logistic baseline (+0.030 AUC on both assets), consistent with the filtering
question being genuinely non-linear.

The control that matters is the bare rule with no ML filter:

| asset | variant | trades | return | Sharpe | PF | win% |
|---|---|---|---|---|---|---|
| gold | trend rule alone | 428 | +16.6% | 0.37 | 1.09 | 51.9% |
| gold | trend + ML filter | 137 | +15.7% | **0.62** | **1.31** | **59.1%** |
| gold | buy & hold | – | +110.1% | 0.95 | – | – |
| silver | trend rule alone | 399 | +8.7% | 0.18 | 1.03 | 51.1% |
| silver | trend + ML filter | 129 | −6.2% | −0.07 | 0.94 | 49.6% |
| silver | buy & hold | – | +153.4% | 0.73 | – | – |

Read this carefully before building on it:
- On gold the filter behaves exactly as a working meta-label should — removes
  68% of trades, keeps the same profit, lifts PF 1.09→1.31 and win rate by 7pp.
  ~~Profit factor also degrades smoothly across neighbouring entry quantiles
  (1.33/1.33/1.31 at 0.70/0.80/0.90), not knife-edge.~~ **RETRACTED in run 5 —
  the sweep varies cluster-entry timing as well as selectivity, so smoothness in
  it is not evidence of robustness. See run 5.**
- **On silver it makes things worse.** One-of-two replication on two highly
  correlated assets is exactly the pattern noise plus multiple comparisons
  produces. This is not yet an established effect.
- **Both still lose heavily to buy-and-hold**, which is the benchmark that
  matters for a long-only metals strategy.
- Gold's AUC edge is ~1.1 SE once the 12-bar label overlap is accounted for.

**Scope note (corrected 28 Jul 2026).** An earlier version of this section named
platinum/palladium as "the next test" of meta-labeling. That is wrong for step 1
and was read as an instruction. **Work stays on gold until step 1 is validated on
gold**; silver is the transferability control and the scope goes no wider. PGMs
are step 3. When a gold result is weak or one-of-two, interrogate that result —
sample size, sensitivity, the bare-rule control — rather than adding assets,
which multiplies comparisons and defers the actual question.

### Run 4 (28 Jul 2026): honesty corrections + gold/silver ratio + live path

Three changes landed together, plus two design bugs they exposed.

**Gold/silver ratio** (`features/crossasset.py`, `features.crossasset: true`).
Three features: z-score of the ratio and its 30/120-bar log changes. Defined
canonically as XAU/XAG in *both* configs, so it describes the environment
rather than the asset. It ranks 3rd and 6th on split importance for gold.
**Feature work is now closed** — CFTC positioning and ETF flows stay unbuilt.
*(Step-1 scope rule: no features added to rescue a weak price-only result. Step 2
reopened feature work on 2026-07-28 under a separate pre-registered program,
`docs/step2_design.md`; the step-1 freeze and this rule's original meaning stand.)*

**Sample weighting by label uniqueness** (`modeling/sample_weights.py`,
`model.sample_weighting: uniqueness`). With a 12-bar horizon, consecutive
labels share 11/12 of their future, so the model was counting one move as up
to twelve independent facts. Kish effective sample size says the last fold's
4,836 rows are worth **3,595 independent observations (74%)**. Kept as a
correctness fix regardless of what it does to the numbers.

**Probability calibration** (`modeling/calibration.py`,
`model.calibration: isotonic`). Fitted per fold on a gapped tail of that
fold's *training* block. Raw Brier 0.3035 against a 0.2496 no-skill baseline;
isotonic closes 80% of the gap on gold, 54% on silver. Fold 0 is too small to
calibrate (needs 250 rows) and is left alone.

Two bugs this turned up, both of the kind that flatter results silently:

1. **Platt scaling inverted the model.** `LogisticRegression` on the score's
   logit is unconstrained, so on folds where the score/outcome relationship was
   negative it learned a negative slope and flipped the ranking — turning the
   logistic baseline's pooled AUC from 0.4820 into 0.5133. Calibrators are now
   probed for monotonicity and fall back to identity if they would invert.
   This dataset sets that trap deliberately: several folds genuinely score
   below 0.50.
2. **Calibrating changed which bars traded.** Entries rank on a *quantile* of
   the prediction distribution; isotonic is a step function that collapsed
   4,845 distinct predictions into ~1,050, moving the trade count 137 -> 71 for
   reasons unrelated to signal. Fixed structurally: `WalkForwardResult` now
   carries `oof_proba` (raw — ranking and entries) and `oof_calibrated`
   (Brier, and position sizing later). The OOF parquet stores both.
   "Calibration changes meaning, not behaviour" is now true by construction
   and pinned by a test.

**The bare-rule control is now printed on every meta-mode backtest**, not
behind a flag — the README rule is too easy to skip, and on these numbers it
is the only thing that distinguishes the two assets:

| asset | variant | trades | return | Sharpe | PF | win% |
|---|---|---|---|---|---|---|
| gold | trend rule alone | 466 | +15.5% | 0.34 | 1.08 | 52.1% |
| gold | trend + ML filter | 120 | **+22.3%** | **0.91** | **1.54** | **58.3%** |
| gold | buy & hold | – | +108.5% | 1.01 | – | – |
| silver | trend rule alone | 434 | **+36.5%** | 0.42 | 1.09 | 52.3% |
| silver | trend + ML filter | 125 | +20.8% | 0.43 | 1.20 | 53.6% |
| silver | buy & hold | – | +139.0% | 0.70 | – | – |

AUC (raw scores): gold 0.5291 vs baseline 0.4820 (+0.047); silver 0.5148 vs
0.4617 (+0.053). LightGBM's edge over the baseline **survives** the uniqueness
correction on both assets — it was +0.057/+0.052 unweighted.

Read the control column, not the absolute numbers:
- Both assets clear the 100-trade minimum (120, 125) for the first time.
- **Gold's filter still behaves like a working meta-label**: a quarter of the
  trades, more total return, PF 1.08 -> 1.54.
- **Silver's still does not.** Absolute return improved a lot versus run 3
  (−6.2% -> +20.8%), which looks like replication until you read the control:
  the bare rule made +36.5% over the same window, so the filter *removed
  profit* while leaving Sharpe flat (0.42 -> 0.43). One of two, again.
- **A caveat that did not exist in run 3:** gold and silver now share a feature
  computed from both series. Agreement between them is correspondingly weaker
  evidence than when the two configs were genuinely independent.
- Both still lose heavily to buy-and-hold.

**Live prediction path** (`scripts/predict.py`, `features/pipeline.py`).
Training and prediction now share one feature-assembly function, so train/live
skew cannot arise from two implementations drifting apart.
`tests/test_predict_parity.py` pins that a feature row built from a trailing
window equals the row training recorded for that bar, and that poisoning
future partner-asset data changes nothing. The artifact bundles model +
calibrator + feature order + entry threshold, so the live path cannot apply
one without the others. Runs end-to-end offline; only the candle-fetch line
changes when IG credentials arrive, and order placement is not wired up.

### Run 5 (28 Jul 2026): gold-only diagnosis + template frozen

No new data, no new features, no config change. This run interrogated the run-4
gold result and **weakened it three times over**. Written up in full, with the
holdout pre-registration, in `docs/step1_freeze_record.md` — read that before
touching the holdout.

**1. The entry-quantile sweep is not a robustness test — retract run 3's claim.**
`debounce_entries` keeps the *first* qualifying bar in each 12-bar window, not
the strongest. That is correct for live trading (bar *t* cannot know bar *t+3*
will score higher), but it means moving the threshold changes *which* trades are
taken, not just how many:

```
q=0.90: 475 bars qualify -> 103 kept.  mean score qualifying 0.8987, kept 0.8909
q=0.80: 921 bars qualify -> 158 kept.  mean score qualifying 0.8515, kept 0.8350
```

Kept bars score *lower* than the pool they came from. The q=0.90 and q=0.80 trade
sets overlap on only **53 entries** — 67 of the 120 trades at q=0.90 are absent
at q=0.80. So the run-4 sweep's ugly q=0.80 row (PF 0.87, below the bare rule's
1.08) is cluster-entry timing, not a signal cliff; and run 3's smooth sweep was
not evidence of robustness either. **The sweep is evidence in neither direction.**
It stays in the backtest output as a diagnostic, correctly labelled.

**2. The score does rank trades — consistently, measured within each fold** so
that pooling cannot manufacture it. Top-scoring decile vs that fold's own base
rate: +28.8, +5.4, +4.8, +7.3, +10.4 pp. Five of five positive, mean +11.3pp;
as a sign test on its own, p = 0.031.

**3. Label overlap then removes most of the significance.** The 485 top-decile
bars are not 485 independent observations — confident bars cluster in time and
each label looks 12 bars forward, giving a mean average-uniqueness of **0.178**
(each confident label shares its future with ~5 others).

| | n | z | one-sided p |
|---|---|---|---|
| naive | 485 | +4.27 | 0.00002 |
| overlap-corrected | 86 | +1.80 | 0.036 |

At trade level — the number actually traded — 58.3% vs the bare rule's 52.1% on
120 trades gives z = +1.36, **p = 0.088**.

**Verdict: not a validated edge.** The effect's direction is consistent across
all five folds, which is not nothing. But its best framing is p ≈ 0.036 on 86
effective observations, against ~25 configurations tried (ledger in the freeze
record), and its trade-level framing is p = 0.088. It also did not replicate on
silver.

**The holdout is not being spent, and the reason is arithmetic.** Counting bars
only, no outcomes read: the gold holdout holds 1,700 bars, 984 with the trend
rule long, giving **93 bare-rule trades** and **~24 filtered trades**. Confirming
the observed 6.2pp lift at 80% power needs **~401 trades**; on 24 trades the win
rate would have to reach 68.9% to clear p=0.05. No gold-only change fixes this —
5.4 years at this trade frequency is what exists. Spending it now would burn a
single-use resource on a test whose only informative outcome is a negative one.
The pre-registered protocol, predictions and decision rules are fixed in the
freeze record for whenever it *is* spent.

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
- **Meta-labeling is the one price-only idea still alive** (runs 3-5), because
  it asks the model a regime question rather than a direction question. But
  after run 5 it is **explicitly not validated**: p ≈ 0.036 on 86 effective
  observations at best, p = 0.088 at trade level, unreplicated on silver, and
  unadjusted for ~25 configurations. It is a lead, not a result. Do not build
  on it, cite it as working, or let its direction-consistency across folds
  stand in for significance.
- **Effective sample size, not row count, is the denominator.** Run 5's single
  most important number: 485 top-decile bars are 86 independent observations,
  because confident predictions cluster in time and each label looks 12 bars
  forward. Any significance claim on overlapping labels must be deflated by
  average uniqueness first — the naive p-value was 1,800× too small.
- **A sensitivity sweep is only a robustness test if it varies one thing.**
  The entry-quantile sweep also changes which bar in each cluster the debouncer
  takes, so both its smooth version (run 3) and its jagged version (run 4) were
  over-read. Check what else moves before treating a sweep as evidence.
- **Always read the bare-rule control before the return column.** Run 4 is the
  cautionary case: silver's return went from −6.2% to +20.8% and looked like a
  replication, while against its own control the filter was destroying profit.
- **Corrections that make numbers worse are working.** Sample weighting and
  calibration exist to remove flattery, not to add signal. Two separate
  mechanisms in run 4 quietly improved metrics by inverting or re-ranking
  predictions; both were bugs.

## Implementation ideas for future steps (not yet built)

- **Step 1 — what is left.** Done in runs 3-5: the holdout fence, the silver
  transferability check, meta-labeling, the gold/silver ratio, sample weighting
  by label uniqueness, probability calibration, the live prediction path, and
  (run 5) the diagnosis plus **the frozen template and holdout pre-registration
  in `docs/step1_freeze_record.md`**. The template is frozen at commit `69f0527`:
  changing the config, label, features, model or entry rule and *then* spending
  the holdout turns the holdout into development data.

  Still open, in order:
  1. **Spend the holdout, once — but not yet.** Run 5's power analysis says it
     yields ~24 filtered trades against the ~401 needed to confirm the observed
     effect, so it cannot answer the question it was reserved for. It can still
     *falsify* (a filtered PF below the bare rule's would retire the idea for
     gold). Protocol, point predictions and decision rules are pre-registered in
     the freeze record — follow them exactly, and do not re-derive the procedure
     after seeing the answer.
  2. Order placement + scheduling on the live path, once IG credentials exist.
     This is the only genuinely credential-blocked item.
  3. Hand-rolled Kalman-filter trend feature — **the user wants to build this
     one personally for the learning value**, so leave it to them.

  Deliberately NOT doing: more features (CFTC positioning, ETF flows,
  gold/copper — feature work closed in run 4), more model capacity, optuna, and
  **other assets** (platinum/palladium are step 3; adding an asset to chase a
  replication is scope creep and multiplies comparisons). Tuning cannot create
  information that isn't present, and every extra variant tried before the
  holdout is spent makes the holdout weaker evidence.
- **Step 2 (sentiment) — STARTED 2026-07-28, design fixed in
  `docs/step2_design.md`.** Two independent feature families for gold: CFTC COT
  positioning (Socrata `72hh-3qpy`, contract 088691, lag 7d, publication-
  interruption table) and GDELT DOC 2.0 tone/attention (daily timelines,
  pinned non-point-in-time snapshots, preview-validated query). Exactly six
  pre-registered features; matched-row ablations via `train.py
  --exclude-family`; retention gate = ΔAUC ≥ +0.015 AND ≥4/5 folds AND beating
  the max of 20 circular-shift placebos. RSS dropped (no archive → not
  backtestable); ETF flows deferred; transformer only if a lexicon-grade
  baseline shows signal. Composite scoring (weighted ML prob + sentiment z) is
  GATED on a family surviving — do not build it early. PGM query
  personalisation (South Africa/Russia supply + auto demand) waits for step 3.
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
