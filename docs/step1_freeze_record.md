# Step 1 — freeze record and holdout pre-registration

**Written 2026-07-28, before the holdout has been read.** Frozen at commit
`69f0527`.

This document exists because a backtest is only evidence if you can say what
else you tried. Development results are not out-of-sample with respect to design
decisions — the label, timeframe and feature set were all chosen *after* reading
earlier results. The holdout at `data.holdout_start: 2025-07-01` is the one slice
that has never informed a decision, and it can be spent once. Whatever it says
is only interpretable against a prediction made in advance, which is what
sections 5–7 are.

Asset scope: **gold (XAUUSD)**. Silver is a transferability control, already run.
Platinum and palladium belong to step 3 and are out of scope here.

---

## 1. What is frozen

No change to any of the following is permitted before the holdout is spent.
Changing one and then spending the holdout makes the holdout development data.

| | value |
|---|---|
| commit | `69f0527` |
| timeframe | H4 |
| history | Dukascopy XAUUSD, 2019-03-27 → 2026-07-23 (11,716 bars) |
| development slice | 10,016 bars (85.5%), through 2025-06-30 |
| holdout slice | 1,700 bars (14.5%), 2025-07-01 → 2026-07-23 |
| label | `meta` — symmetric triple barrier, `horizon_bars: 12`, `atr_mult: 1.5`, `timeout: sign` |
| primary rule | `trend`, `ema_span: 50` (close > EMA50) |
| features | technical + macro (FRED) + crossasset (gold/silver ratio) |
| model | LightGBM, library defaults, no tuning |
| sample weighting | `uniqueness` |
| calibration | `isotonic`, fitted per fold inside training data |
| CV | `TimeSeriesSplit`, 5 splits, `gap_bars: 12` |
| entry rule | expanding-window quantile, `entry_quantile: 0.90` |
| costs | `fee_pct: 0.0001` per side |

Environment: Python 3.13.14, lightgbm 4.7.0, scikit-learn 1.9.0, pandas 2.3.3,
numpy 2.4.6.

**Reproducibility verified 2026-07-28.** `scripts/train.py` run twice on the same
data produced bit-identical OOF predictions (`proba` and `proba_calibrated`, max
absolute difference 0.0 across 4,845 rows). Models are seeded at
`models.RANDOM_STATE = 42`. Caveat: this was checked on one machine; LightGBM can
differ across thread counts, so a reproduction on different hardware should
compare OOF predictions rather than assume them. A freeze that cannot be
reproduced is not a freeze.

**What the freeze hash covers, and what may change after it.** `69f0527` pins the
*modelling* code: config, label construction, feature assembly, model, primary
rule and entry rule. Tests, guards, docs and refactors that do not alter
predictions are still permitted afterwards — but each must be verified, not
assumed, by re-running `scripts/train.py` and confirming the OOF predictions are
bit-identical. The changes made on 2026-07-28 after `69f0527` (the dataset
contract guard, the extracted decision module, the integration tests) were
verified this way: pooled OOF AUC 0.5291 before and after, and `predict.py`
returned an identical score of 0.7591 against the same 0.8494 threshold. If a
change ever *does* move the numbers, it is not a refactor and the freeze is
broken.

**Pipeline coverage.** 111 tests, all offline on synthetic data. Added at freeze
time: `tests/test_pipeline_integration.py` covers the seams between the four
scripts — previously every test was unit-level, so a column rename could keep the
whole suite green while changing what the model sees — and
`tests/test_predict_decision.py` covers the artifact round-trip and the live
decision rule. The dataset contract now lives in one place,
`commodity_ml/dataset.py`, and refuses a dataset that does not match the current
code rather than treating unrecognised columns as features.

---

## 2. Ledger of everything tried before the freeze

This is the multiple-comparisons record. Each row is a variant whose result was
read on development data. A p-value from the best of these is not a p-value.

| dimension | variants tried |
|---|---|
| timeframe | H1, H4, D |
| label mode | `long_threshold`, `updown`, `triple_barrier`, `meta` |
| label horizon (daily sweep) | 3, 5, 10, 20, 40 days |
| feature families | technical only, macro only, both, both + crossasset |
| model | LightGBM, LogisticRegression baseline |
| sample weighting | none, uniqueness |
| calibration | none, isotonic, sigmoid |
| entry quantile (reported sweep) | 0.70, 0.80, 0.90, 0.95, 0.99 |
| assets | gold, silver |

Roughly 20–30 meaningfully distinct configurations were evaluated. Under a crude
Bonferroni correction at 25 looks, a nominal α of 0.05 becomes 0.002. Nothing
below has reached that.

Deliberately **not** tried, and still off the table: XGBoost or neural nets,
hyperparameter tuning, optuna, CFTC positioning, ETF flows, gold/copper. The
constraint identified in runs 1–2 was information, not model capacity, and every
additional variant weakens the holdout further.

*(Update 2026-07-28: step 2 reopened CFTC positioning and news sentiment under
its own pre-registered program with a separate append-only ledger —
`docs/step2_design.md` §8. Both ledgers count against the same single holdout;
whenever it is spent, interpret against their union.)*

---

## 3. Development results (gold, as frozen)

Walk-forward OOF over 4,845 scored bars, base rate 0.5197.

**Model quality**

| metric | value |
|---|---|
| pooled OOF AUC (raw scores) | 0.5291 |
| logistic baseline AUC | 0.4820 |
| edge over baseline | +0.047 |
| Brier (calibrated) | 0.2603 |
| Brier (raw) / no-skill | 0.3035 / 0.2496 |
| effective training sample, last fold | 3,595 of 4,836 rows (74%) |

**Backtest, 5.37 years**

| variant | trades | return | Sharpe | PF | win% | maxDD |
|---|---|---|---|---|---|---|
| trend rule alone | 466 | +15.5% | 0.34 | 1.08 | 52.1% | −10.6% |
| trend + ML filter | 120 | +22.3% | 0.91 | 1.54 | 58.3% | −7.1% |
| buy & hold | – | +108.5% | 1.01 | – | – | −21.7% |

The filter keeps a quarter of the trades and improves every trade-level metric.
It loses heavily to buy-and-hold, which remains the benchmark that matters for a
long-only metals strategy.

---

## 4. What the evidence actually supports (corrected 2026-07-28)

Three corrections to how runs 3–4 were written up. All three make the result
weaker, and all three were found by interrogating the existing gold result rather
than by adding data.

**4.1 The entry-quantile sweep is not a robustness test.** It was cited in run 3
as evidence the effect was not knife-edge ("PF degrades smoothly: 1.33/1.33/1.31").
That reading is invalid. `debounce_entries` keeps the *first* qualifying bar in
each 12-bar window, not the strongest — correct for live trading, since bar *t*
cannot know that bar *t+3* will score higher, but it means changing the threshold
changes *which* trades are taken, not merely how many:

```
q=0.90: 475 bars qualify -> 103 kept.  mean score qualifying 0.8987, kept 0.8909
q=0.80: 921 bars qualify -> 158 kept.  mean score qualifying 0.8515, kept 0.8350
```

Kept bars score lower than the pool they are drawn from. The q=0.90 and q=0.80
trade sets overlap on only **53 entries**; 67 of the 120 trades at q=0.90 are
absent at q=0.80. The sweep therefore varies selectivity and cluster-entry timing
simultaneously and is evidence in neither direction. It stays in the output as a
diagnostic, correctly labelled.

**4.2 The score does rank trades, consistently across folds.** Measured within
each fold, so pooling cannot manufacture it:

| fold | top-decile win | fold base rate | lift |
|---|---|---|---|
| 0 | 78.4% | 49.5% | +28.8pp |
| 1 | 56.7% | 51.3% | +5.4pp |
| 2 | 52.6% | 47.8% | +4.8pp |
| 3 | 63.9% | 56.7% | +7.3pp |
| 4 | 65.0% | 54.6% | +10.4pp |

Five of five positive, mean +11.3pp. As a sign test alone, p = 0.031.

**4.3 Label overlap removes most of the significance.** The 485 top-decile bars
are not 485 independent observations. Confident bars cluster in time and each
label looks 12 bars forward, so mean average-uniqueness among them is **0.178** —
each confident label shares its future with about five others.

| | n | z | one-sided p |
|---|---|---|---|
| naive | 485 | +4.27 | 0.00002 |
| overlap-corrected | 86 | +1.80 | 0.036 |

At trade level, the number that actually gets traded: 58.3% vs the bare rule's
52.1% on 120 trades gives z = +1.36, **p = 0.088**.

**Verdict.** The gold meta-label is not nothing — the direction of the effect is
consistent across all five folds. But its best framing is p ≈ 0.036 on 86
effective observations, unadjusted for the ~25 configurations in section 2, and
its trade-level framing is p = 0.088. **It is not a validated edge.** It also did
not replicate on silver, where the filter removed profit relative to the bare rule.

---

## 5. Why the holdout is not being spent yet

Counting only (no outcomes read): the gold holdout contains 1,700 bars, 984 of
which the trend rule wants to be long, yielding **93 bare-rule trades** after
debouncing and, at the frozen 0.90 quantile, an expected **~24 filtered trades**.

| question | trades needed | trades available |
|---|---|---|
| confirm the observed 6.2pp win-rate lift (80% power, one-sided α=0.05) | ~401 | ~24 |

On 24 trades the filtered win rate would have to reach **68.9%** to clear p=0.05
against the bare rule. The observed development lift is 58.3%. The same shortfall
applies to every other framing: an AUC test on all 984 holdout signal bars has
roughly 177 effective observations after the overlap correction, which is still
short of what a 0.03 AUC edge requires.

**The holdout cannot confirm this hypothesis, and no gold-only change makes it
able to.** 5.4 years of development data at this trade frequency is what exists.
Spending it now would consume a single-use resource for a test whose only
possible informative outcome is a negative one.

---

## 6. Pre-registered protocol, for whenever it is spent

Fixed now so the procedure cannot be chosen to suit the answer.

1. Train the frozen configuration on the development slice only. Verify the
   artifact's `trained_through` precedes `2025-07-01`.
2. Score every holdout bar where the primary rule is long, using the raw model
   score (not the calibrated probability — entries rank, and ranking uses raw).
3. Continue the expanding-window entry threshold seeded from the development OOF
   distribution. Do not recompute a fresh quantile on the holdout.
4. Report, on the holdout window only: the ML-filtered strategy, the bare trend
   rule, and buy-and-hold — same barriers, fees, debouncing and window.
5. Report trade count alongside every metric, and the overlap-corrected effective
   sample size.
6. Record the spend in this document with the date and commit.

---

## 7. Pre-registered predictions and decision rules

Written before seeing anything. Point predictions for the filtered strategy on
the gold holdout:

| quantity | prediction |
|---|---|
| filtered trades | 20–28 |
| filtered win rate | 52–56%, i.e. close to the bare rule |
| filtered vs bare-rule PF | filtered higher, but within noise |
| filtered vs buy-and-hold | buy-and-hold wins |
| statistical significance | none; the test is underpowered by ~16× |

Decision rules, fixed in advance:

- **Refuted** — filtered profit factor below 1.0, *or* below the bare rule's on
  the same window. This is the one question the sample size can answer, and a
  clear negative here retires meta-labeling for gold.
- **Not refuted, not confirmed** — anything between. The expected outcome. Treat
  the hypothesis as unresolved and do not build on it.
- **Confirmed** — filtered win rate ≥ 68.9% on ≥ 20 trades *and* profit factor
  above the bare rule's. Deliberately a high bar, because at this sample size
  nothing less is distinguishable from luck. I do not expect this.

A result in the middle band must not be written up as support. That is the
specific failure this document exists to prevent.

---

## 8. Rules once it is spent

- No tuning of anything, on any asset, afterwards. Not the entry quantile, not
  the primary rule, not the feature set.
- The holdout becomes development data the moment it is read. Any later
  out-of-sample claim needs a genuinely new period or a new asset.
- If the outcome is "not refuted, not confirmed", the honest report is that step 1
  produced no confirmed directional or regime-filter edge for gold, and the open
  hypotheses move to step 2 (sentiment) and step 4 (cross-sectional ranking) as
  the README always argued.

---

## 9. Spend ledger

| date | commit | asset | outcome |
|---|---|---|---|
| — | — | — | *not yet spent* |
