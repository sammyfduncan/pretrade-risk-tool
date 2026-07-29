# Step 2 — alt-data layer: pre-registration and design record

**Written 2026-07-28, before any COT or GDELT number has been computed on real
data.** No snapshot has been fetched; no IC, AUC or backtest involving these
sources has been read.

This document is to step 2 what `docs/step1_freeze_record.md` is to step 1: the
thing that makes results interpretable. Step 1's central lesson was that a
number is only evidence relative to what was specified *before* it was read —
the hypotheses, the exact features, the decision rules, and the count of looks.
All of those are fixed here first.

Asset scope: **gold (XAUUSD)** only. Per-asset personalisation of queries and
contract codes is deliberately config-driven so silver/PGMs later are
yaml-only changes — but nothing beyond gold is fetched, built or evaluated in
step 2.

---

## 0. Status and freeze linkage

- Step 1 is frozen at commit `69f0527`; the holdout (`data.holdout_start:
  2025-07-01`) is **unspent** and stays fenced throughout step 2. All step-2
  work uses the development slice only.
- The step-1 frozen baseline: pooled OOF AUC **0.5291** (LightGBM, meta mode),
  logistic baseline 0.4820, backtest 120 trades / PF 1.54 — not a validated
  edge (freeze record §4).
- **Supersession note.** CLAUDE.md's run-4 statement "feature work is now
  closed — CFTC positioning and ETF flows stay unbuilt" was a *step-1 scope
  rule*: it forbade adding features to rescue a weak price-only result. This
  document opens a new, pre-registered feature program. The step-1 freeze
  stands untouched: the new feature flags default to **false**, and the frozen
  configuration remains reproducible.
- **One holdout, two ledgers.** Step 2 spends comparisons against the *same*
  single holdout as step 1. Every look at development data taken here is
  recorded in §8; whenever the holdout is eventually spent, it must be
  interpreted against the union of the step-1 ledger (freeze record §2) and
  §8 of this document.
- **Freeze-verification protocol for step-2 code changes** (the FRED hole):
  "re-run and diff" only works against a fixed input. `train.py` bit-identity
  is checked against the **existing** dataset parquet. `build_dataset.py`
  changes are checked by running the old and new code back-to-back (minutes
  apart) and diffing frames — FRED revises published values, so if only macro
  columns differ, re-run the *old* code twice to attribute the drift to FRED
  rather than to the change.

| freeze check | when | result |
|---|---|---|
| train.py bit-identical OOF on existing dataset, flags off | Phase 1 | *pending* |
| back-to-back build_dataset comparison, flags off | Phase 4 | *pending* |

---

## 1. Hypotheses

Both are stated with an expected direction so they are falsifiable, but neither
direction is strongly held — the falsifiable core is "information content
distinguishable from zero, with a consistent sign".

**H-COT (positioning / crowdedness — exploratory).** Managed-money net
positioning in COMEX gold futures, at extremes relative to its own two-year
history, is informative about forward returns and/or forward volatility at
5–20 day horizons. Expected direction: *contrarian* — stretched net-long
(high `cot_mm_net_z`) precedes weaker forward returns, because a crowded trade
has fewer marginal buyers left. Producer/merchant positioning is approximately
the mirror image (they take the other side). Explicitly **exploratory**: §7
shows the weekly sample cannot confirm realistic effect sizes; the honest
deliverables are the sign, stability, and an effect-size estimate — not a
significance claim.

**H-GDELT (news tone / attention).** Shifts in the tone of, and attention to,
gold-relevant news (monetary policy, central banks, gold-market coverage) lead
gold returns at 5–20 day horizons. Expected direction: *safe-haven* — falling
tone (fear) and rising attention precede gold strength. Attention
(`gdelt_vol_smooth7_z`) is additionally expected to predict forward
*volatility* regardless of direction — consistent with step 1's finding that
volatility is the predictable quantity here (best feature vs forward vol:
0.62). A result where GDELT features predict vol but not direction is a
*useful positive* for the step-4 sizing/allocation layer, not a failure.

What would retire each hypothesis is defined in §6 (Tier-2 gate); Tier 1 (§5)
informs interpretation but gates nothing.

---

## 2. Data sources and point-in-time caveats

### 2.1 CFTC Commitments of Traders (positioning)

- **Source:** CFTC Public Reporting Socrata API, dataset `72hh-3qpy`
  (Disaggregated, **Futures Only**), weekly since June 2006. No API key.
  Gold: `cftc_contract_market_code = 088691`
  ("GOLD - COMMODITY EXCHANGE INC."). Verified live 2026-07-28 (current
  through the 2026-07-21 report).
- **Release arithmetic → publication lag 7 days.** Reports are *dated* Tuesday
  (positions as of Tuesday close) and *released* Friday ~15:30 ET. Alignment
  shifts by whole days on the observation index, so: lag 3 = Friday 00:00 UTC
  — leaks all of Friday's bars before the afternoon release. Lag 4–5 survives
  normal weeks only because metals don't trade Saturday. Holiday weeks slip
  the release to Monday ~15:30 ET, so lag 6 = Monday 00:00 UTC leaks Monday's
  bars. **Lag 7 (next Tuesday 00:00 UTC) is the smallest whole-day lag safe in
  both cases.** House rule: overstating a lag weakens results, understating
  one manufactures them. The cost is negligible — COT features are step
  functions for a week regardless.
- **Publication interruptions.** A fixed lag assumes the release schedule
  held. It did not, inside the development window: the **ION Cleared
  Derivatives cyberattack (Feb 2023)** suspended COT publication for roughly
  six weeks, with reports dated ~2023-01-31 through ~2023-03-07 released on a
  catch-up schedule ending mid-March 2023. A fixed 7-day lag would hand the
  backtest those reports weeks before any trader had them. `features/cot.py`
  therefore carries a `PUBLICATION_INTERRUPTIONS` table —
  `(first_report_date, last_report_date, catch_up_completed)` — applied as
  `knowable = max(report_date + lag, catch_up_completed)`; combined with the
  15-day staleness guard this yields honest NaNs for ~4–5 weeks of Feb–Mar
  2023 (~130–160 H4 bars, ~1.4% of the 10,016 dev bars). Exact dates are to be
  verified against CFTC press releases in Phase 2 and pinned here:
  *[TO VERIFY — Phase 2]*. The **Oct/Nov 2025 government-shutdown gap falls in
  the holdout** — whenever the holdout is spent with COT enabled, the table
  must cover it first.
- **Extending to another asset (verified 2026-07-29, no code changes needed).**
  A new metal needs exactly two yaml lines: `features.cot: true` and its own
  `cot.contract_code`. Codes confirmed live against the API — gold `088691`,
  silver `084691`, platinum `076651`, palladium `075651`, all four current
  through the 2026-07-21 report on the same dataset. Proven end-to-end by
  fetching silver's history and building its features through the unmodified
  gold code path (1,050 reports; features well-scaled; identical 141-bar
  Feb-2023 gap, as expected for a CFTC-wide outage). `contract_code` has **no
  default**: a default would be one asset's code silently inherited by every
  other config, so a new asset that enables COT without setting it is refused
  rather than trained on the wrong market's positioning.
- **Corrections-in-place caveat (accepted).** CFTC occasionally amends
  published figures; the Socrata dataset reflects amended values, not the
  first print. Same class of problem as GDELT's non-point-in-time index but
  far smaller in magnitude; accepted and documented rather than engineered
  around.
- **Fetch details:** full history since 2006 in one request (~1,050 rows —
  warmup is never the constraint). Socrata's default `$limit` is **1,000**,
  which would *silently truncate* — the fetch passes an explicit limit,
  orders by report date, and asserts the row count is below the limit. Values
  arrive as JSON strings → explicit parsing, offline-tested. Field-name quirk
  pinned in code: `swap__positions_short_all` (double underscore, real).

### 2.2 GDELT DOC 2.0 (news tone / attention)

- **Source:** `https://api.gdeltproject.org/api/v2/doc/doc`, no key, full-text
  archive back to **2017-01-01** (verified 2026-07-28). Three timeline modes
  fetched into one snapshot: `timelinetone` (average tone of matching
  articles per day), `timelinevol` (matching volume as a share of *all* GDELT
  coverage — normalized for index growth; this is the feature input), and
  `timelinevolraw` (raw article counts — used only as a mask: a day with zero
  articles has *no* tone, and tone 0.0 is otherwise a valid value).
- **NOT point-in-time — the structural caveat.** The API queries *today's*
  index about the past. Article dedup, removals and index rebuilds mean a
  re-fetch months later returns slightly different aggregates than a live
  system would have seen. Mitigations: (a) daily tone/volume are aggregates
  over many articles, far less sensitive to index churn than any single
  article; (b) the snapshot is fetched **once** and pinned on disk with a
  sidecar recording query + fetch time — every evaluation reads the same
  snapshot; (c) a re-fetch is treated as a *new sample* and must be recorded
  in §11. There is no free way to do better; the residual risk is accepted
  and stated.
- **Query design is validated, not guessed.** "gold" is contaminated (Olympic
  medals, Gold Coast, golden retrievers). The final query is chosen by
  eyeballing `artlist` previews over fixed windows that *include* Olympic
  months (2021-07/08, 2022-02) plus one calm and one stress month, before any
  timeline data is fetched. The validated query is recorded verbatim in §4
  and in `config/gold.yaml`; the config loader refuses `features.gdelt: true`
  with an empty query.
- **Mechanics:** publication lag 1 day (a day's aggregate is knowable at the
  next UTC midnight; the current partial UTC day is dropped at fetch time);
  zero-article days masked to NaN then ffilled with a 3-day limit; 3-day
  staleness tolerance at alignment (a longer news drought becomes NaN
  features, not fossils); yearly fetch chunks with polite delays and retries.

---

## 3. Pre-registered features — exactly six

Windows follow existing repo conventions (`macro.py`: 252-day z with
min_periods 126; here 104 weeks ≈ 2 years with min_periods 52 for weekly
data). **They are conventions, not tuned values, and are fixed before any
evaluation.** No "maybe" features: two earlier candidates (a 3-year COT-index
percentile; `tone_chg_20d`) were dropped *at design time* as rank/speed
transforms of features already in the list — second looks at the same
hypothesis.

**COT** (computed on the weekly report index, then aligned):

| feature | formula |
|---|---|
| `cot_mm_net_z` | `mm_net_frac = (m_money_long − m_money_short) / open_interest`; z over 104 weekly obs, min_periods 52 |
| `cot_mm_net_chg_4w` | `mm_net_frac.diff(4)` |
| `cot_prod_net_z` | `prod_net_frac = (prod_merc_long − prod_merc_short) / open_interest`; same z |

Alignment: observation index + 7 days → interruption override (§2.1) →
`merge_asof` backward onto the H4 bar index, tolerance 15 days.

**GDELT** (computed on the daily calendar index, then aligned):

| feature | formula |
|---|---|
| `gdelt_tone_z` | daily tone (zero-article days NaN, ffill limit 3); z over 252 daily obs, min_periods 126 |
| `gdelt_tone_chg_5d` | `tone.diff(5)` on the ffilled series |
| `gdelt_vol_smooth7_z` | 7-day rolling mean of normalized volume (kills day-of-week news seasonality), then the same 252-day z |

Alignment: index + 1 day → `merge_asof` backward, tolerance 3 days.

Raw levels never reach the model, matching macro.py. All six features are
covered by the same test batteries as existing families: publication-lag pin,
truncation causality, warmup-NaNs, staleness-NaN, and live/train parity with
future-data poisoning.

---

## 4. GDELT query validation record

**Validated 2026-07-28, before any timeline data was fetched.** One iteration
was sufficient.

Final query, verbatim (matches `config/gold.yaml` and the snapshot sidecar):

```
("gold price" OR "gold prices" OR "price of gold" OR bullion OR "gold futures") sourcelang:english
```

Preview protocol: `fetch_altdata.py --preview` sampled 250 articles from each
of four fixed windows — Tokyo Olympics (2021-07-23→08-08), Beijing Olympics
(2022-02-04→02-20), COVID stress (2020-03), calm (2023-06) — chosen before any
preview was run, with the Olympic windows included deliberately as the known
contamination trap for the word "gold".

Result: **0 of 1,000 sampled titles** matched contamination terms (olympic,
medal, gold coast, golden globe, gold cup, golden retriever), and manual
reading of the samples found only gold-market coverage in all four windows,
including both Olympic ones. Phrase-matching is why: Olympic articles say
"gold medal", never "gold price".

Known characteristic, accepted: the sample skews toward high-frequency retail
price coverage (Indian/Pakistani "gold rate today" dailies). These are
formulaic and will give the tone series a stable baseline; the rolling z-score
absorbs the level, and narrative shifts still move the aggregates. No
domain filtering — that would be a tuning knob with no pre-registered basis.

Not pursued: a central-bank/monetary-theme variant (the README's original
sketch). The market-coverage query already captures central-bank gold stories
that mention price/bullion, previewed clean on the first attempt, and each
additional variant is design surface. Revisit only as a pre-registered
addendum if step 2 produces a reason to.

---

## 5. Tier-1 protocol — diagnostics, not gates

Purpose: understand what each feature actually measures before the pipeline
runs — step 1's run-1 lesson (the "signal" was volatility wearing a
direction costume) makes this non-optional. **Nothing is retained or retired
on Tier-1 numbers**; they inform interpretation of Tier 2.

For each of the six features, on the development slice only:

1. **Spearman IC vs forward log-return** at 5-day and 20-day horizons.
2. **Spearman IC vs forward realized volatility** over the same horizons.
3. Computed at the feature's **native frequency** — COT on weekly report
   dates (~327 obs), GDELT on calendar days (~2,300 obs) — never on ffilled
   H4 bars, which would fake the sample size (step-1's effective-n rule).
4. SEs corrected for overlap/persistence (Newey–West with lag = horizon;
   effective-n reported alongside).
5. **Stability split:** the IC on each half of the dev window, reported next
   to the full-window IC. Sign flips between halves are the regime-instability
   signature that step 1's sub-0.50 AUCs hinted at.

Every number read is a look and goes in §8.

---

## 6. Tier-2 protocol and the retention gate

### Matched-row ablations

Toggling a feature family in yaml changes which rows survive the dataset's
`dropna` (staleness/warmup NaNs — e.g. the Feb-2023 COT hole), which moves
`TimeSeriesSplit` fold boundaries, which contaminates any AUC delta with
sample composition. Therefore: **one dataset is built with both families on**,
and ablations run via `train.py --exclude-family {cot,gdelt}`, which filters
the feature list after the dataset contract check. Identical rows, identical
folds; deltas attributable to columns only. The flag is **evaluation-only**:
a retained family is expressed by setting yaml flags and retraining, so the
live path never depends on an exclusion flag. The matched control's AUC will
differ slightly from the frozen 0.5291 because the row set differs; the frozen
baseline is reported alongside for reference.

### Run matrix (8 primary runs)

{matched control, +cot, +gdelt, +both} × {`meta`, `triple_barrier`} — each
with the logistic baseline alongside, and (meta runs) the backtest with its
bare-rule control. Commands and outputs archived to `reports/step2/`; every
run is a row in §8.

### Retention gate — fixed now, before any result

A family is **retained** only if all three hold, on the meta-mode matched-row
comparison:

1. **ΔAUC ≥ +0.015** pooled OOF over the matched control;
2. **same-direction improvement in ≥ 4 of 5 folds**;
3. **placebo calibration:** the observed Δ exceeds the **maximum** Δ across
   20 placebo runs in which that family's source series (the weekly COT frame
   / daily GDELT frame) is circularly time-shifted by ≥ 1 year (wrapping)
   and pushed through the *identical* pipeline. Max-of-20 ≈ empirical
   p < 0.05, and the placebos inherit every artifact of the real pipeline —
   row set, folds, label overlap, autocorrelation — which analytic SEs do
   not. This is the same move as the bare-rule control: calibrate against
   the null *of this exact procedure*.

Condition 1 alone is a screen, not evidence: at these effective sample sizes
+0.015 AUC is ~1.2–1.9 SE. The gate's evidential content is condition 3.

**On failure** (the expected outcome): the family's yaml flag returns to
false, the snapshot stays on disk, the result is written to §9 and CLAUDE.md
as an honest negative. **On success:** yaml expresses the retained set, the
model is retrained, and Phase D (§10) opens with its own pre-registered
addendum.

**Interpretive notes fixed in advance.** (a) A `triple_barrier` (direction)
improvement without a `meta` improvement, or vice versa, is reported as what
it is — one of two, not a replication. (b) A vol-only result (Tier 1 shows
vol IC, Tier 2 shows no AUC gain) retires the family for *direction* but
records it as a candidate input for step-4 sizing. (c) Nothing in step 2
touches the meta-label hypothesis itself — it remains "a lead, not a result"
regardless of what happens here.

---

## 7. Power analysis — stated before evaluation

Dev window 2019-03-27 → 2025-06-30 ≈ 6.26 years.

| source | native obs | detectable |r| at 80% power, α=0.05 |
|---|---|---|
| COT | ~327 weekly | ≈ 0.155 |
| GDELT vs 5d horizon | ~2,290 daily, n_eff ≈ n/5 ≈ 460 | ≈ 0.13 |
| GDELT vs 20d horizon | n_eff ≈ n/20 ≈ 115 | ≈ 0.26 |

(2.8/√n rule; GDELT effective n deflated for overlapping forward windows —
persistent features like `gdelt_tone_z` deflate further, toward the 20d row.
Empirical Newey–West SEs in Tier 1 supersede these planning numbers.)

Realistic ICs in this domain are 0.02–0.05. **The honest statement, made in
advance: this development window cannot *confirm* an effect of realistic size
from either source.** What it can do: estimate sign and magnitude, check
stability, and — through the Tier-2 placebo gate, which aggregates
information across all six features and five folds — detect an effect large
enough to matter. A family that cannot clear a max-of-20-placebos bar on 6.3
years of data is not carrying tradeable information, whatever its p-value
against a naive null. Expected outcome, stated now: **honest negative on
direction for both families; GDELT-attention vs volatility is the likeliest
positive.**

---

## 8. Ledger of looks (append-only)

Every number read on real (non-synthetic) data is a row. A p-value quoted
without this table is not interpretable.

| # | date | action | numbers read | notes |
|---|---|---|---|---|
| 0 | 2026-07-28 | pre-registration written | none | this document, before any fetch |

---

## 9. Outcomes

*[Written after Tier 2, against the §6 gate. Nothing here yet.]*

---

## 10. Phase D — composite scoring (GATED)

Built **only** if a family survives §6. Sketch, not a spec: composite score =
`w · ML_probability + (1 − w) · sentiment_z`, with `w` validated walk-forward
(README Track B design). Requires its own pre-registered addendum before
implementation — including how the composite is compared against both of its
components alone (the composite's version of the bare-rule control). Nothing
is built now.

---

## 11. Snapshot / re-fetch ledger

Snapshots are pinned artifacts; a GDELT re-fetch is a new sample.

| date | snapshot | action | query/params | reason |
|---|---|---|---|---|
| 2026-07-28 | `XAUUSD_cot.parquet` | initial fetch | contract 088691, full history | 1,050 weekly reports, 2006-06-13 → 2026-07-21; median cadence 7d, 11 gaps > 7d (holidays + known interruptions) |
