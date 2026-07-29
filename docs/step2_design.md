# Step 2 — alt-data layer: pre-registration and design record

**Written 2026-07-28, before any COT or GDELT number has been computed on real
data.** No snapshot has been fetched; no IC, AUC or backtest involving these
sources has been read.

This document is to step 2 what `docs/step1_freeze_record.md` is to step 1: the
thing that makes results interpretable. Step 1's central lesson was that a
number is only evidence relative to what was specified *before* it was read —
the hypotheses, the exact features, the decision rules, and the count of looks.
All of those are fixed here first.

Asset scope: **gold (XAUUSD) and silver (XAGUSD)**. Gold is the hypothesis;
silver is the transferability control it was in step 1, run with identical
windows, lags, thresholds and gates. *(Scope amended 2026-07-29 — this
document originally said gold only. Silver's role, its per-asset expectations
and the joint decision rule were fixed in §1.1 and §6.1 before any silver
outcome was computed; the COT snapshot pinned that day is descriptive only,
§11.)* Per-asset personalisation is config-driven, so a metal costs two yaml
identifiers — its CFTC contract code and its GDELT query — and no code.
**PGMs remain step 3** and are not fetched, built or evaluated here, including
as a tie-break if gold and silver disagree (§6.1).

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
| train.py bit-identical OOF on existing dataset, flags off | Phase 1 (config plumbing) | **pass** — max\|diff\| 0.0, AUC 0.5291 |
| back-to-back build_dataset comparison, flags off | Phase 4 (wiring) | **pass** — max\|diff\| 0.0, AUC 0.5291; predict.py 0.7591 / thr 0.8494 / NO TRADE |
| train.py bit-identical OOF, after removing COTConfig's default code | 2026-07-29 | **pass** — max\|diff\| 0.0, AUC 0.5291 |
| train.py bit-identical OOF, after GDELT chunking/pacing + `_check_daily_resolution` | 2026-07-29 (silver generalisation) | **pass** — max\|diff\| 0.0 on all 4 columns × 4,845 rows, AUC 0.5291 |

The last row is why the fetch-layer rewrite in §4.2 was safe to make mid-step:
the changed code is reachable only during a network fetch, and gold's flags are
off, so the frozen path could not move — but "could not move" is checked, not
asserted.

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

### 1.1 Extension to silver — expectations fixed 2026-07-29

Silver joins step 2 in the role it held in step 1: a **transferability
control**, run with the same six features, windows, lags and gates, differing
only in the two identifiers that name the asset in an external dataset (§4.1).
Nothing about silver's configuration may be adjusted to improve a silver
result. Directional expectations, recorded before any silver number is read:

- **H-COT is *more* plausible on silver than on gold.** Silver's futures market
  is far smaller and more speculative, with a well-documented history of
  crowded positioning; if managed-money crowdedness carries information
  anywhere in the metals, silver is where it should be largest. Sample size is
  unchanged (~327 weekly dev observations), so this remains exploratory
  regardless of outcome.
- **H-GDELT is *weaker and directionally ambiguous* on silver.** Gold's version
  is a pure safe-haven story: fear up, gold up. Silver is roughly half
  industrial demand (solar PV, electronics, brazing), so a fear shock pushes
  its monetary leg up and its industrial leg down at the same time. The two
  effects partially cancel. **Therefore: a weaker GDELT result on silver than
  on gold is consistent with the mechanism and must not be read as the gold
  result failing to replicate.** The converse is the informative surprise — a
  *stronger* safe-haven signal on silver than gold would be evidence against
  the safe-haven story as stated, and would need explaining before anything
  built on it.

This asymmetry is why §6.1's joint rule is written in terms of *both* assets
passing, rather than treating silver as a straight replication test.

#### 1.1a A confound in H-GDELT-on-silver, measured 2026-07-29 before any outcome

The BigQuery migration (§4.3) made article counts directly measurable, and the
first thing they showed is that **the H-GDELT silver prediction above is
confounded**. Over calendar 2022, with the pinned phrase lists:

| asset | distinct stories/day | days with zero coverage |
|---|---|---|
| gold | **20.9** | 0 of 365 |
| silver | **5.3** | 17 of 365 |

**Confirmed on the full snapshot (2026-07-29, §11), which is worse:** median
28.0 stories/day for gold against **7.0** for silver over 2019-09 → 2026-07,
and **137 zero-story days against 17**. The 4× ratio is stable across both
measurements, so this is a structural property of coverage rather than an
artifact of the 2022 sample.

A daily mean tone over ~7 articles is roughly **2× noisier** than one over ~28,
by sampling alone. So there are now two independent reasons to expect a weaker
silver GDELT result — the industrial/monetary cancellation above (mechanism)
and thin sampling (measurement) — and **this design cannot distinguish them.**

Recorded as a confound rather than repaired, deliberately. Three options were
considered and rejected on 2026-07-29, before any Tier-1 or Tier-2 GDELT number
existed:

1. *Minimum-articles mask* — introduces a new threshold chosen after seeing the
   data, and thins silver further.
2. *Smooth tone before z-scoring* — changes a §3 feature definition post-hoc.
3. *Broaden silver's phrases* — the density gap is not query breadth, it is that
   the world writes 4× more about gold; widening it buys contamination
   (§4.1's trap list) rather than coverage.

The reasoning for leaving it alone: **noise costs power, it cannot manufacture
signal.** A noisy feature that still clears the §6 gate is a *stronger* result,
not a weaker one. This is the same call as run 5's holdout decision — state the
power problem, do not redesign the test around it.

**Consequence for §6.1, accepted in advance:** if GDELT passes on gold and fails
on silver, the joint rule still returns *not retained* — but the write-up must
report that silver's failure is **uninterpretable** between mechanism and
measurement, rather than claiming it as evidence against the gold result. The
threshold is not relaxed; the interpretation is qualified. A negative on gold is
unaffected by any of this.

Which features this touches: counting is unaffected at 5 stories/day, averaging
is. So `gdelt_vol_smooth7_z` carries its full weight on both assets;
`gdelt_tone_z` and `gdelt_tone_chg_5d` are the compromised pair on silver only.

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

**Re-validated 2026-07-29 under headline matching.** The BigQuery migration
(§4.3) moved matching from article bodies to `<PAGE_TITLE>`, so the 2026-07-28
DOC 2.0 record does not carry over; it is preserved at the end of this section
as superseded. Both assets were re-previewed from scratch before either
snapshot was fetched. One iteration was sufficient for each.

Phrases, verbatim (match `config/gold.yaml` and the snapshot sidecar):

```yaml
title_phrases: ["gold price", "price of gold", "gold futures", "gold bullion"]
```

compiled to `<page_title>[^<]*(gold\ price|price\ of\ gold|gold\ futures|gold\ bullion)`.

Preview protocol, fixed before any preview ran and unchanged from 2026-07-28:
four windows — Tokyo Olympics (2021-07-23→08-08), Beijing Olympics
(2022-02-04→02-20), COVID stress (2020-03), calm (2023-06) — with the Olympic
windows included deliberately as the known contamination trap for the word
"gold". Each window returned **250 distinct stories** (the preview cap).

Result: **no contamination term appeared in either Olympic window**, and
manual reading of all four samples found gold-market coverage throughout.
Phrase-matching is why, and it is strictly stronger here than under full text:
Olympic coverage says "gold medal", never "gold price" — and a passing mention
of the gold price *inside the body* of an Olympics report can no longer pull
the article in at all, because only the headline is searched.

**One false positive, recorded rather than fixed:** *"Black Diamond Ice Cream
23 Karat Gold Price in Dubai"* — gold-leaf ice cream; one story in 250. No
phrase was changed to exclude it. A filter written to remove a specific
headline someone has already read is a decision made after seeing the data,
and one story cannot move a daily mean over ~28 stories/day. Leaving it in
costs a little noise; removing it would cost the pre-registration.

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

**Superseded — the 2026-07-28 DOC 2.0 record**, kept because the ledger is
append-only and because it is the baseline the headline mechanism replaced.
Query: `("gold price" OR "gold prices" OR "price of gold" OR bullion OR "gold
futures") sourcelang:english`. Same four windows, 250 articles each; result
**0 of 1,000 sampled titles** matched contamination terms (olympic, medal,
gold coast, golden globe, gold cup, golden retriever). No snapshot was ever
fetched under this query, so nothing downstream depends on it.

### 4.1 Silver — query and validation record

**Status 2026-07-29: VALIDATED.** The candidate below was recorded *before*
validation, so the reasoning could not be adjusted after seeing headlines; the
preview cleared it on the first attempt. (It had been blocked for two days by
DOC 2.0 rate limiting, §4.2 — the BigQuery migration is what unblocked it.)

Phrases, verbatim (match `config/silver.yaml` and the snapshot sidecar):

```yaml
title_phrases: ["silver price", "price of silver", "silver futures", "silver bullion"]
```

**Structural rule applied:** the phrase list is gold's, with the metal's name
substituted and nothing else changed — pinned as an assertion in
`tests/test_config_altdata.py`. Under DOC 2.0 the two lists could not be made
parallel: bare `bullion` in English news is dominated by *gold* bullion
coverage, so silver had to use the qualified `"silver bullion"` while gold kept
the bare term, and §4.1 recorded that asymmetry as accepted-but-regrettable.
**The migration removed the need for it** — headline matching made
`"gold bullion"` a viable phrase for gold, so both lists are now exactly
parallel and the asymmetry is gone. This is the one place where the forced
re-validation improved the design rather than merely repeating it.

**Result: 0/720 titles matched contamination terms**, across five windows —
Tokyo Olympics, Beijing Olympics (143 stories between them), silver squeeze
(196), COVID stress (161), calm month (220). Manual reading found no Nate
Silver, no Adam Silver, no Silver Lake, no silver linings.

**The positive control fired**, which is the half of the test that a clean
contamination grep cannot give you: the Jan/Feb 2021 window returned *"Reddit
Day-Trader Frenzy Sends Silver Prices Soaring"* and *"Silver price surges as
GameStop traders move into precious metals"*. So the query is neither too
broad (no contamination) nor too narrow (it catches the biggest retail silver
event in the sample). Both failure directions are excluded, not just one.

Note the window counts against gold's flat 250/250/250/250: only the calm
month approaches the cap. That is the density confound of §1.1a visible in the
preview itself — silver simply has less coverage, and it is a property of the
world, not of the query.

**Why silver needs a bigger trap list than gold.** Gold's contamination was
one word in one context (Olympic medals), and phrase-matching killed it
outright. "Silver" is also a surname (Nate Silver; the NBA's Adam Silver), a
private-equity firm (Silver Lake), two US towns, a US missing-person alert, a
restaurant chain, a racetrack, and a dozen idioms (silver lining, silver
bullet, silver screen, silverware). `scripts/fetch_altdata.py` therefore
carries per-asset `ASSET_CONTAMINATION_TERMS` and `ASSET_PREVIEW_WINDOWS`,
with **no fallback** for an unregistered asset — screening a new metal's query
against another metal's trap words would report a clean preview that validated
nothing.

Silver's windows add a **positive control** the gold set did not need: the
retail "silver squeeze" of late Jan/early Feb 2021 is real silver-market news,
so a query that *misses* it is too narrow, not too clean. Gold's four windows
are unchanged and were simply re-run under the new mechanism (§4).

### 4.2 GDELT rate limiting — an operational constraint on the design

Recorded because it changed the fetch code, not just the schedule. The
documented limit is one request per 5 seconds. Observed behaviour across
2026-07-28/29: an 8-minute exponential-backoff ladder never escaped a 429; a
9-minute silent probe still got 429; a day later the API answered immediately,
then 429'd again after ~5 requests spaced 10 s apart. The binding constraint
is therefore **request count over a long window**, not request rate, and
retrying during a penalty extends it.

Mitigations tried: chunk width raised to `CHUNK_YEARS = 3` (~9 requests/asset)
at 60 s spacing, with a daily-resolution guard so a wider chunk could not
silently coarsen the series. **These were not enough.** On 2026-07-29 a preview
run that had idled 45 minutes was 429'd on its *first* request. The REST API is
not usable for an 8.5-year backfill from a single IP, and no amount of patience
demonstrably fixes it. Superseded by §4.3.

### 4.3 Migration to BigQuery — 2026-07-29

The same corpus is a Google BigQuery public dataset
(`gdelt-bq.gdeltv2.gkg_partitioned`), unthrottled, with a 1 TB/month free tier.
The REST path was removed rather than kept alongside: a dead import path that
still runs is a trap. Auth is Application Default Credentials, not an API key,
so nothing enters `.env`.

**This changes the sample, and the changes are not cosmetic.** All three were
discovered by probing the table *before* any feature was built, and none of
them would have raised an error if ported naively:

1. **No full text → matching moved to headlines.** DOC 2.0 searched article
   bodies; GKG stores the headline in `Extras` as `<PAGE_TITLE>` (populated in
   10/10 sampled rows; `V2Themes` was `None` in 6/10 and is unusable). Title
   matching is *tighter* than full text — it selects articles **about** the
   metal rather than ones mentioning it in passing, which is the entity
   filtering the README argued for. The config representation changed
   accordingly: `gdelt.query` (a DOC 2.0 string) → `gdelt.title_phrases` (a
   list), and the phrase lists were made **exactly parallel** between gold and
   silver, removing the old bare-`bullion`-vs-`"silver bullion"` asymmetry.
   **Both assets' §4/§4.1 validation records must be redone under this
   mechanism** — the 2026-07-28 gold preview does not carry over.

2. **No deduplication → aggregates run over distinct titles.** DOC 2.0 clusters
   syndicated copies; raw GKG stores one row per outlet. In a 10-row sample,
   8 were the same Associated Press story. Measured over Nov 2020–Apr 2021 this
   is **not** a constant offset a z-score would absorb:

   | | value |
   |---|---|
   | median copies/story | 1.00 |
   | p90 / max copies/story | 1.12 / 2.61 |
   | corr(copies-per-story, stories/day) | **+0.514** |
   | corr(copies-per-story, mean tone) | −0.312 |
   | mean tone, Feb 2021: raw vs deduped | 0.999 vs **1.549** |

   Duplication spikes precisely on the highest-attention days — Feb 1–3 2021,
   the silver squeeze, ran at 2.0–2.6× — i.e. exactly the days a sentiment
   feature is supposed to fire on, and it moved one month's mean tone by 55%.
   Every aggregate therefore collapses to one row per (day, asset, title)
   *before* the daily mean, pinned by
   `test_sql_deduplicates_by_title_before_counting`.

3. **Cost, not rate, is now the constraint → one scan for all assets.** The
   8.5-year pull scans ~536 GB (54% of the monthly free tier), because billing
   is on bytes scanned in the columns touched and `Extras` is large. A per-asset
   scan would exhaust the quota, so `fetch_gdelt_bq` takes a mapping of assets
   and writes every snapshot from a single pass. Better provenance as a side
   effect: gold and silver demonstrably come from the same scan of the same
   index, which matters because §6.1 compares them. The daily-total denominator
   is a separate query and is free (`COUNT(*)` over partition metadata).

4. **History is 21 months shorter — `<PAGE_TITLE>` does not exist before
   2019-09-22.** Discovered when the snapshot fetch, asked for 2018-01-01,
   returned data starting 2019-09-22 on *both* assets. Verified directly
   (0.68 GB, five sampled partitions) rather than assumed, because "my query
   returned nothing" and "the field is empty" are different problems:

   | partition | rows | non-empty `Extras` | contains `<PAGE_TITLE>` |
   |---|---|---|---|
   | 2018-06-01 | 621,870 | 549,011 | **0** |
   | 2019-06-01 | 333,269 | 255,397 | **0** |
   | 2019-09-15 | 285,005 | 220,957 | **0** |
   | 2019-09-25 | 517,812 | 517,812 | **517,812** |
   | 2020-06-01 | 405,580 | 405,580 | **405,580** |

   The table itself is partitioned from 2015-02-17, so the articles are there;
   GDELT simply began populating the page title between 15 and 25 Sept 2019,
   as a step change from 0% to 100% rather than a ramp. **The headline
   mechanism therefore has a hard floor at 2019-09-22 and no query can move
   it.** DOC 2.0's full-text search reached back to 2017; this is a real cost
   of the migration, paid for the tighter entity filtering in (1).

   Consequence for the evaluation, stated before any outcome is read: with the
   252-day z-window (min_periods 126) the first usable GDELT bar is
   **2020-02-02**, against H4 candles that begin 2019-01-01. That removes
   1,775 development bars — **8,615 of 10,390 dev bars survive (82.9%)**,
   identically on both assets. The loss falls entirely at the *head* of the
   window, so it shortens fold 0 rather than punching a hole in the middle.
   §7's power analysis is restated on these counts.

Unchanged by the migration: the six features and their formulas (§3), the
publication lag and staleness rules (§2.2), the Tier-1 and Tier-2 protocols
(§5, §6), and the retention gate (§6, §6.1). **This is a data-source
substitution, not a change to the hypothesis or the decision rule.** The
density confound it exposed is recorded in §1.1a; the shortened window is
finding 4 above and costs power, which the gate does not compensate for.

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

### 5.1 Tier-1 COT results — read 2026-07-29 (gold + silver, n = 332 each)

GDELT is absent (no snapshot yet), so this is half the Tier-1 picture.
Reminder from §5: **nothing is retained or retired on these numbers.**

**Direction: nothing, on either metal.** `cot_mm_net_z` vs forward return is
−0.003 (5d) / +0.086 (20d) on gold — sign flips — and −0.079 / −0.123 on
silver. Silver's are at least *contrarian-signed*, as H-COT predicted, but at
z = −1.40 and −1.53 they are indistinguishable from noise. This matches the
pre-registered expectation and step 1's finding.

**Volatility: one consistent candidate.** Managed-money crowdedness precedes
higher realized volatility, positive in **4 of 4** asset × horizon cells:

| | gold | silver |
|---|---|---|
| `cot_mm_net_z` vs vol, 5d | +0.130 (z 2.38) | +0.136 (z 2.27) |
| `cot_mm_net_z` vs vol, 20d | +0.124 (z 1.47) | +0.199 (z 1.98) |
| half-sample signs | +/+ , +/+ | +/+ , +/+ |

Every half is same-signed, which the unstable cells below are not. The
mechanism is plausible without any story-fitting: crowded positioning is
unwind risk, and unwinds are large moves. It also lands exactly where §7 said
a positive was most likely — volatility, not direction — and where step 1 put
the only genuinely predictable quantity (best feature vs forward vol: 0.62).

**Discount these, despite large |z|:** gold `cot_prod_net_z` vs 20d return
(−0.188, z −2.27) has halves **+0.023 / −0.324**, a full sign flip; gold
`cot_prod_net_z` vs vol (z −2.81, −2.70) has halves −0.308/−0.023 and
−0.407/−0.057, i.e. almost entirely first-half. These are the regime-
instability signature §5 was written to expose.

**Multiple comparisons:** 24 cells per asset, 48 total. At |z| > 2 roughly
2–3 cells are expected by chance alone, and about that many appeared — so no
single cell is evidence. The `cot_mm_net_z`-vs-volatility pattern is
interesting because of its *consistency across two assets, two horizons and
both halves*, not because of any one z.

**Underpowered, as pre-registered.** §7 put the detectable |ic| at ~0.155 for
n ≈ 327; the observed vol ICs are 0.12–0.20, straddling that line. Suggestive,
not established — and per §6.1 the COT replication between gold and silver is
the *weak* kind (one report, one schedule, correlated flows).

**Carried forward:** `cot_mm_net_z` → forward volatility is logged as a
**step-4 position-sizing candidate**, which §6's interpretive note (b) already
defined as a useful positive that does *not* retain the family for direction.
Tier 2 will still test direction, per the pre-registered matrix.

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

### 6.1 Joint rule across gold and silver — fixed 2026-07-29, before any silver result

Running the same evaluation on a second asset **doubles the comparison count**
against the same single holdout. That cost buys something only if the decision
rule is fixed first — otherwise "it worked on one of them" is available after
the fact, which is precisely the pattern that left meta-labeling as a lead
rather than a result in step 1 (run 3: gold yes, silver no).

The gate in §6 is evaluated **per asset, unchanged**. The program-level
decision is then:

| gold | silver | decision |
|---|---|---|
| pass | pass | **Retained.** Family goes live in both yamls; Phase D (§10) opens. |
| pass | fail | **Not retained.** Logged as a lead, exactly meta-labeling's status. |
| fail | pass | **Not retained.** Same treatment — the rule is symmetric, and was written that way before either number existed. |
| fail | fail | Retired. Honest negative to §9 and CLAUDE.md. |

**Thresholds are not relaxed to accommodate a one-of-two outcome**, and a
one-of-two outcome does not license a third asset to break the tie — adding
platinum to resolve a gold/silver disagreement would be the scope creep
CLAUDE.md's asset-scope rule exists to prevent, and would multiply comparisons
again. PGMs remain step 3.

**How much is silver replication actually worth?** Less than two independent
assets, and the two families differ in how much less:

- **GDELT replication is the stronger evidence of the two — but by less than
  this section originally claimed.** The first version argued that because the
  two phrase lists share no terms, the news series are near-independent inputs.
  Sharing no *terms* does not mean matching no *articles*: precious-metals
  coverage routinely reports both metals in one headline ("Gold and Silver
  Prices Plummet"). Measured over 2022 (2026-07-29, before any GDELT outcome):

  | | stories | shared with the other asset |
  |---|---|---|
  | gold | 7,624 | 334 (**4.4%**) |
  | silver | 1,831 | 334 (**18.2%**) |

  So the input overlap is real, asymmetric, and falls on silver — the same
  direction as the §1.1a density confound, and for a related reason (silver's
  corpus is smaller, so a fixed pool of joint stories is a larger share of it).
  The ordering still holds: **82% of silver's stories are silver-only**, which
  is far more independence than COT offers. But "close to independent" was
  wrong and is withdrawn; the correct statement is *substantially but not
  wholly independent, with silver the more contaminated leg*.
- **COT replication is the weaker.** Both codes come from the same weekly
  report on the same release schedule, so availability is perfectly correlated
  (the Feb-2023 ION outage blanks identical bars on both assets), and
  managed-money flows across the two metals are themselves correlated. Silver
  agreeing with gold on COT is close to one observation seen twice.
- Underneath both: gold and silver **returns** correlate ~0.8, so the
  *outcome* side is shared no matter how independent the inputs are. Neither
  family's replication is worth two clean assets.

---

## 7. Power analysis — stated before evaluation

Dev window 2019-03-27 → 2025-06-30 ≈ 6.26 years for COT. **GDELT's window is
shorter and was not known to be until the snapshot was fetched** — the
`<PAGE_TITLE>` floor plus z-warmup puts its first usable bar at 2020-02-02
(§4.3 finding 4), giving 5.4 years.

| source | native obs | detectable \|r\| at 80% power, α=0.05 |
|---|---|---|
| COT | ~327 weekly | ≈ 0.155 |
| GDELT vs 5d horizon | 1,960 gold / 1,856 silver daily, n_eff ≈ n/5 ≈ 392 / 371 | ≈ 0.14 |
| GDELT vs 20d horizon | n_eff ≈ n/20 ≈ 98 / 93 | ≈ 0.28 |

(2.8/√n rule; GDELT effective n deflated for overlapping forward windows —
persistent features like `gdelt_tone_z` deflate further, toward the 20d row.
Empirical Newey–West SEs in Tier 1 supersede these planning numbers.)

**These are worse than the numbers this section carried before the fetch**
(≈0.13 and ≈0.26 on ~2,290 daily observations). The shortfall is the 21 months
of history that `<PAGE_TITLE>` does not cover, and it is recorded here rather
than quietly re-based, because a power analysis revised *downward* after the
data arrives is the honest direction and the one that is easy to omit.
Silver's daily count is a further 5% below gold's — the 137 zero-story days of
§1.1a — so silver is the weaker leg on sample size as well as on density.

**Silver's power is the same, and does not add to gold's.** Identical dev
window, identical native frequencies — silver's COT snapshot is 1,050 reports
over the same dates as gold's (§11). Because the two assets' returns correlate
~0.8, pooling them would *not* give ~654 independent weekly observations; the
combined evidence is much closer to gold's alone than to twice it. Silver is
therefore run as a replication check under §6.1, never pooled with gold to
manufacture sample size.

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

**Running a second asset doubles the ledger.** Every Tier-1 and Tier-2 look
below is taken once per asset, so silver's inclusion doubles the comparison
count charged against the one holdout. That cost is accepted because §6.1
fixes the joint decision rule in advance; it would not be acceptable as a
free second chance to find a positive.

| # | date | action | numbers read | notes |
|---|---|---|---|---|
| 0 | 2026-07-28 | pre-registration written | none | this document, before any fetch |
| 1 | 2026-07-29 | scope amended to include silver; §1.1 expectations, §4.1 query rule and §6.1 joint gate fixed | none | no outcome computed; COT snapshot sanity in §11 is feature distributions only, no forward return touched |
| 2 | 2026-07-29 | Tier-1 COT, gold + silver — **VOID, discarded** | 24 cells × 2 assets | ran with the `forward_outcomes` clamping bug (below). Numbers were wrong, not merely noisy; recorded here because the ledger is append-only and the run happened |
| 3 | 2026-07-29 | Tier-1 COT, gold + silver, after the fix | 24 cells × 2 assets, n=332 each | §5.1. GDELT not included (no snapshot) |
| 4 | 2026-07-29 | GDELT re-validation previews under headline matching, gold + silver | none | §4/§4.1. Headlines and syndication counts only; no price series loaded, no forward outcome computed. Not a look — see the §8 note below |
| 5 | 2026-07-29 | GDELT snapshots fetched + coverage/`PAGE_TITLE` probes | none | §11, §4.3 finding 4. Row counts, date ranges, story densities, partition metadata. Descriptive only |

**Look 2 was void — the bug, recorded because it is the instructive part.**
`diagnose_altdata.forward_outcomes` located each observation's forward window
with `searchsorted`, which **clamps to index 0** for any date before the price
series. The COT snapshot starts 2006-06 and the candles 2019-03, so ~600 of
936 observations were matched to *the same* March-2019 window. Effects: n
inflated 332 → 936, Newey–West SEs shrunk ~1.7×, every z inflated to match,
and the rank correlation distorted by a large block of tied outcomes. It threw
no error and produced a completely plausible table with three cells at |z| > 2.

The tell was the **stability column**: `ic_half1` was NaN in all 24 rows,
because that half's outcomes were constant and rank correlation against a
constant is undefined. The half-sample split was added to detect regime
instability and caught a data bug instead — an argument for reporting
diagnostics you do not strictly need.

Fixed both ends (an out-of-range date returns NaN; the observation filter now
bounds the head as well as the tail) and pinned by `tests/test_diagnostics.py`,
including a test that the Newey–West SE must *exceed* the naive SE under
overlap — the direction of that inequality is what keeps z-scores honest.

**Not looks: the 2026-07-29 BigQuery probes.** Four queries were run against
the GKG table while designing the migration (§4.3) — table schema, sample rows,
syndication ratios over Nov 2020–Apr 2021, and article density over 2022. They
are recorded here for completeness but are **not entries in this ledger**,
because none of them touched a price, a return or a label: they measure how
many articles exist and how often the same story is duplicated. That is
data-quality inspection of the kind §11 exists for, and it is the reason the
dedup rule and the §1.1a confound could both be fixed *before* any outcome was
readable. The distinction that matters: looking at **the data** costs nothing;
looking at **the relationship between the data and returns** is what this
ledger counts.

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
| 2026-07-29 | `XAGUSD_cot.parquet` | initial fetch | contract 084691, full history | 1,050 weekly reports, 2006-06-13 → 2026-07-21; median cadence 7d, 11 gaps > 7d — identical shape to gold, as expected for one CFTC-wide release schedule |
| 2026-07-29 | `XAUUSD_gdelt.parquet` | initial fetch | BigQuery `gkg_partitioned`, 4 title phrases, dedup by title, requested 2018-01-01 → now | 2,502 days, 2019-09-22 → 2026-07-28; median 28.0 stories/day; 17 zero-story/missing days |
| 2026-07-29 | `XAGUSD_gdelt.parquet` | initial fetch | same scan, same params, silver's 4 phrases | 2,501 days, 2019-09-23 → 2026-07-28; median 7.0 stories/day; 137 zero-story/missing days |

**Both GDELT snapshots came from one scan** (540 GB, 54% of the monthly free
tier), so the sidecars cross-reference each other via `co_fetched_with`. §6.1
compares the two assets, and this is the strongest provenance available for
that comparison: same index, same query date, same dedup, same code path.

**GDELT snapshot sanity (descriptive only — no forward outcome touched):**

| | gold | silver |
|---|---|---|
| first / last day | 2019-09-22 → 2026-07-28 | 2019-09-23 → 2026-07-28 |
| stories/day, median (mean) | 28.0 (28.4) | 7.0 (8.5) |
| zero-story or missing days | 17 (0.7%) | 137 (5.5%) |
| longest gap | 17d, 2025-06-15 → 2025-07-01 | 17d, **same dates** |
| usable dev H4 bars | 8,615 / 10,390 (82.9%) | 8,615 / 10,390 (82.9%) |
| first usable bar | 2020-02-02 | 2020-02-02 |

Three things to read off this table:

1. **The 17-day gap is identical on both assets**, so it is a GDELT-side
   outage, not a news drought in either metal — two queries sharing no terms
   do not go quiet on the same seventeen days by coincidence. It ends on
   2025-07-01, i.e. it occupies the final fortnight of the development window
   and touches the holdout boundary exactly. Nothing is done about it; it is
   ~0.7% of the sample and `MAX_STALENESS` already turns it into honest NaNs
   rather than three-week-old tone presented as current.
2. **Silver has 8× gold's count of blank days** (137 vs 17), which is §1.1a's
   density confound in its most concrete form: at ~7 stories/day a zero-story
   day is an ordinary occurrence, at ~28 it is not. Most are singletons that
   `ffill(limit=3)` bridges, which is why silver still ends with the same
   8,615 usable bars — but a bridged day is a repeated observation, not a
   fresh one, so silver's *effective* sample is smaller than the shared count
   implies. Read this alongside §7.
3. **The identical 82.9% is the head-truncation of §4.3 finding 4**, not a
   coincidence between the assets: both are clipped by the same
   `<PAGE_TITLE>` floor and the same 126-day warmup, and both feature sets
   start on the same bar.

**Snapshot sanity, both COT assets (descriptive only — no forward outcome
touched, so not a §8 look).** Built through the pinned snapshots and the
sidecar contract check onto 10,391 development H4 bars each:

| | gold | silver |
|---|---|---|
| NaN bars | 141 (1.36%) | 141 (1.36%) |
| NaN span | 2023-02-15 → 2023-03-17 | 2023-02-15 → 2023-03-17 |
| `cot_mm_net_z` mean / std | 0.122 / 1.164 | 0.185 / 1.111 |
| `cot_prod_net_z` mean / std | 0.280 / 1.165 | −0.052 / 1.057 |
| `cot_mm_net_chg_4w` std | 0.068 | 0.088 |

The identical 141-bar hole is the ION outage on both assets — one release
schedule, one interruption, and the reason §6.1 treats COT replication as the
weaker of the two. Silver's 4-week positioning change is ~30% more volatile
than gold's, consistent with §1.1's expectation that crowdedness effects
should be larger in the smaller, more speculative market.
