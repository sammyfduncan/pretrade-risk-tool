# Step 2b — alt-data → volatility: pre-registration

**Status: DESIGN — nothing fetched, nothing evaluated.** Successor to
`docs/step2_design.md`, whose direction program closed negative on 4 of 4
combinations (§9 there). This document must be committed before any Tier
number is read; the dirty-tree guard convention applies to it.

## 0. What this program is, and is not

Step 2 asked alt-data a **direction** question and got a clean no. Its own
diagnostics, however, put measured, stable information in **volatility**:

| finding (all pre-outcome-committed protocols) | where |
|---|---|
| `cot_mm_net_z` → forward vol positive in 4/4 asset × horizon cells, every half same-signed | step2 §5.1 |
| gold `gdelt_vol_smooth7_z` → forward vol ic 0.269/0.277 (z 5.68/3.44), stable halves | step2 §5.2 |
| price features → forward vol 0.62 vs → direction 0.05 | step 1, run 1 |

This program asks the question those results actually license: **do alt-data
features improve a volatility forecast beyond what price history already
provides?** The output, if any family passes, is an input to **step-4 position
sizing and regime filtering** — never a trade direction signal.

Not in scope, fixed now: no direction claims of any kind; no holdout data; no
change to `config/*.yaml` or the frozen step-1 template (this is offline
research; step 4 decides what to consume); no variance-risk-premium direction
probe (that would be its own addendum); no redefinition of any existing COT or
GDELT feature — the pinned snapshots and §3 formulas of step 2 are reused
verbatim.

The step-2 §8 ledger discipline continues here (§6 below). Comparisons
accumulate against the same single holdout.

## 1. Hypotheses, with per-asset predictions fixed in advance

- **H-VOL-COT** — crowded managed-money positioning precedes higher realized
  vol (crowding = unwind risk; unwinds are large moves). Tier-1 support 4/4.
  *Prediction: stronger on silver* (smaller, more speculative market; its
  Tier-1 20d cell was 0.199 vs gold's 0.124, and its `cot_mm_net_chg_4w` is
  ~30% more volatile).
- **H-VOL-ATTN** — news attention precedes higher vol. Tier-1 gold ic 0.27,
  but step2 §5.2's caveat is the entire question here: attention rises when
  markets move, so it may only proxy *recent realized* vol, which the baseline
  already carries. *Prediction: adds ≈ nothing beyond the baseline on gold;
  uninterpretable on silver* (density confound + unstable halves, step2 §1.1a,
  §5.2).
- **H-VOL-IV** — options-implied vol (GVZ) carries forward-looking information
  beyond backward-looking realized vol. The standard result in every asset
  class; *prediction: passes, and by the largest margin*. Its role here is
  double: a candidate family, and — once admitted — the **strongest free
  baseline** that the other families must then beat.
- **H-VOL-CAL** — scheduled macro events (FOMC, CPI, NFP) raise near-horizon
  vol. Deterministic, known ex ante, cannot leak. *Prediction: small reliable
  increment at the 5d horizon, ≈ nothing at 20d.*

**Expected outcome, stated now:** IV passes on gold; calendar passes at 5d;
GDELT-attention fails to add beyond the baseline (its Tier-1 ic was mostly
realized-vol proxy); COT is the genuinely open cell — positioning is the one
series here that price cannot see.

## 2. Data sources

| source | series | range | lag | notes |
|---|---|---|---|---|
| FRED | `GVZCLS` (Cboe Gold ETF Volatility Index) | 2008-06-03 → current | 2 trading days, matching `VIXCLS`'s convention in `features/macro.py` | rides the existing FRED fetch path; index closes do not revise materially, so no pinned snapshot needed — same treatment as the other FRED series |
| — | ~~VXSLV~~ | — | — | **excluded, decision fixed now.** Discontinued 2022-02-11 (FRED `VXSLVCLS`), relaunched by Cboe 2025-06-16 with recalculated history only back to 2022-02-14 — a methodology splice mid-dev-window on the silver leg. **F-IV therefore runs gold-only and is exempt from the §4 joint rule.** A missing silver IV result is a data limitation, not a failed replication. If a continuous single-methodology series ever exists, adding it is a new pre-registered decision. |
| Fed / BLS | FOMC meeting dates; CPI and Employment Situation release dates | published in advance, archived for the full window | none — schedules are known ex ante | verify archive completeness at build time; any missing year fails loudly, no silent gap |
| pinned snapshots | `*_cot.parquet`, `*_gdelt.parquet` | as pinned (step2 §11) | as built | **no refetch** — the snapshot is the sample |

## 3. Target, features, model — exact, fixed

**Observation grid:** daily, the same grid as step-2 Tier-1
(`diagnose_altdata.py`), so results are comparable.

**Target:** forward realized volatility over h ∈ {5, 20} trading days — the
same `forward_outcomes` vol definition Tier-1 used — log-transformed for
regression (vol is right-skewed; log makes errors comparable across regimes).

**Model:** OLS on standardized features. Deliberately linear: HAR-RV — the
standard vol-forecasting benchmark — is linear, and both step-1 runs showed the
constraint in this repo is information, not capacity. Walk-forward with the
same 5-fold, gapped scheme (gap ≥ horizon).

**Baselines and families:**

- **B0 (price-only, HAR-style):** log realized vol over trailing 1d, 5d, 22d.
  Three features.
- **F-IV** (gold only): log `GVZCLS` level, and its 5-day change. Two features.
- **B1 = B0 + F-IV** on gold; **B1 = B0** on silver (no clean IV series).
- **F-COT:** `cot_mm_net_z`, `cot_mm_net_chg_4w`, `cot_prod_net_z` — the
  step-2 trio, formulas unchanged.
- **F-ATTN:** `gdelt_vol_smooth7_z` — attention only; the tone features were
  direction hypotheses and earned no vol claim in Tier-1.
- **F-CAL:** `fomc_within_h`, `cpi_or_nfp_within_h` — binary, is a scheduled
  event inside the forecast horizon. Two features, h-matched.

No other features. No interactions. Windows and z-conventions are the ones
already pre-registered in step 2 §3.

## 4. The gate — fixed before any number

For each family F, matched rows (one feature frame per asset, all families
present, ablated by column — the step-2 §6 argument for why row sets must not
move):

- ΔR²_OOS = R²_OOS(B + F) − R²_OOS(B), pooled over folds, where **B = B1**
  for F-COT, F-ATTN, F-CAL (they must beat price *and* implied vol where IV
  exists) and **B = B0** for F-IV itself.

A family qualifies as a step-4 sizing input only if all three hold, at either
horizon:

1. ΔR²_OOS > 0 pooled, with the improvement in **≥ 4 of 5 folds**;
2. **placebo condition:** observed Δ exceeds the **max of 20 circular
   time-shifts** of that family's source series through the identical pipeline
   (`evaluation/placebo.py` — shift policies extend to the new sources: GVZ
   daily ≥ 365d in whole weeks; calendar dates shifted ≥ 1y). **There is
   deliberately no absolute Δ threshold** — run 6 measured the fixed-threshold
   approach as beatable by noise in 3 of 4 cells, so the placebo max *is* the
   bar;
3. **joint rule** across gold and silver for families present on both
   (F-COT, F-ATTN, F-CAL). F-IV is gold-only by §2. One-of-two = a lead,
   logged, not retained.

On failure: result logged in §7, nothing built. On success: the family is a
**named candidate input for the step-4 sizing design**, which will carry its
own pre-registration before anything trades on it.

## 5. Power — why this program is runnable where the direction one was not

Step 2's direction gate could detect |r| ≈ 0.14–0.28 while realistic direction
ICs are 0.02–0.05 — structurally underpowered, stated in advance and borne
out. Here the measured univariate effects are 0.13–0.27, i.e. **at or above
the detectable floor**, and the estimation problem (continuous target,
persistent signal) is easier. The risk inverts: with a strong autocorrelated
baseline, the danger is not missing a real effect but crediting a redundant
one — which is what the matched-row ablation against B1 and the placebo bar
exist to prevent.

## 6. Ledger of looks (append-only)

| # | date | action | numbers read | notes |
|---|---|---|---|---|
| 0 | 2026-07-30 | this document written; GVZ/VXSLV availability verified (dates only) | none | VXSLV splice discovered and excluded before any outcome existed |
| 1 | 2026-07-30 | B0 bug fixed in `vol_layer.build_b0` | none — NaN counts only | The 22d HAR rung had **zero** valid rows on real data: Dukascopy's 1–2 bar Sunday stubs are correctly NaN'd, and rolling over the raw daily series let one stub void every window touching it, so `dropna()` emptied the dataset and the gate crashed. Fixed to roll over *completed* trading days. Disclosure: this is a harness change made after §§0–6 were committed. It was made **before any ΔR² was read** (the only numbers inspected were per-column non-NaN counts), and it did not alter the §3 specification — three rungs, trailing 1/5/22 days, log. Pinned by `test_b0_survives_interleaved_sunday_stubs`. **Guard limitation now known:** `vol_gate.py`'s dirty-tree check covers only this document, not the harness, so it could not have caught a post-outcome harness edit. Future programs should hash the harness too. |
| 2 | 2026-07-30 | **the §4 gate, both assets** | 14 observed ΔR² + 280 placebo ΔR² | §7 below. Gold 4 families × 2 horizons = 8 cells; silver 3 × 2 = 6 |

## 7. Outcomes

**Verdict, 2026-07-30: F-IV and F-CAL qualify as step-4 sizing inputs; F-COT
and F-ATTN do not.** The first pre-registered gate this project has run that
anything passed. Nothing here is a direction signal, nothing trades, and the
one qualification that relied on the joint rule is weaker than the rule makes
it look — see §7.3, which is the part to read.

### 7.1 The gate, all 14 cells

Baselines per §3: **B1** (price + implied vol) for COT/ATTN/CAL on gold;
**B0** (price only) for F-IV itself, and for every silver family since silver
has no clean IV series (§2). c1 = Δ>0 and ≥4/5 folds; c2 = Δ > max of 20
placebos.

**Gold**

| family | horizon | ΔR² | R² without → with | folds | placebo max | emp. p | c1 | c2 |
|---|---|---|---|---|---|---|---|---|
| **IV** | 5d | **+0.0634** | 0.0906 → 0.1540 | 4/5 | +0.0015 | 0.048 | PASS | **PASS** |
| IV | 20d | +0.0998 | 0.0111 → 0.1108 | 3/5 | +0.0151 | 0.048 | FAIL | PASS |
| COT | 5d | −0.0429 | 0.1540 → 0.1111 | 2/5 | +0.0117 | 0.238 | FAIL | FAIL |
| COT | 20d | −0.0932 | 0.1108 → 0.0177 | 2/5 | −0.0605 | 0.238 | FAIL | FAIL |
| ATTN | 5d | +0.0112 | 0.1540 → 0.1651 | 3/5 | +0.0033 | 0.048 | FAIL | PASS |
| ATTN | 20d | +0.0307 | 0.1108 → 0.1415 | 4/5 | +0.0514 | 0.286 | PASS | FAIL |
| **CAL** | 5d | **+0.0207** | 0.1540 → 0.1747 | 4/5 | +0.0076 | 0.048 | PASS | **PASS** |
| CAL | 20d | −0.0172 | 0.1108 → 0.0937 | 2/5 | +0.0123 | 0.714 | FAIL | FAIL |

**Silver** (no IV family, so baseline is B0 throughout)

| family | horizon | ΔR² | R² without → with | folds | placebo max | emp. p | c1 | c2 |
|---|---|---|---|---|---|---|---|---|
| COT | 5d | −0.0780 | 0.0555 → −0.0225 | 1/5 | +0.0000 | 0.429 | FAIL | FAIL |
| COT | 20d | −0.4354 | 0.0152 → −0.4203 | 1/5 | +0.2525 | 0.667 | FAIL | FAIL |
| ATTN | 5d | +0.0060 | 0.0555 → 0.0616 | 2/5 | +0.0308 | 0.143 | FAIL | FAIL |
| ATTN | 20d | +0.0641 | 0.0152 → 0.0793 | 2/5 | +0.1371 | 0.095 | FAIL | FAIL |
| **CAL** | 5d | **+0.0241** | 0.0555 → 0.0797 | **5/5** | +0.0098 | 0.048 | PASS | **PASS** |
| CAL | 20d | −0.0002 | 0.0152 → 0.0150 | 3/5 | +0.0080 | 0.190 | FAIL | FAIL |

Applying §4: **F-IV qualifies** (gold, 5d; gold-only and joint-rule-exempt by
§2). **F-CAL qualifies** (5d, passing on both assets). **F-COT and F-ATTN do
not qualify** at any horizon on either asset.

### 7.2 All four §1 predictions were correct, including the open one

Written before the harness existed:

| prediction (§1) | outcome |
|---|---|
| "IV passes, and by the largest margin" | correct — +0.063 at 5d, the largest passing Δ; +0.100 at 20d is larger still but fails fold consistency |
| "calendar passes at 5d, ≈ nothing at 20d" | correct, precisely — 5d +0.021/+0.024 both pass; 20d −0.017/−0.000 both fail |
| "GDELT-attention adds ≈ nothing beyond the baseline" | correct — no cell passes both conditions; §5.2's caveat that its ic 0.27 was largely realized-vol proxy is now confirmed by direct test |
| "COT is the genuinely open cell" | resolved **negative**, and emphatically: all four cells negative (−0.043, −0.093, −0.078, −0.435) |

The GDELT one is the substantive result. Tier-1 measured attention → forward
vol at ic 0.269 (z 5.68), the largest effect step 2 produced, and §5.2 warned
in advance that most of it was probably a proxy for recent realized
volatility. The matched-row test against a HAR baseline confirms it: once the
model already knows trailing 1/5/22-day realized vol, news attention adds
+0.011 (3/5 folds) at 5d. **An ic of 0.27 became a ΔR² of 0.011.** That gap
is the entire reason Tier-1 is labelled "diagnostics, not gates."

### 7.3 The F-CAL qualification is weaker than the joint rule implies

**A design gap, stated plainly.** §4 condition 3 requires a family to pass on
both assets, and its justification — inherited from step 2 §6.1 — is that two
assets are two chances to be fooled. That logic assumes the family has
**per-asset inputs**: COT has a different contract code per metal, GDELT a
different query. **F-CAL has no per-asset component at all** — gold and silver
receive the *identical* two columns, built from the same FOMC/CPI/NFP dates.
Since the two metals' returns correlate ~0.8 and their vol co-moves harder
still, "passes on both" is close to **one observation reported twice**, not a
replication.

So F-CAL's honest status is: **one cell, p = 0.048, on a mechanism that is
independently well documented** (scheduled macro events raise near-term vol)
and that cannot be lookahead by construction. It qualifies under the letter of
the pre-registered rule, and I am not relaxing or re-writing that rule after
the fact. But the joint rule did no work here, and any future family without
per-asset inputs should have that noted *before* it is run.

F-IV never claimed joint support — §2 exempted it in advance for the VXSLV
splice — so its single-asset status is a documented design fact rather than a
shortfall.

### 7.4 The placebo nulls are centred well below zero, unlike run 6's

| cell | placebo mean | sd | max |
|---|---|---|---|
| gold IV 5d | −0.033 | 0.022 | +0.002 |
| gold CAL 5d | −0.005 | 0.006 | +0.008 |
| gold COT 20d | −0.680 | 1.676 | −0.061 |
| silver COT 20d | −0.507 | 0.537 | +0.253 |

Run 6's ΔAUC placebos averaged ≈0.000; these average strongly negative. The
reason is the model, not a bug: **OLS with R²_OOS punishes uninformative
columns** — they add fitting variance with no signal, and out-of-sample R²
charges for that — whereas LightGBM ranking by AUC can largely ignore a junk
feature. So the null distribution here answers "what does adding noise columns
to an OLS do?" and the answer is "it hurts."

Consequences, both directions:

- The empirical p remains **valid and distribution-free** — it asks whether
  the observed Δ falls in the tail of the no-information distribution, and
  p = 1/21 = 0.048 is the strongest statement 20 placebos can make (0 of 20
  exceeded). Three cells hit exactly that floor.
- But the **margin looks more impressive than the evidence is**. "+0.063
  versus a placebo max of +0.002" is partly "signal helps" versus "noise
  hurts", which is a weaker contrast than "signal helps more than noise
  appears to help". Read the empirical p, not the gap.
- COT's 20d nulls have sd 1.68 (gold) and 0.54 (silver), and its observed
  silver R² is **−0.42** — far worse than predicting the training mean. At
  this horizon the COT arm is unstable enough that its point estimate carries
  little information beyond "not helping". The verdict does not depend on the
  magnitude: all four cells are negative and 1–2 of 5 folds improve.

### 7.5 Multiple comparisons, and what is NOT claimed

14 cells were run. At a 0.048 threshold roughly 0.7 false passes are expected
by chance; three cells passed both conditions (gold IV 5d, gold CAL 5d, silver
CAL 5d), of which the two CAL cells are near-duplicates per §7.3. This is
consistent with two real effects, both of which have independent
well-documented mechanisms — but it is not strong evidence on its own, and the
step-4 addendum must treat both as candidates to be re-tested, not as
established inputs.

Not claimed: any direction edge (none was tested); any holdout result (none
touched); that F-IV or F-CAL improves *trading* outcomes (a vol forecast is
not a P&L); that COT or GDELT are useless for vol in general — only that they
add nothing once a HAR baseline, and on gold implied vol, are already present.

**Step-2 §9's carried-forward candidates are now resolved:** `cot_mm_net_z`
and gold's `gdelt_vol_smooth7_z` were logged there as step-4 sizing
candidates on Tier-1 evidence. This program tested exactly that and **retires
both**. The surviving sizing candidates are implied vol (gold) and the event
calendar (both), at the 5-day horizon only.
