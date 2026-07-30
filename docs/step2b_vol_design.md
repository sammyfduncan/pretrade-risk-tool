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

## 7. Outcomes

*[Written after the gate runs. Nothing here yet.]*
