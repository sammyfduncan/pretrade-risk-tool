# Joint intraday volatility forecast experiment

**Frozen:** 2026-07-31 after the separate-family gate and before generating a
joint-model outcome  
**Development:** gold M30 decisions from `2021-01-01` through `2025-06-30`  
**Confirmation:** `2025-07-01` onward remains unopened  
**Deployment:** impossible from this experiment alone; paper sizing remains
disabled without a retained primary strategy and a registered economic test

**Outcome:** PASS. J1 achieved pooled out-of-sample R2 `0.2284974` versus
`0.2242097` for J0, for delta R2 `0.0042877`; four of five folds improved and
the maximum registered placebo delta was `0.0008232`.

## 1. Known evidence and question

The registered separate-family experiment is already known. B0 achieved pooled
out-of-sample R2 `0.2031819`. F-IV added `0.0210278` and improved all five
folds. F-CAL added `0.0052251` and improved four of five folds. Both exceeded
their registered placebo maxima. The frozen policy therefore selected B0
alone rather than choosing a family after seeing their relative results.

This experiment asks one new question:

> Does the already-registered event calendar add stable, non-spurious
> information when GVZ is already present in the causal B0 forecast?

It does not revisit the horizon, target, lags, features, model family, session,
or separate-family results.

## 2. Fixed models and data

The data, matched-row construction, target, exclusions, five expanding folds,
four-row purge gap, source snapshots, and ordinary-least-squares fitting are
identical to `docs/intraday_risk_design.md`.

- **Reference J0:** B0 plus `gvz_log` and `gvz_chg_5d`.
- **Joint J1:** J0 plus `bls_release_next_120m` and
  `fomc_release_next_120m`.

J0 is fixed as the reference because F-IV was the stronger family in the
already-read separate-family result. There is no alternative model, interaction,
regularizer, horizon, event window, or feature selection.

## 3. Registered placebo and gate

The observed release schedule is circularly shifted by 20 registered calendar
offsets drawn without replacement from seed `20260731`. GVZ, B0, prices,
targets, rows, and folds remain fixed. A shift may not reproduce a near-original
alignment, and every placebo must reproduce the exact observed matched-row
index.

J1 passes only if every condition holds:

1. joint pooled out-of-sample R2 is positive;
2. joint pooled R2 is strictly above J0;
3. joint-minus-J0 R2 is positive in at least four of five folds;
4. observed joint-minus-J0 R2 is strictly above the maximum of all 20 placebo
   deltas; and
5. matched coverage remains at least 90%.

If J1 passes, the registered forecast selection becomes B0 plus GVZ plus the
calendar. If it fails, B0 remains the selected forecast under the earlier
both-pass rule. Either result writes an audit artifact with
`paper_enabled=False`; it cannot activate sizing.

## 4. Economic boundary

No primary gold strategy currently passes its frozen retention gate. This
experiment therefore reads no trade labels, directions, returns, profit
factors, drawdowns, or sizing outcomes. A later matched constant-risk versus
volatility-risk comparison must be registered before it is run and must refuse
any strategy whose primary gate did not pass.

## 5. Guarded command and look ledger

The design and evaluation harness must be committed and clean. The only real
data command will be:

```bash
python scripts/evaluate_intraday_joint_risk.py
```

It will expose no model, feature, fold, placebo-count, seed, or threshold
arguments.

| Look | Date | Item | Outcome available? |
|---:|---|---|---|
| 0 | 2026-07-31 | Joint design frozen from the already-known separate-family result | no joint result |
| 1 | 2026-07-31 | Registered joint evaluation on 28,513 matched development rows and 20 calendar shifts | PASS; B0 plus GVZ plus calendar selected; `paper_enabled=False`; confirmation unopened |
