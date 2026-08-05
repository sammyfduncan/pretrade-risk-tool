"""Placebo calibration for the step-2 retention gate — docs/archive/step2_design.md §6, condition 3.

    python scripts/placebo_altdata.py --config config/gold.yaml --family gdelt
    python scripts/placebo_altdata.py --config config/silver.yaml --family cot --n 20

Conditions 1 and 2 of the gate (ΔAUC ≥ +0.015, same direction in ≥4/5 folds)
are a *screen*. At these effective sample sizes +0.015 AUC is ~1.2-1.9 SE, so
on its own it says almost nothing. This script supplies the gate's actual
evidential content: the null distribution of ΔAUC produced by **this exact
procedure** when the alt-data carries no information at all.

Why a placebo rather than an analytic standard error. The real pipeline has
overlapping labels, autocorrelated features, uniqueness weighting, per-fold
isotonic calibration, a meta-mode row filter and five time-ordered folds.
Every one of those inflates or deflates a naive SE in a direction nobody can
compute in closed form — run 5's naive p-value was 1,800x too small for
exactly this reason. A placebo inherits all of it, because it *is* the
pipeline, with one thing changed.

What is changed: the family's **source snapshot** is circularly shifted in
time (§6 specifies the source series, not the finished feature columns, so
that the z-score windows, staleness bridges and publication lags are all
recomputed on the shifted data — the placebo goes through the real code, not
past it). Values roll, the index stays. The resulting series has this family's
genuine internal structure — autocorrelation, spikiness, missingness, weekly
news seasonality — and no true correspondence with price.

Reading the output: the gate asks whether the observed Δ exceeds the **maximum**
of 20 placebo Δs. Max-of-20 is an empirical one-sided p < 0.05, and unlike a
t-test it needs no distributional assumption. An observed Δ buried inside the
placebo spread means the family's apparent contribution is what this pipeline
manufactures from noise.

This script reads no holdout data and makes no retention decision — it prints
the comparison the pre-registered gate is then applied to.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from commodity_ml.config import AssetConfig, load_config
from commodity_ml.dataset import feature_columns
from commodity_ml.evaluation.ml_metrics import fold_metrics_table, overall_metrics
from commodity_ml.evaluation.placebo import circular_shift, draw_shifts, empirical_p
from commodity_ml.features.cot import cot_feature_names, load_cot_snapshot
from commodity_ml.features.gdelt import gdelt_feature_names, load_gdelt_snapshot
from commodity_ml.features.macro import fetch_macro
from commodity_ml.features.pipeline import (
    build_feature_frame,
    load_partner_candles,
    partner_symbols,
)
from commodity_ml.labels import make_labels
from commodity_ml.modeling.walkforward import walk_forward
from commodity_ml.primary import SIGNAL_COLUMN, make_primary_signal

FAMILY_FEATURES = {"cot": cot_feature_names, "gdelt": gdelt_feature_names}


def assemble(cfg: AssetConfig, raw, macro, partners, cot, gdelt) -> pd.DataFrame:
    """build_dataset.py's assembly, same calls in the same order.

    Fidelity is not asserted by inspection — `main` checks that a zero-shift
    pass through this function reproduces the committed dataset parquet exactly.
    If this ever drifts from build_dataset.py, that check fails loudly rather
    than quietly invalidating every placebo.
    """
    features, _ = build_feature_frame(
        cfg, raw, macro=macro, partners=partners, cot=cot, gdelt=gdelt
    )
    labels = make_labels(raw, cfg.label)
    primary = make_primary_signal(raw, cfg.primary)
    return pd.concat([features, labels, primary, raw["close"]], axis=1).dropna()


def evaluation_slice(cfg: AssetConfig, dataset: pd.DataFrame) -> pd.DataFrame:
    """Development rows only, meta-filtered — the §6 gate is on meta mode."""
    ds = dataset[dataset.index < cfg.data.holdout_start]
    if cfg.label.mode == "meta":
        ds = ds[ds[SIGNAL_COLUMN] == 1]
    return ds


def delta_auc(cfg: AssetConfig, dataset: pd.DataFrame, family: str) -> dict:
    """AUC(with the family) − AUC(without it), on identical rows and folds.

    Both arms come from the same `dataset`, so the row set, the fold
    boundaries, the label overlap and the sample weights are shared and the
    difference is attributable to the six columns alone. This is the
    matched-row ablation of §6 computed in-process; `train.py
    --exclude-family` computes the same quantity from the command line.
    """
    ds = evaluation_slice(cfg, dataset)
    names = feature_columns(ds)
    family_cols = [c for c in FAMILY_FEATURES[family]() if c in names]
    if not family_cols:
        raise SystemExit(
            f"No {family} columns in the dataset — rebuild with features.{family}: true"
        )
    without = [c for c in names if c not in family_cols]

    y = ds["label"]
    options = dict(
        n_splits=cfg.cv.n_splits,
        gap=cfg.cv.gap_bars,
        exit_time=ds["exit_time"],
        sample_weighting=cfg.model.sample_weighting,
        calibration=cfg.model.calibration,
    )
    wf_with = walk_forward(ds[names], y, model_kind=cfg.model.type, **options)
    wf_without = walk_forward(ds[without], y, model_kind=cfg.model.type, **options)

    auc_with = overall_metrics(y, wf_with.oof_proba)["auc"]
    auc_without = overall_metrics(y, wf_without.oof_proba)["auc"]
    per_fold = (
        fold_metrics_table(y, wf_with.oof_proba, wf_with.folds)["auc"]
        - fold_metrics_table(y, wf_without.oof_proba, wf_without.folds)["auc"]
    )
    return {
        "delta": auc_with - auc_without,
        "auc_with": auc_with,
        "auc_without": auc_without,
        "per_fold": per_fold,
        "n_rows": len(ds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--family", required=True, choices=sorted(FAMILY_FEATURES))
    parser.add_argument("--n", type=int, default=20,
                        help="placebo runs; the §6 gate is calibrated on 20")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    family = args.family

    if not getattr(cfg.features, family):
        raise SystemExit(
            f"features.{family} is false in {args.config}. The matched-row ablation "
            "needs one dataset built with BOTH families on (§6)."
        )

    raw = pd.read_parquet(cfg.raw_path())
    macro = fetch_macro(start=cfg.data.start, end=cfg.data.end) if cfg.features.macro else None
    partners = load_partner_candles(cfg)
    snapshots = {
        "cot": load_cot_snapshot(cfg) if cfg.features.cot else None,
        "gdelt": load_gdelt_snapshot(cfg) if cfg.features.gdelt else None,
    }
    truth = snapshots[family]
    print(f"=== Placebo calibration: {cfg.asset} / {family} ===")
    print(f"  snapshot: {len(truth):,} observations "
          f"({truth.index[0].date()} -> {truth.index[-1].date()})")
    print(f"  partner legs: {', '.join(partner_symbols(cfg))}")

    # Fidelity gate. Every placebo delta is meaningless unless this function
    # reproduces the real pipeline, so prove it against the committed parquet
    # before spending 20 runs.
    observed_dataset = assemble(cfg, raw, macro, partners, snapshots["cot"], snapshots["gdelt"])
    committed = pd.read_parquet(cfg.dataset_path)
    if not observed_dataset.index.equals(committed.index):
        raise SystemExit(
            f"Zero-shift assembly gives {len(observed_dataset):,} rows, the committed "
            f"dataset has {len(committed):,}. This harness has drifted from "
            "build_dataset.py — fix that before trusting any placebo."
        )
    shared = [c for c in committed.columns if c in observed_dataset.columns]
    if len(shared) != len(committed.columns):
        raise SystemExit(
            f"Zero-shift assembly is missing columns the committed dataset has: "
            f"{sorted(set(committed.columns) - set(shared))}"
        )
    # Split by dtype: the primary signal is boolean and `-` is undefined on it,
    # and an exact-equality check is the stronger test there anyway.
    numeric = [c for c in shared if pd.api.types.is_float_dtype(committed[c])]
    exact = [c for c in shared if c not in numeric]
    drift = (observed_dataset[numeric] - committed[numeric]).abs().max().max()
    mismatched = [c for c in exact if not observed_dataset[c].equals(committed[c])]
    if mismatched:
        raise SystemExit(
            f"Zero-shift assembly disagrees with the committed dataset on "
            f"{mismatched} — this harness has drifted from build_dataset.py."
        )
    print(f"  fidelity check: index identical, {len(exact)} non-float column(s) exact, "
          f"max |Δ| on {len(numeric)} float column(s) = {drift:.2e}")

    observed = delta_auc(cfg, observed_dataset, family)
    print(f"\n  OBSERVED  ΔAUC = {observed['delta']:+.4f}  "
          f"({observed['auc_without']:.4f} -> {observed['auc_with']:.4f}, "
          f"n={observed['n_rows']:,})")
    folds = observed["per_fold"]
    print(f"    per fold: {', '.join(f'{d:+.4f}' for d in folds)}")
    print(f"    same-direction folds: {int((folds > 0).sum())}/{len(folds)} "
          f"(gate condition 2 needs >= 4)")

    rng = np.random.default_rng(args.seed)
    shifts = draw_shifts(len(truth), args.n, family, rng)
    print(f"\n  {args.n} placebos, shifts (periods): {shifts}")

    rows = []
    t0 = time.time()
    for i, shift in enumerate(shifts, 1):
        placebo_snapshots = dict(snapshots)
        placebo_snapshots[family] = circular_shift(truth, shift)
        dataset = assemble(
            cfg, raw, macro, partners,
            placebo_snapshots["cot"], placebo_snapshots["gdelt"],
        )
        result = delta_auc(cfg, dataset, family)
        rows.append({"shift": shift, "delta": result["delta"],
                     "auc_with": result["auc_with"], "auc_without": result["auc_without"],
                     "n_rows": result["n_rows"]})
        elapsed = time.time() - t0
        print(f"    [{i:2d}/{args.n}] shift {shift:5d}  ΔAUC {result['delta']:+.4f}  "
              f"n={result['n_rows']:,}  ({elapsed/i:.0f}s/run)")

    placebos = pd.DataFrame(rows)
    worst = placebos["delta"].max()
    exceeded = int((placebos["delta"] >= observed["delta"]).sum())
    p_emp = empirical_p(observed["delta"], placebos["delta"])

    print(f"\n=== Result: {cfg.asset} / {family} ===")
    print(f"  observed ΔAUC      {observed['delta']:+.4f}")
    print(f"  placebo max        {worst:+.4f}")
    print(f"  placebo mean/sd    {placebos['delta'].mean():+.4f} / {placebos['delta'].std():.4f}")
    print(f"  placebo 90th pct   {placebos['delta'].quantile(0.90):+.4f}")
    print(f"  placebos >= obs    {exceeded}/{args.n}   empirical p = {p_emp:.3f}")

    c1 = observed["delta"] >= 0.015
    c2 = int((folds > 0).sum()) >= 4
    c3 = observed["delta"] > worst
    print(f"\n  gate condition 1 (ΔAUC >= +0.015):     {'PASS' if c1 else 'FAIL'}")
    print(f"  gate condition 2 (>= 4/5 folds):        {'PASS' if c2 else 'FAIL'}")
    print(f"  gate condition 3 (Δ > max placebo):     {'PASS' if c3 else 'FAIL'}")
    print(f"  -> {family} on {cfg.asset}: "
          f"{'PASSES all three' if (c1 and c2 and c3) else 'DOES NOT pass the gate'}")
    print("\n  Per §6.1 retention also requires the SAME verdict on the other asset;\n"
          "  one-of-two is a lead, not a result. Ledger this run in §8.")

    # Keyed by symbol AND family — reports_dir is shared across assets, so an
    # unkeyed name would have silver's placebos overwrite gold's.
    out = cfg.reports_dir / "step2" / f"placebo_{cfg.symbol}_{family}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    placebos.to_csv(out, index=False)
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
