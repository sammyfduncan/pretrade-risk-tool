"""Step 2b gate: does alt-data improve a volatility forecast? (design §4)

    python scripts/vol_gate.py --config config/gold.yaml
    python scripts/vol_gate.py --config config/silver.yaml

Development slice only. Per horizon (5d, 20d) and per family, prints the
matched-row ΔR²_OOS against the pre-registered baseline — B1 (price + implied
vol) for COT/attention/calendar, B0 (price only) for implied vol itself — and
calibrates it against 20 circular time-shifts of that family's source series.
There is deliberately no absolute Δ threshold: run 6 measured a fixed
threshold as beatable by noise, so the placebo max IS the bar.

Silver runs without the IV family (design §2: VXSLV has a mid-window
methodology splice), so its baseline is B0 and its verdict covers three
families, not four. The §4 joint rule is applied by reading both assets'
outputs together — this script prints one asset and says so.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.evaluation.placebo import circular_shift, draw_shifts, empirical_p
from commodity_ml.evaluation.vol_gate import ablation_delta
from commodity_ml.features import event_dates, vol_layer
from commodity_ml.features.cot import load_cot_snapshot
from commodity_ml.features.gdelt import load_gdelt_snapshot

N_SPLITS = 5
DESIGN_DOC = "docs/archive/step2b_vol_design.md"


def refuse_dirty_preregistration(allow: bool) -> None:
    """Same contract as diagnose_altdata's guard, pointed at the 2b doc."""
    result = subprocess.run(
        ["git", "status", "--porcelain", DESIGN_DOC],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parents[1],
    )
    if result.stdout.strip():
        message = (
            f"{DESIGN_DOC} has uncommitted changes. Commit the pre-registration "
            "BEFORE reading any gate number, or the numbers cannot be "
            "interpreted against it."
        )
        if not allow:
            raise SystemExit(message + "\n(--allow-dirty-preregistration to override.)")
        print(f"WARNING: {message} (override active)\n")


def shifted_event_days(events: pd.DatetimeIndex, shift_days: int) -> pd.DatetimeIndex:
    """Roll event dates through the verified coverage span, wrapping.

    The events become a daily binary series so the shift machinery (and its
    whole-week step, which preserves the FOMC-Wednesday / NFP-Friday weekday
    structure) applies to the calendar exactly as it does to every other
    source.
    """
    span = pd.date_range(event_dates.COVERAGE_START, event_dates.COVERAGE_END, freq="D")
    flags = pd.DataFrame(
        {"event": np.isin(span, events.normalize()).astype(float)}, index=span
    )
    rolled = circular_shift(flags, shift_days)
    return pd.DatetimeIndex(rolled.index[rolled["event"] > 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--n-placebos", type=int, default=20,
                        help="the §4 gate is calibrated on 20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty-preregistration", action="store_true")
    args = parser.parse_args()
    refuse_dirty_preregistration(args.allow_dirty_preregistration)

    cfg = load_config(args.config)
    candles = pd.read_parquet(cfg.raw_path())
    candles = cfg.development_slice(candles)
    h4_close = candles["close"]
    print(f"=== Step 2b vol gate: {cfg.asset} ===")
    print(f"Development slice only: candles through {h4_close.index[-1].date()}")

    cot_raw = load_cot_snapshot(cfg)
    gdelt_raw = load_gdelt_snapshot(cfg)

    # Implied vol is gold-only (design §2). Fetched live like the other FRED
    # series — index closes do not revise, so there is no snapshot to pin.
    gvz = None
    if cfg.asset == "gold":
        print("  FRED GVZCLS   Cboe gold implied vol — the F-IV family")
        gvz = vol_layer.fetch_gvz()
    else:
        print("  F-IV absent for this asset (design §2: VXSLV splice) — baseline is B0")

    sources = {"cot": cot_raw, "gdelt": gdelt_raw}
    rng = np.random.default_rng(args.seed)
    report_lines: list[str] = []

    for horizon in vol_layer.TARGET_HORIZONS:
        dataset = vol_layer.assemble_vol_dataset(
            h4_close, cot_raw, gdelt_raw, gvz, horizon
        )
        target = f"fwd_log_vol_{horizon}d"
        b0 = vol_layer.FAMILY_COLUMNS["b0"]
        iv_cols = vol_layer.FAMILY_COLUMNS["iv"] if gvz is not None else []
        b1 = b0 + iv_cols
        cal_cols = vol_layer.calendar_columns(horizon)

        # family -> (baseline it must beat, its columns, its shiftable source)
        arms: dict[str, tuple[list[str], list[str], str]] = {}
        if gvz is not None:
            arms["iv"] = (b0, iv_cols, "gvz")
        arms["cot"] = (b1, vol_layer.FAMILY_COLUMNS["cot"], "cot")
        arms["attn"] = (b1, vol_layer.FAMILY_COLUMNS["attn"], "gdelt")
        arms["cal"] = (b1, cal_cols, "calendar")

        header = (f"\n--- horizon {horizon}d: {len(dataset):,} matched rows, "
                  f"{dataset.index[0].date()} -> {dataset.index[-1].date()} ---")
        print(header)
        report_lines.append(header)

        for family, (baseline, columns, source_kind) in arms.items():
            observed = ablation_delta(
                dataset, target, baseline, columns, N_SPLITS, gap=horizon
            )
            base_name = "B0" if baseline == b0 else "B1"
            print(f"\n  {family.upper()} vs {base_name}: "
                  f"ΔR² = {observed['delta']:+.4f} "
                  f"({observed['r2_without']:.4f} -> {observed['r2_with']:.4f}), "
                  f"{observed['folds_improved']}/{N_SPLITS} folds improved")

            # ---- placebos: shift this family's source, rebuild, re-measure
            if source_kind == "gvz":
                n_obs = len(gvz)
            elif source_kind == "calendar":
                n_obs = (event_dates.COVERAGE_END - event_dates.COVERAGE_START).days + 1
            else:
                n_obs = len(sources[source_kind])
            shifts = draw_shifts(n_obs, args.n_placebos, source_kind, rng)

            deltas = []
            t0 = time.time()
            for i, shift in enumerate(shifts, 1):
                shifted_cot, shifted_gdelt, shifted_gvz = cot_raw, gdelt_raw, gvz
                calendar_override = None
                if source_kind == "cot":
                    shifted_cot = circular_shift(cot_raw, shift)
                elif source_kind == "gdelt":
                    shifted_gdelt = circular_shift(gdelt_raw, shift)
                elif source_kind == "gvz":
                    shifted_gvz = circular_shift(gvz.to_frame(), shift)["gvz"]
                elif source_kind == "calendar":
                    calendar_override = {
                        "fomc": shifted_event_days(event_dates.fomc_days(), shift),
                        "cpi_nfp": shifted_event_days(event_dates.cpi_nfp_days(), shift),
                    }
                placebo_ds = vol_layer.assemble_vol_dataset(
                    h4_close, shifted_cot, shifted_gdelt, shifted_gvz, horizon,
                    calendar_override=calendar_override,
                )
                result = ablation_delta(
                    placebo_ds, target, baseline, columns, N_SPLITS, gap=horizon
                )
                deltas.append(result["delta"])
            elapsed = time.time() - t0

            worst = max(deltas)
            p_emp = empirical_p(observed["delta"], np.array(deltas))
            c1 = observed["delta"] > 0 and observed["folds_improved"] >= 4
            c2 = observed["delta"] > worst
            verdict = "PASSES c1+c2" if (c1 and c2) else "does NOT pass"
            lines = [
                f"  {family.upper()} vs {base_name} @ {horizon}d: "
                f"ΔR² {observed['delta']:+.4f}, folds {observed['folds_improved']}/5",
                f"    placebo max {worst:+.4f}, mean "
                f"{np.mean(deltas):+.4f}, sd {np.std(deltas):.4f}, "
                f"emp. p = {p_emp:.3f}  ({args.n_placebos} shifts, {elapsed:.0f}s)",
                f"    c1 (Δ>0 and >=4/5 folds): {'PASS' if c1 else 'FAIL'}   "
                f"c2 (Δ > placebo max): {'PASS' if c2 else 'FAIL'}   -> {verdict}",
            ]
            for line in lines:
                print(line)
            report_lines.extend(lines)

    footer = ("\nJoint rule (§4 condition 3): a family present on both assets is "
              "retained only on the same verdict for BOTH. Read this output "
              "alongside the other asset's before concluding anything. Ledger "
              "this run in the step2b §6 table.")
    print(footer)
    report_lines.append(footer)

    out = cfg.reports_dir / "step2b" / f"vol_gate_{cfg.symbol}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report_lines))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
