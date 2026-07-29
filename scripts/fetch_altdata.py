"""Fetch and pin the step-2 alt-data snapshots (CFTC COT + GDELT).

Usage:
    # COT is per-asset and cheap — one small Socrata request each.
    python scripts/fetch_altdata.py --config config/gold.yaml --source cot

    # GDELT preview: validate one asset's headline phrases. Writes nothing.
    python scripts/fetch_altdata.py --config config/silver.yaml --source gdelt \
        --preview --project ml-commodity

    # GDELT snapshot: ALL assets at once, deliberately. See below.
    python scripts/fetch_altdata.py --source gdelt --project ml-commodity

Why the GDELT fetch is not per-asset
------------------------------------
The 8.5-year BigQuery scan costs ~536 GB against a 1 TB/month free tier, and
cost is bytes scanned in the columns touched — so scanning once per asset would
blow the quota. Every asset's yaml is read, the phrase lists are combined into
one query, and each asset's snapshot is written from that single scan. This is
also better provenance: gold and silver demonstrably come from the same scan of
the same index, which matters because they are compared against each other
under the §6.1 joint rule.

Why "pin" and not just "cache": GDELT indexes today's view of the past, so a
re-fetch months later returns a slightly different sample than a live system
would have seen. The snapshot parquet IS the evaluation sample —
build_dataset.py and predict.py only ever read it, a sidecar .meta.json records
its provenance, and overwriting one requires --force plus a row in
docs/step2_design.md §11.

--preview is the query-validation step and writes NOTHING: it prints sample
headlines over fixed windows — including that metal's known homonym traps, e.g.
Olympic months for gold and silver — so the phrases can be eyeballed and
iterated (--phrase overrides the yaml while iterating) before any data is
committed to. Validated phrases are then recorded in the yaml and in
docs/step2_design.md §4. Each asset needs its own phrases and its own trap
list; see preview_plan().
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.features import cot as cot_module
from commodity_ml.features import gdelt as gdelt_module

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

# Preview windows, fixed before any preview is run so every phrase iteration is
# judged against the same evidence. Two are shared — a stress month and a calm
# month, testing that the phrases return sensible market coverage at all — and
# the rest are per-asset traps for that metal's specific homonyms.
SHARED_PREVIEW_WINDOWS = (
    ("COVID stress", "2020-03-01", "2020-03-31"),
    ("calm month", "2023-06-01", "2023-06-30"),
)
ASSET_PREVIEW_WINDOWS = {
    "gold": (
        ("Tokyo Olympics", "2021-07-23", "2021-08-08"),
        ("Beijing Olympics", "2022-02-04", "2022-02-20"),
    ),
    # Silver gets the Olympic windows too ("silver medal"), plus a positive
    # control: the Jan/Feb-2021 retail "silver squeeze" is real silver-market
    # news, so a query that misses it is too narrow rather than too broad.
    "silver": (
        ("Tokyo Olympics", "2021-07-23", "2021-08-08"),
        ("Beijing Olympics", "2022-02-04", "2022-02-20"),
        ("silver squeeze (positive control)", "2021-01-28", "2021-02-05"),
    ),
}

# Per-asset, because the homonyms are. Gold's trap is essentially one word in
# one context (Olympic medals); "silver" is also a surname (Nate, and the NBA's
# Adam Silver), a private-equity firm, two US towns, a missing-person alert, a
# restaurant chain, a racetrack and a dozen idioms.
ASSET_CONTAMINATION_TERMS = {
    "gold": (
        "olympic", "medal", "gold coast", "golden globe", "gold cup",
        "golden retriever",
    ),
    "silver": (
        "olympic", "medal", "adam silver", "nate silver", "silver lake",
        "silver spring", "silver alert", "long john silver", "silverstone",
        "silverware", "silver screen", "silver lining", "silver bullet",
        "silver anniversary", "silverado", "quicksilver",
    ),
}
_TITLES_SHOWN = 12


def preview_plan(asset: str) -> tuple[tuple, tuple]:
    """Windows and contamination terms for one asset.

    No fallback for an unknown asset, for the same reason cot.contract_code has
    no default: silently screening platinum's phrases against gold's trap words
    would report a clean preview that validated nothing.
    """
    if asset not in ASSET_PREVIEW_WINDOWS:
        raise SystemExit(
            f"No preview plan for asset '{asset}'. Before validating phrases for a "
            "new metal, add its windows to ASSET_PREVIEW_WINDOWS and its homonyms "
            "to ASSET_CONTAMINATION_TERMS in this file — screening a new asset "
            "against another's trap words would pass without testing anything."
        )
    return (
        ASSET_PREVIEW_WINDOWS[asset] + SHARED_PREVIEW_WINDOWS,
        ASSET_CONTAMINATION_TERMS[asset],
    )


def write_snapshot(path: Path, frame: pd.DataFrame, meta: dict, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"{path} already exists. A snapshot is a pinned sample — for GDELT a "
            "re-fetch is a DIFFERENT sample (the index is not point-in-time). If you "
            "mean it, pass --force and record the re-fetch in docs/step2_design.md §11."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    meta_path = path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(f"  -> {path}")
    print(f"  -> {meta_path}")


def fetch_cot_snapshot(cfg, force: bool) -> None:
    print("=== CFTC COT (disaggregated futures-only) ===")
    frame = cot_module.fetch_cot(cfg.cot.contract_code)

    gaps = frame.index.to_series().diff().dt.days.dropna()
    print(f"  {len(frame):,} weekly reports, {frame.index[0].date()} -> {frame.index[-1].date()}")
    print(f"  cadence: median {gaps.median():.0f} days between reports; "
          f"{int((gaps > 7).sum())} gaps longer than a week "
          "(holidays and the 2013/2018-19/2025 publication interruptions)")
    if gaps.median() != 7:
        raise SystemExit("Median report cadence is not weekly — inspect before pinning.")

    write_snapshot(
        cfg.cot_snapshot_path,
        frame,
        {
            "contract_code": cfg.cot.contract_code,
            "source": cot_module.COT_URL,
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
            "rows": len(frame),
            "first_report": str(frame.index[0].date()),
            "last_report": str(frame.index[-1].date()),
        },
        force,
    )


def preview_gdelt(phrases: list[str], asset: str, project: str) -> None:
    windows, terms = preview_plan(asset)
    print(f"=== GDELT preview for {asset} (writes nothing) ===")
    print(f"phrases: {phrases}")
    print(f"regex:   {gdelt_module.build_title_regex(phrases)}\n")

    total, contaminated = 0, 0
    for label, start, end in windows:
        rows = gdelt_module.preview_titles(phrases, project, start, end)
        total += len(rows)
        hits = [r for r in rows if any(t in (r["title"] or "").lower() for t in terms)]
        contaminated += len(hits)

        print(f"--- {label} ({start} -> {end}): {len(rows)} distinct stories ---")
        for row in rows[:_TITLES_SHOWN]:
            copies = f" x{row['copies']}" if row["copies"] > 1 else ""
            print(f"  {row['day']}  [{row['domain']}]{copies}  {(row['title'] or '')[:95]}")
        if hits:
            print(f"  !! {len(hits)} title(s) match contamination terms, e.g.:")
            for hit in hits[:3]:
                print(f"     {(hit['title'] or '')[:95]}")
        print()

    rate = contaminated / total if total else 0.0
    print(f"Contamination grep across all samples: {contaminated}/{total} titles ({rate:.1%})")
    print("(The grep is crude — the decision is made by reading titles, not by this number.)")
    print("\nWhen the phrases read clean: record them in the yaml (gdelt.title_phrases)")
    print("and in docs/step2_design.md §4, then fetch the snapshot (no --preview).")


def fetch_gdelt_snapshots(start: str, end: str | None, project: str, force: bool) -> None:
    """Every asset's snapshot from ONE scan — see the module docstring."""
    configs = {}
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(path)
        if not cfg.gdelt.title_phrases:
            print(f"  skipping {path.name}: no gdelt.title_phrases set")
            continue
        configs[cfg.asset] = cfg
    if not configs:
        raise SystemExit(
            "No config in config/ has gdelt.title_phrases set. Validate phrases "
            "first with --preview."
        )

    print("=== GDELT (BigQuery, one scan for all assets) ===")
    print(f"  assets: {', '.join(configs)}")
    for cfg in configs.values():
        if cfg.gdelt_snapshot_path.exists() and not force:
            raise SystemExit(
                f"{cfg.gdelt_snapshot_path} already exists and --force was not given. "
                "Refusing BEFORE spending ~536 GB of quota on a scan whose output "
                "could not be written."
            )

    frames = gdelt_module.fetch_gdelt_bq(
        {a: list(c.gdelt.title_phrases) for a, c in configs.items()},
        project=project, start=start, end=end,
    )

    for asset, cfg in configs.items():
        frame = frames[asset]
        dead = int((frame["n_articles"].isna() | (frame["n_articles"] == 0)).sum())
        observed = frame["tone"].notna()
        runs = (observed != observed.shift()).cumsum()[~observed]
        longest_gap = int(runs.value_counts().max()) if len(runs) else 0
        print(f"\n  {asset}: {len(frame):,} days, "
              f"{frame.index[0].date()} -> {frame.index[-1].date()}")
        print(f"    zero-story/missing days: {dead} (masked to NaN); "
              f"longest gap {longest_gap} days")
        print(f"    stories/day: median {frame['n_articles'].median():.1f}")

        write_snapshot(
            cfg.gdelt_snapshot_path,
            frame,
            {
                "title_phrases": list(cfg.gdelt.title_phrases),
                "source": f"bigquery:{gdelt_module.BQ_TABLE}",
                "matching": "headline (<PAGE_TITLE>), deduplicated by distinct title",
                "start": start,
                "end": end or "now",
                "fetched_at": pd.Timestamp.utcnow().isoformat(),
                "rows": len(frame),
                "zero_article_days": dead,
                "co_fetched_with": sorted(a for a in configs if a != asset),
            },
            force,
        )
    print("\nRecord this fetch in docs/step2_design.md §11.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml",
                        help="COT and --preview are per-asset; the GDELT snapshot "
                             "fetch ignores this and reads every config/*.yaml")
    parser.add_argument("--source", choices=("cot", "gdelt", "all"), default="all")
    parser.add_argument("--preview", action="store_true",
                        help="GDELT only: print sample headlines for phrase "
                             "validation; writes nothing")
    parser.add_argument("--phrase", action="append", default=None, dest="phrases",
                        help="override gdelt.title_phrases while iterating on "
                             "--preview; repeat the flag once per phrase")
    parser.add_argument("--project", default=None,
                        help="Google Cloud project for BigQuery billing/quota "
                             "(the data itself lives in the public gdelt-bq project)")
    parser.add_argument("--start", default=gdelt_module.SNAPSHOT_START,
                        help="GDELT only (COT always fetches full history)")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing snapshot (ledger it)")
    args = parser.parse_args()

    if args.preview:
        cfg = load_config(args.config)
        phrases = args.phrases or list(cfg.gdelt.title_phrases)
        if not phrases:
            raise SystemExit("No phrases: pass --phrase or set gdelt.title_phrases.")
        if not args.project:
            raise SystemExit("--project is required (BigQuery billing/quota project).")
        preview_gdelt(phrases, cfg.asset, args.project)
        return

    if args.source in ("cot", "all"):
        fetch_cot_snapshot(load_config(args.config), force=args.force)
    if args.source in ("gdelt", "all"):
        if not args.project:
            raise SystemExit("--project is required (BigQuery billing/quota project).")
        fetch_gdelt_snapshots(args.start, args.end, args.project, args.force)


if __name__ == "__main__":
    main()
