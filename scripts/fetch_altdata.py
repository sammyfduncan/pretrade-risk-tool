"""Fetch and pin the step-2 alt-data snapshots (CFTC COT + GDELT).

Usage:
    python scripts/fetch_altdata.py --config config/gold.yaml                     # both
    python scripts/fetch_altdata.py --config config/gold.yaml --source cot
    python scripts/fetch_altdata.py --config config/gold.yaml --source gdelt --preview
    python scripts/fetch_altdata.py --config config/gold.yaml --source gdelt --force

Why "pin" and not just "cache": GDELT's API queries today's index about the
past, so a re-fetch months later returns a slightly different sample than a
live system would have seen. The snapshot parquet IS the evaluation sample —
build_dataset.py and predict.py only ever read it, a sidecar .meta.json
records its provenance, and overwriting one requires --force plus a row in
docs/step2_design.md §11.

--preview (GDELT only) is the query-validation step and writes NOTHING: it
prints sample headlines over fixed windows — deliberately including Olympic
months, the known contamination trap for the word "gold" — so the query can
be eyeballed and iterated (--query overrides the yaml while iterating) before
any timeline data is committed to. The validated query is then recorded in
the yaml and in docs/step2_design.md §4.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from commodity_ml.config import load_config
from commodity_ml.features import cot as cot_module
from commodity_ml.features import gdelt as gdelt_module

# Fixed preview windows: two Olympic months (the "gold" contamination trap),
# one stress month, one calm month. Fixed so every query iteration is judged
# against the same evidence.
PREVIEW_WINDOWS = (
    ("Tokyo Olympics", "2021-07-23", "2021-08-08"),
    ("Beijing Olympics", "2022-02-04", "2022-02-20"),
    ("COVID stress", "2020-03-01", "2020-03-31"),
    ("calm month", "2023-06-01", "2023-06-30"),
)
CONTAMINATION_TERMS = (
    "olympic", "medal", "gold coast", "golden globe", "gold cup", "golden retriever",
)
_ARTICLES_SHOWN = 12


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


def preview_gdelt(query: str) -> None:
    print(f"=== GDELT preview (writes nothing) ===\nquery: {query}\n")
    total, contaminated = 0, 0
    for i, (label, start, end) in enumerate(PREVIEW_WINDOWS):
        if i:
            time.sleep(gdelt_module._REQUEST_PAUSE_S)
        articles = gdelt_module.preview_articles(query, start, end)
        total += len(articles)
        hits = [
            a for a in articles
            if any(t in a.get("title", "").lower() for t in CONTAMINATION_TERMS)
        ]
        contaminated += len(hits)

        print(f"--- {label} ({start} -> {end}): {len(articles)} articles sampled ---")
        for article in articles[:_ARTICLES_SHOWN]:
            date = article.get("seendate", "?")[:8]
            print(f"  {date}  [{article.get('domain', '?')}]  {article.get('title', '')[:100]}")
        if hits:
            print(f"  !! {len(hits)} title(s) match contamination terms, e.g.:")
            for hit in hits[:3]:
                print(f"     {hit.get('title', '')[:100]}")
        print()

    rate = contaminated / total if total else 0.0
    print(f"Contamination grep across all samples: {contaminated}/{total} titles ({rate:.1%})")
    print("(The grep is crude — the decision is made by reading titles, not by this number.)")
    print("\nWhen the query reads clean: record it in the yaml (gdelt.query) and in")
    print("docs/step2_design.md §4, then fetch the timeline snapshot (no --preview).")


def fetch_gdelt_snapshot(cfg, query: str, start: str, end: str | None, force: bool) -> None:
    print("=== GDELT timelines (tone / vol / volraw) ===")
    frame = gdelt_module.fetch_gdelt(query, start=start, end=end)

    dead = int((frame["n_articles"].isna() | (frame["n_articles"] == 0)).sum())
    observed = frame["tone"].notna()
    runs = (observed != observed.shift()).cumsum()[~observed]
    longest_gap = int(runs.value_counts().max()) if len(runs) else 0
    print(f"  {len(frame):,} days, {frame.index[0].date()} -> {frame.index[-1].date()}")
    print(f"  zero-article/missing days: {dead} (masked to NaN); longest gap {longest_gap} days")

    write_snapshot(
        cfg.gdelt_snapshot_path,
        frame,
        {
            "query": query,
            "source": gdelt_module.GDELT_URL,
            "start": start,
            "end": end or "now",
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
            "rows": len(frame),
            "zero_article_days": dead,
        },
        force,
    )
    print("\nRecord this fetch in docs/step2_design.md §11.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--source", choices=("cot", "gdelt", "all"), default="all")
    parser.add_argument("--preview", action="store_true",
                        help="GDELT only: print sample article titles for query "
                             "validation; writes nothing")
    parser.add_argument("--query", default=None,
                        help="override gdelt.query while iterating on --preview")
    parser.add_argument("--start", default=gdelt_module.SNAPSHOT_START,
                        help="GDELT only (COT always fetches full history)")
    parser.add_argument("--end", default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing snapshot (ledger it)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.preview:
        query = args.query or cfg.gdelt.query
        if not query:
            raise SystemExit("No query: pass --query or set gdelt.query in the yaml.")
        preview_gdelt(query)
        return

    if args.source in ("cot", "all"):
        fetch_cot_snapshot(cfg, force=args.force)
    if args.source in ("gdelt", "all"):
        query = args.query or cfg.gdelt.query
        if not query:
            raise SystemExit(
                "gdelt.query is empty — validate one first:\n"
                f"    python scripts/fetch_altdata.py --config {args.config} "
                "--source gdelt --preview --query '...'"
            )
        fetch_gdelt_snapshot(cfg, query, start=args.start, end=args.end, force=args.force)


if __name__ == "__main__":
    main()
