"""GDELT news tone and attention — step 2's unstructured alt-data leg.

What this measures
------------------
GDELT machine-reads global news continuously and scores each article's tone
(positive minus negative language, roughly -10..+10). For one per-asset title
filter (config: gdelt.title_phrases) we aggregate three daily series:

  tone        mean tone across DISTINCT STORIES that day
  vol         distinct stories as a share of ALL GDELT coverage that day;
              normalized, so the index growing over the years does not
              masquerade as rising attention
  n_articles  distinct story count; kept as a mask, because a zero-story day
              has NO tone and tone 0.0 is otherwise a valid value

The pre-registered hypothesis (docs/step2_design.md §1): falling tone (fear)
and rising attention lead gold strength at 5-20 days — and attention should
predict forward *volatility* regardless of direction, which would be useful for
step 4's sizing even if direction fails.

Source: BigQuery, not the REST API (migrated 2026-07-29)
--------------------------------------------------------
The DOC 2.0 REST API is unusable for a backfill: its binding limit is request
COUNT over a multi-hour window, and it returned 429 on the first request after
a 45-minute idle wait. The same corpus lives in the `gdelt-bq` BigQuery public
dataset with no throttling. Three consequences, all recorded in §4.1:

1. NO FULL TEXT. DOC 2.0 searched article bodies; the GKG table stores the
   headline in Extras as <PAGE_TITLE>. So matching is on TITLES. This is
   tighter than DOC 2.0 — it selects articles ABOUT the metal rather than ones
   that mention it in passing — but it is a different sample, so both assets'
   queries were re-validated under this mechanism.

2. NO DEDUPLICATION. DOC 2.0 clusters syndicated copies; raw GKG does not. One
   Associated Press story appears once per outlet that ran it. Measured over
   Nov 2020-Apr 2021, duplication is not a constant offset that a z-score would
   absorb: median 1.00 copies/story on quiet days but 2.0-2.6x on the busiest,
   with corr(ratio, stories/day) = +0.514 — i.e. it inflates precisely the
   high-attention days a sentiment feature is supposed to fire on, and it moved
   mean tone by 55% (0.999 -> 1.549) over one month. Hence every aggregate here
   is computed over DISTINCT TITLES, never raw rows.

3. COST IS THE BINDING CONSTRAINT, not rate. The 8.5-year pull scans ~536 GB
   against a 1 TB/month free tier, so all assets are fetched in ONE scan
   (fetch_gdelt_bq takes a mapping, not a single query) and the result is
   pinned. The daily-total denominator is a separate, free query — COUNT(*)
   over partition metadata touches no columns.

Language: DOC 2.0 had sourcelang:english. There is no equivalent filter here,
but matching English title phrases is an implicit and near-equivalent one.

The point-in-time caveat, stated plainly
----------------------------------------
GDELT indexes today's view of the past. Deduplication and removals mean a
re-fetch months later is a slightly different sample than a live system would
have seen. Daily aggregates over many stories blunt this, but nothing removes
it — so the snapshot is fetched ONCE by scripts/fetch_altdata.py, pinned with a
provenance sidecar, and every consumer reads that file. A re-fetch requires
--force and a ledger entry (design record §11).

Density is asymmetric between assets and that is a KNOWN CONFOUND (§1.1):
over 2022, gold averaged 20.9 distinct stories/day and silver only 5.3. Silver's
daily mean tone is therefore ~2x noisier by sampling alone, which predicts a
weaker silver result for a reason that has nothing to do with the pre-registered
industrial-demand mechanism. The features were deliberately NOT changed in
response — noise costs power, it cannot manufacture signal — but a weak silver
GDELT result is ambiguous between the two causes and must be reported as such.

Timing: a day's aggregate is treated as knowable at the next UTC midnight
(lag 1 day), and the current, incomplete UTC day is dropped at fetch time.
"""

from __future__ import annotations

import json
import re

import pandas as pd

BQ_TABLE = "gdelt-bq.gdeltv2.gkg_partitioned"

# Warms the 252-day z-window well before the first dataset row (2019-03-27,
# itself set by the crossasset warmup). The GKG v2 archive reaches back to 2015.
SNAPSHOT_START = "2018-01-01"

PUBLICATION_LAG_DAYS = 1
MAX_STALENESS = pd.Timedelta(days=3)

Z_WINDOW_DAYS = 252  # matches macro.py's convention, min_periods half of it
TONE_CHANGE_DAYS = 5
VOL_SMOOTH_DAYS = 7  # news flow dips at weekends; a 252d z does not remove
                     # that weekly seasonality, a 7-day mean does
FFILL_LIMIT_DAYS = 3

SNAPSHOT_COLUMNS = ("tone", "vol", "n_articles")


def build_title_regex(phrases: list[str]) -> str:
    """Phrases -> a regex matching them inside a lowercased <PAGE_TITLE>.

    Anchoring to the title tag is what makes this an entity filter rather than
    a substring search over the whole Extras blob, which also contains page
    links, author names and AMP URLs — 'silver' in an author byline or a
    related-links URL is not silver news.
    """
    if not phrases:
        raise ValueError("No title phrases: the query would match every article.")
    alternation = "|".join(re.escape(p.lower()) for p in phrases)
    return rf"<page_title>[^<]*({alternation})"


def _matched_stories_sql(regexes: dict[str, str], start: str, end: str) -> str:
    """One scan, every asset. Cost is bytes scanned in the columns touched, so
    a per-asset query would multiply a ~536 GB pull by the number of assets.

    Deduplication happens in `per_story`: articles are collapsed to one row per
    (day, asset, title) BEFORE any daily aggregate, so a wire story syndicated
    to 200 outlets counts once. An article whose title matches two assets
    legitimately counts for both, which is why this is a UNION ALL over flags
    rather than a single asset column.
    """
    flags = ",\n           ".join(
        f"REGEXP_CONTAINS(LOWER(Extras), r'{rx}') AS is_{asset}"
        for asset, rx in regexes.items()
    )
    any_match = " OR ".join(f"is_{asset}" for asset in regexes)
    per_asset = "\n  UNION ALL\n".join(
        f"  SELECT day, '{asset}' AS asset, title, AVG(tone) AS tone\n"
        f"  FROM matched WHERE is_{asset} AND title IS NOT NULL\n"
        f"  GROUP BY day, title"
        for asset in regexes
    )
    return f"""
WITH matched AS (
  SELECT DATE(_PARTITIONTIME) AS day,
         REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
         CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) AS tone,
         {flags}
  FROM `{BQ_TABLE}`
  WHERE DATE(_PARTITIONTIME) BETWEEN '{start}' AND '{end}'
),
filtered AS (SELECT * FROM matched WHERE {any_match}),
per_story AS (
{per_asset.replace("FROM matched", "FROM filtered")}
)
SELECT day, asset, COUNT(*) AS n_articles, AVG(tone) AS tone
FROM per_story
GROUP BY day, asset
ORDER BY day, asset
"""


def _daily_totals_sql(start: str, end: str) -> str:
    """Denominator for the attention share. Free: COUNT(*) grouped by the
    partition column reads table metadata and touches no data columns."""
    return f"""
SELECT DATE(_PARTITIONTIME) AS day, COUNT(*) AS total
FROM `{BQ_TABLE}`
WHERE DATE(_PARTITIONTIME) BETWEEN '{start}' AND '{end}'
GROUP BY day ORDER BY day
"""


def estimate_scan_gb(client, sql: str) -> float:
    """Bytes a query WOULD scan, without running it. The pull is a
    non-trivial fraction of a monthly quota, so it is always priced first."""
    from google.cloud import bigquery

    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False))
    return job.total_bytes_processed / 1e9


def fetch_gdelt_bq(
    phrases_by_asset: dict[str, list[str]],
    project: str,
    start: str = SNAPSHOT_START,
    end: str | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """Every asset's daily tone/attention series from ONE table scan.

    Returns {asset: DataFrame[tone, vol, n_articles]} on a daily index — the
    same contract the REST fetch produced, so everything downstream of the
    snapshot (derive_daily, alignment, the three features) is unchanged.
    """
    from google.cloud import bigquery

    end = end or str(pd.Timestamp.utcnow().normalize().date())
    regexes = {a: build_title_regex(p) for a, p in phrases_by_asset.items()}
    client = bigquery.Client(project=project)

    stories_sql = _matched_stories_sql(regexes, start, end)
    if verbose:
        gb = estimate_scan_gb(client, stories_sql)
        print(f"  scanning {gb:,.0f} GB ({gb / 10:.0f}% of the 1TB monthly free tier)"
              f" for {len(regexes)} asset(s), {start} -> {end}", flush=True)

    stories = client.query(stories_sql).result().to_dataframe()
    totals = client.query(_daily_totals_sql(start, end)).result().to_dataframe()

    out = {}
    for asset in phrases_by_asset:
        frame = assemble_snapshot(stories, totals, asset)
        if verbose:
            print(f"  {asset}: {int(frame['n_articles'].notna().sum()):,} days with "
                  f"coverage, {frame['n_articles'].median():.1f} stories/day median",
                  flush=True)
        out[asset] = frame
    return out


def assemble_snapshot(stories: pd.DataFrame, totals: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Query rows -> one asset's pinned daily frame. Pure, so the whole shape
    contract is testable without BigQuery.

    `stories` is one row per (day, asset) already deduplicated to distinct
    titles by the SQL; `totals` is one row per day of all GKG coverage.
    """
    block = stories[stories["asset"] == asset]
    if block.empty:
        raise ValueError(
            f"GDELT returned no matching stories for {asset!r} — check the title "
            "phrases. A silently empty result would become an all-NaN feature "
            "column rather than an error."
        )
    denominator = totals.set_index(pd.to_datetime(totals["day"]))["total"]

    frame = pd.DataFrame(index=pd.to_datetime(block["day"].to_numpy()))
    frame["tone"] = block["tone"].to_numpy(dtype=float)
    frame["n_articles"] = block["n_articles"].to_numpy(dtype=float)

    aligned = denominator.reindex(frame.index)
    if aligned.isna().any():
        # Both queries read the same partitions of the same table, so a day
        # with matched stories but no total is impossible unless the two
        # ranges drifted apart. Dividing by a silently missing denominator
        # would zero out attention on exactly those days.
        missing = aligned.index[aligned.isna()][:3]
        raise ValueError(
            f"GDELT {asset}: {int(aligned.isna().sum())} day(s) have matched "
            f"stories but no total-volume denominator (e.g. {list(missing)}). "
            "The two queries' date ranges disagree — do not normalize over it."
        )
    frame["vol"] = frame["n_articles"] / aligned.to_numpy()
    frame = frame[list(SNAPSHOT_COLUMNS)]

    # A continuous daily calendar: days with no matching story are absent from
    # the GROUP BY and become NaN rows, which the mask then treats identically
    # to an explicit zero.
    full_days = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    frame = frame.reindex(full_days)
    frame.index.name = "date"
    return _mask_zero_article_days(_drop_incomplete_day(frame))


def preview_titles(
    phrases: list[str], project: str, start: str, end: str, limit: int = 250
) -> list[dict]:
    """Sample matching headlines for human eyeballing — the query-validation
    step that must happen before any snapshot is pulled. Deduplicated by title,
    because 8 copies of one wire story tell you nothing 1 copy does not."""
    from google.cloud import bigquery

    sql = f"""
    SELECT title, ANY_VALUE(domain) AS domain, MIN(day) AS day, COUNT(*) AS copies
    FROM (
      SELECT DATE(_PARTITIONTIME) AS day,
             REGEXP_EXTRACT(Extras, r'<PAGE_TITLE>(.*?)</PAGE_TITLE>') AS title,
             SourceCommonName AS domain
      FROM `{BQ_TABLE}`
      WHERE DATE(_PARTITIONTIME) BETWEEN '{start}' AND '{end}'
        AND REGEXP_CONTAINS(LOWER(Extras), r'{build_title_regex(phrases)}')
    )
    WHERE title IS NOT NULL
    GROUP BY title ORDER BY day LIMIT {limit}
    """
    client = bigquery.Client(project=project)
    return [dict(row) for row in client.query(sql).result()]


def _drop_incomplete_day(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the current UTC day: its aggregate only covers articles so far."""
    now = now if now is not None else pd.Timestamp.utcnow().tz_localize(None)
    return frame.loc[frame.index < now.normalize()]


def _mask_zero_article_days(frame: pd.DataFrame) -> pd.DataFrame:
    """A day with no matching stories has no tone — and no observed attention
    either (an indexing gap is indistinguishable from a genuine zero, so both
    are treated as missing rather than inventing a 0.0)."""
    masked = frame.copy()
    dead = masked["n_articles"].isna() | (masked["n_articles"] == 0)
    masked.loc[dead, ["tone", "vol"]] = float("nan")
    return masked


def derive_daily(gdelt: pd.DataFrame) -> pd.DataFrame:
    """The three pre-registered features on the daily index — the one place
    the formulas live (docs/step2_design.md §3). Short gaps are bridged with a
    limited ffill; anything longer stays NaN and the alignment tolerance turns
    it into missing features rather than fossils."""
    daily = gdelt.reindex(pd.date_range(gdelt.index.min(), gdelt.index.max(), freq="D"))
    tone = daily["tone"].ffill(limit=FFILL_LIMIT_DAYS)
    vol = daily["vol"].ffill(limit=FFILL_LIMIT_DAYS)

    derived = pd.DataFrame(index=daily.index)
    derived["gdelt_tone_z"] = _rolling_z(tone)
    derived["gdelt_tone_chg_5d"] = tone.diff(TONE_CHANGE_DAYS)
    derived["gdelt_vol_smooth7_z"] = _rolling_z(
        vol.rolling(VOL_SMOOTH_DAYS, min_periods=VOL_SMOOTH_DAYS).mean()
    )
    return derived


def build_gdelt_features(gdelt: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive the daily features and align them to bars."""
    return _align_to_bars(derive_daily(gdelt), bar_index)


def _rolling_z(series: pd.Series) -> pd.Series:
    rolling = series.rolling(Z_WINDOW_DAYS, min_periods=Z_WINDOW_DAYS // 2)
    return (series - rolling.mean()) / rolling.std()


def _align_to_bars(derived: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join daily observations onto bars: a bar on day D sees at most
    day D-1's complete aggregate. Same mechanism as macro.py/cot.py, with the
    all-NaN rows dropped first so the staleness tolerance measures distance to
    the last REAL observation, not to a gap row."""
    if derived.index.tz is None:
        derived = derived.copy()
        derived.index = derived.index.tz_localize("UTC")

    block = derived.dropna(how="all").copy()
    block.index = block.index + pd.Timedelta(days=PUBLICATION_LAG_DAYS)

    bars = pd.DataFrame(index=bar_index).sort_index()
    return pd.merge_asof(
        bars,
        block,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=MAX_STALENESS,
    )


def gdelt_feature_names() -> list[str]:
    return ["gdelt_tone_z", "gdelt_tone_chg_5d", "gdelt_vol_smooth7_z"]


def load_gdelt_snapshot(cfg) -> pd.DataFrame:
    """Read the pinned snapshot, refusing anything without matching provenance."""
    path = cfg.gdelt_snapshot_path
    if not path.exists():
        raise FileNotFoundError(
            f"features.gdelt is enabled but there is no GDELT snapshot at {path}.\n"
            "Snapshots are fetched once and pinned (GDELT is not point-in-time "
            "reproducible), and ALL assets come from one scan:\n"
            "    python scripts/fetch_altdata.py --source gdelt --project <gcp-project>"
        )

    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"GDELT snapshot {path.name} has no sidecar {meta_path.name}, so its "
            "provenance is unknown. Refetch it with scripts/fetch_altdata.py."
        )
    _check_gdelt_sidecar(json.loads(meta_path.read_text()), cfg)
    return pd.read_parquet(path)


def _check_gdelt_sidecar(meta: dict, cfg) -> None:
    """Pure so the refusal is testable without touching the filesystem."""
    have, want = meta.get("title_phrases"), list(cfg.gdelt.title_phrases)
    if have != want:
        raise ValueError(
            f"GDELT snapshot was fetched with title phrases {have!r} but the config "
            f"says {want!r}. The snapshot IS the sample — either restore the yaml or "
            "refetch with scripts/fetch_altdata.py --force and record it in "
            "docs/step2_design.md §11."
        )
