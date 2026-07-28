"""GDELT news tone and attention — step 2's unstructured alt-data leg.

What this measures
------------------
GDELT machine-reads global news continuously and scores each article's tone
(positive minus negative language, roughly −10..+10). For one per-asset query
(config: gdelt.query — validated by eyeballing article previews before any
timeline is fetched, see scripts/fetch_altdata.py --preview) we aggregate three
daily series from the DOC 2.0 API:

  tone        timelinetone   — average tone of matching articles that day
  vol         timelinevol    — matching volume as a share of ALL GDELT coverage;
                               normalized, so the index growing over the years
                               does not masquerade as rising attention
  n_articles  timelinevolraw — raw article count; kept only as a mask, because
                               a zero-article day has NO tone and tone 0.0 is
                               otherwise a perfectly valid value

The pre-registered hypothesis (docs/step2_design.md §1): falling tone (fear)
and rising attention lead gold strength at 5–20 days — and attention should
predict forward *volatility* regardless of direction, which would be useful for
step 4's sizing even if direction fails.

The point-in-time caveat, stated plainly
----------------------------------------
The API queries TODAY'S index about the past. Deduplication and removals mean
a re-fetch months later is a slightly different sample than a live system
would have seen. Daily aggregates over many articles blunt this, but nothing
removes it — so the snapshot is fetched ONCE by scripts/fetch_altdata.py,
pinned with a provenance sidecar, and every consumer reads that file. A
re-fetch requires --force and a ledger entry (design record §11).

Timing: a day's aggregate is treated as knowable at the next UTC midnight
(lag 1 day), and the current, incomplete UTC day is dropped at fetch time — a
timeline point for "today" only counts articles indexed so far.
"""

from __future__ import annotations

import json
import re
import time

import pandas as pd

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Warms the 252-day z-window well before the first dataset row (2019-03-27,
# itself set by the crossasset warmup). The archive reaches back to 2017-01-01.
SNAPSHOT_START = "2018-01-01"

PUBLICATION_LAG_DAYS = 1
MAX_STALENESS = pd.Timedelta(days=3)

Z_WINDOW_DAYS = 252  # matches macro.py's convention, min_periods half of it
TONE_CHANGE_DAYS = 5
VOL_SMOOTH_DAYS = 7  # news flow dips at weekends; a 252d z does not remove
                     # that weekly seasonality, a 7-day mean does
FFILL_LIMIT_DAYS = 3

TIMELINE_MODES = {"tone": "timelinetone", "vol": "timelinevol", "n_articles": "timelinevolraw"}

# GDELT throttles aggressively (HTTP 429 well under one request per second);
# its informal guidance is one request every ~5 seconds, and a burst can land
# the client in a multi-minute penalty box — observed: a 429 that outlasted
# 40s of backoff. The schedule below tolerates ~8 minutes of cool-down.
_REQUEST_PAUSE_S = 5.0
_BACKOFF_S = (15, 30, 60, 120, 240)


def fetch_gdelt(
    query: str, start: str = SNAPSHOT_START, end: str | None = None, verbose: bool = True
) -> pd.DataFrame:
    """Download the three daily timelines for one query, in yearly chunks.

    Chunking is politeness, not necessity — the API serves multi-year ranges,
    but GDELT is a free public service and a failed 8-year request restarts
    from zero. Each chunk is retried with backoff.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.utcnow().tz_localize(None)

    pieces = []
    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = min(pd.Timestamp(chunk_start.year, 12, 31), end_ts)
        if verbose:
            print(f"  GDELT {chunk_start.date()} -> {chunk_end.date()}")

        columns = {}
        for name, mode in TIMELINE_MODES.items():
            payload = _request_with_retries(
                query, mode, chunk_start, chunk_end + pd.Timedelta(hours=23, minutes=59)
            )
            columns[name] = _parse_timeline(payload, name)
            time.sleep(_REQUEST_PAUSE_S)
        pieces.append(pd.DataFrame(columns))
        chunk_start = chunk_end + pd.Timedelta(days=1)

    frame = pd.concat(pieces).sort_index()
    if frame.index.has_duplicates:
        raise ValueError(
            "GDELT chunks overlap — duplicate dates in the assembled timeline. "
            "The chunking arithmetic is wrong; do not average over it."
        )

    # A continuous daily calendar: days the API omitted become NaN rows, which
    # the zero-article mask treats the same as zero-article days.
    full_days = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    frame = frame.reindex(full_days)
    frame.index.name = "date"

    frame = _drop_incomplete_day(frame)
    return _mask_zero_article_days(frame)


def _request_with_retries(
    query: str,
    mode: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    extra: dict | None = None,
) -> dict:
    import requests

    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "STARTDATETIME": start.strftime("%Y%m%d%H%M%S"),
        "ENDDATETIME": end.strftime("%Y%m%d%H%M%S"),
        **(extra or {}),
    }
    last_error: Exception | None = None
    for wait in _BACKOFF_S:
        try:
            response = requests.get(GDELT_URL, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(wait)
    raise ConnectionError(
        f"GDELT {mode} request failed after {len(_BACKOFF_S)} attempts: {last_error}"
    )


def _parse_timeline(payload: dict, name: str) -> pd.Series:
    """DOC 2.0 timeline JSON -> daily series. Pure, tested on canned payloads.

    Shape: {"timeline": [{"series": ..., "data": [{"date": ..., "value": ...}]}]}.
    Date strings vary ("20180101T000000Z" vs "20180101000000"), so parsing
    strips to digits rather than trusting one format.
    """
    timeline = payload.get("timeline")
    if not timeline or "data" not in timeline[0]:
        raise ValueError(
            f"GDELT response for {name!r} has no timeline data — usually a "
            "malformed query (unbalanced quotes/parens) or an API error page."
        )
    points = timeline[0]["data"]
    dates = pd.to_datetime(
        [re.sub(r"\D", "", p["date"])[:8] for p in points], format="%Y%m%d"
    )
    series = pd.Series([float(p["value"]) for p in points], index=dates, name=name)
    # Sub-daily points appear if a chunk boundary lands mid-day; collapse to
    # daily means (tone) — counts/volume shares are per-point averages too at
    # this API resolution, so mean is the right collapse for all three.
    return series.groupby(level=0).mean()


def _drop_incomplete_day(frame: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Drop the current UTC day: its aggregate only covers articles so far."""
    now = now if now is not None else pd.Timestamp.utcnow().tz_localize(None)
    return frame.loc[frame.index < now.normalize()]


def _mask_zero_article_days(frame: pd.DataFrame) -> pd.DataFrame:
    """A day with no matching articles has no tone — and no observed attention
    either (an API gap is indistinguishable from a genuine zero, so both are
    treated as missing rather than inventing a 0.0)."""
    masked = frame.copy()
    dead = masked["n_articles"].isna() | (masked["n_articles"] == 0)
    masked.loc[dead, ["tone", "vol"]] = float("nan")
    return masked


def build_gdelt_features(gdelt: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive the three pre-registered features and align to bars.

    Formulas fixed in docs/step2_design.md §3. Short gaps are bridged with a
    limited ffill; anything longer stays NaN and the alignment tolerance turns
    it into missing features rather than fossils.
    """
    daily = gdelt.reindex(pd.date_range(gdelt.index.min(), gdelt.index.max(), freq="D"))
    tone = daily["tone"].ffill(limit=FFILL_LIMIT_DAYS)
    vol = daily["vol"].ffill(limit=FFILL_LIMIT_DAYS)

    derived = pd.DataFrame(index=daily.index)
    derived["gdelt_tone_z"] = _rolling_z(tone)
    derived["gdelt_tone_chg_5d"] = tone.diff(TONE_CHANGE_DAYS)
    derived["gdelt_vol_smooth7_z"] = _rolling_z(
        vol.rolling(VOL_SMOOTH_DAYS, min_periods=VOL_SMOOTH_DAYS).mean()
    )

    return _align_to_bars(derived, bar_index)


def _rolling_z(series: pd.Series) -> pd.Series:
    rolling = series.rolling(Z_WINDOW_DAYS, min_periods=Z_WINDOW_DAYS // 2)
    return (series - rolling.mean()) / rolling.std()


def _align_to_bars(derived: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join daily observations onto bars: a bar on day D sees at most
    day D−1's complete aggregate. Same mechanism as macro.py/cot.py, with the
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


def preview_articles(query: str, start: str, end: str, max_records: int = 250) -> list[dict]:
    """Sample matching articles for human eyeballing — the query-validation
    step that must happen before any timeline snapshot is pulled."""
    payload = _request_with_retries(
        query,
        "artlist",
        pd.Timestamp(start),
        pd.Timestamp(end),
        extra={"maxrecords": max_records},
    )
    return payload.get("articles", [])


def load_gdelt_snapshot(cfg) -> pd.DataFrame:
    """Read the pinned snapshot, refusing anything without matching provenance."""
    path = cfg.gdelt_snapshot_path
    if not path.exists():
        raise FileNotFoundError(
            f"features.gdelt is enabled but there is no GDELT snapshot at {path}.\n"
            "Snapshots are fetched once and pinned (GDELT is not point-in-time "
            "reproducible):\n"
            f"    python scripts/fetch_altdata.py --config config/{cfg.asset}.yaml --source gdelt"
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
    have, want = meta.get("query"), cfg.gdelt.query
    if have != want:
        raise ValueError(
            f"GDELT snapshot was fetched with query {have!r} but the config says "
            f"{want!r}. The snapshot IS the sample — either restore the yaml or "
            "refetch with scripts/fetch_altdata.py --force and record it in "
            "docs/step2_design.md §11."
        )
