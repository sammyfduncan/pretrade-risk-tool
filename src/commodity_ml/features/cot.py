"""CFTC Commitments of Traders positioning — step 2's structured alt-data leg.

What this measures
------------------
Every Tuesday the CFTC records who holds COMEX gold futures, split by trader
type. Two groups matter here. "Managed money" is hedge funds and CTAs — fast,
momentum-chasing capital whose positioning stretches to extremes. "Producer/
merchant" is miners and refiners hedging physical metal — slow capital that
mechanically takes the other side. The pre-registered hypothesis
(docs/step2_design.md §1) is *crowdedness*: when managed money is unusually
net-long relative to its own two-year history, the marginal buyer is already
in, and forward returns should be weaker. This is positioning, not sentiment —
it is deliberately a separate feature family from GDELT news tone.

Net position is expressed as a fraction of open interest before standardizing,
so a growing market does not masquerade as growing conviction.

Lookahead risk lives in the release schedule
--------------------------------------------
Reports are *dated* Tuesday but *released* Friday ~15:30 ET — and alignment
shifts by whole days, so the arithmetic is unforgiving: a 3-day lag would leak
all of Friday's bars, and a 6-day lag would leak the Monday bars of holiday
weeks (when the release slips a business day). Seven days — next Tuesday
00:00 UTC — is the smallest whole-day lag that is safe in both cases. House
rule: overstating a lag only weakens results; understating one manufactures
them. COT features are step functions for a week regardless, so the cost of
the extra caution is negligible.

A fixed lag also assumes the schedule held. It did not: publication has been
interrupted (see PUBLICATION_INTERRUPTIONS), and inside those windows a fixed
lag would hand the backtest reports that no trader had. The table caps when
each affected report became knowable; combined with MAX_STALENESS this turns
the gap into honest NaNs rather than manufactured foresight.

Unlike macro (tiny, stable, refetched each build), COT is fetched once by
scripts/fetch_altdata.py into a pinned snapshot that build_dataset.py and
predict.py only ever read. The CFTC occasionally amends published figures in
place; the snapshot reflects amended values — accepted and documented in the
design record rather than engineered around.
"""

from __future__ import annotations

import json

import pandas as pd

COT_URL = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"

# Socrata's default $limit is 1,000 — BELOW gold's ~1,050 weekly reports since
# 2006, and truncation is silent. fetch_cot passes this limit explicitly and
# refuses a response that fills it.
FETCH_LIMIT = 5000

PUBLICATION_LAG_DAYS = 7

# One skipped release means up to 14 days between knowable report dates;
# anything staler than this becomes NaN rather than a fossil.
MAX_STALENESS = pd.Timedelta(days=15)

Z_WINDOW_WEEKS = 104  # ~2 years of weekly reports, min_periods half of it
CHANGE_WINDOW_WEEKS = 4

# Socrata field -> short name. Pins the dataset's real field spellings in one
# place — including the double underscore in swap__positions_short_all, which
# is not a typo here because it is not a typo there.
FIELDS = {
    "report_date_as_yyyy_mm_dd": "report_date",
    "open_interest_all": "oi",
    "m_money_positions_long_all": "mm_long",
    "m_money_positions_short_all": "mm_short",
    "prod_merc_positions_long": "prod_long",
    "prod_merc_positions_short": "prod_short",
    "swap_positions_long_all": "swap_long",
    "swap__positions_short_all": "swap_short",
}

# Windows where the weekly release schedule broke down and the fixed lag would
# lie. (first_report_date, last_report_date, knowable_from): every report dated
# inside [first, last] is treated as unknowable before `knowable_from` 00:00 UTC.
#
# ION Cleared Derivatives cyberattack: the report dated 2023-01-31 (scheduled
# for Friday 2023-02-03) was postponed; publication resumed 2023-02-24 and the
# backlog was released sequentially, with the schedule fully current on
# 2023-03-17 (CFTC press releases 8655-23 / 8662-23 and the Historical Special
# Announcements page). Using the day after full catch-up for every affected
# report is deliberately MORE conservative than the per-report reality.
#
# The Oct/Nov 2025 government-shutdown backlog falls inside the HOLDOUT — this
# table must be extended to cover it before the holdout is ever spent with COT
# enabled (docs/step2_design.md §2.1).
PUBLICATION_INTERRUPTIONS = (
    ("2023-01-31", "2023-03-07", "2023-03-18"),
)


def fetch_cot(contract_code: str, verbose: bool = True) -> pd.DataFrame:
    """Download the full weekly history for one contract from the CFTC API.

    Full history since 2006 is ~1,050 rows — one request, and warmup is never
    the constraint, so there is no start parameter to get wrong.
    """
    import requests

    if verbose:
        print(f"  CFTC disaggregated futures-only, contract {contract_code}")

    response = requests.get(
        COT_URL,
        params={
            "cftc_contract_market_code": contract_code,
            "$select": ",".join(FIELDS),
            "$order": "report_date_as_yyyy_mm_dd",
            "$limit": FETCH_LIMIT,
        },
        timeout=60,
    )
    response.raise_for_status()
    rows = response.json()

    if not rows:
        raise ValueError(
            f"CFTC returned no rows for contract code {contract_code!r}. "
            "Codes are per-market — verify it against the API before trusting it "
            "(gold COMEX is 088691)."
        )
    if len(rows) >= FETCH_LIMIT:
        raise ValueError(
            f"CFTC response filled the ${FETCH_LIMIT} row limit — the history has "
            "outgrown a single request and silent truncation is now possible. "
            "Raise FETCH_LIMIT or page with $offset."
        )
    return _parse_socrata_rows(rows)


def _parse_socrata_rows(rows: list[dict]) -> pd.DataFrame:
    """Socrata JSON -> typed weekly frame. Pure, so it is testable offline.

    Socrata returns every value as a string and omits null fields per record,
    so types and presence are both checked rather than assumed.
    """
    df = pd.DataFrame.from_records(rows)

    missing = [f for f in FIELDS if f not in df.columns]
    if missing:
        raise ValueError(
            f"CFTC response is missing expected fields: {missing}. "
            "The dataset schema has drifted — reconcile FIELDS before refetching."
        )

    df = df[list(FIELDS)].rename(columns=FIELDS)
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.set_index("report_date").sort_index()

    if df.index.has_duplicates:
        dupes = df.index[df.index.duplicated()].unique()
        raise ValueError(
            f"Duplicate report dates for one contract: {list(dupes[:5])}. "
            "One market should produce one row per week — the query is wrong."
        )
    return df.astype(float)


def build_cot_features(cot: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive the three pre-registered positioning features and align to bars.

    Everything is computed on the weekly report index first (rolling windows
    are backward-looking there), then aligned; raw positions never leave this
    module. Formulas are fixed in docs/step2_design.md §3 — windows are repo
    conventions, not tunables.
    """
    mm_net_frac = (cot["mm_long"] - cot["mm_short"]) / cot["oi"]
    prod_net_frac = (cot["prod_long"] - cot["prod_short"]) / cot["oi"]

    derived = pd.DataFrame(index=cot.index)
    derived["cot_mm_net_z"] = _rolling_z(mm_net_frac)
    derived["cot_mm_net_chg_4w"] = mm_net_frac.diff(CHANGE_WINDOW_WEEKS)
    derived["cot_prod_net_z"] = _rolling_z(prod_net_frac)

    return _align_to_bars(derived, bar_index)


def _rolling_z(series: pd.Series) -> pd.Series:
    rolling = series.rolling(Z_WINDOW_WEEKS, min_periods=Z_WINDOW_WEEKS // 2)
    return (series - rolling.mean()) / rolling.std()


def _align_to_bars(derived: pd.DataFrame, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of join weekly observations onto bars, respecting when each report
    actually became knowable.

    Same mechanism as macro.py's aligner (shift the observation index, then
    merge_asof backward) with two additions macro does not need: the
    publication-interruption override, and a staleness tolerance so a broken
    release schedule yields NaN instead of a months-old value. Kept local
    rather than shared — macro.py is freeze-sensitive and its aligner handles
    per-column lags this module does not have.
    """
    if derived.index.tz is None:
        derived = derived.copy()
        derived.index = derived.index.tz_localize("UTC")

    knowable = derived.index + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    for first, last, knowable_from in PUBLICATION_INTERRUPTIONS:
        affected = (derived.index >= pd.Timestamp(first, tz="UTC")) & (
            derived.index <= pd.Timestamp(last, tz="UTC")
        )
        override = pd.Timestamp(knowable_from, tz="UTC")
        knowable = pd.DatetimeIndex(
            [max(k, override) if hit else k for k, hit in zip(knowable, affected)]
        )

    block = derived.copy()
    block.index = knowable
    # An interruption makes several reports knowable at the same instant;
    # merge_asof backward on duplicate keys takes the last row, i.e. the most
    # recent report — exactly what a trader reading the catch-up batch would
    # use. The index stays non-decreasing because the override only pushes
    # dates forward, so only a stability-preserving sort is safe here.
    block = block.sort_index(kind="stable")

    bars = pd.DataFrame(index=bar_index).sort_index()
    merged = pd.merge_asof(
        bars,
        block,
        left_index=True,
        right_index=True,
        direction="backward",
        tolerance=MAX_STALENESS,
    )
    return merged


def cot_feature_names() -> list[str]:
    return ["cot_mm_net_z", "cot_mm_net_chg_4w", "cot_prod_net_z"]


def load_cot_snapshot(cfg) -> pd.DataFrame:
    """Read the pinned snapshot, refusing anything without matching provenance."""
    path = cfg.cot_snapshot_path
    if not path.exists():
        raise FileNotFoundError(
            f"features.cot is enabled but there is no COT snapshot at {path}.\n"
            "Snapshots are fetched once and pinned:\n"
            f"    python scripts/fetch_altdata.py --config config/{cfg.asset}.yaml --source cot"
        )

    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"COT snapshot {path.name} has no sidecar {meta_path.name}, so its "
            "provenance is unknown. Refetch it with scripts/fetch_altdata.py."
        )
    _check_cot_sidecar(json.loads(meta_path.read_text()), cfg)
    return pd.read_parquet(path)


def _check_cot_sidecar(meta: dict, cfg) -> None:
    """Pure so the refusal is testable without touching the filesystem."""
    have, want = meta.get("contract_code"), cfg.cot.contract_code
    if have != want:
        raise ValueError(
            f"COT snapshot was fetched for contract {have!r} but the config says "
            f"{want!r}. Either restore the yaml or refetch with "
            "scripts/fetch_altdata.py --force."
        )
