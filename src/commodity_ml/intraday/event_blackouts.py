"""Exact scheduled-event entry blackouts for the intraday execution engine."""

from __future__ import annotations

import pandas as pd

from commodity_ml.features import event_dates
from commodity_ml.intraday.execution import EntryBlackout

NEW_YORK = "America/New_York"


def scheduled_blackout_state(
    decision_time: pd.Timestamp,
) -> tuple[bool, tuple[str, ...]]:
    """Return the active exact-time blackout state from verified schedules."""

    current = _utc(decision_time)
    # Widest registered window is FOMC: 30 minutes before through 60 after.
    windows = scheduled_entry_blackouts(
        current - pd.Timedelta(hours=2),
        current + pd.Timedelta(hours=2),
    )
    reasons = tuple(
        item.reason for item in windows if item.start <= current < item.end
    )
    return bool(reasons), reasons


def scheduled_release_times(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Return verified regular release timestamps on a half-open UTC range.

    The index is the exact scheduled release instant.  Outcomes are
    deliberately absent: this is a point-in-time schedule, not news data.
    """

    start_utc, end_utc = _verified_bounds(start, end)
    rows: list[tuple[pd.Timestamp, str]] = []
    for kind, dates in (
        ("cpi", event_dates.CPI_RELEASES),
        ("nfp", event_dates.NFP_RELEASES),
    ):
        rows.extend((_release_timestamp(value, "08:30"), kind) for value in dates)
    rows.extend(
        (_release_timestamp(value, "14:00"), "fomc")
        for value in event_dates.fomc_days()
    )
    selected = sorted(
        (timestamp, kind)
        for timestamp, kind in rows
        if start_utc <= timestamp < end_utc
    )
    index = pd.DatetimeIndex(
        [timestamp for timestamp, _ in selected],
        tz="UTC",
        name="release_time",
    )
    return pd.DataFrame(
        {"kind": [kind for _, kind in selected]},
        index=index,
    )


def scheduled_entry_blackouts(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[EntryBlackout, ...]:
    """Build the frozen CPI/NFP/FOMC windows over verified calendar coverage.

    BLS CPI and Employment Situation releases are scheduled for 08:30 New York
    time. FOMC statements are scheduled for 14:00 New York time. Localizing in
    America/New_York before converting to UTC handles historical DST changes.
    """

    start_utc, end_utc = _verified_bounds(start, end)

    blackouts: list[EntryBlackout] = []
    releases = scheduled_release_times(start_utc, end_utc)
    for release, row in releases.iterrows():
        kind = str(row["kind"])
        after = 60 if kind == "fomc" else 30
        blackouts.append(
            EntryBlackout(
                start=release - pd.Timedelta(minutes=30),
                end=release + pd.Timedelta(minutes=after),
                reason=kind,
            )
        )
    return tuple(sorted(blackouts, key=lambda item: item.start))


def _release_timestamp(value, clock: str) -> pd.Timestamp:
    day = pd.Timestamp(value).strftime("%Y-%m-%d")
    return pd.Timestamp(f"{day} {clock}", tz=NEW_YORK).tz_convert("UTC")


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("Event blackout bounds must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _verified_bounds(
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_utc, end_utc = _utc(start), _utc(end)
    if start_utc >= end_utc:
        raise ValueError("Event blackout range requires start < end")
    coverage_start = event_dates.COVERAGE_START.tz_localize("UTC")
    coverage_end = (
        event_dates.COVERAGE_END + pd.Timedelta(days=1)
    ).tz_localize("UTC")
    if start_utc < coverage_start or end_utc > coverage_end:
        raise ValueError(
            "Intraday event blackout coverage is verified only over "
            f"[{coverage_start}, {coverage_end})"
        )
    return start_utc, end_utc
