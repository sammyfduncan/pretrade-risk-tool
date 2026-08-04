"""Scheduled event timestamps and DST-aware blackout widths."""

import pandas as pd
import pytest

from commodity_ml.intraday.event_blackouts import (
    scheduled_blackout_state,
    scheduled_entry_blackouts,
)


def test_bls_release_uses_new_york_dst_and_frozen_half_hour_window():
    blackouts = scheduled_entry_blackouts(
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2021-07-01", tz="UTC"),
    )
    january_nfp = next(
        item
        for item in blackouts
        if item.reason == "nfp"
        and item.start.date() == pd.Timestamp("2021-01-08").date()
    )
    june_nfp = next(
        item
        for item in blackouts
        if item.reason == "nfp"
        and item.start.date() == pd.Timestamp("2021-06-04").date()
    )

    assert january_nfp.start == pd.Timestamp("2021-01-08 13:00", tz="UTC")
    assert january_nfp.end == pd.Timestamp("2021-01-08 14:00", tz="UTC")
    assert june_nfp.start == pd.Timestamp("2021-06-04 12:00", tz="UTC")
    assert june_nfp.end == pd.Timestamp("2021-06-04 13:00", tz="UTC")


def test_fomc_release_has_thirty_before_and_sixty_after():
    blackouts = scheduled_entry_blackouts(
        pd.Timestamp("2021-06-01", tz="UTC"),
        pd.Timestamp("2021-07-01", tz="UTC"),
    )
    fomc = next(item for item in blackouts if item.reason == "fomc")

    assert fomc.start == pd.Timestamp("2021-06-16 17:30", tz="UTC")
    assert fomc.end == pd.Timestamp("2021-06-16 19:00", tz="UTC")


def test_unverified_calendar_range_is_refused():
    with pytest.raises(ValueError, match="coverage"):
        scheduled_entry_blackouts(
            pd.Timestamp("2019-01-01", tz="UTC"),
            pd.Timestamp("2020-02-01", tz="UTC"),
        )


def test_current_state_uses_verified_2026_schedule_and_exact_half_open_window():
    # August CPI is 08:30 New York = 12:30 UTC under EDT.
    assert scheduled_blackout_state(
        pd.Timestamp("2026-08-12 12:00", tz="UTC")
    ) == (True, ("cpi",))
    assert scheduled_blackout_state(
        pd.Timestamp("2026-08-12 13:00", tz="UTC")
    ) == (False, ())
    assert scheduled_blackout_state(
        pd.Timestamp("2026-08-04 12:30", tz="UTC")
    ) == (False, ())
