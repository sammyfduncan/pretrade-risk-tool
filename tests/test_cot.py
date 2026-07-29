"""CFTC COT positioning features — offline, on synthetic weekly reports.

Mirrors test_macro.py's battery (publication-lag pin, truncation causality,
warmup NaNs) plus the failure modes weekly release schedules add: stale
reports fossilizing, and the Feb-2023 publication interruption, where a fixed
lag would hand the backtest reports no trader had.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_cot

from commodity_ml.config import AssetConfig, COTConfig
from commodity_ml.features.cot import (
    FIELDS,
    Z_WINDOW_WEEKS,
    _check_cot_sidecar,
    _parse_socrata_rows,
    build_cot_features,
    cot_feature_names,
)


def bars(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="4h", tz="UTC")


def independent_derived(cot: pd.DataFrame) -> pd.DataFrame:
    """The §3 formulas recomputed with plain pandas, independent of the module,
    so a sign swap or wrong window in cot.py cannot hide behind itself."""
    out = pd.DataFrame(index=cot.index)
    for name, net in (
        ("cot_mm_net", (cot["mm_long"] - cot["mm_short"]) / cot["oi"]),
        ("cot_prod_net", (cot["prod_long"] - cot["prod_short"]) / cot["oi"]),
    ):
        roll = net.rolling(Z_WINDOW_WEEKS, min_periods=Z_WINDOW_WEEKS // 2)
        out[f"{name}_z"] = (net - roll.mean()) / roll.std()
    out["cot_mm_net_chg_4w"] = ((cot["mm_long"] - cot["mm_short"]) / cot["oi"]).diff(4)
    return out


@pytest.fixture(scope="module")
def cot() -> pd.DataFrame:
    return make_cot()


def test_every_declared_feature_is_produced(cot):
    index = bars("2021-01-04", "2021-03-01")
    features = build_cot_features(cot, index)

    assert list(features.columns) == cot_feature_names()
    assert features.index.equals(index)


def test_values_respect_publication_lag(cot):
    """A Tuesday-dated report is knowable 7 days later. The last bar here is a
    Tuesday evening: the naive join would already use that morning's report
    date; the lagged join must still be on the previous week's."""
    index = bars("2021-05-03", "2021-06-29 20:00")
    features = build_cot_features(cot, index)
    expected = independent_derived(cot)

    lagged = expected.loc["2021-06-22"]  # knowable 2021-06-29 00:00 UTC
    naive = expected.loc["2021-06-29"]  # dated before the last bar, NOT knowable
    for feature in cot_feature_names():
        assert lagged[feature] != naive[feature], "fixture too flat to catch the bug"
        assert features[feature].iloc[-1] == pytest.approx(lagged[feature], rel=1e-9)


def test_cot_features_are_causal(cot):
    """Truncating future reports must not change past bars."""
    index = bars("2019-01-07", "2019-06-28")
    full = build_cot_features(cot, index)
    truncated = build_cot_features(cot.loc[:"2019-10-15"], index)

    pd.testing.assert_frame_equal(full, truncated, rtol=1e-9)


def test_warmup_produces_nans_not_silent_zeros():
    """Before the 104-week z-window has min_periods, the z features must be
    NaN (dropped by the dataset builder), never fabricated values."""
    young = make_cot(n_weeks=30)
    features = build_cot_features(young, bars("2015-07-01", "2015-08-05"))

    assert features["cot_mm_net_z"].isna().all()
    assert features["cot_prod_net_z"].isna().all()
    # The 4-week change needs only 5 reports — it should be flowing already.
    assert features["cot_mm_net_chg_4w"].notna().any()


def test_stale_reports_produce_nan_not_fossils(cot):
    """If publication stops, bars beyond MAX_STALENESS must go NaN rather than
    carrying a months-old report forward as if it were current."""
    ends_march = cot.loc[:"2021-03-30"]  # last report knowable 2021-04-06
    features = build_cot_features(ends_march, bars("2021-03-01", "2021-05-15"))

    fresh = features.loc[: pd.Timestamp("2021-04-05", tz="UTC")]
    stale = features.loc[pd.Timestamp("2021-04-25", tz="UTC") :]
    assert fresh.notna().all().all()
    assert stale.isna().all().all()


def test_features_step_only_on_knowable_dates(cot):
    """Weekly data on H4 bars is a step function: values may change only at
    Tuesday 00:00 UTC, when the week-old report becomes knowable."""
    features = build_cot_features(cot, bars("2021-05-05", "2021-05-26"))
    z = features["cot_mm_net_z"]

    changed = z.index[(z != z.shift()) & z.notna() & z.shift().notna()]
    assert len(changed) >= 2, "range should span several report releases"
    for t in changed:
        assert t.dayofweek == 1 and t.hour == 0, f"feature stepped mid-week at {t}"


def test_interruption_delays_availability(cot):
    """The ION window, against the real PUBLICATION_INTERRUPTIONS table.

    Under the normal 7-day lag the 2023-01-31 report would be knowable
    2023-02-07 — but it was not published until the catch-up. A February bar
    must still carry the 2023-01-24 report; the staleness guard must then blank
    March's first half; and after 2023-03-18 the backlog (through the Mar-07
    report) becomes available at once.
    """
    features = build_cot_features(cot, bars("2023-01-02", "2023-04-10"))
    expected = independent_derived(cot)
    z = features["cot_mm_net_z"]

    feb_bar = z.loc[pd.Timestamp("2023-02-10 12:00", tz="UTC")]
    assert feb_bar == pytest.approx(expected.loc["2023-01-24", "cot_mm_net_z"], rel=1e-9)
    assert feb_bar != pytest.approx(expected.loc["2023-01-31", "cot_mm_net_z"], rel=1e-9)

    assert np.isnan(z.loc[pd.Timestamp("2023-03-01 12:00", tz="UTC")]), (
        "bars beyond staleness during the outage must be NaN, not a January fossil"
    )

    after_catchup = z.loc[pd.Timestamp("2023-03-20 12:00", tz="UTC")]
    assert after_catchup == pytest.approx(
        expected.loc["2023-03-07", "cot_mm_net_z"], rel=1e-9
    ), "after catch-up the most recent backlogged report should be in effect"

    back_to_normal = z.loc[pd.Timestamp("2023-03-22 12:00", tz="UTC")]
    assert back_to_normal == pytest.approx(
        expected.loc["2023-03-14", "cot_mm_net_z"], rel=1e-9
    ), "the first post-outage report should follow the normal 7-day lag again"


def test_net_fraction_sign(cot):
    """More managed-money longs must push cot_mm_net_z up — catches a
    long/short swap, which no causality test would notice."""
    index = bars("2021-05-03", "2021-06-28")
    baseline = build_cot_features(cot, index)["cot_mm_net_z"].iloc[-1]

    bumped = cot.copy()
    recent = bumped.index >= "2021-05-01"
    bumped.loc[recent, "mm_long"] += 0.1 * bumped.loc[recent, "oi"]
    raised = build_cot_features(bumped, index)["cot_mm_net_z"].iloc[-1]

    assert raised > baseline


# --------------------------------------------------------------------------
# Offline parsing and provenance
# --------------------------------------------------------------------------

def canned_row(date: str, **overrides) -> dict:
    row = {
        "report_date_as_yyyy_mm_dd": f"{date}T00:00:00.000",
        "open_interest_all": "500000",
        "m_money_positions_long_all": "150000",
        "m_money_positions_short_all": "50000",
        "prod_merc_positions_long": "60000",
        "prod_merc_positions_short": "180000",
        "swap_positions_long_all": "70000",
        "swap__positions_short_all": "90000",
    }
    row.update(overrides)
    return row


def test_socrata_rows_parse_offline():
    """Socrata returns strings and omits null fields per record; parsing must
    type them and pin the swap__ double-underscore spelling."""
    second = canned_row("2023-01-31", open_interest_all="510000")
    del second["swap__positions_short_all"]  # per-record null omission
    parsed = _parse_socrata_rows([second, canned_row("2023-01-24")])  # unsorted on purpose

    assert parsed.index.is_monotonic_increasing
    assert parsed["oi"].dtype == float
    assert parsed.loc["2023-01-24", "mm_long"] == 150000.0
    assert np.isnan(parsed.loc["2023-01-31", "swap_short"])


def test_schema_drift_is_refused():
    rows = [canned_row("2023-01-24")]
    for row in rows:
        del row["prod_merc_positions_long"]  # column absent entirely

    with pytest.raises(ValueError, match="missing expected fields"):
        _parse_socrata_rows(rows)


def test_duplicate_report_dates_are_refused():
    with pytest.raises(ValueError, match="Duplicate report dates"):
        _parse_socrata_rows([canned_row("2023-01-24"), canned_row("2023-01-24")])


def test_sidecar_contract_mismatch_is_refused():
    cfg = AssetConfig(
        asset="gold", symbol="XAUUSD", granularity="H4",
        cot=COTConfig(contract_code="088691"),
    )

    _check_cot_sidecar({"contract_code": "088691"}, cfg)  # matching: no raise
    with pytest.raises(ValueError, match="084691.*088691"):
        _check_cot_sidecar({"contract_code": "084691"}, cfg)


def test_fields_map_matches_feature_inputs():
    """Every column build_cot_features reads must come out of the parser."""
    parsed_columns = set(FIELDS.values()) - {"report_date"}
    needed = {"oi", "mm_long", "mm_short", "prod_long", "prod_short"}
    assert needed <= parsed_columns
