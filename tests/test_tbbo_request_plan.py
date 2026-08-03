"""Tests for outcome-blind sparse TBBO acquisition planning."""

import importlib.util
from pathlib import Path

import pandas as pd


_SCRIPT = Path(__file__).parents[1] / "scripts" / "plan_cme_gc_tbbo_windows.py"
_SPEC = importlib.util.spec_from_file_location("plan_cme_gc_tbbo_windows", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_merge_overlapping_windows = _MODULE._merge_overlapping_windows
_quote_plan_costs = _MODULE._quote_plan_costs


def test_request_plan_merges_only_same_symbol_overlaps() -> None:
    plan = pd.DataFrame(
        {
            "dataset": ["GLBX.MDP3"] * 3,
            "schema": ["tbbo"] * 3,
            "stype_in": ["raw_symbol"] * 3,
            "symbol": ["GCM5", "GCM5", "GCQ5"],
            "start": pd.to_datetime(
                ["2025-05-01T10:00Z", "2025-05-01T10:20Z", "2025-05-01T10:20Z"]
            ),
            "end_exclusive": pd.to_datetime(
                ["2025-05-01T10:30Z", "2025-05-01T10:50Z", "2025-05-01T10:50Z"]
            ),
        }
    )

    merged = _merge_overlapping_windows(plan)

    assert len(merged) == 2
    gcm = merged.loc[merged["symbol"] == "GCM5"].iloc[0]
    assert gcm["start"] == pd.Timestamp("2025-05-01T10:00Z")
    assert gcm["end_exclusive"] == pd.Timestamp("2025-05-01T10:50Z")
    assert set(merged["symbol"]) == {"GCM5", "GCQ5"}


def test_cost_quote_uses_only_metadata_get_cost() -> None:
    class FakeMetadata:
        def __init__(self) -> None:
            self.calls = []

        def get_cost(self, **kwargs) -> float:
            self.calls.append(kwargs)
            return 0.125

    plan = pd.DataFrame(
        {
            "dataset": ["GLBX.MDP3"],
            "schema": ["tbbo"],
            "stype_in": ["raw_symbol"],
            "symbol": ["GCM5"],
            "start": pd.to_datetime(["2025-05-01T10:00Z"]),
            "end_exclusive": pd.to_datetime(["2025-05-01T10:30Z"]),
        }
    )
    metadata = FakeMetadata()

    costs = _quote_plan_costs(metadata, plan)

    assert costs == [0.125]
    assert metadata.calls == [
        {
            "dataset": "GLBX.MDP3",
            "symbols": "GCM5",
            "schema": "tbbo",
            "stype_in": "raw_symbol",
            "start": pd.Timestamp("2025-05-01T10:00Z"),
            "end": pd.Timestamp("2025-05-01T10:30Z"),
        }
    ]
