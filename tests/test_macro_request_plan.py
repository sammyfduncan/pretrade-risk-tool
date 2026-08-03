"""Outcome-blind tests for the frozen P4 acquisition planner."""

import importlib.util
from pathlib import Path

import pandas as pd

_SCRIPT = Path(__file__).parents[1] / "scripts" / "plan_cme_macro_drivers.py"
_SPEC = importlib.util.spec_from_file_location("plan_cme_macro_drivers", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_registered_plan_has_only_the_four_frozen_requests() -> None:
    plan = _MODULE.registered_plan()

    assert set(zip(plan["symbol"], plan["schema"], strict=True)) == {
        ("ZN.FUT", "ohlcv-1m"),
        ("ZN.FUT", "definition"),
        ("6E.FUT", "ohlcv-1m"),
        ("6E.FUT", "definition"),
    }
    assert set(plan["stype_in"]) == {"parent"}
    assert set(plan["stype_out"]) == {"instrument_id"}
    assert set(plan["encoding"]) == {"dbn"}
    assert set(plan["compression"]) == {"zstd"}
    assert plan["start"].nunique() == 1
    assert plan["end_exclusive"].nunique() == 1


def test_quote_path_uses_metadata_get_cost_only() -> None:
    class FakeMetadata:
        def __init__(self) -> None:
            self.calls = []

        def get_cost(self, **kwargs) -> float:
            self.calls.append(kwargs)
            return 0.25

    metadata = FakeMetadata()
    costs = _MODULE.quote_costs(metadata, _MODULE.registered_plan())

    assert costs == [0.25] * 4
    assert len(metadata.calls) == 4
    assert {call["symbols"] for call in metadata.calls} == {"ZN.FUT", "6E.FUT"}
    assert {call["schema"] for call in metadata.calls} == {
        "ohlcv-1m",
        "definition",
    }


def test_identical_completed_quote_can_be_reused(tmp_path: Path) -> None:
    plan = _MODULE.registered_plan()
    path = tmp_path / "plan.csv"
    saved = plan.copy()
    saved["quoted_cost_usd"] = [1.0, 2.0, 3.0, 4.0]
    saved.to_csv(path, index=False)

    assert _MODULE.cached_quotes(path, plan) == [1.0, 2.0, 3.0, 4.0]

    changed = plan.copy()
    changed.loc[0, "end_exclusive"] = pd.Timestamp("2025-06-01T00:00:00Z")
    assert _MODULE.cached_quotes(path, changed) is None
