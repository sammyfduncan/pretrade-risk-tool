"""Safety tests for the paid sparse TBBO download boundary."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


_SCRIPT = Path(__file__).parents[1] / "scripts" / "download_cme_gc_tbbo_windows.py"
_SPEC = importlib.util.spec_from_file_location("download_cme_gc_tbbo_windows", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _plan(tmp_path: Path, cost: float = 0.25) -> tuple[Path, Path]:
    path = tmp_path / "plan.csv"
    frame = pd.DataFrame(
        {
            "dataset": ["GLBX.MDP3"],
            "schema": ["tbbo"],
            "stype_in": ["raw_symbol"],
            "symbol": ["GCM5"],
            "start": [pd.Timestamp("2025-05-01T10:00Z")],
            "end_exclusive": [pd.Timestamp("2025-05-01T10:30Z")],
            "quoted_cost_usd": [cost],
        }
    )
    frame.to_csv(path, index=False)
    manifest = path.with_suffix(".manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "output_sha256": _MODULE._sha256(path),
                "request_rows": 1,
                "quoted_cost_usd": cost,
                "paid_endpoint_called": False,
            }
        )
    )
    return path, manifest


def test_authorization_refuses_quote_above_ceiling(tmp_path: Path) -> None:
    path, manifest = _plan(tmp_path, cost=0.25)

    with pytest.raises(ValueError, match="exceeds authorization"):
        _MODULE._load_authorized_plan(path, manifest, authorized_max_usd=0.24)


def test_authorization_accepts_exact_frozen_quote(tmp_path: Path) -> None:
    path, manifest = _plan(tmp_path, cost=0.25)

    plan, loaded = _MODULE._load_authorized_plan(
        path, manifest, authorized_max_usd=0.25
    )

    assert len(plan) == 1
    assert loaded["paid_endpoint_called"] is False
