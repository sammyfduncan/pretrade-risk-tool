"""Config plumbing for the step-2 alt-data families (COT, GDELT).

The load-bearing test here is the first one: both flags defaulting to False is
what keeps the step-1 frozen template reproducible after this code lands. If a
default ever flips to True, pulling the repo silently changes what the frozen
configuration means — that is a freeze break, not a convenience.
"""

from pathlib import Path

import pytest

from commodity_ml.config import (
    AssetConfig,
    COTConfig,
    FeatureConfig,
    GDELTConfig,
    load_config,
)


def test_altdata_flags_default_off():
    """The freeze-safety pin (docs/step2_design.md §0)."""
    features = FeatureConfig()
    assert features.cot is False
    assert features.gdelt is False


def test_unknown_yaml_keys_fail_loudly():
    """load_config unpacks yaml blocks with **kwargs into frozen dataclasses,
    so a typo'd key must be a TypeError, not a silently ignored setting."""
    with pytest.raises(TypeError):
        COTConfig(**{"contract": "088691"})  # should be contract_code
    with pytest.raises(TypeError):
        GDELTConfig(**{"search": "gold"})  # should be query


def test_snapshot_paths_are_symbol_keyed():
    cfg = AssetConfig(asset="gold", symbol="XAUUSD", granularity="H4")

    assert cfg.cot_snapshot_path.name == "XAUUSD_cot.parquet"
    assert cfg.gdelt_snapshot_path.name == "XAUUSD_gdelt.parquet"
    # Same directory as raw candles — snapshots are raw inputs, not artifacts.
    assert cfg.cot_snapshot_path.parent == cfg.raw_path().parent


def test_cot_enabled_without_contract_code_is_refused():
    """A default code would be one asset's, inherited by every other config —
    enabling COT on silver without setting it would train silver on gold's
    positioning silently. Extending to a new asset must be a decision."""
    with pytest.raises(ValueError, match="cot.contract_code is empty"):
        AssetConfig(
            asset="silver",
            symbol="XAGUSD",
            granularity="H4",
            features=FeatureConfig(cot=True),
        )


def test_cot_enabled_with_contract_code_is_accepted():
    cfg = AssetConfig(
        asset="silver",
        symbol="XAGUSD",
        granularity="H4",
        features=FeatureConfig(cot=True),
        cot=COTConfig(contract_code="084691"),
    )
    assert cfg.cot.contract_code == "084691"
    assert cfg.cot_snapshot_path.name == "XAGUSD_cot.parquet"


def test_gdelt_enabled_without_query_is_refused():
    """The query is a validated, per-asset judgment call (--preview first).
    Enabling the family with no query means that step was skipped."""
    with pytest.raises(ValueError, match="gdelt.query is empty"):
        AssetConfig(
            asset="gold",
            symbol="XAUUSD",
            granularity="H4",
            features=FeatureConfig(gdelt=True),
        )


def test_gdelt_enabled_with_query_is_accepted():
    cfg = AssetConfig(
        asset="gold",
        symbol="XAUUSD",
        granularity="H4",
        features=FeatureConfig(gdelt=True),
        gdelt=GDELTConfig(query='"gold price" OR bullion'),
    )
    assert cfg.features.gdelt is True


def test_checked_in_gold_yaml_parses_with_the_new_blocks():
    """Round-trip the real config file: the cot/gdelt blocks must map onto the
    dataclasses (this catches a yaml/dataclass drift the unit tests above,
    which construct dataclasses directly, would miss)."""
    path = Path(__file__).resolve().parents[1] / "config" / "gold.yaml"
    cfg = load_config(path)

    assert cfg.cot.contract_code == "088691"
    assert isinstance(cfg.gdelt.query, str)
