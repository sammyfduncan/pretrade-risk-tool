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
        GDELTConfig(**{"search": "gold"})  # should be title_phrases


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
    with pytest.raises(ValueError, match="gdelt.title_phrases is empty"):
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
        gdelt=GDELTConfig(title_phrases=("gold price", "gold bullion")),
    )
    assert cfg.features.gdelt is True


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


@pytest.mark.parametrize(
    "name,code,asset_word,other_word",
    [("gold.yaml", "088691", "gold", "silver"),
     ("silver.yaml", "084691", "silver", "gold")],
)
def test_checked_in_yaml_parses_with_the_new_blocks(name, code, asset_word, other_word):
    """Round-trip the real config files: the cot/gdelt blocks must map onto the
    dataclasses (this catches a yaml/dataclass drift the unit tests above,
    which construct dataclasses directly, would miss)."""
    cfg = load_config(CONFIG_DIR / name)

    assert cfg.cot.contract_code == code
    # A tuple, not a list: the config is frozen, so a snapshot sidecar checked
    # against it cannot then be invalidated by mutating the phrases in place.
    assert isinstance(cfg.gdelt.title_phrases, tuple) and cfg.gdelt.title_phrases
    assert all(isinstance(p, str) and p for p in cfg.gdelt.title_phrases)


def test_each_config_names_its_own_asset_in_the_altdata_identifiers():
    """The copy-paste failure a new asset invites, and the reason both
    identifiers exist: a config cloned from gold's keeps gold's contract code
    and gold's GDELT query, and then trains on gold's positioning and gold's
    news under a silver name. Nothing downstream would raise — the shapes are
    identical and the numbers look plausible.

    The contract-code half is already refused at construction (a missing code
    is an error, so a *wrong* one is the only way through); this pins the
    per-asset distinctness that guard cannot check.
    """
    gold = load_config(CONFIG_DIR / "gold.yaml")
    silver = load_config(CONFIG_DIR / "silver.yaml")

    assert gold.cot.contract_code != silver.cot.contract_code
    assert gold.gdelt.title_phrases != silver.gdelt.title_phrases
    assert not any("silver" in p.lower() for p in gold.gdelt.title_phrases)
    assert not any("gold" in p.lower() for p in silver.gdelt.title_phrases)
    # Structurally parallel, differing only in the metal's name — which is what
    # makes silver a transferability control rather than a second experiment.
    # (Under DOC 2.0 gold used bare `bullion` and silver the qualified phrase;
    # the BigQuery/headline mechanism removed the need for that asymmetry.)
    assert len(gold.gdelt.title_phrases) == len(silver.gdelt.title_phrases)
    assert [p.replace("gold", "M") for p in gold.gdelt.title_phrases] == \
           [p.replace("silver", "M") for p in silver.gdelt.title_phrases]
    # Snapshots are symbol-keyed, so two assets can never share a pinned file.
    assert gold.cot_snapshot_path != silver.cot_snapshot_path
    assert gold.gdelt_snapshot_path != silver.gdelt_snapshot_path


def test_silver_differs_from_gold_only_in_asset_identity():
    """silver.yaml is a transferability control: every window, threshold, lag
    and gate must match gold's, or a divergent result is just a divergent
    config. Step 2 widened 'identity' from 3 fields to 5 — this pins that it
    did not widen further."""
    gold = load_config(CONFIG_DIR / "gold.yaml")
    silver = load_config(CONFIG_DIR / "silver.yaml")

    assert gold.features == silver.features
    assert gold.label == silver.label
    assert gold.cv == silver.cv
    assert gold.model == silver.model
    assert gold.backtest == silver.backtest
    assert gold.primary == silver.primary
    assert gold.granularity == silver.granularity
    # The ratio describes the environment, so it is deliberately identical.
    assert gold.crossasset == silver.crossasset
