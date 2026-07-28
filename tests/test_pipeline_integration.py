"""The seams between the four scripts.

Every other test file exercises one function. Nothing checked that
`build_dataset.py -> train.py -> backtest.py` actually chain together, and they
communicate through parquet files with contracts that live in two places at
once. A rename that keeps every unit test green can still change which columns
the model sees.

These run offline on synthetic candles: macro is disabled so there is no FRED
call, and cross-asset is disabled so no partner file is needed.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import make_ohlcv

from commodity_ml.backtest.engine import run_backtest, run_rule_backtest
from commodity_ml.config import (
    AssetConfig,
    BacktestConfig,
    CVConfig,
    FeatureConfig,
    LabelConfig,
    ModelConfig,
    PrimaryConfig,
    bars_per_year,
)
from commodity_ml.dataset import METADATA_COLUMNS, feature_columns
from commodity_ml.features.pipeline import build_feature_frame
from commodity_ml.labels import LABEL_COLUMNS, make_labels
from commodity_ml.modeling.walkforward import walk_forward
from commodity_ml.primary import SIGNAL_COLUMN, make_primary_signal

HORIZON = 12


def cfg_for_tests() -> AssetConfig:
    return AssetConfig(
        asset="test",
        symbol="XAUUSD",
        granularity="H1",
        features=FeatureConfig(technical=True, macro=False, crossasset=False),
        primary=PrimaryConfig(rule="trend", ema_span=50),
        label=LabelConfig(mode="meta", horizon_bars=HORIZON, atr_mult=1.0),
        cv=CVConfig(n_splits=3, gap_bars=HORIZON),
        model=ModelConfig(
            type="lightgbm", sample_weighting="uniqueness", calibration="isotonic"
        ),
        backtest=BacktestConfig(entry_quantile=0.80),
    )


def build_dataset(cfg: AssetConfig, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Exactly what scripts/build_dataset.py assembles, minus the printing."""
    features, feature_names = build_feature_frame(cfg, raw)
    labels = make_labels(raw, cfg.label)
    primary = make_primary_signal(raw, cfg.primary)
    dataset = pd.concat([features, labels, primary, raw["close"]], axis=1).dropna()
    return dataset, feature_names


@pytest.fixture(scope="module")
def built():
    cfg = cfg_for_tests()
    raw = make_ohlcv(n=3000)
    dataset, feature_names = build_dataset(cfg, raw)
    return cfg, raw, dataset, feature_names


# --------------------------------------------------------------------------
# The contract between build_dataset.py and train.py
# --------------------------------------------------------------------------

def test_dataset_columns_partition_into_features_and_metadata(built):
    _, _, dataset, feature_names = built

    assert set(dataset.columns) == set(feature_names) | set(METADATA_COLUMNS)
    assert not set(feature_names) & set(METADATA_COLUMNS), "no column may be both"


def test_train_infers_the_same_features_build_dataset_wrote(built):
    """The two scripts derive the feature list independently — they must agree."""
    _, _, dataset, feature_names = built

    assert sorted(feature_columns(dataset)) == sorted(feature_names)


def test_no_label_derived_column_can_reach_the_model(built):
    """The lookahead that would not crash.

    exit_bars and exit_time describe how the trade finished. If either is ever
    handed to the model as a feature the result is not an error, it is an AUC
    near 1.0 that looks like a breakthrough.
    """
    _, _, dataset, _ = built
    inferred = feature_columns(dataset)

    for leaky in LABEL_COLUMNS:
        assert leaky not in inferred, f"{leaky} is label-derived and must never be a feature"
    assert SIGNAL_COLUMN not in inferred, "the rule being judged is not an input"


def test_a_stale_dataset_is_refused_rather_than_trained_on(built):
    """The realistic failure: datasets are gitignored artifacts that outlive the
    code that built them, so a rebuild can be skipped after a rename."""
    _, _, dataset, _ = built
    stale = dataset.rename(columns={"exit_bars": "n_bars_held"})

    with pytest.raises(ValueError, match="missing expected metadata columns"):
        feature_columns(stale)


def test_a_non_numeric_feature_column_is_refused(built):
    _, _, dataset, _ = built
    contaminated = dataset.copy()
    contaminated["some_timestamp"] = dataset.index

    with pytest.raises(ValueError, match="Non-numeric columns"):
        feature_columns(contaminated)


def test_parquet_round_trip_preserves_the_contract(built, tmp_path):
    """exit_time is tz-aware and label is boolean-ish; parquet must not alter
    either, or train.py and backtest.py silently disagree about the same file."""
    _, _, dataset, feature_names = built
    path = tmp_path / "dataset.parquet"
    dataset.to_parquet(path)
    reloaded = pd.read_parquet(path)

    assert sorted(feature_columns(reloaded)) == sorted(feature_names)
    assert reloaded.index.tz is not None, "candle index must stay UTC-aware"
    assert reloaded["exit_time"].dt.tz is not None, "exit_time must stay UTC-aware"
    # check_freq=False: parquet does not persist a DatetimeIndex's `freq`, so a
    # synthetic frame built with date_range loses it on the round trip. Nothing
    # depends on it — real candles are always read back from parquet and so
    # always have freq=None, and the backtest takes its frequency explicitly.
    pd.testing.assert_frame_equal(dataset, reloaded, check_freq=False)


# --------------------------------------------------------------------------
# The contract between train.py and backtest.py
# --------------------------------------------------------------------------

def test_full_chain_runs_and_each_stage_consumes_the_previous_one(built):
    cfg, raw, dataset, _ = built

    # --- train.py: meta mode trains only on bars the primary rule proposes
    training = dataset[dataset[SIGNAL_COLUMN]]
    assert len(training) < len(dataset), "the rule should not fire on every bar"

    feature_names = feature_columns(training)
    wf = walk_forward(
        training[feature_names], training["label"],
        n_splits=cfg.cv.n_splits, gap=cfg.cv.gap_bars, model_kind=cfg.model.type,
        exit_time=training["exit_time"],
        sample_weighting=cfg.model.sample_weighting,
        calibration=cfg.model.calibration,
    )

    oof = pd.DataFrame({
        "proba": wf.oof_proba,
        "proba_calibrated": wf.oof_calibrated,
        "label": training["label"].loc[wf.oof_proba.index],
        "barrier_pct": training["barrier_pct"].loc[wf.oof_proba.index],
    })
    assert not oof.isna().any().any(), "the OOF frame backtest.py reads must be complete"

    # --- backtest.py: consumes that frame plus raw close
    result = run_backtest(
        raw["close"], oof["proba"], oof["barrier_pct"], cfg.backtest,
        cfg.granularity, bars_per_year(cfg.granularity), cfg.label.horizon_bars,
    )
    assert result.stats["n_trades"] >= 0
    assert set(result.stats) >= {"total_return", "sharpe", "max_drawdown", "n_trades"}

    # --- the bare-rule control runs on the same window, as backtest.py does
    rule = run_rule_backtest(
        raw["close"], dataset[SIGNAL_COLUMN], oof["barrier_pct"], cfg.backtest,
        cfg.granularity, bars_per_year(cfg.granularity), cfg.label.horizon_bars,
        window=result.portfolio.wrapper.index,
    )
    assert len(result.portfolio.wrapper.index) == len(rule.portfolio.wrapper.index), (
        "strategy and control must be measured over an identical window"
    )


def test_backtest_only_ever_sees_out_of_fold_bars(built):
    """Rule 1 of the engine: no prediction may come from a model that trained
    on that bar. The OOF index is the first fold's test start onward."""
    cfg, _, dataset, _ = built
    training = dataset[dataset[SIGNAL_COLUMN]]
    feature_names = feature_columns(training)

    wf = walk_forward(
        training[feature_names], training["label"],
        n_splits=cfg.cv.n_splits, gap=cfg.cv.gap_bars, model_kind="logistic",
        exit_time=training["exit_time"],
    )

    assert wf.oof_proba.index.is_monotonic_increasing
    assert wf.oof_proba.index.min() == wf.folds[0].test_start
    assert wf.oof_proba.index.min() > wf.folds[0].train_end, (
        "OOF predictions must start after the first fold's training data ends"
    )
    for fold in wf.folds:
        gap_bars = training.index.searchsorted(fold.test_start) - training.index.searchsorted(
            fold.train_end
        )
        assert gap_bars >= cfg.cv.gap_bars, (
            "the train/test gap must cover the label horizon or forward labels leak"
        )


def test_meta_mode_predictions_never_land_on_off_signal_bars(built):
    """backtest.py reindexes a sparse OOF series onto the full timeline; an
    entry on a bar the primary rule never proposed would be untradeable."""
    cfg, raw, dataset, _ = built
    training = dataset[dataset[SIGNAL_COLUMN]]
    feature_names = feature_columns(training)

    wf = walk_forward(
        training[feature_names], training["label"],
        n_splits=cfg.cv.n_splits, gap=cfg.cv.gap_bars, model_kind="logistic",
        exit_time=training["exit_time"],
    )
    off_signal = dataset.index[~dataset[SIGNAL_COLUMN]]
    assert wf.oof_proba.index.intersection(off_signal).empty

    result = run_backtest(
        raw["close"], wf.oof_proba, training["barrier_pct"], cfg.backtest,
        cfg.granularity, bars_per_year(cfg.granularity), cfg.label.horizon_bars,
    )
    # Buys only. Exits are barrier touches and land wherever price takes them,
    # which is legitimately any bar — including ones the rule never proposed.
    orders = result.portfolio.orders.records_readable
    buys = orders.loc[orders["Side"] == "Buy"]
    assert len(buys) > 0, "the fixture should produce at least one trade to check"

    entered_at = pd.to_datetime(buys["Timestamp"]).unique()
    # Entries execute one bar after the signal (SIGNAL_DELAY), so map back.
    positions = raw.index.searchsorted(entered_at) - 1
    proposed = dataset[SIGNAL_COLUMN].reindex(raw.index, fill_value=False).astype(bool).to_numpy()
    assert proposed[np.clip(positions, 0, len(raw) - 1)].all(), (
        "every entry must trace back to a bar where the primary rule was long"
    )
