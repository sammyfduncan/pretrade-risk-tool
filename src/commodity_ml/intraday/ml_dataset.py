"""Candidate-only ML dataset contract for the intraday rebuild."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

from commodity_ml.intraday.execution import (
    IntradayExecutionConfig,
    TradeCandidate,
    simulate_candidates,
)

COMMON_FEATURES = [
    "h1_ema20_dist",
    "h1_ema50_dist",
    "h1_ema50_slope3_atr",
    "h1_atr_pct",
    "m30_ema20_dist",
    "m30_atr_pct",
    "m30_bb_position",
    "m30_bb_width_pct",
    "m30_body_pct",
    "m30_range_pct",
    "previous_session_range_atr",
    "previous_session_return",
    "minutes_since_entry_start",
    "minutes_until_last_entry",
]

STRATEGY_FEATURES = {
    "P1": ["opening_range_compression", "setup_distance_atr"],
    "P2": ["setup_distance_atr", "h1_regime_distance_atr"],
    "P3": ["setup_distance_atr", "h1_regime_distance_atr"],
}

METADATA_COLUMNS = [
    "candidate_id",
    "strategy",
    "side",
    "entry_time",
    "exit_time",
    "exit_reason",
    "entry_price",
    "exit_price",
    "initial_r_pct",
    "net_return",
    "r_multiple",
    "label",
]


@dataclass(frozen=True)
class CandidateDatasetResult:
    dataset: pd.DataFrame
    ledger: pd.DataFrame
    feature_names: tuple[str, ...]


ARTIFACT_SCHEMA_VERSION = 1


def build_candidate_dataset(
    frame: pd.DataFrame,
    candidates: list[TradeCandidate],
    *,
    strategy: str,
    execution_config: IntradayExecutionConfig | None = None,
) -> CandidateDatasetResult:
    """Resolve one bare rule and attach only features known at signal close."""

    if strategy not in STRATEGY_FEATURES:
        raise ValueError(f"Unknown strategy {strategy!r}")
    wrong = [candidate.strategy for candidate in candidates if candidate.strategy != strategy]
    if wrong:
        raise ValueError("All candidates must belong to the requested strategy")

    ledger = simulate_candidates(frame, candidates, execution_config)
    executed = ledger.loc[ledger["status"] == "executed"].copy()
    feature_names = tuple(COMMON_FEATURES + STRATEGY_FEATURES[strategy] + ["is_long"])
    if executed.empty:
        empty = pd.DataFrame(columns=METADATA_COLUMNS + list(feature_names))
        empty.index = pd.DatetimeIndex([], name="time", tz="UTC")
        return CandidateDatasetResult(empty, ledger, feature_names)

    signal_index = pd.DatetimeIndex(executed["signal_time"])
    missing_signals = signal_index.difference(frame.index)
    if len(missing_signals):
        raise ValueError(f"Executed signals missing from feature frame: {list(missing_signals)}")
    rows = frame.loc[signal_index].copy()
    rows.index = signal_index
    sides = executed.set_index("signal_time")["side"].reindex(signal_index)
    rows["is_long"] = (sides == "long").astype("int8").to_numpy()
    direction = np.where(rows["is_long"].to_numpy() == 1, 1.0, -1.0)

    if strategy == "P1":
        boundary = np.where(
            direction > 0,
            rows["opening_range_high"],
            rows["opening_range_low"],
        )
        rows["setup_distance_atr"] = (
            direction * (rows["close"].to_numpy() - boundary)
            / rows["m30_atr14"].to_numpy()
        )
    elif strategy == "P2":
        rows["setup_distance_atr"] = (
            (rows["close"] - rows["m30_ema20"]).abs() / rows["m30_atr14"]
        )
        rows["h1_regime_distance_atr"] = (
            direction
            * (rows["h1_ema20"] - rows["h1_ema50"]).to_numpy()
            / rows["h1_atr14"].to_numpy()
        )
    else:
        band = np.where(
            direction > 0,
            rows["m30_bb_lower"],
            rows["m30_bb_upper"],
        )
        rows["setup_distance_atr"] = (
            direction * (rows["close"].to_numpy() - band)
            / rows["m30_atr14"].to_numpy()
        )
        rows["h1_regime_distance_atr"] = (
            (rows["h1_ema20"] - rows["h1_ema50"]).abs() / rows["h1_atr14"]
        )

    missing_features = [
        column for column in feature_names if column not in rows.columns
    ]
    if missing_features:
        raise ValueError(f"Candidate feature frame missing columns: {missing_features}")
    non_finite = ~np.isfinite(rows.loc[:, feature_names].astype(float).to_numpy())
    if non_finite.any():
        bad_rows, bad_columns = np.where(non_finite)
        examples = sorted(
            {
                f"{rows.index[row]}:{feature_names[column]}"
                for row, column in zip(bad_rows[:10], bad_columns[:10])
            }
        )
        raise ValueError(
            "Candidate features are incomplete; trim the deterministic warmup "
            f"before generating candidates. Examples: {examples}"
        )

    executed = executed.set_index("signal_time").reindex(signal_index)
    dataset = pd.DataFrame(index=signal_index)
    dataset.index.name = "time"
    dataset["candidate_id"] = [
        f"{strategy}:{timestamp.isoformat()}" for timestamp in signal_index
    ]
    for column in METADATA_COLUMNS[1:]:
        dataset[column] = executed[column].to_numpy()
    for column in feature_names:
        dataset[column] = rows[column].astype(float).to_numpy()
    return CandidateDatasetResult(dataset, ledger, feature_names)


def candidate_feature_columns(dataset: pd.DataFrame, strategy: str) -> list[str]:
    """Refuse stale candidate artifacts rather than infer feature columns."""

    expected = COMMON_FEATURES + STRATEGY_FEATURES[strategy] + ["is_long"]
    missing_metadata = [column for column in METADATA_COLUMNS if column not in dataset]
    missing_features = [column for column in expected if column not in dataset]
    unexpected = [
        column
        for column in dataset.columns
        if column not in METADATA_COLUMNS and column not in expected
    ]
    if missing_metadata or missing_features or unexpected:
        raise ValueError(
            "Candidate dataset contract mismatch: "
            f"missing metadata={missing_metadata}, missing features={missing_features}, "
            f"unexpected={unexpected}. Rebuild the intraday candidate dataset."
        )
    non_numeric = [
        column for column in expected if not ptypes.is_numeric_dtype(dataset[column])
    ]
    if non_numeric:
        raise ValueError(f"Candidate model features must be numeric: {non_numeric}")
    return expected


def write_candidate_bundle(
    result: CandidateDatasetResult,
    output: str | Path,
    *,
    strategy: str,
    source: str,
    cost_pct_per_side: float,
    confirmation_cutoff: pd.Timestamp | None,
) -> Path:
    """Atomically publish dataset + ledger + checksummed manifest.

    The manifest is written last. A crash between parquet replacements leaves
    either no manifest or a checksum mismatch, so training fails closed instead
    of combining files from different builds.
    """

    output = Path(output)
    ledger_path = _ledger_path(output)
    manifest_path = _manifest_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    _reconcile_candidate_artifacts(result.dataset, result.ledger, strategy)
    expected_features = tuple(candidate_feature_columns(result.dataset, strategy))
    if expected_features != tuple(result.feature_names):
        raise ValueError(
            "Candidate result feature order differs from the frozen dataset contract"
        )
    if not np.isfinite(cost_pct_per_side) or cost_pct_per_side < 0:
        raise ValueError("cost_pct_per_side must be finite and non-negative")

    temporary_dataset = _temporary_path(output)
    temporary_ledger = _temporary_path(ledger_path)
    temporary_manifest = _temporary_path(manifest_path)
    temporary_paths = (temporary_dataset, temporary_ledger, temporary_manifest)
    try:
        result.dataset.to_parquet(temporary_dataset)
        result.ledger.to_parquet(temporary_ledger)
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "program": "intraday_v1",
            "strategy": strategy,
            "source": source,
            "decision_granularity": "M30",
            "cost_pct_per_side": float(cost_pct_per_side),
            "confirmation_cutoff": (
                None
                if confirmation_cutoff is None
                else _utc_timestamp(confirmation_cutoff).isoformat()
            ),
            "dataset_rows": len(result.dataset),
            "ledger_rows": len(result.ledger),
            "feature_names": list(expected_features),
            "dataset_sha256": _sha256(temporary_dataset),
            "ledger_sha256": _sha256(temporary_ledger),
        }
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dataset, output)
        os.replace(temporary_ledger, ledger_path)
        os.replace(temporary_manifest, manifest_path)
    finally:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    return manifest_path


def load_candidate_bundle(
    output: str | Path,
    *,
    expected_strategy: str,
    expected_source: str = "dukascopy",
    expected_cost_pct_per_side: float | None = None,
) -> CandidateDatasetResult:
    """Verify hashes, provenance, schema, and row-level ledger parity."""

    output = Path(output)
    ledger_path = _ledger_path(output)
    manifest_path = _manifest_path(output)
    missing = [
        str(path)
        for path in (output, ledger_path, manifest_path)
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"Candidate artifact bundle is incomplete: {missing}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Candidate artifact manifest is corrupt: {exc}") from exc

    required = {
        "schema_version",
        "program",
        "strategy",
        "source",
        "decision_granularity",
        "cost_pct_per_side",
        "confirmation_cutoff",
        "dataset_rows",
        "ledger_rows",
        "feature_names",
        "dataset_sha256",
        "ledger_sha256",
    }
    missing_fields = sorted(required - set(manifest))
    unexpected_fields = sorted(set(manifest) - required)
    if missing_fields or unexpected_fields:
        raise ValueError(
            "Candidate artifact manifest schema mismatch: "
            f"missing={missing_fields}, unexpected={unexpected_fields}"
        )
    if manifest["schema_version"] != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported candidate artifact schema {manifest['schema_version']}"
        )
    if manifest["program"] != "intraday_v1":
        raise ValueError(f"Candidate program mismatch: {manifest['program']}")
    if manifest["decision_granularity"] != "M30":
        raise ValueError(
            "Candidate decision granularity mismatch: "
            f"{manifest['decision_granularity']}"
        )
    if manifest["strategy"] != expected_strategy:
        raise ValueError(
            f"Candidate strategy mismatch: {manifest['strategy']} != {expected_strategy}"
        )
    if manifest["source"] != expected_source:
        raise ValueError(
            f"Candidate source mismatch: {manifest['source']} != {expected_source}"
        )
    if expected_cost_pct_per_side is not None and not np.isclose(
        manifest["cost_pct_per_side"],
        expected_cost_pct_per_side,
        rtol=0,
        atol=1e-15,
    ):
        raise ValueError(
            "Candidate cost assumption mismatch: "
            f"{manifest['cost_pct_per_side']} != {expected_cost_pct_per_side}"
        )
    if _sha256(output) != manifest["dataset_sha256"]:
        raise ValueError("Candidate dataset checksum mismatch")
    if _sha256(ledger_path) != manifest["ledger_sha256"]:
        raise ValueError("Candidate ledger checksum mismatch")

    dataset = pd.read_parquet(output)
    ledger = pd.read_parquet(ledger_path)
    if len(dataset) != manifest["dataset_rows"] or len(ledger) != manifest["ledger_rows"]:
        raise ValueError("Candidate artifact row counts differ from manifest")
    feature_names = tuple(candidate_feature_columns(dataset, expected_strategy))
    if list(feature_names) != manifest["feature_names"]:
        raise ValueError("Candidate feature order differs from manifest")
    _reconcile_candidate_artifacts(dataset, ledger, expected_strategy)
    return CandidateDatasetResult(dataset, ledger, feature_names)


def _reconcile_candidate_artifacts(
    dataset: pd.DataFrame,
    ledger: pd.DataFrame,
    strategy: str,
) -> None:
    required_ledger = {
        "strategy",
        "side",
        "signal_time",
        "status",
        "entry_time",
        "exit_time",
        "exit_reason",
        "entry_price",
        "exit_price",
        "initial_r_pct",
        "net_return",
        "r_multiple",
        "label",
    }
    missing = sorted(required_ledger - set(ledger))
    if missing:
        raise ValueError(f"Candidate ledger missing reconciliation columns: {missing}")
    executed = ledger.loc[ledger["status"] == "executed"].copy()
    signal_index = pd.DatetimeIndex(
        pd.to_datetime(executed["signal_time"], utc=True),
        name="time",
    )
    if not signal_index.is_unique:
        raise ValueError("Executed candidate signal times are not unique")
    executed.index = signal_index
    dataset_index = pd.DatetimeIndex(
        pd.to_datetime(dataset.index, utc=True),
        name="time",
    )
    if not dataset_index.is_unique:
        raise ValueError("Candidate dataset index is not unique")
    if not dataset_index.equals(signal_index):
        raise ValueError("Candidate dataset signals differ from executed ledger")
    if len(dataset) and (
        (dataset["strategy"] != strategy).any()
        or (executed["strategy"] != strategy).any()
    ):
        raise ValueError("Candidate artifact contains the wrong strategy")

    expected_ids = [
        f"{strategy}:{timestamp.isoformat()}" for timestamp in dataset_index
    ]
    if list(dataset["candidate_id"]) != expected_ids:
        raise ValueError("Candidate IDs do not match strategy and signal time")
    for column in ("strategy", "side", "exit_reason"):
        if not dataset[column].astype(str).equals(executed[column].astype(str)):
            raise ValueError(f"Candidate {column} differs from executed ledger")
    for column in ("entry_time", "exit_time"):
        left = pd.to_datetime(dataset[column], utc=True)
        right = pd.to_datetime(executed[column], utc=True)
        if not left.equals(right):
            raise ValueError(f"Candidate {column} differs from executed ledger")
    if not dataset["label"].astype("int8").equals(
        executed["label"].astype("int8")
    ):
        raise ValueError("Candidate labels differ from executed ledger")
    for column in (
        "entry_price",
        "exit_price",
        "initial_r_pct",
        "net_return",
        "r_multiple",
    ):
        if not np.allclose(
            dataset[column].astype(float).to_numpy(),
            executed[column].astype(float).to_numpy(),
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError(f"Candidate {column} differs from executed ledger")


def _ledger_path(output: Path) -> Path:
    return output.with_name(output.stem + "_ledger.parquet")


def _manifest_path(output: Path) -> Path:
    return output.with_name(output.stem + "_manifest.json")


def _temporary_path(path: Path) -> Path:
    descriptor, value = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    return Path(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("confirmation_cutoff must be timezone-aware")
    return timestamp.tz_convert("UTC")
