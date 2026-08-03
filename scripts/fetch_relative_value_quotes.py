"""Fetch/pin outcome-blind XAU/XAG M30 bid and ask candles for RV1.

Dukascopy access is free and requires no API key.  This stage validates only
source structure and writes no residual, candidate, forward return, or P&L.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from commodity_ml.config import PROJECT_ROOT, load_config
from commodity_ml.data import dukascopy
from commodity_ml.data.schema import validate_candles

START = pd.Timestamp("2020-01-01T00:00:00Z")
END_EXCLUSIVE = pd.Timestamp("2025-07-01T00:00:00Z")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "relative_value_quotes"),
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 1,
        "program": "rv1_quote_acquisition",
        "source": "dukascopy",
        "granularity": "M30",
        "start": START.isoformat(),
        "end_exclusive": END_EXCLUSIVE.isoformat(),
        "artifacts": {},
        "outcomes_calculated": False,
        "signals_calculated": False,
        "strategy_evaluated": False,
    }
    for leg, config_path in (("gold", "config/gold.yaml"), ("silver", "config/silver.yaml")):
        cfg = load_config(config_path)
        for side in ("bid", "ask"):
            target = output / f"{leg}_{side}_M30.parquet"
            if target.is_file():
                frame = validate_candles(pd.read_parquet(target), f"rv1:{leg}:{side}")
                source = "existing_pinned_artifact"
            else:
                frame, source = _obtain(cfg, side)
                temporary = target.with_suffix(target.suffix + ".tmp")
                frame.to_parquet(temporary)
                temporary.replace(target)
            _validate_period(frame, f"{leg} {side}")
            report["artifacts"][f"{leg}_{side}"] = {
                "path": str(target),
                "sha256": _sha256(target),
                "rows": len(frame),
                "first": frame.index.min().isoformat(),
                "last": frame.index.max().isoformat(),
                "duplicate_timestamps": int(frame.index.duplicated().sum()),
                "source_action": source,
            }

    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print("No residual, candidate, forward return, or strategy P&L was calculated.")


def _obtain(cfg, side: str) -> tuple[pd.DataFrame, str]:
    canonical = cfg.raw_path("dukascopy", granularity="M30")
    if side == "bid" and canonical.is_file():
        existing = validate_candles(
            pd.read_parquet(canonical),
            f"dukascopy:{cfg.symbol}:M30",
        )
        selected = existing.loc[
            (existing.index >= START) & (existing.index < END_EXCLUSIVE)
        ].copy()
        if len(selected) and selected.index.min() <= START + pd.Timedelta(days=7):
            return selected, "sliced_existing_bid"

    data_cfg = replace(
        cfg.data,
        start=START.strftime("%Y-%m-%d"),
        end=END_EXCLUSIVE.strftime("%Y-%m-%d"),
        dukascopy=replace(cfg.data.dukascopy, offer_side=side),
    )
    frame = dukascopy.fetch_candles(data_cfg, "M30")
    frame = frame.loc[(frame.index >= START) & (frame.index < END_EXCLUSIVE)].copy()
    return frame, "downloaded"


def _validate_period(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        raise ValueError(f"{label} quote history is empty")
    if frame.index.min() < START or frame.index.max() >= END_EXCLUSIVE:
        raise ValueError(f"{label} quote history crosses its frozen period")
    if frame.index.min() > START + pd.Timedelta(days=7):
        raise ValueError(f"{label} quote history starts unexpectedly late")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
