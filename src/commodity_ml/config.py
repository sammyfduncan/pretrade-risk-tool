"""Typed config loaded from config/<asset>.yaml.

Everything tunable lives in the yaml; modules receive these dataclasses and
contain no magic numbers of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class HistoryConfig:
    start: str
    end: str | None = None


@dataclass(frozen=True)
class LabelConfig:
    mode: str = "long_threshold"  # long_threshold | updown
    horizon_bars: int = 12
    atr_mult: float = 1.0


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    gap_bars: int = 12


@dataclass(frozen=True)
class ModelConfig:
    type: str = "lightgbm"  # lightgbm | logistic


@dataclass(frozen=True)
class BacktestConfig:
    prob_entry: float = 0.60
    prob_exit: float = 0.50
    fee_pct: float = 0.0001
    init_cash: float = 10_000.0


@dataclass(frozen=True)
class AssetConfig:
    asset: str
    instrument: str
    granularity: str
    history: HistoryConfig
    label: LabelConfig = field(default_factory=LabelConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    def __post_init__(self) -> None:
        if self.cv.gap_bars < self.label.horizon_bars:
            raise ValueError(
                f"cv.gap_bars ({self.cv.gap_bars}) must be >= label.horizon_bars "
                f"({self.label.horizon_bars}), otherwise forward-looking labels "
                "overlap the train/test boundary and leak future information."
            )

    # ---- artifact paths (all gitignored) ----
    @property
    def raw_path(self) -> Path:
        return PROJECT_ROOT / "data" / "raw" / f"{self.instrument}_{self.granularity}.parquet"

    @property
    def dataset_path(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / f"{self.asset}_dataset.parquet"

    @property
    def oof_path(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / f"{self.asset}_oof.parquet"

    @property
    def model_path(self) -> Path:
        return PROJECT_ROOT / "models" / f"{self.asset}_{self.model.type}.joblib"

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / "reports"


def load_config(path: str | Path) -> AssetConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return AssetConfig(
        asset=raw["asset"],
        instrument=raw["instrument"],
        granularity=raw["granularity"],
        history=HistoryConfig(**raw.get("history", {})),
        label=LabelConfig(**raw.get("label", {})),
        cv=CVConfig(**raw.get("cv", {})),
        model=ModelConfig(**raw.get("model", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )


def bars_per_year(granularity: str) -> float:
    """Approximate trading bars per year for annualizing metrics.

    OANDA metals trade ~24h/day, 5 days/week, ~52 weeks -> hours per year.
    """
    hours_per_year = 24 * 5 * 52
    per_hour = {"M5": 12, "M15": 4, "M30": 2, "H1": 1, "H4": 1 / 4, "D": 1 / 24}
    if granularity not in per_hour:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return hours_per_year * per_hour[granularity]
