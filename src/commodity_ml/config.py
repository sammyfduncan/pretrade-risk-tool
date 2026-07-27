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
class DukascopyConfig:
    instrument: str = "XAU/USD"
    offer_side: str = "bid"


@dataclass(frozen=True)
class IGConfig:
    epic: str | None = None


@dataclass(frozen=True)
class DataConfig:
    source: str = "dukascopy"  # dukascopy | ig
    start: str = "2019-01-01"
    end: str | None = None
    dukascopy: DukascopyConfig = field(default_factory=DukascopyConfig)
    ig: IGConfig = field(default_factory=IGConfig)


@dataclass(frozen=True)
class LabelConfig:
    mode: str = "triple_barrier"  # triple_barrier | updown | long_threshold (legacy)
    horizon_bars: int = 12
    atr_mult: float = 1.0
    timeout: str = "sign"  # triple_barrier only: label timeouts by sign, or drop them


@dataclass(frozen=True)
class FeatureConfig:
    technical: bool = True
    macro: bool = True


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    gap_bars: int = 12


@dataclass(frozen=True)
class ModelConfig:
    type: str = "lightgbm"  # lightgbm | logistic


@dataclass(frozen=True)
class BacktestConfig:
    entry_quantile: float = 0.90
    fee_pct: float = 0.0001
    init_cash: float = 10_000.0

    def __post_init__(self) -> None:
        if not 0.0 < self.entry_quantile < 1.0:
            raise ValueError(
                f"entry_quantile must be strictly between 0 and 1, got {self.entry_quantile}"
            )


@dataclass(frozen=True)
class AssetConfig:
    asset: str
    symbol: str
    granularity: str
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
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
        if self.data.source not in ("dukascopy", "ig"):
            raise ValueError(f"Unknown data.source: {self.data.source!r}")
        if not (self.features.technical or self.features.macro):
            raise ValueError("At least one feature family must be enabled.")

    # ---- artifact paths (all gitignored) ----
    def raw_path(self, source: str | None = None) -> Path:
        """Raw candles are namespaced by source so two providers can coexist
        on disk and be compared (scripts/compare_sources.py)."""
        source = source or self.data.source
        return PROJECT_ROOT / "data" / "raw" / f"{self.symbol}_{self.granularity}_{source}.parquet"

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

    data_raw = dict(raw.get("data", {}))
    data = DataConfig(
        source=data_raw.get("source", "dukascopy"),
        start=data_raw.get("start", "2019-01-01"),
        end=data_raw.get("end"),
        dukascopy=DukascopyConfig(**(data_raw.get("dukascopy") or {})),
        ig=IGConfig(**(data_raw.get("ig") or {})),
    )
    return AssetConfig(
        asset=raw["asset"],
        symbol=raw["symbol"],
        granularity=raw["granularity"],
        data=data,
        features=FeatureConfig(**raw.get("features", {})),
        label=LabelConfig(**raw.get("label", {})),
        cv=CVConfig(**raw.get("cv", {})),
        model=ModelConfig(**raw.get("model", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )


def bars_per_year(granularity: str) -> float:
    """Approximate trading bars per year for annualizing metrics.

    Metals trade ~24h/day, 5 days/week, ~52 weeks -> hours per year.
    """
    hours_per_year = 24 * 5 * 52
    per_hour = {"M5": 12, "M15": 4, "M30": 2, "H1": 1, "H4": 1 / 4, "D": 1 / 24}
    if granularity not in per_hour:
        raise ValueError(f"Unsupported granularity: {granularity}")
    return hours_per_year * per_hour[granularity]
