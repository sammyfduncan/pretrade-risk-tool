"""Typed config loaded from config/<asset>.yaml.

Everything tunable lives in the yaml; modules receive these dataclasses and
contain no magic numbers of their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
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
    # Everything from this date on is fenced off from development. See
    # AssetConfig.development_slice for why this exists.
    holdout_start: str | None = None
    dukascopy: DukascopyConfig = field(default_factory=DukascopyConfig)
    ig: IGConfig = field(default_factory=IGConfig)


@dataclass(frozen=True)
class LabelConfig:
    # meta = triple-barrier outcomes, but only on bars where the primary rule
    # fires; the model then predicts whether obeying the rule pays off.
    mode: str = "triple_barrier"  # triple_barrier | meta | updown | long_threshold (legacy)
    horizon_bars: int = 12
    atr_mult: float = 1.0
    timeout: str = "sign"  # triple_barrier only: label timeouts by sign, or drop them


@dataclass(frozen=True)
class FeatureConfig:
    technical: bool = True
    macro: bool = True
    crossasset: bool = True
    # Step-2 alt-data families (docs/step2_design.md). Default FALSE on purpose:
    # the step-1 template frozen at 69f0527 must stay reproducible without them,
    # and turning one on is a pre-registered decision, not a code side effect.
    cot: bool = False    # CFTC COT positioning (features/cot.py)
    gdelt: bool = False  # GDELT news tone/attention (features/gdelt.py)


@dataclass(frozen=True)
class CrossAssetConfig:
    """The gold/silver ratio. See features/crossasset.py.

    Deliberately defined the same way for every precious metal — the ratio
    describes the *environment* both metals sit in, so gold.yaml and
    silver.yaml carry an identical block and the transferability control
    stays a three-line diff.
    """

    numerator_symbol: str = "XAUUSD"
    denominator_symbol: str = "XAGUSD"


@dataclass(frozen=True)
class COTConfig:
    """CFTC Commitments of Traders — which futures contract is 'this asset'.

    Only the contract code is per-asset; the release schedule, publication lag
    and staleness tolerance are properties of the CFTC's publishing process and
    live as constants in features/cot.py, exactly as FRED lags live in macro.py.
    """

    # Disaggregated Futures-Only dataset, cftc_contract_market_code.
    # Gold = "GOLD - COMMODITY EXCHANGE INC.". Silver later: "084691".
    contract_code: str = "088691"


@dataclass(frozen=True)
class GDELTConfig:
    """GDELT DOC 2.0 news query — the per-asset entity/theme filter.

    The query is the one genuinely per-asset judgment call in the alt-data
    layer (README: a blended 'commodities sentiment' is rejected as too
    diffuse). It must be validated by eyeballing article previews BEFORE any
    timeline data is fetched — see scripts/fetch_altdata.py --preview and
    docs/step2_design.md §4 — so an empty query with the family enabled is a
    refusal, not a default.
    """

    query: str = ""


@dataclass(frozen=True)
class PrimaryConfig:
    """The direction rule used when label.mode == 'meta'. See primary.py."""

    rule: str = "trend"  # trend | breakout | always
    ema_span: int = 50  # trend rule
    lookback: int = 20  # breakout rule


@dataclass(frozen=True)
class CVConfig:
    n_splits: int = 5
    gap_bars: int = 12


@dataclass(frozen=True)
class ModelConfig:
    type: str = "lightgbm"  # lightgbm | logistic

    # Overlapping labels make consecutive bars near-duplicates of each other.
    # Weighting each sample by how unique its forward window is stops the model
    # counting one move as `horizon_bars` independent facts. See
    # modeling/sample_weights.py.
    sample_weighting: str = "none"  # none | uniqueness

    # Maps raw scores onto probabilities that mean what they say. Fitted inside
    # each walk-forward fold on a held-out tail of that fold's *training* data —
    # fitting it on test data would be leakage. See modeling/calibration.py.
    calibration: str = "none"  # none | isotonic | sigmoid

    def __post_init__(self) -> None:
        if self.sample_weighting not in ("none", "uniqueness"):
            raise ValueError(
                f"model.sample_weighting must be 'none' or 'uniqueness', "
                f"got {self.sample_weighting!r}"
            )
        if self.calibration not in ("none", "isotonic", "sigmoid"):
            raise ValueError(
                f"model.calibration must be 'none', 'isotonic' or 'sigmoid', "
                f"got {self.calibration!r}"
            )


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
    crossasset: CrossAssetConfig = field(default_factory=CrossAssetConfig)
    cot: COTConfig = field(default_factory=COTConfig)
    gdelt: GDELTConfig = field(default_factory=GDELTConfig)
    primary: PrimaryConfig = field(default_factory=PrimaryConfig)
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
        if not (self.features.technical or self.features.macro or self.features.crossasset):
            raise ValueError("At least one feature family must be enabled.")
        if self.features.gdelt and not self.gdelt.query:
            raise ValueError(
                "features.gdelt is enabled but gdelt.query is empty. The query must "
                "be validated first — run scripts/fetch_altdata.py --preview, eyeball "
                "the article list, and record the final query in the yaml and in "
                "docs/step2_design.md §4."
            )

    # ---- development / holdout split ----
    #
    # Walk-forward out-of-fold predictions are honest in the narrow sense that no
    # model saw its own test bars. They stop being an honest estimate of *future*
    # performance once a human has read them and changed the design in response —
    # at that point the choice of label, timeframe and feature set has been fitted
    # to that period by hand. This is the multiple-comparisons trap in README
    # section 4, and it is the reason a good backtest is easy to manufacture.
    #
    # So a final slice is fenced off and never looked at during development. Every
    # script defaults to the development slice; touching the holdout takes an
    # explicit flag. It is a single-use resource — once you have seen it and tuned
    # against it, it is development data too.

    @property
    def holdout_ts(self) -> pd.Timestamp | None:
        if self.data.holdout_start is None:
            return None
        return pd.Timestamp(self.data.holdout_start, tz="UTC")

    def development_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rows available for training, tuning and inspection."""
        cutoff = self.holdout_ts
        return df if cutoff is None else df.loc[df.index < cutoff]

    def holdout_slice(self, df: pd.DataFrame) -> pd.DataFrame:
        """The untouched final period. Do not use during development."""
        cutoff = self.holdout_ts
        if cutoff is None:
            raise ValueError("No data.holdout_start configured, so there is no holdout.")
        return df.loc[df.index >= cutoff]

    # ---- artifact paths (all gitignored) ----
    def raw_path(self, source: str | None = None, symbol: str | None = None) -> Path:
        """Raw candles are namespaced by source so two providers can coexist
        on disk and be compared (scripts/compare_sources.py).

        `symbol` overrides the asset's own symbol, which is how cross-asset
        features reach the partner leg's candles (gold needs silver's file).
        """
        source = source or self.data.source
        symbol = symbol or self.symbol
        return PROJECT_ROOT / "data" / "raw" / f"{symbol}_{self.granularity}_{source}.parquet"

    @property
    def cot_snapshot_path(self) -> Path:
        """Pinned COT snapshot (scripts/fetch_altdata.py). Symbol-keyed like raw
        candles; no source/granularity segments because neither applies. A
        sidecar `<name>.meta.json` written next to it carries provenance."""
        return PROJECT_ROOT / "data" / "raw" / f"{self.symbol}_cot.parquet"

    @property
    def gdelt_snapshot_path(self) -> Path:
        """Pinned GDELT snapshot. GDELT is not point-in-time reproducible, so
        this file IS the sample — build_dataset.py and predict.py only ever
        read it, never silently refetch."""
        return PROJECT_ROOT / "data" / "raw" / f"{self.symbol}_gdelt.parquet"

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
        holdout_start=data_raw.get("holdout_start"),
        dukascopy=DukascopyConfig(**(data_raw.get("dukascopy") or {})),
        ig=IGConfig(**(data_raw.get("ig") or {})),
    )
    return AssetConfig(
        asset=raw["asset"],
        symbol=raw["symbol"],
        granularity=raw["granularity"],
        data=data,
        features=FeatureConfig(**raw.get("features", {})),
        crossasset=CrossAssetConfig(**raw.get("crossasset", {})),
        cot=COTConfig(**raw.get("cot", {})),
        gdelt=GDELTConfig(**raw.get("gdelt", {})),
        primary=PrimaryConfig(**raw.get("primary", {})),
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
