"""Validate executable bid/ask paths and measure realised paper-trade risk."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from commodity_ml.intraday.pretrade_observations import PaperTradeOutcome


Side = Literal["long", "short"]


@dataclass(frozen=True)
class TradeExecutionRecord:
    """Observed broker facts required to construct one paper outcome."""

    intent_id: str
    side: Side
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    filled_size: float
    expected_entry_price: float
    actual_entry_price: float
    actual_exit_price: float
    value_per_price_unit: float
    fees: float
    peak_margin: float
    realized_pnl: float
    source: str
    source_reference: str
    notes: str | None = None

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError("execution side must be long or short")
        if not self.intent_id or not self.source or not self.source_reference:
            raise ValueError(
                "intent_id, source and source_reference must be non-empty"
            )
        entry = _utc(self.entry_time, "entry_time")
        exit_time = _utc(self.exit_time, "exit_time")
        if exit_time <= entry:
            raise ValueError("exit_time must be after entry_time")
        for name in (
            "filled_size",
            "expected_entry_price",
            "actual_entry_price",
            "actual_exit_price",
            "value_per_price_unit",
        ):
            if not _positive(getattr(self, name)):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("fees", "peak_margin"):
            if not _non_negative(getattr(self, name)):
                raise ValueError(f"{name} must be finite and non-negative")
        if not _finite(self.realized_pnl):
            raise ValueError("realized_pnl must be finite")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TradeExecutionRecord":
        fields = set(cls.__dataclass_fields__)
        unknown = sorted(set(value) - fields)
        if unknown:
            raise ValueError(f"unexpected execution fields: {', '.join(unknown)}")
        missing = sorted(name for name in fields if name != "notes" and value.get(name) is None)
        if missing:
            raise ValueError(f"missing execution fields: {', '.join(missing)}")
        payload = dict(value)
        payload["entry_time"] = _utc(payload["entry_time"], "entry_time")
        payload["exit_time"] = _utc(payload["exit_time"], "exit_time")
        return cls(**payload)


@dataclass(frozen=True)
class QuotePathAudit:
    quote_count: int
    executable_side: str
    first_quote_time: str
    last_quote_time: str
    entry_boundary_gap_seconds: float
    exit_boundary_gap_seconds: float
    maximum_internal_gap_seconds: float
    allowed_boundary_gap_seconds: float
    allowed_internal_gap_seconds: float
    entry_reference_price: float
    exit_reference_price: float
    price_path_pnl_after_fees: float
    broker_pnl_difference: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def outcome_from_executable_quotes(
    execution: TradeExecutionRecord,
    quotes: pd.DataFrame,
    *,
    max_boundary_gap_seconds: float = 5.0,
    max_internal_gap_seconds: float = 120.0,
) -> tuple[PaperTradeOutcome, QuotePathAudit]:
    """Build measured MAE/MFE and execution costs from a valid quote path.

    Quotes must be top-of-book observations, not OHLC candles. For a long the
    liquidation path is bid; for a short it is ask. No midpoint substitution,
    interpolation, fill, resampling or forward-fill is performed.
    """

    if not _positive(max_boundary_gap_seconds):
        raise ValueError("max_boundary_gap_seconds must be positive")
    if not _positive(max_internal_gap_seconds):
        raise ValueError("max_internal_gap_seconds must be positive")
    frame = _validated_quotes(quotes)
    entry_time = _utc(execution.entry_time, "entry_time")
    exit_time = _utc(execution.exit_time, "exit_time")
    index = frame.index

    entry_position = int(index.searchsorted(entry_time, side="left"))
    exit_position = int(index.searchsorted(exit_time, side="right")) - 1
    if entry_position >= len(frame) or exit_position < 0 or entry_position > exit_position:
        raise ValueError("quote path does not bracket the executed holding period")
    path = frame.iloc[entry_position : exit_position + 1]
    if len(path) < 2:
        raise ValueError("quote path requires at least two observations")
    entry_gap = (path.index[0] - entry_time).total_seconds()
    exit_gap = (exit_time - path.index[-1]).total_seconds()
    if entry_gap < 0 or exit_gap < 0:
        raise RuntimeError("internal quote boundary selection error")
    if entry_gap > max_boundary_gap_seconds:
        raise ValueError(f"entry_quote_boundary_gap:{entry_gap:g}s")
    if exit_gap > max_boundary_gap_seconds:
        raise ValueError(f"exit_quote_boundary_gap:{exit_gap:g}s")
    gaps = path.index.to_series().diff().dropna().dt.total_seconds()
    maximum_gap = float(gaps.max())
    if maximum_gap > max_internal_gap_seconds:
        raise ValueError(f"internal_quote_gap:{maximum_gap:g}s")

    entry_spread = float(path["ask"].iloc[0] - path["bid"].iloc[0])
    exit_spread = float(path["ask"].iloc[-1] - path["bid"].iloc[-1])
    if execution.side == "long":
        liquidation = path["bid"]
        maximum_adverse = max(
            0.0, execution.actual_entry_price - float(liquidation.min())
        )
        maximum_favorable = max(
            0.0, float(liquidation.max()) - execution.actual_entry_price
        )
        entry_slippage = max(
            0.0, execution.actual_entry_price - execution.expected_entry_price
        )
        exit_reference = float(path["bid"].iloc[-1])
        exit_slippage = max(0.0, exit_reference - execution.actual_exit_price)
        direction = 1.0
        executable_side = "bid"
        entry_reference = float(path["ask"].iloc[0])
    else:
        liquidation = path["ask"]
        maximum_adverse = max(
            0.0, float(liquidation.max()) - execution.actual_entry_price
        )
        maximum_favorable = max(
            0.0, execution.actual_entry_price - float(liquidation.min())
        )
        entry_slippage = max(
            0.0, execution.expected_entry_price - execution.actual_entry_price
        )
        exit_reference = float(path["ask"].iloc[-1])
        exit_slippage = max(0.0, execution.actual_exit_price - exit_reference)
        direction = -1.0
        executable_side = "ask"
        entry_reference = float(path["bid"].iloc[0])

    path_pnl = (
        direction
        * (execution.actual_exit_price - execution.actual_entry_price)
        * execution.filled_size
        * execution.value_per_price_unit
        - execution.fees
    )
    outcome = PaperTradeOutcome(
        intent_id=execution.intent_id,
        entry_time=entry_time,
        exit_time=exit_time,
        filled_size=execution.filled_size,
        expected_entry_price=execution.expected_entry_price,
        actual_entry_price=execution.actual_entry_price,
        actual_exit_price=execution.actual_exit_price,
        max_adverse_excursion_points=maximum_adverse,
        max_favorable_excursion_points=maximum_favorable,
        entry_spread_points=entry_spread,
        exit_spread_points=exit_spread,
        entry_adverse_slippage_points=entry_slippage,
        exit_adverse_slippage_points=exit_slippage,
        fees=execution.fees,
        peak_margin=execution.peak_margin,
        realized_pnl=execution.realized_pnl,
        source=execution.source,
        source_reference=execution.source_reference,
        notes=execution.notes,
    )
    audit = QuotePathAudit(
        quote_count=len(path),
        executable_side=executable_side,
        first_quote_time=path.index[0].isoformat(),
        last_quote_time=path.index[-1].isoformat(),
        entry_boundary_gap_seconds=float(entry_gap),
        exit_boundary_gap_seconds=float(exit_gap),
        maximum_internal_gap_seconds=maximum_gap,
        allowed_boundary_gap_seconds=float(max_boundary_gap_seconds),
        allowed_internal_gap_seconds=float(max_internal_gap_seconds),
        entry_reference_price=entry_reference,
        exit_reference_price=exit_reference,
        price_path_pnl_after_fees=float(path_pnl),
        broker_pnl_difference=float(execution.realized_pnl - path_pnl),
    )
    return outcome, audit


def _validated_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(quotes, pd.DataFrame) or quotes.empty:
        raise ValueError("quote path must be a non-empty DataFrame")
    if not {"bid", "ask"}.issubset(quotes.columns):
        raise ValueError("quote path requires bid and ask columns")
    index = pd.DatetimeIndex(quotes.index)
    if index.tz is None:
        raise ValueError("quote timestamps must be timezone-aware")
    if not index.is_monotonic_increasing:
        raise ValueError("quote timestamps must be sorted")
    if index.has_duplicates:
        raise ValueError("quote timestamps must be unique")
    frame = quotes.loc[:, ["bid", "ask"]].copy()
    frame.index = index.tz_convert("UTC")
    frame["bid"] = pd.to_numeric(frame["bid"], errors="coerce")
    frame["ask"] = pd.to_numeric(frame["ask"], errors="coerce")
    values = frame[["bid", "ask"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("quote path prices must be finite and positive")
    if (frame["ask"] < frame["bid"]).any():
        raise ValueError("quote path contains crossed quotes")
    return frame


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _finite(value: Any) -> bool:
    return _number(value) is not None


def _positive(value: Any) -> bool:
    number = _number(value)
    return number is not None and number > 0


def _non_negative(value: Any) -> bool:
    number = _number(value)
    return number is not None and number >= 0


def _utc(value: Any, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")
