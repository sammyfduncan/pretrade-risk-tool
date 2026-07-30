"""Timestamp and return-parity diagnostics for training and broker feeds."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParityThresholds:
    min_matched_bars: int = 50
    pass_return_correlation: float = 0.99
    caution_return_correlation: float = 0.95
    pass_coverage: float = 0.95
    caution_coverage: float = 0.80
    material_shift_improvement: float = 0.005


@dataclass(frozen=True)
class SourceParityReport:
    verdict: str
    matched_bars: int
    contiguous_return_pairs: int
    ig_coverage: float
    dukascopy_coverage: float
    return_correlation: float
    tracking_error: float
    median_level_difference: float
    median_level_difference_pct: float
    best_shift_bars: int
    best_shift_correlation: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def compare_price_sources(
    dukascopy: pd.DataFrame,
    ig: pd.DataFrame,
    *,
    bar_duration: pd.Timedelta,
    thresholds: ParityThresholds | None = None,
    max_shift_bars: int = 2,
) -> SourceParityReport:
    """Compare exact timestamps and contiguous close-to-close returns."""

    thresholds = thresholds or ParityThresholds()
    if bar_duration <= pd.Timedelta(0):
        raise ValueError("bar_duration must be positive")
    if isinstance(max_shift_bars, bool) or not isinstance(max_shift_bars, int):
        raise ValueError("max_shift_bars must be an integer")
    if max_shift_bars < 0:
        raise ValueError("max_shift_bars cannot be negative")

    duka_close = _close_series(dukascopy, "dukascopy")
    ig_close = _close_series(ig, "ig")
    window_start = max(duka_close.index[0], ig_close.index[0])
    window_end = min(duka_close.index[-1], ig_close.index[-1])
    if window_start > window_end:
        raise ValueError("The two sources have no overlapping time window")

    duka_window = duka_close.loc[window_start:window_end]
    ig_window = ig_close.loc[window_start:window_end]
    overlap = duka_window.index.intersection(ig_window.index)
    if len(overlap) < thresholds.min_matched_bars:
        raise ValueError(
            f"Only {len(overlap)} exact timestamps overlap; need at least "
            f"{thresholds.min_matched_bars}"
        )

    aligned = pd.concat(
        [
            duka_window.loc[overlap].rename("dukascopy"),
            ig_window.loc[overlap].rename("ig"),
        ],
        axis=1,
    )
    returns = _contiguous_returns(aligned, bar_duration)
    if len(returns) < thresholds.min_matched_bars - 1:
        raise ValueError(
            f"Only {len(returns)} contiguous matched returns; check missing bars"
        )
    correlation = float(returns["dukascopy"].corr(returns["ig"]))
    tracking_error = float((returns["ig"] - returns["dukascopy"]).std())
    if not np.isfinite(correlation) or not np.isfinite(tracking_error):
        raise ValueError("Matched returns have insufficient variation for parity")

    shift_scores: dict[int, float] = {}
    for shift in range(-max_shift_bars, max_shift_bars + 1):
        shifted = ig_close.copy()
        shifted.index = shifted.index + shift * bar_duration
        shifted_overlap = duka_close.index.intersection(shifted.index)
        if len(shifted_overlap) < thresholds.min_matched_bars:
            shift_scores[shift] = float("nan")
            continue
        pair = pd.concat(
            [
                duka_close.loc[shifted_overlap].rename("dukascopy"),
                shifted.loc[shifted_overlap].rename("ig"),
            ],
            axis=1,
        )
        shifted_returns = _contiguous_returns(pair, bar_duration)
        shift_scores[shift] = (
            float(shifted_returns["dukascopy"].corr(shifted_returns["ig"]))
            if len(shifted_returns) >= thresholds.min_matched_bars - 1
            else float("nan")
        )

    finite_shifts = {
        shift: score for shift, score in shift_scores.items() if np.isfinite(score)
    }
    if not finite_shifts:
        raise ValueError("No timestamp shift has enough variable returns to compare")
    best_shift = max(finite_shifts, key=finite_shifts.get)
    best_shift_correlation = finite_shifts[best_shift]
    shifted_materially_better = (
        best_shift != 0
        and best_shift_correlation
        >= correlation + thresholds.material_shift_improvement
    )

    level_difference = aligned["ig"] - aligned["dukascopy"]
    ig_coverage = len(overlap) / len(ig_window)
    duka_coverage = len(overlap) / len(duka_window)
    coverage = min(ig_coverage, duka_coverage)

    reasons: list[str] = []
    if shifted_materially_better:
        reasons.append(f"timestamp_shift_suspected:{best_shift:+d}_bars")
    if coverage < thresholds.pass_coverage:
        reasons.append("timestamp_coverage_below_95pct")
    if correlation < thresholds.pass_return_correlation:
        reasons.append("return_correlation_below_0.99")

    if (
        not shifted_materially_better
        and coverage >= thresholds.pass_coverage
        and correlation >= thresholds.pass_return_correlation
    ):
        verdict = "PASS"
    elif (
        not shifted_materially_better
        and coverage >= thresholds.caution_coverage
        and correlation >= thresholds.caution_return_correlation
    ):
        verdict = "CAUTION"
    else:
        verdict = "FAIL"

    return SourceParityReport(
        verdict=verdict,
        matched_bars=len(overlap),
        contiguous_return_pairs=len(returns),
        ig_coverage=ig_coverage,
        dukascopy_coverage=duka_coverage,
        return_correlation=correlation,
        tracking_error=tracking_error,
        median_level_difference=float(level_difference.median()),
        median_level_difference_pct=float(
            level_difference.median() / aligned["dukascopy"].median()
        ),
        best_shift_bars=best_shift,
        best_shift_correlation=best_shift_correlation,
        reasons=tuple(reasons),
    )


def _close_series(candles: pd.DataFrame, source: str) -> pd.Series:
    if "close" not in candles.columns:
        raise ValueError(f"{source} candles have no close column")
    if not isinstance(candles.index, pd.DatetimeIndex) or candles.index.tz is None:
        raise ValueError(f"{source} candles need a timezone-aware DatetimeIndex")
    if not candles.index.is_monotonic_increasing or not candles.index.is_unique:
        raise ValueError(f"{source} candle index must be sorted and unique")
    close = candles["close"].astype(float).copy()
    close.index = close.index.tz_convert("UTC")
    if not np.isfinite(close.to_numpy()).all() or (close <= 0).any():
        raise ValueError(f"{source} close contains invalid prices")
    return close


def _contiguous_returns(
    aligned_prices: pd.DataFrame,
    bar_duration: pd.Timedelta,
) -> pd.DataFrame:
    returns = np.log(aligned_prices).diff()
    contiguous = aligned_prices.index.to_series().diff().eq(bar_duration)
    return returns.loc[contiguous.to_numpy()].dropna()
