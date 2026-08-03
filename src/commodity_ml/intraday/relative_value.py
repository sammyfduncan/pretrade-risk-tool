"""Causal gold–silver state and two-leg execution for frozen RV1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import time
from typing import Literal

import numpy as np
import pandas as pd

from commodity_ml.data.schema import validate_candles
from commodity_ml.intraday.data import LONDON, M30_DURATION
from commodity_ml.intraday.execution import EntryBlackout

HEDGE_OBSERVATIONS = 1_000
REFERENCE_OBSERVATIONS = 60
MIN_BETA = 0.10
MAX_BETA = 1.50
MIN_CORRELATION = 0.50
Z_THRESHOLD = 2.0

ResidualSide = Literal["long_residual", "short_residual"]
RVExitReason = Literal["target", "stop", "timeout", "cutoff"]


@dataclass(frozen=True)
class SynchronizedQuotes:
    frame: pd.DataFrame
    report: dict


@dataclass(frozen=True)
class RelativeValueState:
    frame: pd.DataFrame
    report: dict


@dataclass(frozen=True)
class RelativeValueCandidate:
    signal_time: pd.Timestamp
    side: ResidualSide
    beta: float
    gold_weight: float
    silver_weight: float
    initial_r: float


@dataclass(frozen=True)
class RelativeValueExecutionConfig:
    bar_duration: pd.Timedelta = M30_DURATION
    session_timezone: str = LONDON
    session_start: time = time(8, 0)
    last_entry: time = time(18, 0)
    flat_cutoff: time = time(21, 30)
    max_holding_bars: int = 6
    additional_cost_per_side: float = 0.00005
    entry_blackouts: tuple[EntryBlackout, ...] = ()


def synchronize_quotes(
    gold_bid: pd.DataFrame,
    gold_ask: pd.DataFrame,
    silver_bid: pd.DataFrame,
    silver_ask: pd.DataFrame,
) -> SynchronizedQuotes:
    """Exact-inner-join four validated candle streams without filling."""

    inputs = {
        "gold_bid": validate_candles(gold_bid, "rv1:gold:bid"),
        "gold_ask": validate_candles(gold_ask, "rv1:gold:ask"),
        "silver_bid": validate_candles(silver_bid, "rv1:silver:bid"),
        "silver_ask": validate_candles(silver_ask, "rv1:silver:ask"),
    }
    common = inputs["gold_bid"].index
    union = inputs["gold_bid"].index
    for frame in tuple(inputs.values())[1:]:
        common = common.intersection(frame.index)
        union = union.union(frame.index)
    common = common.sort_values()
    union = union.sort_values()
    if common.empty:
        raise ValueError("RV1 quote streams have no shared timestamps")

    out = pd.DataFrame(index=pd.DatetimeIndex(common, name="time"))
    for name, frame in inputs.items():
        selected = frame.reindex(common)
        for field in ("open", "high", "low", "close"):
            out[f"{name}_{field}"] = selected[field].astype(float)
    invalid_quote = pd.Series(False, index=out.index)
    for leg in ("gold", "silver"):
        for field in ("open", "close"):
            invalid_quote |= out[f"{leg}_ask_{field}"] < out[f"{leg}_bid_{field}"]
            out[f"{leg}_mid_{field}"] = (
                out[f"{leg}_ask_{field}"] + out[f"{leg}_bid_{field}"]
            ) / 2.0
    if invalid_quote.any():
        raise ValueError(
            f"RV1 contains {int(invalid_quote.sum())} crossed open/close quotes"
        )
    if not np.isfinite(out.to_numpy()).all() or (out <= 0).any().any():
        raise ValueError("RV1 synchronized quotes contain invalid numeric values")

    report = {
        "rows_by_stream": {name: len(frame) for name, frame in inputs.items()},
        "shared_rows": len(out),
        "union_rows": len(union),
        "shared_coverage": len(out) / len(union),
        "first": out.index.min().isoformat(),
        "last": out.index.max().isoformat(),
        "crossed_open_close_quotes": 0,
        "filled_rows": 0,
    }
    return SynchronizedQuotes(frame=out, report=report)


def build_relative_value_state(quotes: pd.DataFrame) -> RelativeValueState:
    """Estimate the prior-only hedge and same-clock robust residual surprise."""

    _validate_quotes(quotes)
    gold_return = np.log(quotes["gold_mid_close"] / quotes["gold_mid_open"])
    silver_return = np.log(quotes["silver_mid_close"] / quotes["silver_mid_open"])
    prior_gold = gold_return.shift(1)
    prior_silver = silver_return.shift(1)
    denominator = prior_silver.pow(2).rolling(
        HEDGE_OBSERVATIONS,
        min_periods=HEDGE_OBSERVATIONS,
    ).sum()
    beta = (prior_gold * prior_silver).rolling(
        HEDGE_OBSERVATIONS,
        min_periods=HEDGE_OBSERVATIONS,
    ).sum() / denominator
    correlation = prior_gold.rolling(
        HEDGE_OBSERVATIONS,
        min_periods=HEDGE_OBSERVATIONS,
    ).corr(prior_silver)
    residual = gold_return - beta * silver_return
    decision_time = quotes.index + M30_DURATION
    london = decision_time.tz_convert(LONDON)
    slot = pd.Series(london.hour * 60 + london.minute, index=quotes.index)
    median, mad, count = _prior_same_slot_reference(residual, slot)
    scale = 1.4826 * mad
    z = ((residual - median) / scale).where(
        (count >= REFERENCE_OBSERVATIONS) & scale.gt(0)
    )
    eligible = (
        beta.between(MIN_BETA, MAX_BETA)
        & correlation.ge(MIN_CORRELATION)
        & z.notna()
        & scale.gt(0)
    )
    state = pd.DataFrame(
        {
            "decision_time": decision_time,
            "gold_return": gold_return,
            "silver_return": silver_return,
            "beta": beta,
            "correlation": correlation,
            "residual": residual,
            "residual_median": median,
            "residual_scale": scale,
            "z": z,
            "eligible": eligible,
        },
        index=quotes.index,
    )
    state.index.name = "time"
    report = {
        "source_rows": len(quotes),
        "hedge_observations": HEDGE_OBSERVATIONS,
        "same_clock_reference_observations": REFERENCE_OBSERVATIONS,
        "eligible_rows": int(eligible.sum()),
        "first_eligible": state.index[eligible].min().isoformat()
        if eligible.any()
        else None,
        "last_eligible": state.index[eligible].max().isoformat()
        if eligible.any()
        else None,
        "ineligible_beta": int((beta.notna() & ~beta.between(MIN_BETA, MAX_BETA)).sum()),
        "ineligible_correlation": int(
            (correlation.notna() & correlation.lt(MIN_CORRELATION)).sum()
        ),
    }
    return RelativeValueState(frame=state, report=report)


def generate_relative_value_candidates(
    state: pd.DataFrame,
    *,
    config: RelativeValueExecutionConfig | None = None,
) -> list[RelativeValueCandidate]:
    """Apply the exact RV1 threshold and first-eligible-per-day rule."""

    config = config or RelativeValueExecutionConfig()
    required = {"eligible", "z", "beta", "residual_scale"}
    if missing := sorted(required - set(state)):
        raise ValueError(f"RV1 state missing columns: {missing}")
    _require_utc_index(state, "RV1 state")
    entry = state.index + config.bar_duration
    local = entry.tz_convert(config.session_timezone)
    minute = local.hour * 60 + local.minute
    in_session = (minute >= 8 * 60) & (minute < 18 * 60)
    blocked = _blackout_mask(entry, config.entry_blackouts)
    long_signal = state["eligible"].eq(True) & state["z"].le(-Z_THRESHOLD)
    short_signal = state["eligible"].eq(True) & state["z"].ge(Z_THRESHOLD)
    signal = pd.Series(in_session & ~blocked, index=state.index) & (
        long_signal | short_signal
    )
    session_dates = np.asarray(local.date)
    candidates: list[RelativeValueCandidate] = []
    for session_date in pd.unique(session_dates):
        times = state.index[signal & (session_dates == session_date)]
        if len(times) == 0:
            continue
        signal_time = times[0]
        beta = float(state.at[signal_time, "beta"])
        scale = float(state.at[signal_time, "residual_scale"])
        if not np.isfinite(beta) or not np.isfinite(scale) or scale <= 0:
            continue
        direction = 1.0 if bool(long_signal.loc[signal_time]) else -1.0
        denominator = 1.0 + beta
        candidates.append(
            RelativeValueCandidate(
                signal_time=signal_time,
                side="long_residual" if direction > 0 else "short_residual",
                beta=beta,
                gold_weight=direction / denominator,
                silver_weight=-direction * beta / denominator,
                initial_r=scale / denominator,
            )
        )
    return candidates


def simulate_relative_value(
    quotes: pd.DataFrame,
    candidates: list[RelativeValueCandidate],
    *,
    config: RelativeValueExecutionConfig | None = None,
) -> pd.DataFrame:
    """Execute RV1 chronologically with close-observed portfolio barriers."""

    config = config or RelativeValueExecutionConfig()
    _validate_quotes(quotes)
    records: list[dict] = []
    busy_until: pd.Timestamp | None = None
    for candidate in sorted(candidates, key=lambda value: value.signal_time):
        base = {
            "strategy": "RV1",
            "signal_time": candidate.signal_time,
            "side": candidate.side,
            "beta": candidate.beta,
            "gold_weight": candidate.gold_weight,
            "silver_weight": candidate.silver_weight,
            "initial_r": candidate.initial_r,
        }
        try:
            outcome = _resolve_candidate(quotes, candidate, config)
            if busy_until is not None and outcome["entry_time"] < busy_until:
                raise ValueError("position_open")
        except ValueError as exc:
            records.append(
                base
                | {
                    "status": "rejected",
                    "reject_reason": str(exc),
                    "entry_time": pd.NaT,
                    "exit_time": pd.NaT,
                }
            )
            continue
        records.append(base | outcome)
        busy_until = outcome["exit_time"]
    if not records:
        return pd.DataFrame(
            columns=[
                "strategy",
                "signal_time",
                "side",
                "status",
                "reject_reason",
                "entry_time",
                "exit_time",
            ]
        )
    return pd.DataFrame.from_records(records)


def remeasure_outcome(
    quotes: pd.DataFrame,
    row: pd.Series,
    *,
    spread_multiplier: float,
    additional_cost_per_side: float,
) -> tuple[float, float]:
    """Remeasure one fixed entry/exit under a quote-spread/cost stress."""

    if spread_multiplier <= 0 or additional_cost_per_side < 0:
        raise ValueError("RV1 cost stress values are invalid")
    entry = quotes.loc[pd.Timestamp(row["entry_time"])]
    exit_bar = quotes.loc[pd.Timestamp(row["exit_bar_time"])]
    gross = _portfolio_return(
        entry,
        exit_bar,
        gold_weight=float(row["gold_weight"]),
        silver_weight=float(row["silver_weight"]),
        spread_multiplier=spread_multiplier,
    )
    net = gross - 2.0 * additional_cost_per_side
    return gross, net


def candidate_coverage(quotes: pd.DataFrame, state: pd.DataFrame) -> dict:
    eligible = state.index[state["eligible"].eq(True)]
    if len(eligible) == 0:
        return {"eligible_quote_rows": 0, "matched_rows": 0, "coverage": 0.0}
    universe = quotes.index >= eligible.min()
    matched = quotes.index.isin(eligible) & universe
    denominator = int(universe.sum())
    return {
        "eligible_quote_rows": denominator,
        "matched_rows": int(matched.sum()),
        "coverage": float(matched.sum() / denominator) if denominator else 0.0,
        "warmup_end": eligible.min().isoformat(),
    }


def _resolve_candidate(
    quotes: pd.DataFrame,
    candidate: RelativeValueCandidate,
    config: RelativeValueExecutionConfig,
) -> dict:
    signal_time = pd.Timestamp(candidate.signal_time).tz_convert("UTC")
    position = quotes.index.get_indexer([signal_time])[0]
    if position < 0:
        raise ValueError("signal_bar_missing")
    entry_pos = position + 1
    if entry_pos >= len(quotes):
        raise ValueError("next_bar_missing")
    entry_time = quotes.index[entry_pos]
    if entry_time != signal_time + config.bar_duration:
        raise ValueError("non_contiguous_entry")
    local_entry = entry_time.tz_convert(config.session_timezone)
    local_clock = local_entry.time().replace(tzinfo=None)
    if not (config.session_start <= local_clock < config.last_entry):
        raise ValueError("outside_entry_window")
    local_cutoff = pd.Timestamp.combine(local_entry.date(), config.flat_cutoff).tz_localize(
        config.session_timezone
    )
    cutoff = local_cutoff.tz_convert("UTC")

    exit_reason: RVExitReason | None = None
    exit_pos = None
    gross_base = None
    for bars_held in range(1, config.max_holding_bars + 1):
        current_pos = entry_pos + bars_held - 1
        if current_pos >= len(quotes):
            raise ValueError("incomplete_future_path")
        expected = entry_time + (bars_held - 1) * config.bar_duration
        if quotes.index[current_pos] != expected:
            raise ValueError("non_contiguous_future_path")
        decision_time = quotes.index[current_pos] + config.bar_duration
        gross = _portfolio_return(
            quotes.iloc[entry_pos],
            quotes.iloc[current_pos],
            gold_weight=candidate.gold_weight,
            silver_weight=candidate.silver_weight,
            spread_multiplier=1.0,
        )
        if gross >= candidate.initial_r:
            exit_reason = "target"
        elif gross <= -candidate.initial_r:
            exit_reason = "stop"
        elif decision_time >= cutoff:
            exit_reason = "cutoff"
        elif bars_held == config.max_holding_bars:
            exit_reason = "timeout"
        if exit_reason is not None:
            exit_pos = current_pos
            gross_base = gross
            break
    if exit_pos is None or exit_reason is None or gross_base is None:
        raise AssertionError("RV1 resolver failed to select an exit")
    exit_bar_time = quotes.index[exit_pos]
    exit_time = exit_bar_time + config.bar_duration
    net_base = gross_base - 2.0 * config.additional_cost_per_side
    return {
        "status": "executed",
        "reject_reason": None,
        "entry_time": entry_time,
        "exit_bar_time": exit_bar_time,
        "exit_time": exit_time,
        "exit_reason": exit_reason,
        "bars_held": exit_pos - entry_pos + 1,
        "gross_return": gross_base,
        "net_return": net_base,
        "r_multiple": net_base / candidate.initial_r,
        "label": int(net_base > 0),
    }


def _portfolio_return(
    entry: pd.Series,
    exit_bar: pd.Series,
    *,
    gold_weight: float,
    silver_weight: float,
    spread_multiplier: float,
) -> float:
    total = 0.0
    for leg, weight in (("gold", gold_weight), ("silver", silver_weight)):
        bid_open, ask_open = _stressed_pair(
            float(entry[f"{leg}_bid_open"]),
            float(entry[f"{leg}_ask_open"]),
            spread_multiplier,
        )
        bid_close, ask_close = _stressed_pair(
            float(exit_bar[f"{leg}_bid_close"]),
            float(exit_bar[f"{leg}_ask_close"]),
            spread_multiplier,
        )
        if weight > 0:
            leg_return = bid_close / ask_open - 1.0
        else:
            leg_return = 1.0 - ask_close / bid_open
        total += abs(weight) * leg_return
    return float(total)


def _stressed_pair(bid: float, ask: float, multiplier: float) -> tuple[float, float]:
    midpoint = (bid + ask) / 2.0
    return (
        midpoint - multiplier * (midpoint - bid),
        midpoint + multiplier * (ask - midpoint),
    )


def _prior_same_slot_reference(
    values: pd.Series,
    slot: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = pd.Series(np.nan, index=values.index, dtype=float)
    mad = pd.Series(np.nan, index=values.index, dtype=float)
    count = pd.Series(0, index=values.index, dtype="int64")
    for _, positions in slot.groupby(slot, sort=False).groups.items():
        observed = values.loc[positions].dropna()
        if observed.empty:
            continue
        prior = observed.shift(1)
        median.loc[observed.index] = prior.rolling(
            REFERENCE_OBSERVATIONS,
            min_periods=REFERENCE_OBSERVATIONS,
        ).median()
        mad.loc[observed.index] = prior.rolling(
            REFERENCE_OBSERVATIONS,
            min_periods=REFERENCE_OBSERVATIONS,
        ).apply(lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True)
        count.loc[observed.index] = prior.expanding().count().clip(
            upper=REFERENCE_OBSERVATIONS
        ).astype("int64")
    return median, mad, count


def _blackout_mask(
    entry: pd.DatetimeIndex,
    blackouts: tuple[EntryBlackout, ...],
) -> np.ndarray:
    blocked = np.zeros(len(entry), dtype=bool)
    for blackout in blackouts:
        blocked |= (entry >= pd.Timestamp(blackout.start).tz_convert("UTC")) & (
            entry < pd.Timestamp(blackout.end).tz_convert("UTC")
        )
    return blocked


def _validate_quotes(frame: pd.DataFrame) -> None:
    required = {
        f"{leg}_{side}_{field}"
        for leg in ("gold", "silver")
        for side in ("bid", "ask")
        for field in ("open", "close")
    } | {
        f"{leg}_mid_{field}"
        for leg in ("gold", "silver")
        for field in ("open", "close")
    }
    if missing := sorted(required - set(frame)):
        raise ValueError(f"RV1 quotes missing columns: {missing}")
    _require_utc_index(frame, "RV1 quotes")
    values = frame.loc[:, sorted(required)].astype(float)
    if not np.isfinite(values.to_numpy()).all() or (values <= 0).any().any():
        raise ValueError("RV1 quotes contain invalid values")
    for leg in ("gold", "silver"):
        for field in ("open", "close"):
            if (frame[f"{leg}_ask_{field}"] < frame[f"{leg}_bid_{field}"]).any():
                raise ValueError("RV1 quotes contain crossed markets")


def _require_utc_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{label} requires a timezone-aware DatetimeIndex")
    if str(frame.index.tz) != "UTC":
        raise ValueError(f"{label} index must be UTC")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{label} index must be sorted and unique")


def candidate_record(candidate: RelativeValueCandidate) -> dict:
    return asdict(candidate)
