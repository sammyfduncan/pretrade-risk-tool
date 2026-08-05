"""One executable trade contract for intraday labels and backtests.

Version-1 semantics are pre-registered in ``docs/archive/intraday_rebuild_plan.md`` §2:

* candle timestamps are bar-open times;
* a signal is known only when its bar closes;
* entry is the next contiguous bar open;
* stop and target are anchored to that actual entry;
* actual high/low determine barrier touches;
* a bar touching both barriers resolves to the stop;
* timeout and the London daily cutoff exit at a bar close; and
* labels describe net results after spread and explicit round-trip costs.

The historical vectorbt/H4 path is deliberately not reused. Keeping this small
engine separate preserves the frozen record while making labels and simulated
trades structurally identical for the new program.

The engine accepts either mid OHLC alone or a complete set of bid/ask OHLC
columns alongside mid OHLC. With quotes, long trades buy at ask and sell/check
barriers at bid; short trades sell at bid and buy/check barriers at ask. The
configured per-side cost is therefore an *additional* commission/slippage
allowance, not a spread proxy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, time
from typing import Literal, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from commodity_ml.data.schema import OHLC_COLUMNS, QUOTE_COLUMNS

Side = Literal["long", "short"]
ExitReason = Literal["target", "stop", "timeout", "cutoff"]
PriceSource = Literal["mid", "quotes"]

_OHLC = tuple(OHLC_COLUMNS)
_QUOTE_OHLC = tuple(QUOTE_COLUMNS)


class CandidateRejected(ValueError):
    """A candidate cannot be executed under the frozen intraday contract."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class EntryBlackout:
    """A half-open UTC interval in which new entries are forbidden."""

    start: pd.Timestamp
    end: pd.Timestamp
    reason: str = "scheduled_event"

    def __post_init__(self) -> None:
        start, end = _utc(self.start), _utc(self.end)
        if start >= end:
            raise ValueError("EntryBlackout requires start < end")
        if not self.reason:
            raise ValueError("EntryBlackout reason must be non-empty")


@dataclass(frozen=True)
class SessionCutoffOverride:
    """Forced-flat time for one London session, already adjusted for closure."""

    session_date: date
    flat_cutoff: time


@dataclass(frozen=True)
class IntradayExecutionConfig:
    """Execution rules shared by labeling, research, and later paper trading."""

    bar_duration: pd.Timedelta = pd.Timedelta(minutes=30)
    session_timezone: str = "Europe/London"
    session_start: time = time(7, 0)
    last_entry: time = time(19, 30)
    flat_cutoff: time = time(21, 30)
    # Additional non-spread friction. When bid/ask candles are supplied, the
    # spread is already embedded in entry, exit, and barrier prices.
    cost_pct_per_side: float = 0.0001
    entry_blackouts: tuple[EntryBlackout, ...] = ()
    cutoff_overrides: tuple[SessionCutoffOverride, ...] = ()
    max_entries_per_day: int = 3
    strategy_daily_entry_limits: tuple[tuple[str, int], ...] = (
        ("P1", 1),
        ("P2", 2),
        ("P3", 1),
        ("P1V2", 1),
        ("P2V2", 2),
        ("P3V2", 1),
        ("P4", 1),
    )

    def __post_init__(self) -> None:
        if self.bar_duration <= pd.Timedelta(0):
            raise ValueError("bar_duration must be positive")
        if not (self.session_start < self.last_entry < self.flat_cutoff):
            raise ValueError("Require session_start < last_entry < flat_cutoff")
        if self.cost_pct_per_side < 0:
            raise ValueError("cost_pct_per_side cannot be negative")
        if (
            isinstance(self.max_entries_per_day, bool)
            or not isinstance(self.max_entries_per_day, int)
            or self.max_entries_per_day < 1
        ):
            raise ValueError("max_entries_per_day must be a positive integer")
        strategy_names = [name for name, _ in self.strategy_daily_entry_limits]
        if len(strategy_names) != len(set(strategy_names)):
            raise ValueError("strategy_daily_entry_limits contains duplicates")
        if any(
            not name
            or isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            for name, limit in self.strategy_daily_entry_limits
        ):
            raise ValueError("strategy daily entry limits must be positive integers")
        minutes = self.bar_duration / pd.Timedelta(minutes=1)
        if not float(minutes).is_integer():
            raise ValueError("bar_duration must be a whole number of minutes")
        minutes = int(minutes)
        if 24 * 60 % minutes:
            raise ValueError("bar_duration must divide a 24-hour day")
        for clock in (self.session_start, self.last_entry, self.flat_cutoff):
            if (clock.hour * 60 + clock.minute) % minutes:
                raise ValueError("session clocks must align to bar_duration")
        override_dates = [override.session_date for override in self.cutoff_overrides]
        if len(override_dates) != len(set(override_dates)):
            raise ValueError("Only one cutoff override is allowed per session date")
        for override in self.cutoff_overrides:
            if (override.flat_cutoff.hour * 60 + override.flat_cutoff.minute) % minutes:
                raise ValueError("cutoff overrides must align to bar_duration")
        try:
            ZoneInfo(self.session_timezone)
        except Exception as exc:  # pragma: no cover - platform timezone failure
            raise ValueError(f"Unknown session timezone: {self.session_timezone}") from exc


@dataclass(frozen=True)
class TradeCandidate:
    """A signal known at the close of ``signal_time``'s candle."""

    signal_time: pd.Timestamp
    side: Side
    stop_distance_pct: float | None
    target_distance_pct: float | None
    max_holding_bars: int
    strategy: str
    # ATR-based primary rules freeze an absolute distance at signal close and
    # anchor it to the actual next-bar entry. Percentage fields remain for
    # legacy/synthetic callers. Exactly one stop mode and one target mode apply.
    stop_distance_abs: float | None = None
    target_r_multiple: float | None = None

    def __post_init__(self) -> None:
        if self.side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")
        stop_modes = (
            self.stop_distance_pct is not None,
            self.stop_distance_abs is not None,
        )
        if sum(stop_modes) != 1:
            raise ValueError("Configure exactly one stop distance mode")
        if self.stop_distance_pct is not None and self.stop_distance_pct <= 0:
            raise ValueError("stop_distance_pct must be positive")
        if self.stop_distance_abs is not None and self.stop_distance_abs <= 0:
            raise ValueError("stop_distance_abs must be positive")
        target_modes = (
            self.target_distance_pct is not None,
            self.target_r_multiple is not None,
        )
        if sum(target_modes) != 1:
            raise ValueError("Configure exactly one target distance mode")
        if self.target_distance_pct is not None and self.target_distance_pct <= 0:
            raise ValueError("target_distance_pct must be positive")
        if self.target_r_multiple is not None and self.target_r_multiple <= 0:
            raise ValueError("target_r_multiple must be positive")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars must be >= 1")
        if not self.strategy:
            raise ValueError("strategy must be non-empty")


@dataclass(frozen=True)
class TradeOutcome:
    """The complete, label-ready result of one executed candidate."""

    strategy: str
    side: Side
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    stop_price: float
    target_price: float
    exit_reason: ExitReason
    bars_held: int
    gross_return: float
    net_return: float
    initial_r_pct: float
    r_multiple: float
    label: int
    price_source: PriceSource

    def to_record(self) -> dict:
        record = asdict(self)
        record["status"] = "executed"
        record["reject_reason"] = None
        return record


def resolve_trade(
    candles: pd.DataFrame,
    candidate: TradeCandidate,
    cfg: IntradayExecutionConfig | None = None,
) -> TradeOutcome:
    """Resolve one candidate without compressing or inventing bars.

    Raises ``CandidateRejected`` when the next executable bar is absent, lies
    outside the entry window, or the future path is incomplete. A rejected
    signal is not queued for a later price.
    """

    cfg = cfg or IntradayExecutionConfig()
    frame = _validate_candles(candles)
    entry_pos, signal_time, entry_time, cutoff = _entry_details(frame, candidate, cfg)

    price_source: PriceSource = "quotes" if _has_quotes(frame) else "mid"
    entry_price = _entry_price(frame.iloc[entry_pos], candidate.side, price_source)
    stop_distance = (
        candidate.stop_distance_abs
        if candidate.stop_distance_abs is not None
        else entry_price * candidate.stop_distance_pct
    )
    target_distance = (
        stop_distance * candidate.target_r_multiple
        if candidate.target_r_multiple is not None
        else entry_price * candidate.target_distance_pct
    )
    if candidate.side == "long":
        stop_price = entry_price - stop_distance
        target_price = entry_price + target_distance
    else:
        stop_price = entry_price + stop_distance
        target_price = entry_price - target_distance

    last_allowed_pos = entry_pos + candidate.max_holding_bars - 1
    for pos in range(entry_pos, last_allowed_pos + 1):
        if pos >= len(frame):
            raise CandidateRejected("insufficient_future")

        expected_start = entry_time + (pos - entry_pos) * cfg.bar_duration
        bar_start = frame.index[pos]
        if bar_start != expected_start:
            raise CandidateRejected("non_contiguous_future")

        bar = _executable_bar(frame.iloc[pos], candidate.side, price_source)
        bar_end = bar_start + cfg.bar_duration
        exit_price, exit_reason = _barrier_exit(
            bar=bar,
            side=candidate.side,
            stop_price=stop_price,
            target_price=target_price,
        )

        if exit_reason is None and bar_end >= cutoff:
            exit_price = float(bar["close"])
            exit_reason = "cutoff"
        elif exit_reason is None and pos == last_allowed_pos:
            exit_price = float(bar["close"])
            exit_reason = "timeout"

        if exit_reason is not None:
            return _outcome(
                candidate=candidate,
                signal_time=signal_time,
                entry_time=entry_time,
                exit_time=min(bar_end, cutoff),
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
                target_price=target_price,
                exit_reason=exit_reason,
                bars_held=pos - entry_pos + 1,
                cfg=cfg,
                price_source=price_source,
                initial_r_pct=stop_distance / entry_price,
            )

    raise CandidateRejected("insufficient_future")  # defensive; loop always exits or raises


def simulate_candidates(
    candles: pd.DataFrame,
    candidates: Sequence[TradeCandidate],
    cfg: IntradayExecutionConfig | None = None,
) -> pd.DataFrame:
    """Execute candidates chronologically with one-position, no-queue behavior.

    The returned ledger contains both executions and reason-coded rejections.
    This prevents an attractive backtest from silently discarding late,
    overlapping, or malformed candidates.
    """

    cfg = cfg or IntradayExecutionConfig()
    frame = _validate_candles(candles)
    ordered = sorted(enumerate(candidates), key=lambda item: (_utc(item[1].signal_time), item[0]))
    busy_until: pd.Timestamp | None = None
    daily_entries: dict[date, int] = {}
    strategy_daily_entries: dict[tuple[date, str], int] = {}
    strategy_limits = dict(cfg.strategy_daily_entry_limits)
    records: list[dict] = []

    for _, candidate in ordered:
        signal_time = _utc(candidate.signal_time)
        base = {
            "strategy": candidate.strategy,
            "side": candidate.side,
            "signal_time": signal_time,
        }
        try:
            _, _, entry_time, _ = _entry_details(frame, candidate, cfg)
            if busy_until is not None and entry_time < busy_until:
                raise CandidateRejected("position_open")
            entry_date = entry_time.tz_convert(cfg.session_timezone).date()
            if daily_entries.get(entry_date, 0) >= cfg.max_entries_per_day:
                raise CandidateRejected("daily_entry_limit")
            strategy_limit = strategy_limits.get(candidate.strategy)
            if (
                strategy_limit is not None
                and strategy_daily_entries.get((entry_date, candidate.strategy), 0)
                >= strategy_limit
            ):
                raise CandidateRejected("strategy_daily_entry_limit")
            outcome = resolve_trade(frame, candidate, cfg)
        except CandidateRejected as exc:
            records.append(
                {
                    **base,
                    "status": "rejected",
                    "reject_reason": exc.reason,
                    "entry_time": pd.NaT,
                    "exit_time": pd.NaT,
                    "label": np.nan,
                    "net_return": np.nan,
                }
            )
            continue

        records.append(outcome.to_record())
        busy_until = outcome.exit_time
        daily_entries[entry_date] = daily_entries.get(entry_date, 0) + 1
        strategy_key = (entry_date, candidate.strategy)
        strategy_daily_entries[strategy_key] = (
            strategy_daily_entries.get(strategy_key, 0) + 1
        )

    if not records:
        return pd.DataFrame(
            columns=[
                "strategy",
                "side",
                "signal_time",
                "status",
                "reject_reason",
                "entry_time",
                "exit_time",
                "label",
                "net_return",
            ]
        )
    return pd.DataFrame.from_records(records)


def labels_from_ledger(ledger: pd.DataFrame) -> pd.Series:
    """Return candidate labels from the exact trades the simulator executed."""

    required = {"status", "signal_time", "label", "net_return"}
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"Ledger missing required columns: {sorted(missing)}")

    executed = ledger.loc[ledger["status"] == "executed"].copy()
    if executed.empty:
        return pd.Series(dtype="int8", name="label")

    expected = (executed["net_return"].astype(float) > 0).astype("int8")
    actual = executed["label"].astype("int8")
    if not expected.equals(actual):
        raise ValueError("Ledger labels diverge from net executed outcomes")

    return pd.Series(
        actual.to_numpy(),
        index=pd.DatetimeIndex(executed["signal_time"]),
        name="label",
        dtype="int8",
    )


def _validate_candles(candles: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in _OHLC if column not in candles.columns]
    if missing:
        raise ValueError(f"Candles missing OHLC columns: {missing}")
    if not isinstance(candles.index, pd.DatetimeIndex):
        raise ValueError("Candles must have a DatetimeIndex")
    if candles.index.tz is None:
        raise ValueError("Candle index must be timezone-aware")
    if not candles.index.is_monotonic_increasing or not candles.index.is_unique:
        raise ValueError("Candle index must be sorted and unique")

    quote_columns_present = [column for column in _QUOTE_OHLC if column in candles.columns]
    if quote_columns_present and len(quote_columns_present) != len(_QUOTE_OHLC):
        missing_quotes = sorted(set(_QUOTE_OHLC) - set(quote_columns_present))
        raise ValueError(
            "Bid/ask candles must provide every quote OHLC column; "
            f"missing: {missing_quotes}"
        )

    columns = (*_OHLC, *_QUOTE_OHLC) if quote_columns_present else _OHLC
    frame = candles.loc[:, columns].astype(float)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Candles contain non-finite OHLC values")
    if (frame <= 0).any().any():
        raise ValueError("Candles contain non-positive OHLC values")

    for prefix in ("", "bid_", "ask_"):
        if prefix and not quote_columns_present:
            continue
        names = {field: f"{prefix}{field}" for field in _OHLC}
        bad = (
            (frame[names["high"]] < frame[[names["open"], names["close"]]].max(axis=1))
            | (frame[names["low"]] > frame[[names["open"], names["close"]]].min(axis=1))
            | (frame[names["high"]] < frame[names["low"]])
        )
        if bad.any():
            label = prefix.rstrip("_") or "mid"
            raise ValueError(
                f"{int(bad.sum())} {label} candles violate OHLC ranges"
            )

    if quote_columns_present:
        crossed = pd.DataFrame(
            {
                field: frame[f"ask_{field}"] < frame[f"bid_{field}"]
                for field in _OHLC
            }
        )
        if crossed.any().any():
            raise ValueError("Bid/ask candles contain crossed quote OHLC values")

    frame.index = frame.index.tz_convert("UTC")
    return frame


def _entry_details(
    candles: pd.DataFrame,
    candidate: TradeCandidate,
    cfg: IntradayExecutionConfig,
) -> tuple[int, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    signal_time = _utc(candidate.signal_time)
    position = candles.index.get_indexer([signal_time])[0]
    if position < 0:
        raise CandidateRejected("signal_bar_missing")
    if position + 1 >= len(candles):
        raise CandidateRejected("next_bar_missing")

    entry_time = candles.index[position + 1]
    if entry_time != signal_time + cfg.bar_duration:
        raise CandidateRejected("non_contiguous_entry")

    local_entry = entry_time.tz_convert(cfg.session_timezone)
    local_clock = local_entry.time().replace(tzinfo=None)
    cutoff_clock = dict(
        (override.session_date, override.flat_cutoff) for override in cfg.cutoff_overrides
    ).get(local_entry.date(), cfg.flat_cutoff)
    local_cutoff = pd.Timestamp.combine(local_entry.date(), cutoff_clock).tz_localize(
        cfg.session_timezone
    )
    cutoff = local_cutoff.tz_convert("UTC")

    latest_by_cutoff = (local_cutoff - cfg.bar_duration).time().replace(tzinfo=None)
    effective_last_entry = min(cfg.last_entry, latest_by_cutoff)
    if not (cfg.session_start <= local_clock < effective_last_entry):
        raise CandidateRejected("outside_entry_window")

    for blackout in cfg.entry_blackouts:
        if _utc(blackout.start) <= entry_time < _utc(blackout.end):
            raise CandidateRejected(f"entry_blackout:{blackout.reason}")

    return position + 1, signal_time, entry_time, cutoff


def _barrier_exit(
    bar: pd.Series,
    side: Side,
    stop_price: float,
    target_price: float,
) -> tuple[float, ExitReason | None]:
    open_price = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])

    # A gap through the stop fills at the worse opening price. A favorable gap
    # receives only the registered target, not an optimistic opening windfall.
    if side == "long":
        if open_price <= stop_price:
            return open_price, "stop"
        if open_price >= target_price:
            return target_price, "target"
        touched_stop = low <= stop_price
        touched_target = high >= target_price
    else:
        if open_price >= stop_price:
            return open_price, "stop"
        if open_price <= target_price:
            return target_price, "target"
        touched_stop = high >= stop_price
        touched_target = low <= target_price

    if touched_stop:  # adverse ordering also handles bars touching both
        return stop_price, "stop"
    if touched_target:
        return target_price, "target"
    return np.nan, None


def _outcome(
    candidate: TradeCandidate,
    signal_time: pd.Timestamp,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    stop_price: float,
    target_price: float,
    exit_reason: ExitReason,
    bars_held: int,
    cfg: IntradayExecutionConfig,
    price_source: PriceSource,
    initial_r_pct: float,
) -> TradeOutcome:
    direction = 1.0 if candidate.side == "long" else -1.0
    gross_return = direction * (exit_price / entry_price - 1)
    net_return = gross_return - 2 * cfg.cost_pct_per_side
    return TradeOutcome(
        strategy=candidate.strategy,
        side=candidate.side,
        signal_time=signal_time,
        entry_time=entry_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_reason=exit_reason,
        bars_held=bars_held,
        gross_return=gross_return,
        net_return=net_return,
        initial_r_pct=initial_r_pct,
        r_multiple=net_return / initial_r_pct,
        label=int(net_return > 0),
        price_source=price_source,
    )


def _has_quotes(candles: pd.DataFrame) -> bool:
    return all(column in candles.columns for column in _QUOTE_OHLC)


def _entry_price(bar: pd.Series, side: Side, price_source: PriceSource) -> float:
    if price_source == "quotes":
        column = "ask_open" if side == "long" else "bid_open"
        return float(bar[column])
    return float(bar["open"])


def _executable_bar(
    bar: pd.Series,
    side: Side,
    price_source: PriceSource,
) -> pd.Series:
    """Return the OHLC path available to close ``side`` at executable quotes."""

    if price_source == "mid":
        return bar.loc[list(_OHLC)]

    quote_side = "bid" if side == "long" else "ask"
    return pd.Series(
        {
            field: float(bar[f"{quote_side}_{field}"])
            for field in _OHLC
        }
    )


def _utc(timestamp: pd.Timestamp) -> pd.Timestamp:
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        raise ValueError("Candidate timestamps must be timezone-aware")
    return value.tz_convert("UTC")
