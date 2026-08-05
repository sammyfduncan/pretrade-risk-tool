"""Pure candidate and immutable-cycle contracts for OANDA shadow operation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from commodity_ml.data.broker import BrokerMarketQuote
from commodity_ml.intraday.execution import TradeCandidate
from commodity_ml.intraday.strategies import generate_candidates


SHADOW_SCHEMA_VERSION = 1
M30 = pd.Timedelta(minutes=30)


class ShadowLedgerError(RuntimeError):
    """A shadow check would cease to be immutable or unambiguous."""


@dataclass(frozen=True)
class ShadowCheck:
    check_id: str
    instrument: str
    strategy: str
    signal_time: pd.Timestamp
    checked_at: pd.Timestamp
    status: str
    intent_id: str | None = None
    decision: str | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.check_id or not self.instrument or not self.strategy:
            raise ValueError("shadow check identifiers must be non-empty")
        if self.status not in (
            "NO_SIGNAL",
            "ASSESSED",
            "ALREADY_ASSESSED",
            "SUPPRESSED",
            "ERROR",
        ):
            raise ValueError(f"unsupported shadow check status: {self.status}")
        _utc(self.signal_time, "signal_time")
        checked = _utc(self.checked_at, "checked_at")
        if checked < _utc(self.signal_time, "signal_time") + M30:
            raise ValueError("shadow check precedes signal-bar close")
        if self.status in ("ASSESSED", "ALREADY_ASSESSED") and (
            not self.intent_id or self.decision not in ("ALLOW", "REDUCE", "BLOCK", "INCOMPLETE")
        ):
            raise ValueError("assessed shadow checks require an intent and decision")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "check_id": self.check_id,
            "instrument": self.instrument,
            "strategy": self.strategy,
            "signal_time": _utc(self.signal_time, "signal_time").isoformat(),
            "status": self.status,
            "intent_id": self.intent_id,
            "decision": self.decision,
            "reasons": list(self.reasons),
        }


class ShadowCheckStore:
    """Append-only record of signal and no-signal checks at each closed bar."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def record(self, check: ShadowCheck) -> bool:
        payload = check.canonical_payload()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT check_sha256 FROM shadow_checks WHERE check_id = ?",
                (check.check_id,),
            ).fetchone()
            if existing is not None:
                if existing[0] == digest:
                    return False
                raise ShadowLedgerError("shadow_check_id_collision")
            connection.execute(
                """INSERT INTO shadow_checks
                   (check_id, instrument, strategy, signal_time, checked_at,
                    status, intent_id, decision, reasons_json, check_sha256)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    check.check_id,
                    check.instrument,
                    check.strategy,
                    _utc(check.signal_time, "signal_time").isoformat(),
                    _utc(check.checked_at, "checked_at").isoformat(),
                    check.status,
                    check.intent_id,
                    check.decision,
                    json.dumps(list(check.reasons), separators=(",", ":")),
                    digest,
                ),
            )
        return True

    def check_result(
        self, check_id: str
    ) -> tuple[str, str | None, tuple[str, ...]] | None:
        """Return a prior immutable check so polling is exactly idempotent."""

        if not check_id:
            raise ValueError("check_id must be non-empty")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status, decision, reasons_json FROM shadow_checks WHERE check_id = ?",
                (check_id,),
            ).fetchone()
        if row is None:
            return None
        reasons = json.loads(row[2])
        if not isinstance(reasons, list):
            raise ShadowLedgerError("stored_shadow_reasons_are_invalid")
        return str(row[0]), row[1], tuple(str(value) for value in reasons)

    def summary(self) -> dict[str, object]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            statuses = connection.execute(
                "SELECT status, COUNT(*) FROM shadow_checks GROUP BY status"
            ).fetchall()
            decisions = connection.execute(
                """SELECT decision, COUNT(*) FROM shadow_checks
                   WHERE decision IS NOT NULL GROUP BY decision"""
            ).fetchall()
            total = connection.execute("SELECT COUNT(*) FROM shadow_checks").fetchone()[0]
        return {
            "schema_version": SHADOW_SCHEMA_VERSION,
            "total_checks": int(total),
            "status_counts": {str(key): int(value) for key, value in statuses},
            "decision_counts": {str(key): int(value) for key, value in decisions},
            "limitations": [
                "No-signal and risk decisions are operational shadow observations, not broker orders.",
                "Shadow counts do not establish strategy profitability.",
            ],
        }

    def read_frame(self) -> pd.DataFrame:
        """Read the immutable checks for descriptive operational reporting."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            frame = pd.read_sql_query(
                """SELECT check_id, instrument, strategy, signal_time, checked_at,
                          status, intent_id, decision, reasons_json
                   FROM shadow_checks ORDER BY signal_time, instrument, strategy""",
                connection,
            )
        if frame.empty:
            frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True)
            frame["checked_at"] = pd.to_datetime(frame["checked_at"], utc=True)
            frame["reasons"] = pd.Series(dtype=object)
            return frame.drop(columns="reasons_json")
        frame["signal_time"] = pd.to_datetime(
            frame["signal_time"], utc=True, errors="raise"
        )
        frame["checked_at"] = pd.to_datetime(
            frame["checked_at"], utc=True, errors="raise"
        )
        frame["reasons"] = frame.pop("reasons_json").map(_stored_reasons)
        return frame

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS shadow_checks (
                check_id TEXT PRIMARY KEY,
                instrument TEXT NOT NULL,
                strategy TEXT NOT NULL,
                signal_time TEXT NOT NULL,
                checked_at TEXT NOT NULL,
                status TEXT NOT NULL,
                intent_id TEXT,
                decision TEXT,
                reasons_json TEXT NOT NULL,
                check_sha256 TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS shadow_checks_status_idx
                ON shadow_checks(status);
            """
        )
        connection.execute(f"PRAGMA user_version = {SHADOW_SCHEMA_VERSION}")
        return connection


def latest_closed_candidates(
    frame: pd.DataFrame,
    strategies: Sequence[str],
    *,
    now: pd.Timestamp,
    max_age: pd.Timedelta = M30,
) -> tuple[pd.Timestamp, Mapping[str, TradeCandidate | None]]:
    """Return at most one signal per strategy for the newest completed M30 bar."""

    current = _utc(now, "now")
    if "decision_time" not in frame or frame.empty:
        raise ValueError("intraday frame has no decision rows")
    decisions = pd.to_datetime(frame["decision_time"], utc=True)
    eligible = frame.loc[decisions <= current]
    if eligible.empty:
        raise RuntimeError("no_completed_m30_signal_bar")
    signal_time = pd.Timestamp(eligible.index[-1]).tz_convert("UTC")
    decision_time = pd.Timestamp(eligible.iloc[-1]["decision_time"]).tz_convert("UTC")
    age = current - decision_time
    if age < pd.Timedelta(0) or age > max_age:
        raise RuntimeError(f"stale_latest_completed_m30_bar:{age}")
    history = frame.loc[:signal_time]
    output: dict[str, TradeCandidate | None] = {}
    for strategy in strategies:
        matches = [
            candidate
            for candidate in generate_candidates(history, strategy)
            if pd.Timestamp(candidate.signal_time).tz_convert("UTC") == signal_time
        ]
        if len(matches) > 1:
            raise RuntimeError(f"multiple_{strategy}_signals_on_one_bar")
        output[strategy] = matches[0] if matches else None
    return signal_time, output


def trade_intent_from_candidate(
    candidate: TradeCandidate,
    *,
    symbol: str,
    quote: BrokerMarketQuote,
    assessed_at: pd.Timestamp,
    slippage_bp_per_side: float = 0.5,
) -> dict[str, object]:
    """Translate a frozen rule candidate into the generic risk contract."""

    if candidate.stop_distance_abs is None or candidate.target_r_multiple is None:
        raise ValueError("shadow runner requires ATR-absolute primary candidates")
    if not np.isfinite(slippage_bp_per_side) or slippage_bp_per_side < 0:
        raise ValueError("slippage_bp_per_side must be finite and non-negative")
    current = _utc(assessed_at, "assessed_at")
    signal = _utc(candidate.signal_time, "candidate.signal_time")
    entry = quote.mid
    direction = 1 if candidate.side == "long" else -1
    stop = entry - direction * candidate.stop_distance_abs
    target = entry + direction * candidate.stop_distance_abs * candidate.target_r_multiple
    intent_id = (
        f"shadow-{symbol.lower()}-{candidate.strategy.lower()}-"
        f"{signal.strftime('%Y%m%dT%H%MZ')}"
    )
    return {
        "intent_id": intent_id,
        "instrument": symbol,
        "side": candidate.side,
        "order_type": "market",
        "decision_time": current.isoformat(),
        "entry_price": entry,
        "stop_price": stop,
        "target_price": target,
        "requested_size": None,
        "holding_period_minutes": candidate.max_holding_bars * 30,
        "allow_overnight": False,
        "max_slippage_points_per_side": entry * slippage_bp_per_side / 10_000,
    }


def shadow_check_id(symbol: str, strategy: str, signal_time: pd.Timestamp) -> str:
    signal = _utc(signal_time, "signal_time")
    return f"{symbol}:{strategy}:{signal.strftime('%Y%m%dT%H%MZ')}"


def _utc(value: pd.Timestamp, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _stored_reasons(value: str) -> tuple[str, ...]:
    reasons = json.loads(value)
    if not isinstance(reasons, list):
        raise ShadowLedgerError("stored_shadow_reasons_are_invalid")
    return tuple(str(item) for item in reasons)
