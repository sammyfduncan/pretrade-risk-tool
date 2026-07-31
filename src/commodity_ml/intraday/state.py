"""Persistent, fail-closed paper runtime and idempotency state.

The state file is deliberately separate from model artifacts. It records the
operational facts needed to survive a process restart: day/week equity anchors,
entry count, expected broker position IDs, kill-switch state, and every intent
ID that has been prepared. No function here submits or modifies a broker order.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, timedelta
import fcntl
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator, Literal

import numpy as np
import pandas as pd

from commodity_ml.intraday.paper import PaperOrderIntent, PaperRuntimeState

SCHEMA_VERSION = 1
LONDON = "Europe/London"
IntentStatus = Literal["prepared", "submitted", "confirmed", "rejected"]


class PaperStateError(RuntimeError):
    """Persistent state is missing, corrupt, stale, or irreconcilable."""


@dataclass(frozen=True)
class IntentRecord:
    intent_id: str
    fingerprint: str
    status: IntentStatus
    created_at: pd.Timestamp
    deal_reference: str | None = None
    position_id: str | None = None

    def __post_init__(self) -> None:
        if not self.intent_id or not self.fingerprint:
            raise ValueError("intent_id and fingerprint must be non-empty")
        if self.status not in ("prepared", "submitted", "confirmed", "rejected"):
            raise ValueError(f"Unknown intent status: {self.status!r}")
        _utc(self.created_at)
        if self.status in ("submitted", "confirmed") and not self.deal_reference:
            raise ValueError(f"{self.status} intent requires deal_reference")
        if self.status == "confirmed" and not self.position_id:
            raise ValueError("confirmed intent requires position_id")


@dataclass(frozen=True)
class PersistentPaperState:
    account_id: str
    revision: int
    updated_at: pd.Timestamp
    london_session_date: date
    london_week_start: date
    day_start_equity: float
    week_start_equity: float
    entries_today: int = 0
    expected_position_ids: tuple[str, ...] = ()
    intents: tuple[IntentRecord, ...] = ()
    kill_switch_active: bool = False
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported paper-state schema {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.account_id:
            raise ValueError("account_id must be non-empty")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 0
        ):
            raise ValueError("revision must be a non-negative integer")
        _utc(self.updated_at)
        for name in ("day_start_equity", "week_start_equity"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.entries_today, bool)
            or not isinstance(self.entries_today, int)
            or self.entries_today < 0
        ):
            raise ValueError("entries_today must be a non-negative integer")
        if len(self.expected_position_ids) != len(set(self.expected_position_ids)):
            raise ValueError("expected_position_ids contains duplicates")
        intent_ids = [record.intent_id for record in self.intents]
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("intent IDs must be unique")
        if self.london_week_start.weekday() != 0:
            raise ValueError("london_week_start must be a Monday")

    def runtime_state(self) -> PaperRuntimeState:
        """Convert persisted facts to the pure paper-intent policy input."""

        known_intents = tuple(record.intent_id for record in self.intents)
        return PaperRuntimeState(
            day_start_equity=self.day_start_equity,
            week_start_equity=self.week_start_equity,
            entries_today=self.entries_today,
            expected_open_positions=len(self.expected_position_ids),
            expected_open_position_ids=self.expected_position_ids,
            submitted_intent_ids=known_intents,
            kill_switch_active=self.kill_switch_active,
        )


class PaperStateStore:
    """Atomic JSON store with an inter-process lock and revision checks."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")

    def initialize(
        self,
        *,
        account_id: str,
        equity: float,
        now: pd.Timestamp,
        broker_position_ids: tuple[str, ...] = (),
    ) -> PersistentPaperState:
        """Create first state only for a verified flat broker account."""

        if broker_position_ids:
            raise PaperStateError(
                "state_initialization_requires_flat_broker:"
                f"{sorted(set(broker_position_ids))}"
            )
        timestamp = _utc(now)
        local_date = timestamp.tz_convert(LONDON).date()
        state = PersistentPaperState(
            account_id=account_id,
            revision=0,
            updated_at=timestamp,
            london_session_date=local_date,
            london_week_start=_week_start(local_date),
            day_start_equity=equity,
            week_start_equity=equity,
        )
        with self._locked():
            if self.path.exists():
                raise PaperStateError(f"paper_state_already_exists:{self.path}")
            self._write_unlocked(state)
        return state

    def load(self) -> PersistentPaperState:
        with self._locked():
            return self._load_unlocked()

    def roll_boundaries(
        self,
        *,
        account_id: str,
        equity: float,
        now: pd.Timestamp,
    ) -> PersistentPaperState:
        """Reset day/week anchors exactly once on London calendar boundaries."""

        timestamp = _utc(now)
        local_date = timestamp.tz_convert(LONDON).date()
        week_start = _week_start(local_date)
        with self._locked():
            state = self._load_unlocked()
            _require_account(state, account_id)
            updates: dict[str, object] = {"updated_at": timestamp}
            if local_date > state.london_session_date:
                updates.update(
                    london_session_date=local_date,
                    day_start_equity=equity,
                    entries_today=0,
                )
            elif local_date < state.london_session_date:
                raise PaperStateError("runtime_clock_moved_before_persisted_session")
            if week_start > state.london_week_start:
                updates.update(
                    london_week_start=week_start,
                    week_start_equity=equity,
                )
            elif week_start < state.london_week_start:
                raise PaperStateError("runtime_clock_moved_before_persisted_week")
            return self._replace_unlocked(state, **updates)

    def reserve_intent(
        self,
        intent: PaperOrderIntent,
    ) -> PersistentPaperState:
        """Persist an intent before a future adapter is allowed to submit it."""

        fingerprint = _intent_fingerprint(intent)
        with self._locked():
            state = self._load_unlocked()
            _require_account(state, intent.account_id)
            existing = {record.intent_id: record for record in state.intents}.get(
                intent.intent_id
            )
            if existing is not None:
                suffix = (
                    "duplicate_intent_id"
                    if existing.fingerprint == fingerprint
                    else "intent_id_fingerprint_collision"
                )
                raise PaperStateError(suffix)
            record = IntentRecord(
                intent_id=intent.intent_id,
                fingerprint=fingerprint,
                status="prepared",
                created_at=_utc(intent.created_at),
            )
            return self._replace_unlocked(
                state,
                updated_at=_utc(intent.created_at),
                intents=state.intents + (record,),
            )

    def mark_submitted(
        self,
        *,
        intent_id: str,
        deal_reference: str,
        now: pd.Timestamp,
    ) -> PersistentPaperState:
        """Record the broker reference; repeating the same transition is safe."""

        if not deal_reference:
            raise ValueError("deal_reference must be non-empty")
        return self._transition_intent(
            intent_id,
            now=now,
            target_status="submitted",
            deal_reference=deal_reference,
        )

    def mark_confirmed(
        self,
        *,
        intent_id: str,
        deal_reference: str,
        position_id: str,
        now: pd.Timestamp,
    ) -> PersistentPaperState:
        """Confirm one fill and atomically add its expected broker position."""

        if not position_id:
            raise ValueError("position_id must be non-empty")
        with self._locked():
            state = self._load_unlocked()
            index, record = _find_intent(state, intent_id)
            if record.status == "confirmed":
                if (
                    record.deal_reference == deal_reference
                    and record.position_id == position_id
                ):
                    return state
                raise PaperStateError("confirmed_intent_reference_mismatch")
            if record.status not in ("prepared", "submitted"):
                raise PaperStateError(f"cannot_confirm_intent_from:{record.status}")
            if record.deal_reference not in (None, deal_reference):
                raise PaperStateError("submitted_intent_reference_mismatch")
            if position_id in state.expected_position_ids:
                raise PaperStateError("position_id_already_expected")
            intents = list(state.intents)
            intents[index] = replace(
                record,
                status="confirmed",
                deal_reference=deal_reference,
                position_id=position_id,
            )
            return self._replace_unlocked(
                state,
                updated_at=_utc(now),
                intents=tuple(intents),
                expected_position_ids=tuple(
                    sorted(state.expected_position_ids + (position_id,))
                ),
                entries_today=state.entries_today + 1,
            )

    def mark_rejected(
        self,
        *,
        intent_id: str,
        now: pd.Timestamp,
    ) -> PersistentPaperState:
        return self._transition_intent(
            intent_id,
            now=now,
            target_status="rejected",
        )

    def mark_position_closed(
        self,
        *,
        position_id: str,
        now: pd.Timestamp,
    ) -> PersistentPaperState:
        """Remove a position only after the broker confirms it is no longer open."""

        with self._locked():
            state = self._load_unlocked()
            if position_id not in state.expected_position_ids:
                raise PaperStateError("closed_position_was_not_expected")
            remaining = tuple(
                value for value in state.expected_position_ids if value != position_id
            )
            return self._replace_unlocked(
                state,
                updated_at=_utc(now),
                expected_position_ids=remaining,
            )

    def reconcile(
        self,
        *,
        account_id: str,
        broker_position_ids: tuple[str, ...],
    ) -> PersistentPaperState:
        """Return state only when exact broker position IDs agree."""

        state = self.load()
        _require_account(state, account_id)
        actual = tuple(sorted(set(broker_position_ids)))
        expected = tuple(sorted(state.expected_position_ids))
        if actual != expected:
            raise PaperStateError(
                f"position_id_reconciliation_mismatch:"
                f"expected={expected},actual={actual}"
            )
        return state

    def set_kill_switch(
        self,
        *,
        active: bool,
        now: pd.Timestamp,
    ) -> PersistentPaperState:
        with self._locked():
            state = self._load_unlocked()
            return self._replace_unlocked(
                state,
                updated_at=_utc(now),
                kill_switch_active=bool(active),
            )

    def _transition_intent(
        self,
        intent_id: str,
        *,
        now: pd.Timestamp,
        target_status: IntentStatus,
        deal_reference: str | None = None,
    ) -> PersistentPaperState:
        with self._locked():
            state = self._load_unlocked()
            index, record = _find_intent(state, intent_id)
            if record.status == target_status:
                if deal_reference is None or record.deal_reference == deal_reference:
                    return state
                raise PaperStateError("idempotent_transition_reference_mismatch")
            allowed = {
                "submitted": ("prepared",),
                "rejected": ("prepared", "submitted"),
            }
            if record.status not in allowed[target_status]:
                raise PaperStateError(
                    f"cannot_mark_{target_status}_from:{record.status}"
                )
            intents = list(state.intents)
            intents[index] = replace(
                record,
                status=target_status,
                deal_reference=deal_reference or record.deal_reference,
            )
            return self._replace_unlocked(
                state,
                updated_at=_utc(now),
                intents=tuple(intents),
            )

    def _replace_unlocked(
        self,
        state: PersistentPaperState,
        **updates: object,
    ) -> PersistentPaperState:
        current = self._load_unlocked()
        if current.revision != state.revision:
            raise PaperStateError(
                f"paper_state_revision_conflict:"
                f"expected={state.revision},actual={current.revision}"
            )
        next_updated = _utc(updates.get("updated_at", state.updated_at))
        if next_updated < _utc(state.updated_at):
            raise PaperStateError("paper_state_update_time_moved_backwards")
        next_state = replace(state, revision=state.revision + 1, **updates)
        self._write_unlocked(next_state)
        return next_state

    def _load_unlocked(self) -> PersistentPaperState:
        if not self.path.exists():
            raise PaperStateError(f"paper_state_missing:{self.path}")
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return _state_from_payload(payload)
        except PaperStateError:
            raise
        except Exception as exc:
            raise PaperStateError(f"paper_state_corrupt:{exc}") from exc

    def _write_unlocked(self, state: PersistentPaperState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            _state_to_payload(state),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _find_intent(
    state: PersistentPaperState,
    intent_id: str,
) -> tuple[int, IntentRecord]:
    for index, record in enumerate(state.intents):
        if record.intent_id == intent_id:
            return index, record
    raise PaperStateError(f"unknown_intent_id:{intent_id}")


def _require_account(state: PersistentPaperState, account_id: str) -> None:
    if state.account_id != account_id:
        raise PaperStateError(
            f"paper_state_account_mismatch:"
            f"expected={state.account_id},actual={account_id}"
        )


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


def _intent_fingerprint(intent: PaperOrderIntent) -> str:
    """Stable human-auditable fingerprint; no secret material is included."""

    fields = (
        intent.strategy,
        intent.epic,
        intent.account_id,
        intent.account_currency,
        intent.direction,
        f"{intent.size:.12g}",
        f"{intent.reference_price:.12g}",
        f"{intent.stop_level:.12g}",
        f"{intent.target_level:.12g}",
    )
    return "|".join(fields)


def _state_to_payload(state: PersistentPaperState) -> dict:
    return {
        "schema_version": state.schema_version,
        "account_id": state.account_id,
        "revision": state.revision,
        "updated_at": _utc(state.updated_at).isoformat(),
        "london_session_date": state.london_session_date.isoformat(),
        "london_week_start": state.london_week_start.isoformat(),
        "day_start_equity": state.day_start_equity,
        "week_start_equity": state.week_start_equity,
        "entries_today": state.entries_today,
        "expected_position_ids": list(state.expected_position_ids),
        "kill_switch_active": state.kill_switch_active,
        "intents": [
            {
                "intent_id": record.intent_id,
                "fingerprint": record.fingerprint,
                "status": record.status,
                "created_at": _utc(record.created_at).isoformat(),
                "deal_reference": record.deal_reference,
                "position_id": record.position_id,
            }
            for record in state.intents
        ],
    }


def _state_from_payload(payload: dict) -> PersistentPaperState:
    required = {
        "schema_version",
        "account_id",
        "revision",
        "updated_at",
        "london_session_date",
        "london_week_start",
        "day_start_equity",
        "week_start_equity",
        "entries_today",
        "expected_position_ids",
        "kill_switch_active",
        "intents",
    }
    missing = sorted(required - set(payload))
    unexpected = sorted(set(payload) - required)
    if missing or unexpected:
        raise PaperStateError(
            f"paper_state_schema_mismatch:missing={missing},unexpected={unexpected}"
        )
    intents = tuple(
        IntentRecord(
            intent_id=item["intent_id"],
            fingerprint=item["fingerprint"],
            status=item["status"],
            created_at=pd.Timestamp(item["created_at"]),
            deal_reference=item.get("deal_reference"),
            position_id=item.get("position_id"),
        )
        for item in payload["intents"]
    )
    return PersistentPaperState(
        schema_version=payload["schema_version"],
        account_id=payload["account_id"],
        revision=payload["revision"],
        updated_at=pd.Timestamp(payload["updated_at"]),
        london_session_date=date.fromisoformat(payload["london_session_date"]),
        london_week_start=date.fromisoformat(payload["london_week_start"]),
        day_start_equity=payload["day_start_equity"],
        week_start_equity=payload["week_start_equity"],
        entries_today=payload["entries_today"],
        expected_position_ids=tuple(payload["expected_position_ids"]),
        kill_switch_active=payload["kill_switch_active"],
        intents=intents,
    )


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("paper state timestamps must be timezone-aware")
    return timestamp.tz_convert("UTC")
