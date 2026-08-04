"""Bridge restart-safe paper state into the quantitative pre-trade contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from commodity_ml.data.ig import IGAccountSnapshot
from commodity_ml.intraday.state import PaperStateStore


@dataclass(frozen=True)
class PaperRiskRuntimeSnapshot:
    day_start_equity: float
    week_start_equity: float
    entries_today: int
    kill_switch_active: bool
    expected_open_positions: int
    state_revision: int
    state_updated_at: str
    london_session_date: str
    london_week_start: str

    def risk_inputs(self) -> dict[str, object]:
        return {
            "day_start_equity": self.day_start_equity,
            "week_start_equity": self.week_start_equity,
            "entries_today": self.entries_today,
            "kill_switch_active": self.kill_switch_active,
        }

    def provenance(self) -> dict[str, object]:
        return asdict(self)


def roll_and_reconcile_paper_runtime(
    store: PaperStateStore,
    *,
    account: IGAccountSnapshot,
    broker_position_ids: tuple[str, ...],
    now: pd.Timestamp,
) -> PaperRiskRuntimeSnapshot:
    """Require exact account/positions, then roll London risk boundaries once."""

    timestamp = _utc(now)
    # Reconcile before changing even local heartbeat/boundary state. Unknown
    # broker positions must not be adopted as a side effect of assessment.
    store.reconcile(
        account_id=account.account_id,
        broker_position_ids=broker_position_ids,
    )
    state = store.roll_boundaries(
        account_id=account.account_id,
        equity=account.equity,
        now=timestamp,
    )
    return PaperRiskRuntimeSnapshot(
        day_start_equity=state.day_start_equity,
        week_start_equity=state.week_start_equity,
        entries_today=state.entries_today,
        kill_switch_active=state.kill_switch_active,
        expected_open_positions=len(state.expected_position_ids),
        state_revision=state.revision,
        state_updated_at=state.updated_at.isoformat(),
        london_session_date=state.london_session_date.isoformat(),
        london_week_start=state.london_week_start.isoformat(),
    )


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("risk runtime now must be timezone-aware")
    return timestamp.tz_convert("UTC")
