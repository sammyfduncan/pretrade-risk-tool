"""Inspect or change only the local paper-risk kill switch; never call a broker."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from commodity_ml.intraday.state import PaperStateError, PaperStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/runtime/oanda_paper_state.json"),
    )
    parser.add_argument("command", choices=("status", "kill-on", "kill-off"))
    args = parser.parse_args()

    store = PaperStateStore(args.state)
    try:
        if args.command == "status":
            state = store.load()
        else:
            state = store.set_kill_switch(
                active=args.command == "kill-on",
                now=pd.Timestamp.now(tz="UTC"),
            )
    except (OSError, RuntimeError, TypeError, ValueError, PaperStateError) as exc:
        raise SystemExit(f"Paper-state operation refused: {exc}") from exc

    print("=== LOCAL PAPER RISK STATE ===")
    print(f"Account suffix: {state.account_id[-4:]}")
    print(f"Revision: {state.revision}")
    print(f"Updated: {state.updated_at.isoformat()}")
    print(f"London session/week: {state.london_session_date} / {state.london_week_start}")
    print(
        f"Day/week equity anchors: {state.day_start_equity:,.2f} / "
        f"{state.week_start_equity:,.2f}"
    )
    print(f"Entries today: {state.entries_today}")
    print(f"Expected open positions: {len(state.expected_position_ids)}")
    print(f"Kill switch: {'ACTIVE' if state.kill_switch_active else 'clear'}")
    print("No broker or order action was performed.")


if __name__ == "__main__":
    main()
