"""Initialize restart-safe paper risk state from a verified flat IG account."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.data import ig
from commodity_ml.intraday.state import PaperStateError, PaperStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path("data/runtime/ig_paper_state.json"),
    )
    args = parser.parse_args()

    load_dotenv()
    try:
        service = ig.get_service()
        account = ig.fetch_account_snapshot(service)
        position_ids = ig.fetch_open_position_ids(service)
        state = PaperStateStore(args.state).initialize(
            account_id=account.account_id,
            equity=account.equity,
            now=pd.Timestamp.now(tz="UTC"),
            broker_position_ids=position_ids,
        )
    except (OSError, RuntimeError, TypeError, ValueError, PaperStateError) as exc:
        raise SystemExit(f"Paper-state initialization refused: {exc}") from exc

    print("=== IG PAPER RISK STATE INITIALIZED ===")
    print(f"Account suffix: {account.account_id[-4:]}")
    print(f"Currency: {account.currency}")
    print(f"Day/week start equity: {state.day_start_equity:,.2f}")
    print(f"State -> {args.state.resolve()}")
    print("The account was verified flat. No order or position action was performed.")


if __name__ == "__main__":
    main()
