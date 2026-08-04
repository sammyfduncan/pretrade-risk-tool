"""Initialize restart-safe paper risk state from a verified flat OANDA account."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.intraday.state import PaperStateError, PaperStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument(
        "--state", type=Path, default=Path("data/runtime/oanda_paper_state.json")
    )
    args = parser.parse_args()
    load_dotenv()
    try:
        cfg = load_config(args.config)
        client = oanda.get_client()
        account = oanda.fetch_account_snapshot(
            client, account_type=cfg.data.oanda.account_type
        )
        state = PaperStateStore(args.state).initialize(
            account_id=account.account_id,
            equity=account.equity,
            now=pd.Timestamp.now(tz="UTC"),
            broker_position_ids=oanda.fetch_open_position_ids(client),
        )
    except (OSError, RuntimeError, TypeError, ValueError, PaperStateError) as exc:
        raise SystemExit(f"Paper-state initialization refused: {exc}") from exc
    print("=== OANDA PAPER RISK STATE INITIALIZED ===")
    print(f"Account suffix: {account.account_id[-4:]}")
    print(f"Currency: {account.currency}")
    print(f"Day/week start equity: {state.day_start_equity:,.2f}")
    print(f"State -> {args.state.resolve()}")
    print("The account was verified flat. No broker action was performed.")


if __name__ == "__main__":
    main()
