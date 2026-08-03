"""Create a quantitative pre-trade report from read-only IG REST snapshots.

After normal IG session authentication, this command makes only account,
market, recent-price and open-position reads through ``trading-ig``. It
contains no order creation, amendment or closure call. Unknown runtime facts
produce an INCOMPLETE report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import ig
from commodity_ml.intraday.ig_risk_snapshot import build_ig_read_only_snapshot
from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing trade_intent")
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--points", type=int, default=250)
    parser.add_argument("--day-start-equity", type=float)
    parser.add_argument("--week-start-equity", type=float)
    parser.add_argument("--entries-today", type=int)
    parser.add_argument(
        "--event-state",
        choices=("unknown", "clear", "blackout"),
        default="unknown",
        help="Current mechanical scheduled-event state; unknown fails closed",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/ig_pretrade_risk.json"))
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    if not cfg.data.ig.epic:
        raise SystemExit(
            "Read-only IG assessment refused: data.ig.epic is unset. Run "
            "scripts/find_epic.py, confirm the exact spot spread-bet market, "
            "then set it in the selected config. No epic is guessed."
        )
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        trade_intent = source.get("trade_intent")
        if not isinstance(trade_intent, dict):
            raise ValueError("input must contain a trade_intent object")
        runtime = source.get("runtime") or {}
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be an object when supplied")

        now = pd.Timestamp.now(tz="UTC")
        service = ig.get_service()
        account = ig.fetch_account_snapshot(service)
        details = ig.fetch_market_details(service, cfg.data.ig.epic)
        quote = ig.market_quote_from_details(details, now=now)
        candles = ig.fetch_recent_candles(
            cfg.data,
            "M30",
            num_points=args.points,
            service=service,
            verbose=False,
        )
        positions = service.fetch_open_positions()

        event_state = {
            "unknown": None,
            "clear": False,
            "blackout": True,
        }[args.event_state]
        snapshot = build_ig_read_only_snapshot(
            instrument_symbol=cfg.symbol,
            trade_intent=trade_intent,
            account=account,
            market_details=details,
            quote=quote,
            recent_candles=candles,
            open_positions=positions,
            size_increment=cfg.data.ig.size_increment,
            value_per_price_unit=cfg.data.ig.value_per_price_unit,
            fallback_margin_rate=cfg.data.ig.margin_rate,
            fallback_margin_rate_max_size=cfg.data.ig.margin_rate_max_size,
            day_start_equity=(
                args.day_start_equity
                if args.day_start_equity is not None
                else runtime.get("day_start_equity")
            ),
            week_start_equity=(
                args.week_start_equity
                if args.week_start_equity is not None
                else runtime.get("week_start_equity")
            ),
            entries_today=(
                args.entries_today
                if args.entries_today is not None
                else runtime.get("entries_today")
            ),
            scheduled_event_blackout_active=(
                event_state
                if args.event_state != "unknown"
                else runtime.get("scheduled_event_blackout_active")
            ),
            now=now,
        )
        if isinstance(source.get("policy"), dict):
            snapshot.payload["policy"] = source["policy"]
        report = evaluate_pretrade_payload(snapshot.payload)
        result = {
            "schema_version": 1,
            "mode": "IG_READ_ONLY",
            "order_authority": snapshot.order_authority,
            "provenance": {
                **snapshot.provenance,
                "historical_price_points_requested": args.points,
            },
            "adapter_issues": snapshot.issues,
            "risk_report": report.to_dict(),
        }
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Read-only IG assessment refused: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"=== IG READ-ONLY PRE-TRADE RISK: {report.decision} ===")
    if report.reasons:
        print(f"Reasons: {', '.join(report.reasons)}")
    if snapshot.issues:
        print(f"Adapter audit: {', '.join(snapshot.issues)}")
    print(f"Report -> {args.output.resolve()}")
    print("No order was created, amended, closed or submitted.")


if __name__ == "__main__":
    main()
