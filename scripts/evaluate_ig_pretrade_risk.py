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
from commodity_ml.intraday.risk_runtime import roll_and_reconcile_paper_runtime
from commodity_ml.intraday.state import PaperStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON containing trade_intent")
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument(
        "--portfolio-config",
        action="append",
        default=[],
        help=(
            "Additional configured metal YAML used only to map existing IG "
            "positions (repeat for gold/silver)"
        ),
    )
    parser.add_argument("--points", type=int, default=250)
    parser.add_argument("--day-start-equity", type=float)
    parser.add_argument("--week-start-equity", type=float)
    parser.add_argument("--entries-today", type=int)
    parser.add_argument(
        "--paper-state",
        type=Path,
        help=(
            "Restart-safe paper state; supplies loss anchors, entry count and "
            "kill switch after exact broker-position reconciliation"
        ),
    )
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
        portfolio_configs = [cfg, *(load_config(path) for path in args.portfolio_config)]
        epic_to_instrument: dict[str, str] = {}
        point_values: dict[str, float] = {}
        for portfolio_cfg in portfolio_configs:
            configured_epic = portfolio_cfg.data.ig.epic
            if configured_epic:
                prior = epic_to_instrument.get(configured_epic)
                if prior is not None and prior != portfolio_cfg.symbol:
                    raise ValueError(
                        f"epic {configured_epic!r} maps to both {prior} and "
                        f"{portfolio_cfg.symbol}"
                    )
                epic_to_instrument[configured_epic] = portfolio_cfg.symbol
            if portfolio_cfg.data.ig.value_per_price_unit is not None:
                point_values[portfolio_cfg.symbol] = (
                    portfolio_cfg.data.ig.value_per_price_unit
                )
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

        paper_runtime = None
        if args.paper_state is not None:
            if any(
                value is not None
                for value in (
                    args.day_start_equity,
                    args.week_start_equity,
                    args.entries_today,
                )
            ):
                raise ValueError(
                    "--paper-state cannot be combined with manual account runtime overrides"
                )
            paper_runtime = roll_and_reconcile_paper_runtime(
                PaperStateStore(args.paper_state),
                account=account,
                broker_position_ids=ig.open_position_ids_from_response(positions),
                now=now,
            )
            risk_runtime = paper_runtime.risk_inputs()
        else:
            risk_runtime = {
                "day_start_equity": (
                    args.day_start_equity
                    if args.day_start_equity is not None
                    else runtime.get("day_start_equity")
                ),
                "week_start_equity": (
                    args.week_start_equity
                    if args.week_start_equity is not None
                    else runtime.get("week_start_equity")
                ),
                "entries_today": (
                    args.entries_today
                    if args.entries_today is not None
                    else runtime.get("entries_today")
                ),
                "kill_switch_active": runtime.get("kill_switch_active"),
            }

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
            day_start_equity=risk_runtime["day_start_equity"],
            week_start_equity=risk_runtime["week_start_equity"],
            entries_today=risk_runtime["entries_today"],
            kill_switch_active=risk_runtime["kill_switch_active"],
            scheduled_event_blackout_active=(
                event_state
                if args.event_state != "unknown"
                else runtime.get("scheduled_event_blackout_active")
            ),
            epic_to_instrument=epic_to_instrument,
            value_per_price_unit_by_instrument=point_values,
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
                "paper_runtime": (
                    paper_runtime.provenance() if paper_runtime is not None else None
                ),
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
