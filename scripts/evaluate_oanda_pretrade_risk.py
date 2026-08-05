"""Create a quantitative pre-trade report from read-only OANDA v20 facts.

This command reads account, instrument, pricing, candle and open-trade data.
It contains no order creation, amendment or closure call.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.features import event_dates
from commodity_ml.intraday.event_blackouts import scheduled_blackout_state
from commodity_ml.intraday.calibration_release import load_admitted_calibrated_risk
from commodity_ml.intraday.oanda_risk_snapshot import build_oanda_read_only_snapshot
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
        help="Additional metal YAML used to map existing OANDA trades",
    )
    parser.add_argument("--points", type=int, default=250)
    parser.add_argument("--day-start-equity", type=float)
    parser.add_argument("--week-start-equity", type=float)
    parser.add_argument("--entries-today", type=int)
    parser.add_argument("--paper-state", type=Path)
    parser.add_argument(
        "--calibration-profile",
        type=Path,
        help="Optional current range-risk profile; requires its admitted release JSON",
    )
    parser.add_argument(
        "--calibration-release",
        type=Path,
        help="Admission JSON matching --calibration-profile exactly",
    )
    parser.add_argument(
        "--event-state",
        choices=("auto", "unknown", "clear", "blackout"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/oanda_pretrade_risk.json"),
    )
    args = parser.parse_args()

    load_dotenv()
    try:
        cfg = load_config(args.config)
        portfolio_configs = [cfg, *(load_config(path) for path in args.portfolio_config)]
        configured = [item.data.oanda.instrument for item in portfolio_configs]
        if any(not value for value in configured):
            raise ValueError("every portfolio config must set data.oanda.instrument")
        if len(configured) != len(set(configured)):
            raise ValueError("portfolio configs contain duplicate OANDA instruments")

        source = json.loads(args.input.read_text(encoding="utf-8"))
        if bool(args.calibration_profile) != bool(args.calibration_release):
            raise ValueError(
                "--calibration-profile and --calibration-release must be supplied together"
            )
        trade_intent = source.get("trade_intent")
        if not isinstance(trade_intent, dict):
            raise ValueError("input must contain a trade_intent object")
        runtime = source.get("runtime") or {}
        if not isinstance(runtime, dict):
            raise ValueError("runtime must be an object when supplied")

        client = oanda.get_client()
        account = oanda.fetch_account_snapshot(
            client,
            account_type=cfg.data.oanda.account_type,
        )
        pricing = client.pricing(configured)
        quotes = oanda.quotes_from_pricing(pricing)
        raw_instruments = client.account_instruments(configured)
        raw_by_name = {str(item.get("name")): item for item in raw_instruments}
        metadata = {
            name: oanda.instrument_metadata_from_responses(
                raw_by_name[name], pricing, account_currency=account.currency
            )
            for name in configured
            if name in raw_by_name
        }
        if set(metadata) != set(configured) or set(quotes) != set(configured):
            raise RuntimeError("OANDA omitted a configured portfolio instrument")
        primary_name = cfg.data.oanda.instrument
        primary_metadata = metadata[primary_name]
        primary_quote = quotes[primary_name]
        candles = oanda.fetch_recent_candles(
            cfg.data,
            "M30",
            num_points=args.points,
            client=client,
            verbose=False,
        )
        trades = client.open_trades()
        now = pd.Timestamp.now(tz="UTC")

        paper_runtime = None
        if args.paper_state is not None:
            if args.event_state != "auto":
                raise ValueError("--paper-state requires --event-state auto")
            if any(
                value is not None
                for value in (args.day_start_equity, args.week_start_equity, args.entries_today)
            ):
                raise ValueError("--paper-state cannot be combined with manual runtime overrides")
            paper_runtime = roll_and_reconcile_paper_runtime(
                PaperStateStore(args.paper_state),
                account=account,
                broker_position_ids=oanda.open_trade_ids_from_response(trades),
                now=now,
            )
            risk_runtime = paper_runtime.risk_inputs()
        else:
            risk_runtime = {
                "day_start_equity": args.day_start_equity if args.day_start_equity is not None else runtime.get("day_start_equity"),
                "week_start_equity": args.week_start_equity if args.week_start_equity is not None else runtime.get("week_start_equity"),
                "entries_today": args.entries_today if args.entries_today is not None else runtime.get("entries_today"),
                "kill_switch_active": runtime.get("kill_switch_active"),
            }

        if args.event_state == "auto":
            event_state, event_reasons = scheduled_blackout_state(now)
            event_provenance = {
                "mode": "verified_schedule",
                "active_reasons": event_reasons,
                "verified_on": event_dates.EVENT_SCHEDULE_VERIFIED_ON,
                "coverage_end": event_dates.COVERAGE_END.isoformat(),
            }
        else:
            event_state = {"unknown": None, "clear": False, "blackout": True}[args.event_state]
            event_provenance = {"mode": f"manual_{args.event_state}", "active_reasons": ()}

        snapshot = build_oanda_read_only_snapshot(
            instrument_symbol=cfg.symbol,
            trade_intent=trade_intent,
            account=account,
            metadata=primary_metadata,
            quote=primary_quote,
            recent_candles=candles,
            open_trades=trades,
            metadata_by_instrument=metadata,
            quotes_by_instrument=quotes,
            symbol_by_instrument={
                item.data.oanda.instrument: item.symbol for item in portfolio_configs
            },
            day_start_equity=risk_runtime["day_start_equity"],
            week_start_equity=risk_runtime["week_start_equity"],
            entries_today=risk_runtime["entries_today"],
            kill_switch_active=risk_runtime["kill_switch_active"],
            scheduled_event_blackout_active=(
                event_state if args.event_state != "unknown"
                else runtime.get("scheduled_event_blackout_active")
            ),
            now=now,
        )
        if isinstance(source.get("policy"), dict):
            snapshot.payload["policy"] = source["policy"]
        calibration = None
        if args.calibration_profile:
            assessed_intent = snapshot.payload["trade_intent"]
            calibration = load_admitted_calibrated_risk(
                args.calibration_profile,
                args.calibration_release,
                instrument=assessed_intent.get("instrument"),
                decision_time=assessed_intent.get("decision_time"),
                horizon_minutes=assessed_intent.get("holding_period_minutes"),
                entry_price=assessed_intent.get("entry_price"),
                side=assessed_intent.get("side"),
            )
        report = evaluate_pretrade_payload(
            snapshot.payload,
            historical_calibration=calibration,
        )
        result = {
            "schema_version": 2,
            "mode": "OANDA_V20_READ_ONLY",
            "order_authority": snapshot.order_authority,
            "provenance": {
                **snapshot.provenance,
                "environment": client.environment,
                "historical_price_points_requested": args.points,
                "paper_runtime": paper_runtime.provenance() if paper_runtime else None,
                "event_calendar": event_provenance,
            },
            "adapter_issues": snapshot.issues,
            "risk_report": report.to_dict(),
        }
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Read-only OANDA assessment refused: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"=== OANDA READ-ONLY PRE-TRADE RISK: {report.decision} ===")
    if report.reasons:
        print(f"Reasons: {', '.join(report.reasons)}")
    if snapshot.issues:
        print(f"Adapter audit: {', '.join(snapshot.issues)}")
    print(f"Report -> {args.output.resolve()}")
    print("No order was created, amended, closed or submitted.")


if __name__ == "__main__":
    main()
