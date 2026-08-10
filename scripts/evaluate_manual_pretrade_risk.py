"""Evaluate a manual GBP spread-bet account using read-only OANDA market data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.features import event_dates
from commodity_ml.intraday.calibration_release import load_admitted_calibrated_risk
from commodity_ml.intraday.event_blackouts import scheduled_blackout_state
from commodity_ml.intraday.oanda_risk_snapshot import recent_same_session_spreads
from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--points", type=int, default=250)
    parser.add_argument("--calibration-profile", type=Path)
    parser.add_argument("--calibration-release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    load_dotenv()
    try:
        source = json.loads(args.input.read_text(encoding="utf-8"))
        trade = source.get("trade_intent")
        manual = source.get("manual_account")
        if not isinstance(trade, dict) or not isinstance(manual, dict):
            raise ValueError("input requires trade_intent and manual_account objects")
        if trade.get("requested_size") is None:
            raise ValueError("manual mode requires stake per quoted-price unit")
        if manual.get("no_other_metals_positions_confirmed") is not True:
            raise ValueError("manual mode requires confirmation of no other metals positions")
        if bool(args.calibration_profile) != bool(args.calibration_release):
            raise ValueError("calibration profile and release must be supplied together")

        cfg = load_config(args.config)
        broker_instrument = cfg.data.oanda.instrument
        if not broker_instrument:
            raise ValueError("config must set data.oanda.instrument")
        client = oanda.get_client()
        pricing = client.pricing([broker_instrument])
        quotes = oanda.quotes_from_pricing(pricing)
        if broker_instrument not in quotes:
            raise RuntimeError("OANDA omitted the configured market instrument")
        quote = quotes[broker_instrument]
        candles = oanda.fetch_recent_candles(
            cfg.data,
            "M30",
            num_points=args.points,
            client=client,
            verbose=False,
        )
        now = pd.Timestamp.now(tz="UTC")
        quote.validate_for_entry(now=now)
        median_spread, p95_spread, spread_rows = recent_same_session_spreads(
            candles,
            decision_time=now,
        )
        event_active, event_reasons = scheduled_blackout_state(now)

        assessed_trade = dict(trade)
        assessed_trade["decision_time"] = now.isoformat()
        if str(assessed_trade.get("order_type") or "").lower() == "market":
            assessed_trade["entry_price"] = quote.mid
        payload = {
            "trade_intent": assessed_trade,
            "account": {
                "account_type": "SPREADBET",
                "status": "ENABLED",
                "currency": "GBP",
                "equity": manual.get("equity"),
                "available_funds": manual.get("available_funds"),
                "day_start_equity": manual.get("day_start_equity"),
                "week_start_equity": manual.get("week_start_equity"),
                "entries_today": manual.get("entries_today"),
                "kill_switch_active": manual.get("kill_switch_active"),
            },
            "instrument": {
                "instrument": cfg.symbol,
                "broker_instrument": broker_instrument,
                "instrument_type": "METAL",
                "expiry": "DFB",
                "account_currency": "GBP",
                "min_size": 0.01,
                "size_increment": 0.01,
                "value_per_price_unit": 1.0,
                "notional_value_per_price_unit": 1.0,
                "margin_rate": manual.get("margin_rate"),
                "margin_rate_max_size": None,
                "commission_per_size_per_side": manual.get(
                    "commission_per_stake_per_side", 0.0
                ),
            },
            "market": {
                "broker_instrument": broker_instrument,
                "as_of": quote.timestamp.isoformat(),
                "bid": quote.bid,
                "ask": quote.ask,
                "market_open": quote.market_status == "TRADEABLE",
                "quote_delay_seconds": quote.delay_seconds,
                "scheduled_event_blackout_active": event_active,
                "median_spread_points": median_spread,
                "p95_spread_points": p95_spread,
                "forecast_range_points": None,
                "forecast_horizon_minutes": None,
                "volatility_percentile": None,
                "volatility_artifact_approved": False,
            },
            "existing_positions": [],
        }
        calibration = None
        if args.calibration_profile:
            calibration = load_admitted_calibrated_risk(
                args.calibration_profile,
                args.calibration_release,
                instrument=assessed_trade.get("instrument"),
                decision_time=assessed_trade.get("decision_time"),
                horizon_minutes=assessed_trade.get("holding_period_minutes"),
                entry_price=assessed_trade.get("entry_price"),
                side=assessed_trade.get("side"),
            )
        report = evaluate_pretrade_payload(payload, historical_calibration=calibration)
        result = {
            "schema_version": 1,
            "mode": "MANUAL_GBP_ACCOUNT_OANDA_MARKET_READ_ONLY",
            "order_authority": False,
            "provenance": {
                "account_source": "user_supplied_not_broker_verified",
                "market_source": "OANDA_V20_READ_ONLY",
                "broker_instrument": broker_instrument,
                "quote_timestamp": quote.timestamp.isoformat(),
                "spread_sample_rows": spread_rows,
                "stake_unit": "GBP_per_quoted_price_unit",
                "margin_source": "user_supplied_external_broker_rate",
                "environment": client.environment,
                "event_calendar": {
                    "mode": "verified_schedule",
                    "active_reasons": event_reasons,
                    "verified_on": event_dates.EVENT_SCHEDULE_VERIFIED_ON,
                    "coverage_end": event_dates.COVERAGE_END.isoformat(),
                },
            },
            "adapter_issues": [
                "manual_account_facts_not_broker_verified",
                "external_margin_rate_user_supplied",
                "external_existing_positions_limited_to_explicit_none_confirmation",
            ],
            "risk_report": report.to_dict(),
        }
        encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Manual external assessment refused: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(f"=== MANUAL EXTERNAL PRE-TRADE RISK: {report.decision} ===")
    print(f"Report -> {args.output.resolve()}")
    print("No order was created, amended, closed or submitted.")


if __name__ == "__main__":
    main()
