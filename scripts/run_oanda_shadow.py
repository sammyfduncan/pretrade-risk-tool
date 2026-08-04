"""Evaluate frozen intraday signals against OANDA without placing orders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.features import event_dates
from commodity_ml.intraday.data import build_intraday_frame
from commodity_ml.intraday.event_blackouts import scheduled_blackout_state
from commodity_ml.intraday.oanda_risk_snapshot import build_oanda_read_only_snapshot
from commodity_ml.intraday.pretrade_observations import PretradeObservationStore
from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload
from commodity_ml.intraday.risk_runtime import roll_and_reconcile_paper_runtime
from commodity_ml.intraday.shadow import (
    ShadowCheck,
    ShadowCheckStore,
    latest_closed_candidates,
    shadow_check_id,
    trade_intent_from_candidate,
)
from commodity_ml.intraday.state import PaperStateStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="portfolio config; defaults to gold and silver",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=("P1", "P2", "P3"),
        default=[],
        help="frozen strategy family; defaults to P1/P2/P3",
    )
    parser.add_argument(
        "--paper-state",
        type=Path,
        default=Path("data/runtime/oanda_paper_state.json"),
    )
    parser.add_argument(
        "--observation-database",
        type=Path,
        default=Path("data/runtime/pretrade_observations.sqlite"),
    )
    parser.add_argument(
        "--shadow-database",
        type=Path,
        default=Path("data/runtime/oanda_shadow_checks.sqlite"),
    )
    parser.add_argument("--m30-points", type=int, default=1500)
    parser.add_argument("--h1-points", type=int, default=750)
    parser.add_argument("--slippage-bp-per-side", type=float, default=0.5)
    parser.add_argument(
        "--duration-seconds",
        type=int,
        help="repeat for this bounded duration; omitted means one cycle",
    )
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.duration_seconds is not None and args.duration_seconds < 1:
        raise SystemExit("--duration-seconds must be positive")
    if args.poll_seconds < 1:
        raise SystemExit("--poll-seconds must be positive")
    if args.m30_points < 1000 or args.h1_points < 100:
        raise SystemExit("shadow warmup requires at least 1000 M30 and 100 H1 points")

    shadow_store = ShadowCheckStore(args.shadow_database)
    if args.summary_only:
        print(json.dumps(shadow_store.summary(), indent=2, sort_keys=True))
        print("No network or broker action was performed.")
        return
    if not args.paper_state.exists():
        raise SystemExit(
            f"Missing persistent paper state: {args.paper_state}\n"
            "Initialize it from a verified flat account with "
            "scripts/initialize_oanda_paper_state.py."
        )

    load_dotenv()
    config_paths = args.config or ["config/gold.yaml", "config/silver.yaml"]
    strategies = tuple(args.strategy or ("P1", "P2", "P3"))
    configs = [load_config(path) for path in config_paths]
    instruments = [cfg.data.oanda.instrument for cfg in configs]
    if any(not value for value in instruments) or len(set(instruments)) != len(instruments):
        raise SystemExit("Every config needs one distinct OANDA instrument")

    deadline = (
        time.monotonic() + args.duration_seconds
        if args.duration_seconds is not None
        else None
    )
    while True:
        try:
            counts = run_shadow_cycle(
                configs=configs,
                strategies=strategies,
                paper_state=args.paper_state,
                observation_database=args.observation_database,
                shadow_store=shadow_store,
                m30_points=args.m30_points,
                h1_points=args.h1_points,
                slippage_bp_per_side=args.slippage_bp_per_side,
            )
            print(
                f"Shadow cycle: {counts['signals']} signals, "
                f"{counts['assessed']} new assessments, "
                f"{counts['no_signal']} no-signal checks, "
                f"{counts['suppressed']} portfolio-priority suppressions"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            print(
                f"Shadow cycle refused safely: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            if deadline is None:
                raise SystemExit(1) from exc
        if deadline is None:
            break
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.poll_seconds, remaining))

    print(json.dumps(shadow_store.summary(), indent=2, sort_keys=True))
    print("No order was created, amended, closed or submitted to OANDA.")


def run_shadow_cycle(
    *,
    configs,
    strategies: tuple[str, ...],
    paper_state: Path,
    observation_database: Path,
    shadow_store: ShadowCheckStore,
    m30_points: int,
    h1_points: int,
    slippage_bp_per_side: float,
) -> dict[str, int]:
    """Run one account-consistent, read-only shadow evaluation cycle."""

    client = oanda.get_client()
    configured = [cfg.data.oanda.instrument for cfg in configs]
    account = oanda.fetch_account_snapshot(
        client, account_type=configs[0].data.oanda.account_type
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
        raise RuntimeError("OANDA omitted a configured shadow instrument")
    trades = client.open_trades()
    trade_ids = oanda.open_trade_ids_from_response(trades)
    now = pd.Timestamp.now(tz="UTC")
    runtime = roll_and_reconcile_paper_runtime(
        PaperStateStore(paper_state),
        account=account,
        broker_position_ids=trade_ids,
        now=now,
    )
    event_blackout, event_reasons = scheduled_blackout_state(now)
    observation_store = PretradeObservationStore(observation_database)
    symbol_by_instrument = {
        cfg.data.oanda.instrument: cfg.symbol for cfg in configs
    }
    counts = {"signals": 0, "assessed": 0, "no_signal": 0, "suppressed": 0}
    portfolio_slot_claimed = False

    for cfg in configs:
        instrument = cfg.data.oanda.instrument
        m30 = oanda.fetch_recent_candles(
            cfg.data, "M30", num_points=m30_points, client=client, verbose=False
        )
        h1 = oanda.fetch_recent_candles(
            cfg.data, "H1", num_points=h1_points, client=client, verbose=False
        )
        assessed_at = pd.Timestamp.now(tz="UTC")
        m30 = m30.loc[m30.index + pd.Timedelta(minutes=30) <= assessed_at]
        h1 = h1.loc[h1.index + pd.Timedelta(hours=1) <= assessed_at]
        frame = build_intraday_frame(h1, m30, source="oanda-shadow")
        signal_time, candidates = latest_closed_candidates(
            frame, strategies, now=assessed_at
        )

        for strategy in strategies:
            candidate = candidates[strategy]
            check_id = shadow_check_id(cfg.symbol, strategy, signal_time)
            prior_check = shadow_store.check_result(check_id)
            if prior_check is not None:
                _, prior_decision, _ = prior_check
                if prior_decision in ("ALLOW", "REDUCE"):
                    portfolio_slot_claimed = True
                continue
            if candidate is None:
                shadow_store.record(
                    ShadowCheck(
                        check_id=check_id,
                        instrument=cfg.symbol,
                        strategy=strategy,
                        signal_time=signal_time,
                        checked_at=assessed_at,
                        status="NO_SIGNAL",
                    )
                )
                counts["no_signal"] += 1
                continue

            counts["signals"] += 1
            trade_intent = trade_intent_from_candidate(
                candidate,
                symbol=cfg.symbol,
                quote=quotes[instrument],
                assessed_at=assessed_at,
                slippage_bp_per_side=slippage_bp_per_side,
            )
            intent_id = str(trade_intent["intent_id"])
            existing = observation_store.assessment_result(intent_id)
            if existing is not None:
                decision, reasons = existing
                shadow_store.record(
                    ShadowCheck(
                        check_id=check_id,
                        instrument=cfg.symbol,
                        strategy=strategy,
                        signal_time=signal_time,
                        checked_at=assessed_at,
                        status="ALREADY_ASSESSED",
                        intent_id=intent_id,
                        decision=decision,
                        reasons=reasons,
                    )
                )
                if decision in ("ALLOW", "REDUCE"):
                    portfolio_slot_claimed = True
                continue
            if portfolio_slot_claimed:
                shadow_store.record(
                    ShadowCheck(
                        check_id=check_id,
                        instrument=cfg.symbol,
                        strategy=strategy,
                        signal_time=signal_time,
                        checked_at=assessed_at,
                        status="SUPPRESSED",
                        intent_id=intent_id,
                        reasons=("one_position_shadow_priority",),
                    )
                )
                counts["suppressed"] += 1
                continue

            snapshot = build_oanda_read_only_snapshot(
                instrument_symbol=cfg.symbol,
                trade_intent=trade_intent,
                account=account,
                metadata=metadata[instrument],
                quote=quotes[instrument],
                recent_candles=m30,
                open_trades=trades,
                metadata_by_instrument=metadata,
                quotes_by_instrument=quotes,
                symbol_by_instrument=symbol_by_instrument,
                day_start_equity=runtime.day_start_equity,
                week_start_equity=runtime.week_start_equity,
                entries_today=runtime.entries_today,
                kill_switch_active=runtime.kill_switch_active,
                scheduled_event_blackout_active=event_blackout,
                now=assessed_at,
            )
            report = evaluate_pretrade_payload(snapshot.payload)
            wrapper = {
                "schema_version": 2,
                "mode": "OANDA_V20_SHADOW",
                "order_authority": False,
                "provenance": {
                    **snapshot.provenance,
                    "environment": client.environment,
                    "strategy": strategy,
                    "signal_time": signal_time.isoformat(),
                    "paper_runtime": runtime.provenance(),
                    "event_calendar": {
                        "mode": "verified_schedule",
                        "active_reasons": event_reasons,
                        "verified_on": event_dates.EVENT_SCHEDULE_VERIFIED_ON,
                        "coverage_end": event_dates.COVERAGE_END.isoformat(),
                    },
                    "portfolio_arbitration": "config_order_then_strategy_order",
                },
                "adapter_issues": snapshot.issues,
                "risk_report": report.to_dict(),
            }
            observation_store.record_assessment(wrapper)
            shadow_store.record(
                ShadowCheck(
                    check_id=check_id,
                    instrument=cfg.symbol,
                    strategy=strategy,
                    signal_time=signal_time,
                    checked_at=assessed_at,
                    status="ASSESSED",
                    intent_id=intent_id,
                    decision=report.decision,
                    reasons=report.reasons,
                )
            )
            counts["assessed"] += 1
            if report.decision in ("ALLOW", "REDUCE"):
                portfolio_slot_claimed = True
            print(
                f"  {cfg.symbol} {strategy} {signal_time}: {report.decision}"
                + (f" ({', '.join(report.reasons)})" if report.reasons else "")
            )
    return counts


if __name__ == "__main__":
    main()
