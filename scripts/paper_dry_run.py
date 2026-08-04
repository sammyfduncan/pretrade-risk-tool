"""Build one fully checked OANDA paper order intent, but never submit it.

This is an operational harness for the broker/risk boundary while the three
primary strategies are still unimplemented. Candidate values are explicit CLI
inputs; no signal is invented by this script.
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.intraday.paper import (
    PaperCandidate,
    PaperRuntimeState,
    build_paper_intent,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--intent-id", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--side", choices=["long", "short"], required=True)
    parser.add_argument("--stop-distance-pct", type=float, required=True)
    parser.add_argument("--target-distance-pct", type=float, required=True)
    parser.add_argument("--expected-open-positions", type=int, required=True)
    parser.add_argument("--entries-today", type=int, default=0)
    parser.add_argument("--day-start-equity", type=float)
    parser.add_argument("--week-start-equity", type=float)
    parser.add_argument("--submitted-intent-id", action="append", default=[])
    parser.add_argument("--kill-switch", action="store_true")
    parser.add_argument("--points", type=int, default=150)
    parser.add_argument("--slippage-pct-per-side", type=float, default=0.00005)
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    if not cfg.data.oanda.instrument:
        raise SystemExit("data.oanda.instrument is unset")

    now = pd.Timestamp.now(tz="UTC")
    client = oanda.get_client()
    try:
        account, market, quote = oanda.fetch_broker_snapshot(
            cfg.data,
            client=client,
        )
    except RuntimeError as exc:
        raise SystemExit(f"Broker snapshot refused: {exc}") from exc
    broker_positions = oanda.fetch_open_position_ids(client)
    quotes = oanda.fetch_recent_candles(
        cfg.data,
        "M30",
        num_points=args.points,
        client=client,
        verbose=False,
    )

    runtime = PaperRuntimeState(
        day_start_equity=(
            account.equity
            if args.day_start_equity is None
            else args.day_start_equity
        ),
        week_start_equity=(
            account.equity
            if args.week_start_equity is None
            else args.week_start_equity
        ),
        entries_today=args.entries_today,
        expected_open_positions=args.expected_open_positions,
        submitted_intent_ids=tuple(args.submitted_intent_id),
        kill_switch_active=args.kill_switch,
    )
    candidate = PaperCandidate(
        intent_id=args.intent_id,
        strategy=args.strategy,
        side=args.side,
        decision_time=now,
        stop_distance_pct=args.stop_distance_pct,
        target_distance_pct=args.target_distance_pct,
    )
    decision = build_paper_intent(
        candidate,
        account=account,
        market=market,
        quote=quote,
        quote_candles=quotes,
        runtime=runtime,
        broker_open_position_ids=broker_positions,
        quote_history_bars=min(16, args.points - 1),
        slippage_pct_per_side=args.slippage_pct_per_side,
    )

    print(f"=== PAPER DRY RUN: {'ALLOW' if decision.allowed else 'REFUSE'} ===")
    print(f"Account equity: {account.equity:,.2f} {account.currency}")
    print(f"Broker positions: {len(broker_positions)}")
    if decision.reasons:
        print(f"Reasons: {', '.join(decision.reasons)}")
    if decision.spread:
        print(
            f"Costs: median {decision.spread.median_cost_r:.3f}R, "
            f"p95 {decision.spread.p95_cost_r:.3f}R"
        )
    if decision.sizing:
        print(
            f"Sizing: {decision.sizing.size:g} units, planned stop loss "
            f"{decision.sizing.planned_stop_loss:,.2f}/"
            f"{decision.sizing.risk_budget:,.2f}, notional "
            f"{decision.sizing.gross_notional:,.2f}"
        )
    if decision.intent:
        intent = decision.intent
        print(
            f"Intent: {intent.direction} {intent.size:g} {intent.broker_instrument} at "
            f"reference {intent.reference_price:g}, stop {intent.stop_level:g}, "
            f"target {intent.target_level:g}"
        )
    if decision.flatten_required:
        print("Risk policy requests flattening of existing positions.")
    print("No order was created or submitted to OANDA.")


if __name__ == "__main__":
    main()
