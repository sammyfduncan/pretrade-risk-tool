"""Audit an OANDA v20 metal for intraday admission without placing an order."""

import argparse

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import oanda
from commodity_ml.data.parity import compare_price_sources
from commodity_ml.intraday.market_data import assess_spread_history, closed_quote_window
from commodity_ml.intraday.risk import InstrumentConstraints, RiskLimits, size_for_stop

BAR_DURATIONS = {
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--granularity", choices=BAR_DURATIONS, default="M30")
    parser.add_argument("--points", type=int, default=500)
    parser.add_argument("--stop-distance-pct", type=float, required=True)
    parser.add_argument("--slippage-pct-per-side", type=float, default=0.00005)
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    client = oanda.get_client()
    account, metadata, quote = oanda.fetch_broker_snapshot(cfg.data, client=client)
    positions = oanda.fetch_open_position_ids(client)
    now = pd.Timestamp.now(tz="UTC")

    print(f"=== OANDA admission audit: {cfg.asset} / {metadata.instrument} ===")
    print(
        f"Account: ...{account.account_id[-4:]} {account.account_type} {account.currency} | "
        f"equity {account.equity:,.2f} (open P&L {account.open_pnl:+,.2f}) | "
        f"hedging API fact {account.hedging_enabled}"
    )
    print(f"Open broker trades: {len(positions)}")
    print(
        f"Quote: bid {quote.bid:g} / ask {quote.ask:g} | spread {quote.spread_fraction:.4%} | "
        f"status {quote.market_status} | age {quote.age(now)}"
    )
    try:
        quote.validate_for_entry(now=now)
        quote_pass = True
        print("Live quote gate: PASS")
    except RuntimeError as exc:
        quote_pass = False
        print(f"Live quote gate: FAIL ({exc})")

    print(
        f"Instrument: {metadata.name} | type {metadata.instrument_type} | unit {metadata.unit}"
    )
    print(
        f"Sizing: min {metadata.min_size:g}, step {metadata.size_increment:g}, "
        f"adverse GBP/USD conversion {metadata.value_per_price_unit:g}, "
        f"margin {metadata.margin_rate:.2%}"
    )
    recent = oanda.fetch_recent_candles(
        cfg.data, args.granularity, num_points=args.points, client=client, verbose=False
    )
    closed = closed_quote_window(
        recent,
        bar_duration=BAR_DURATIONS[args.granularity],
        now=now,
        min_bars=min(16, args.points - 1),
    )
    spread = assess_spread_history(
        closed,
        stop_distance_pct=args.stop_distance_pct,
        slippage_pct_per_side=args.slippage_pct_per_side,
    )
    print(
        f"Cost gate ({spread.observations} closed bars): median {spread.median_cost_r:.3f}R, "
        f"p95 {spread.p95_cost_r:.3f}R, latest {spread.current_cost_r:.3f}R — "
        f"{'PASS' if spread.allowed else 'FAIL'}"
    )

    entry = quote.ask
    stop = entry * (1 - args.stop_distance_pct)
    explicit_loss = 2 * entry * metadata.value_per_price_unit * args.slippage_pct_per_side
    sizing = size_for_stop(
        equity=account.equity,
        entry_price=entry,
        stop_price=stop,
        notional_per_size_unit=metadata.notional_per_size_unit(entry),
        instrument=InstrumentConstraints(
            min_size=metadata.min_size,
            size_increment=metadata.size_increment,
            value_per_price_unit=metadata.value_per_price_unit,
        ),
        limits=RiskLimits(),
        additional_loss_per_size_unit=explicit_loss,
    )
    print(
        f"Minimum-size/risk gate: {'PASS' if sizing.allowed else 'FAIL'} | size {sizing.size:g}, "
        f"stop loss {sizing.planned_stop_loss:,.2f}/{sizing.risk_budget:,.2f}, "
        f"notional {sizing.gross_notional:,.2f}"
    )

    duka_path = cfg.raw_path("dukascopy", granularity=args.granularity)
    if duka_path.exists():
        try:
            parity = compare_price_sources(
                pd.read_parquet(duka_path),
                recent,
                bar_duration=BAR_DURATIONS[args.granularity],
            )
        except ValueError as exc:
            parity_pass = False
            print(f"Source parity: NOT RUN — {exc}")
        else:
            print(
                f"Source parity: {parity.verdict} | corr {parity.return_correlation:.4f}, "
                f"OANDA coverage {parity.broker_coverage:.1%}, best shift {parity.best_shift_bars:+d} bars"
            )
            parity_pass = parity.verdict == "PASS"
    else:
        parity_pass = False
        print(f"Source parity: NOT RUN — missing {duka_path}")

    admitted = quote_pass and metadata.instrument_type == "METAL" and spread.allowed and sizing.allowed and parity_pass
    print(f"\nOverall automated subset: {'PASS' if admitted else 'FAIL'}")
    print("No order was created or submitted.")


if __name__ == "__main__":
    main()
