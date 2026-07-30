"""Audit an IG market for intraday admission without placing any order.

This spends a bounded number of historical price points, reads account/market
metadata, validates quote freshness and spread history, checks broker minimum
size against the frozen risk caps, and compares recent IG prices with a cached
Dukascopy file when available.

Example:
    python scripts/audit_ig.py --config config/gold.yaml --granularity M30 \
        --stop-distance-pct 0.01 --points 500
"""

import argparse

import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.data import ig
from commodity_ml.data.parity import compare_price_sources
from commodity_ml.intraday.market_data import (
    assess_spread_history,
    closed_quote_window,
)
from commodity_ml.intraday.risk import (
    InstrumentConstraints,
    RiskLimits,
    size_for_stop,
)

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
    parser.add_argument(
        "--stop-distance-pct",
        type=float,
        required=True,
        help="representative executable entry-to-stop distance, e.g. 0.01 for 1%%",
    )
    parser.add_argument(
        "--slippage-pct-per-side",
        type=float,
        default=0.00005,
        help="additional adverse slippage allowance; default 0.5 bp per side",
    )
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)
    if not cfg.data.ig.epic:
        raise SystemExit(
            "data.ig.epic is unset. Run scripts/find_epic.py and confirm the "
            "account-specific spot market first."
        )

    service = ig.get_service()
    now = pd.Timestamp.now(tz="UTC")
    account = ig.fetch_account_snapshot(service)
    details = ig.fetch_market_details(service, cfg.data.ig.epic)
    quote = ig.market_quote_from_details(details, now=now)
    positions = ig.fetch_open_position_ids(service)

    print(f"=== IG admission audit: {cfg.asset} / {cfg.data.ig.epic} ===")
    print(
        f"Account: {account.account_id} {account.account_type} "
        f"{account.currency} | equity {account.equity:,.2f} "
        f"(open P&L {account.open_pnl:+,.2f})"
    )
    print(f"Open broker positions: {len(positions)}")
    print(
        f"Quote: bid {quote.bid:g} / ask {quote.ask:g} | "
        f"spread {quote.spread_fraction:.4%} | status {quote.market_status} | "
        f"age {quote.age(now)} | delay {quote.delay_seconds}s"
    )
    try:
        quote.validate_for_entry(now=now)
        quote_pass = True
        print("Live quote gate: PASS")
    except RuntimeError as exc:
        quote_pass = False
        print(f"Live quote gate: FAIL ({exc})")

    try:
        metadata = ig.market_metadata_from_details(
            details,
            account_currency=account.currency,
            size_increment=cfg.data.ig.size_increment,
            value_per_price_unit=cfg.data.ig.value_per_price_unit,
        )
    except RuntimeError as exc:
        raise SystemExit(
            f"Sizing metadata: FAIL ({exc})\n"
            "Confirm both values from the demo order ticket; do not infer them "
            "from another account or regional epic."
        ) from exc

    print(
        f"Instrument: {metadata.name} | type {metadata.instrument_type} | "
        f"unit {metadata.unit} | expiry {metadata.expiry}"
    )
    print(
        f"Sizing facts: min {metadata.min_size:g}, step "
        f"{metadata.size_increment:g}, account-currency value/price-unit "
        f"{metadata.value_per_price_unit:g}"
    )

    recent = ig.fetch_recent_candles(
        cfg.data,
        args.granularity,
        num_points=args.points,
        service=service,
        verbose=False,
    )
    closed = closed_quote_window(
        recent,
        bar_duration=BAR_DURATIONS[args.granularity],
        now=now,
        min_bars=min(100, args.points - 1),
    )
    spread = assess_spread_history(
        closed,
        stop_distance_pct=args.stop_distance_pct,
        slippage_pct_per_side=args.slippage_pct_per_side,
    )
    print(
        f"Cost gate ({spread.observations} closed bars): "
        f"median {spread.median_cost_r:.3f}R, p95 {spread.p95_cost_r:.3f}R, "
        f"latest {spread.current_cost_r:.3f}R — "
        f"{'PASS' if spread.allowed else 'FAIL'}"
    )
    if spread.reasons:
        print(f"  reasons: {', '.join(spread.reasons)}")

    entry = quote.ask
    stop = entry * (1 - args.stop_distance_pct)
    explicit_loss_per_size = (
        2
        * entry
        * metadata.value_per_price_unit
        * args.slippage_pct_per_side
    )
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
        additional_loss_per_size_unit=explicit_loss_per_size,
    )
    print(
        f"Minimum-size/risk gate: {'PASS' if sizing.allowed else 'FAIL'} | "
        f"size {sizing.size:g}, stop loss {sizing.planned_stop_loss:,.2f}/"
        f"{sizing.risk_budget:,.2f}, notional {sizing.gross_notional:,.2f}"
    )
    if sizing.reason:
        print(f"  reason: {sizing.reason}")

    duka_path = cfg.raw_path(
        "dukascopy",
        granularity=args.granularity,
    )
    if duka_path.exists():
        parity = compare_price_sources(
            pd.read_parquet(duka_path),
            recent,
            bar_duration=BAR_DURATIONS[args.granularity],
        )
        print(
            f"Source parity: {parity.verdict} | corr "
            f"{parity.return_correlation:.4f}, IG coverage "
            f"{parity.ig_coverage:.1%}, Duka coverage "
            f"{parity.dukascopy_coverage:.1%}, best shift "
            f"{parity.best_shift_bars:+d} bars"
        )
        if parity.reasons:
            print(f"  reasons: {', '.join(parity.reasons)}")
        parity_pass = parity.verdict == "PASS"
    else:
        parity_pass = False
        print(
            f"Source parity: NOT RUN — missing {duka_path}\n"
            f"  fetch it with: python scripts/fetch_data.py --config "
            f"{args.config} --source dukascopy --granularity {args.granularity}"
        )

    admitted = (
        quote_pass
        and metadata.instrument_type == "COMMODITIES"
        and spread.allowed
        and sizing.allowed
        and parity_pass
    )
    print(
        "\nOverall automated subset: "
        f"{'PASS' if admitted else 'FAIL'} "
        "(published dealing hours and the human epic/market confirmation remain "
        "manual admission checks)."
    )
    print("No order was created or submitted.")


if __name__ == "__main__":
    main()
