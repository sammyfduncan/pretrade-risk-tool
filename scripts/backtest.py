"""Backtest the out-of-fold predictions as a long-only strategy via vectorbt.

Usage:
    python scripts/backtest.py --config config/gold.yaml

Reads data/processed/<asset>_oof.parquet (produced by train.py) plus raw
prices. Entries are the top `entry_quantile` of predictions (computed on an
expanding window, so no future information sets today's threshold). The
historical simulator uses the label's barrier percentages and horizon, but is
not fully label-equivalent because it uses close-only bars and a delayed entry
anchor. Treat its P&L as provisional; see `docs/archive/intraday_rebuild_plan.md`.

Writes reports/<asset>_backtest.txt.
"""

import argparse

import pandas as pd

from commodity_ml.backtest.engine import quantile_sweep, run_backtest, run_rule_backtest
from commodity_ml.config import bars_per_year, load_config
from commodity_ml.primary import SIGNAL_COLUMN


def fmt_stats(stats: dict) -> str:
    lines = []
    for key, value in stats.items():
        if isinstance(value, float):
            if key in ("total_return", "cagr", "max_drawdown", "win_rate", "time_in_market"):
                lines.append(f"  {key:<15} {value:>10.2%}")
            else:
                lines.append(f"  {key:<15} {value:>10.2f}")
        else:
            lines.append(f"  {key:<15} {value:>10}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # The OOF file already reflects whichever slice train.py ran on, so the
    # backtest inherits the holdout decision rather than making its own.
    close = pd.read_parquet(cfg.raw_path())["close"]
    oof = pd.read_parquet(cfg.oof_path)
    proba, barrier_pct = oof["proba"], oof["barrier_pct"]
    if cfg.holdout_ts is not None and proba.index.max() < cfg.holdout_ts:
        print(f"(Holdout from {cfg.data.holdout_start} is excluded — inherited from train.py)")
    ppy = bars_per_year(cfg.granularity)

    print(f"Backtesting {cfg.asset} on OOF window {proba.index[0]} -> {proba.index[-1]}")
    print(f"  entry: top {1 - cfg.backtest.entry_quantile:.0%} of predictions "
          f"(expanding-window quantile)")
    print(f"  exit:  +/-{cfg.label.atr_mult} ATR barriers, or {cfg.label.horizon_bars} bars")
    print(f"  fees:  {cfg.backtest.fee_pct:.2%}/side\n")

    result = run_backtest(
        close, proba, barrier_pct, cfg.backtest, cfg.granularity, ppy, cfg.label.horizon_bars
    )

    strategy_txt = f"=== Strategy (long-only, OOF predictions) ===\n{fmt_stats(result.stats)}"
    bh_txt = f"=== Buy & hold (same window) ===\n{fmt_stats(result.buy_hold_stats)}"
    sections = [strategy_txt, bh_txt]

    # The control that decides whether the model earned anything. Printed
    # unconditionally, not behind a flag: a meta-label that cannot beat its own
    # primary rule is decoration, and that comparison is too easy to skip.
    if cfg.label.mode == "meta":
        dataset = pd.read_parquet(cfg.dataset_path)
        rule = run_rule_backtest(
            close, dataset[SIGNAL_COLUMN], barrier_pct, cfg.backtest,
            cfg.granularity, ppy, cfg.label.horizon_bars,
            window=result.portfolio.wrapper.index,
        )
        sections.insert(1, f"=== Control: primary rule '{cfg.primary.rule}' alone, "
                           f"no ML filter (same window) ===\n{fmt_stats(rule.stats)}")

        filtered, bare = result.stats, rule.stats
        kept = filtered.get("n_trades", 0) / max(bare.get("n_trades", 0), 1)
        sections.insert(2, (
            "=== Did the filter earn its place? ===\n"
            f"  trades kept      {kept:>10.1%}  ({filtered.get('n_trades', 0)} of "
            f"{bare.get('n_trades', 0)})\n"
            f"  profit factor    {bare.get('profit_factor', 0):.2f} -> "
            f"{filtered.get('profit_factor', 0):.2f}\n"
            f"  sharpe           {bare['sharpe']:.2f} -> {filtered['sharpe']:.2f}\n"
            f"  win rate         {bare.get('win_rate', 0):.1%} -> "
            f"{filtered.get('win_rate', 0):.1%}\n"
            f"  total return     {bare['total_return']:.1%} -> "
            f"{filtered['total_return']:.1%}"
        ))

    print("\n\n".join(sections))

    sweep = quantile_sweep(
        close, proba, barrier_pct, cfg.backtest, cfg.granularity, ppy, cfg.label.horizon_bars
    )
    sweep_txt = ("=== Entry-selectivity sensitivity (WEAK DIAGNOSTIC — not a robustness "
                 "test) ===\n"
                 "Rows are not nested: changing the quantile also changes which bar in "
                 "each cluster\nsurvives debouncing, so smooth is not robust and jagged "
                 "is not fragile. Do not\ncherry-pick a row, and do not read this as "
                 "evidence either way (see engine.py).\n"
                 + sweep.to_string())
    print("\n" + sweep_txt)

    # Entries are debounced (engine.debounce_entries), so their label horizons
    # do not overlap mechanically. Regime clustering can still reduce the
    # effective sample; the raw count is not proof of independence.
    n_trades = result.stats.get("n_trades", 0)
    if n_trades < 100:
        print(f"\nNOTE: {n_trades} non-overlapping trades, below the pre-registered "
              "minimum of 100 (README methodology). No conclusions yet.")

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"{cfg.asset}_backtest.txt"
    report_path.write_text("\n\n".join(sections + [sweep_txt]) + "\n")
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
