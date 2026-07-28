"""Backtest the out-of-fold predictions as a long-only strategy via vectorbt.

Usage:
    python scripts/backtest.py --config config/gold.yaml

Reads data/processed/<asset>_oof.parquet (produced by train.py) plus raw
prices. Entries are the top `entry_quantile` of predictions (computed on an
expanding window, so no future information sets today's threshold); exits are
the label's own barriers, so the simulated trade matches the labeled one.

Writes reports/<asset>_backtest.txt.
"""

import argparse

import pandas as pd

from commodity_ml.backtest.engine import quantile_sweep, run_backtest
from commodity_ml.config import bars_per_year, load_config


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
    print(strategy_txt + "\n\n" + bh_txt)

    sweep = quantile_sweep(
        close, proba, barrier_pct, cfg.backtest, cfg.granularity, ppy, cfg.label.horizon_bars
    )
    sweep_txt = ("=== Entry-selectivity sensitivity (DIAGNOSTIC — do not cherry-pick "
                 "the best row) ===\n" + sweep.to_string())
    print("\n" + sweep_txt)

    # Entries are debounced (engine.debounce_entries), so trades never overlap
    # and each one is an independent bet — the count can be taken at face value.
    n_trades = result.stats.get("n_trades", 0)
    if n_trades < 100:
        print(f"\nNOTE: {n_trades} non-overlapping trades, below the pre-registered "
              "minimum of 100 (README methodology). No conclusions yet.")

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"{cfg.asset}_backtest.txt"
    report_path.write_text(strategy_txt + "\n\n" + bh_txt + "\n\n" + sweep_txt + "\n")
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
