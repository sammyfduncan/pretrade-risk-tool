"""Backtest the out-of-fold predictions as a long-only strategy via vectorbt.

Usage:
    python scripts/backtest.py --config config/gold.yaml

Reads data/processed/<asset>_oof.parquet (produced by train.py) plus raw
prices; prints strategy-vs-buy-and-hold metrics and a threshold sensitivity
table; writes reports/<asset>_backtest.txt.
"""

import argparse

import pandas as pd

from commodity_ml.backtest.engine import run_backtest, threshold_sweep
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

    close = pd.read_parquet(cfg.raw_path)["close"]
    oof = pd.read_parquet(cfg.oof_path)["proba"]
    ppy = bars_per_year(cfg.granularity)

    print(f"Backtesting {cfg.asset} on OOF window {oof.index[0]} -> {oof.index[-1]}")
    print(f"  entry: P >= {cfg.backtest.prob_entry}, exit: P <= {cfg.backtest.prob_exit}, "
          f"fees: {cfg.backtest.fee_pct:.2%}/side\n")

    result = run_backtest(close, oof, cfg.backtest, cfg.granularity, ppy)

    strategy_txt = f"=== Strategy (long-only, OOF predictions) ===\n{fmt_stats(result.stats)}"
    bh_txt = f"=== Buy & hold (same window) ===\n{fmt_stats(result.buy_hold_stats)}"
    print(strategy_txt + "\n\n" + bh_txt)

    sweep = threshold_sweep(close, oof, cfg.backtest, cfg.granularity, ppy)
    sweep_txt = ("=== Entry-threshold sensitivity (DIAGNOSTIC — do not cherry-pick "
                 "the best row) ===\n" + sweep.to_string())
    print("\n" + sweep_txt)

    n_trades = result.stats.get("n_trades", 0)
    if n_trades < 100:
        print(f"\nNOTE: only {n_trades} trades — below the pre-registered minimum "
              "sample (100+). No conclusions yet (README methodology).")

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"{cfg.asset}_backtest.txt"
    report_path.write_text(strategy_txt + "\n\n" + bh_txt + "\n\n" + sweep_txt + "\n")
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
