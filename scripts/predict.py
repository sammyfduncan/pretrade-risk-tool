"""Score the latest bar with the saved model — the live prediction path.

Usage:
    python scripts/predict.py --config config/gold.yaml            # cached candles
    python scripts/predict.py --config config/gold.yaml --refresh  # fetch first

STATUS: groundwork. The decision logic and the feature path are complete and
covered by tests/test_predict_parity.py, which runs offline. What is NOT yet
exercised is fetching live candles from IG and placing an order — that needs
API credentials. When those arrive, only the data-loading step below changes;
everything after it is already the tested path.

What this does
--------------
1. Load the config and the artifact saved by train.py.
2. Load trailing candles (>= MIN_HISTORY_BARS, so every feature is converged).
3. Build features through features/pipeline.py — the *same* function
   build_dataset.py uses. This is the whole train/live-skew defence: there is
   no second implementation to drift.
4. Score the last bar and apply the fold-fitted calibrator.
5. Compare against the primary rule and the stored entry threshold.

Why the decision needs two things, not one
------------------------------------------
Under meta-labeling the model never picks a side. The primary rule decides
whether to be long at all; the model only decides whether that particular
long is worth taking. So a high probability on a bar where the trend rule is
flat means nothing — there is no trade to filter. Both must agree.
"""

import argparse

import joblib
import pandas as pd
from dotenv import load_dotenv

from commodity_ml.config import load_config
from commodity_ml.decision import decide
from commodity_ml.features.macro import fetch_macro
from commodity_ml.features.pipeline import (
    MIN_HISTORY_BARS,
    latest_feature_row,
    load_partner_candles,
)
from commodity_ml.primary import make_primary_signal


def load_candles(cfg, refresh: bool) -> pd.DataFrame:
    """Trailing candles for the asset.

    THE LINE THAT CHANGES FOR LIVE TRADING. Today this reads the cached
    Dukascopy parquet. With IG credentials, `--refresh` pulls recent bars from
    the broker instead — deliberately a small window, because IG allows only
    10,000 historical points per week (see data/ig.py's allowance guard).
    Everything downstream is source-agnostic by construction: both providers
    return the same validated shape from data/schema.py.
    """
    if refresh:
        if cfg.data.source == "ig":
            from commodity_ml.data import ig as provider
        else:
            from commodity_ml.data import dukascopy as provider
        return provider.fetch_candles(cfg.data, cfg.granularity)

    path = cfg.raw_path()
    if not path.exists():
        raise SystemExit(
            f"No cached candles at {path}\n"
            f"Fetch them: python scripts/fetch_data.py --config {cfg.asset}.yaml"
        )
    return pd.read_parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/gold.yaml")
    parser.add_argument("--refresh", action="store_true",
                        help="fetch fresh candles instead of using the cache")
    args = parser.parse_args()

    load_dotenv()
    cfg = load_config(args.config)

    artifact = joblib.load(cfg.model_path)

    raw = load_candles(cfg, refresh=args.refresh)
    # Only the trailing window is needed, but keep a wide margin over the
    # minimum so EWM-based features are fully converged, not merely defined.
    window = raw.iloc[-(MIN_HISTORY_BARS * 2):]

    macro_raw = fetch_macro(start=str(window.index[0].date()), verbose=False) \
        if cfg.features.macro else None
    partners = load_partner_candles(cfg)

    features = latest_feature_row(cfg, window, macro=macro_raw, partners=partners)
    decision = decide(
        artifact,
        features,
        primary_long=bool(make_primary_signal(window, cfg.primary).iloc[-1]),
    )
    bar_time = decision.bar_time

    print(f"=== {cfg.asset} ({cfg.symbol} {cfg.granularity}) ===")
    print(f"  bar:              {bar_time}  close={window['close'].iloc[-1]:,.2f}")
    print(f"  model trained to: {artifact['trained_through']}")
    if bar_time <= artifact["trained_through"]:
        print("  NOTE: this bar is inside the training period — not a live prediction.")
    elif cfg.holdout_ts is not None and bar_time >= cfg.holdout_ts:
        # Predicting past the fence is exactly what live trading does, and is
        # fine. What is not fine is letting a run of these stand in for the
        # formal holdout test — reading them and then adjusting the design
        # spends the holdout just as surely as backtesting it would.
        print(f"  NOTE: past the holdout fence ({cfg.data.holdout_start}). Fine for a "
              "live decision;\n        not a substitute for the single-use holdout "
              "evaluation, and not something\n        to tune against.")
    print()
    print(f"  primary rule ({cfg.primary.rule}): "
          f"{'LONG' if decision.primary_long else 'flat'}")
    print(f"  model score:         {decision.score:.4f}  "
          f"(entry threshold {decision.threshold:.4f} = "
          f"OOF q{artifact['entry_quantile']:.2f})")
    print(f"  calibrated P(win):   {decision.calibrated_proba:.4f}  "
          "(what the score means, not what it ranks)")
    print(f"  model says:          {'take it' if decision.model_agrees else 'skip'}")
    print()
    print(f"  DECISION: {'ENTER LONG' if decision.take_trade else 'NO TRADE'}")
    if decision.take_trade:
        barrier = cfg.label.atr_mult
        print(f"    exit: +/-{barrier} ATR barriers, or {cfg.label.horizon_bars} bars, "
              "whichever comes first")

    print("\n(Order placement is not wired up — this reports a decision only.)")


if __name__ == "__main__":
    main()
