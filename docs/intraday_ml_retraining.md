# Intraday ML data and retraining protocol

**Status:** training complete; no intraday filter retained for version 1

## Role of ML

ML does not predict every M30 bar and does not choose direction. P1, P2, or P3
must first create a complete long/short candidate with its registered ATR stop,
target in `R`, timeout, and next-bar entry contract. A separate model for that
family then asks whether this otherwise valid candidate is worth taking.

If a family has fewer than 100 executed development candidates, it remains a
bare rule. ML is not fitted to manufacture an answer from a small sample.

## Initial training data

The first model uses:

- Dukascopy H1 candles for completed higher-timeframe context;
- Dukascopy M30 candles for candidates, ATR stops, and historical outcomes;
- the development period ending before `2025-07-01`;
- flat conservative costs while deep historical bid/ask data is unavailable;
- OANDA only for recent H1/M30 timestamp/return parity and observed spread stress.

Dukascopy is used for deep, pinned multi-year H1/M30 training. OANDA remains
the account-specific execution truth. A model is
inadmissible if recent cross-source parity fails.

The build is:

```text
Dukascopy H1 + M30
        ↓ causal H1 close availability
shared intraday base
        ↓ P1, P2, or P3 only
primary candidates
        ↓ exact next-open OHLC execution and net label
candidate-only dataset
        ↓ purged expanding walk-forward
logistic baseline + LightGBM challenger
```

H1 bar-open timestamps are shifted to their close/availability time before
joining. An M30 decision can therefore use only the last fully completed H1
bar. The confirmation window is never included by these builders.

## Features

The initial models use a small, explicit mechanism-based contract:

- normalized H1 EMA state, EMA50 slope, and ATR;
- normalized M30 EMA, ATR, Bollinger position/width, body, and range;
- previous-session range/return;
- time remaining in the London entry window;
- candidate direction; and
- one or two setup-specific distances for the relevant primary family.

Absolute prices, exit fields, future duration, label-derived columns, and
non-candidate bars cannot enter the model. The dataset loader refuses missing,
unexpected, nonnumeric, or non-finite feature columns.

Historical OANDA spread is not silently fabricated for Dukascopy rows. Current
spread remains an execution/admission input. A prospective spread feature needs
enough point-in-time OANDA history before it can become a model input.

## Validation and selection

- A minimum of 100 executed candidates is required to start training.
- Five expanding, time-ordered folds are used.
- A training candidate is purged if its trade had not exited before the test
  block began; candidate spacing is irregular, so a fixed row gap is not
  sufficient.
- Overlapping trades receive uniqueness weights.
- Logistic regression is always the baseline.
- LightGBM uses fixed conservative defaults and no hyperparameter search.
- Raw OOF scores measure ranking; calibration is evaluated separately.
- No deployment threshold is selected by the training command.
- Any proposed filter must be compared with the same bare rule, entries,
  period, and costs.

The initial artifact is marked
`research_only_threshold_not_registered`. The paper path cannot consume it.

Candidate parquet files are not trusted on filename alone. Each build publishes
the candidate dataset, its complete reason-coded execution ledger, and a
manifest containing provenance, program/granularity, cost assumption, row
counts, feature order, and SHA-256 hashes. Training rechecks the hashes, cost
configuration, and exact signal/entry/exit/label/return parity before
constructing `X` and `y`.

## Retraining lifecycle

1. Build the initial development model once, then register the candidate filter
   and threshold before opening any confirmation or performance period.
2. Operational-shadow fixes may create a new version. Prediction-logic changes
   reset the evaluation clock.
3. The frozen paper-performance stage does not retrain.
4. After that gate, a challenger may be refit on an expanding window at a
   scheduled review no more frequently than every four weeks and only after at
   least 50 genuinely new executed candidates have accumulated.
5. The incumbent remains active until the challenger passes the same purged
   walk-forward, calibration, cost, bare-rule, drift, and source-parity gates.
6. A promoted artifact records its data cutoff, source, feature order, rule
   version, costs, and OOF evidence. A failed challenger is logged and discarded.

This is intentionally slower than continuous online learning. Commodity
candidate samples are small and clustered; frequent refits would mostly chase
the latest regime.

## Commands

```bash
python scripts/fetch_data.py --config config/gold.yaml \
  --source dukascopy --granularity H1
python scripts/fetch_data.py --config config/gold.yaml \
  --source dukascopy --granularity M30
python scripts/build_intraday_data.py --config config/gold.yaml
python scripts/build_intraday_candidates.py --config config/gold.yaml --strategy P1
python scripts/train_intraday.py --config config/gold.yaml --strategy P1 --model both
```

Repeat candidate building/training for P2 and P3. This is not a license to
retain all three or either model family; every retention decision still uses
the pre-registered gates.

## Version-1 outcome

Gold candidate models were trained on 2026-07-31. Six strategy/model
combinations were screened, after which P1-logistic and P2-LightGBM entered the
single frozen nested filter gate in `docs/intraday_filter_gate.md`. A later
correctness audit found that scheduled-event blackouts had not been passed into
candidate execution; all candidate bundles, models, and gates were regenerated
over the verified 2020–2025 development span without changing the registered
model choices. Both filters still failed. The model files remain research/audit artifacts with
`deployment_threshold=None`; none may enter paper or live decisions.
