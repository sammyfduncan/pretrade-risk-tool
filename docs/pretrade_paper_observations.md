# Prospective paper-risk observation ledger

**Status:** storage, validation and descriptive summary implemented; the first
live shadow cycle has recorded six no-signal checks, while no trade assessment
or outcome exists yet
**Purpose:** compare pre-trade risk assessments with later measured execution
and path outcomes without changing history

## Boundary

Nothing in this ledger is predicted or invented. The workflow has two separate
times:

1. Before entry, store the already-generated risk assessment exactly as it
   existed.
2. After the paper trade closes, store measured broker/path observations for
   that same `intent_id`.

The SQLite database accepts no update or delete operation. Repeating identical
input is idempotent; reusing an intent id with different assessment or outcome
content is refused. Outcomes are accepted only for a prior `ALLOW` or `REDUCE`
assessment, and their entry timestamp cannot precede the assessment.

This prevents an outcome from influencing the risk report it is supposedly
validating. It also prevents blocked or incomplete hypothetical trades from
being treated as executed evidence.

## Recorded outcome facts

`examples/pretrade_outcome.template.json` defines the observation contract:

- actual timezone-aware entry and exit times;
- filled size and expected/actual entry plus actual exit price;
- measured maximum adverse and favourable excursion in price points;
- realised entry/exit spreads and entry/exit adverse slippage in points;
- actual fees, peak margin and realised P&L; and
- an explicit source such as `OANDA_PRACTICE`, a broker trade/source reference, and
  optional notes.

All prices and filled size must be positive. Excursion, spread, slippage, fees
and margin are non-negative. P&L is signed. A trade-path collector must later
derive MAE/MFE from valid executable broker quotes; this foundation does not
pretend that midpoint or research candles are broker executions.

The offline executable-path builder and its stricter bid/ask rules are defined
in `docs/pretrade_quote_path.md`. The OANDA recorder and read-only shadow runner
have both passed their first authenticated practice-account smoke checks.

## Commands

Record the assessment immediately after generating it and before the paper
outcome is known:

```bash
python3 scripts/log_pretrade_observation.py assessment \
  reports/oanda_pretrade_risk.json
```

After the trade has closed and the outcome JSON has been populated from valid
paper/broker records:

```bash
python3 scripts/log_pretrade_observation.py outcome \
  examples/pretrade_outcome.json
```

Produce descriptive monitoring statistics:

```bash
python3 scripts/log_pretrade_observation.py summary
```

The default database is `data/runtime/pretrade_observations.sqlite`, which is
inside the repository's ignored data area. Use `--database` before the
subcommand to select another path.

## Current summary scope

The summary reports assessment counts, completed outcomes, realised P&L, MAE,
entry/exit spread and slippage distributions, and how often realised loss exceeded the
registered size-adjusted planned-stop loss. These are descriptive operational
and calibration diagnostics. They do not establish strategy profitability,
forecast skill or causality.

No result is meaningful until prospectively collected paper observations have
accumulated. A stop-loss exceedance is a review trigger: it may indicate a gap,
slippage, fees, a sizing mismatch or invalid source data.
