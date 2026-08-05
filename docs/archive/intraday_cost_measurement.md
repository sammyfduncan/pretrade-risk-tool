# Intraday execution-cost measurement

## Purpose

The fixed Dukascopy research allowance is two basis points round trip. That is
a conservative placeholder, not a fact about the user's OANDA account.
Promotion requires measured costs from the exact v20 account and instrument.

The cost stack is separated into:

1. **spread:** observed OANDA bid/ask quotes;
2. **slippage:** fill price versus the executable quote captured before order
   submission;
3. **commission/fees:** explicit broker charges where applicable; and
4. **financing:** excluded by the forced-flat intraday mandate, but any
   accidental overnight position must record it.

`scripts/measure_oanda_costs.py` implements the spread profile offline. Realized
slippage requires demo fills and cannot be inferred from candles.

## Collecting a bounded quote sample

Using the configured practice account, collect bounded recent M30 and M15 quote
candles. OANDA broker pulls remain diagnostic rather than research backfills.
For example, choose dates appropriate to the day the command is run:

```bash
python scripts/fetch_data.py --config config/gold.yaml --source oanda \
  --granularity M30 --start 2026-08-01

python scripts/fetch_data.py --config config/gold.yaml --source oanda \
  --granularity M15 --start 2026-08-01
```

The fetcher refuses an oversized pull and writes canonical midpoint plus
bid/ask OHLC parquet files.

## Building the profile

Use a representative stop matching the actual strategy. Recent gold candidate
median stops have been around `0.0018`–`0.0021` (0.18%–0.21%), not 1%:

```bash
python scripts/measure_oanda_costs.py --config config/gold.yaml \
  --granularity M30 --stop-distance-pct 0.0018 \
  --slippage-bp-per-side 0.5

python scripts/measure_oanda_costs.py --config config/gold.yaml \
  --granularity M15 --stop-distance-pct 0.0020 \
  --slippage-bp-per-side 0.5
```

The report contains overall basis-point quantiles, each London clock slot,
weekday profiles, and median/p95 cost in stop-risk units. The spread proxy uses
the worse synchronized open/close spread in each candle; bid/ask highs and lows
are not subtracted because their extrema may occur at different instants.

An overlapping execution ledger can be supplied with `--ledger`. The report
then measures coverage, candidate-entry median/p95 costs, cost in R, the gross
break-even cost, and estimated PF after the observed quote-cost proxy.

## Admission rules

- median round-trip spread plus slippage must be at most `0.10R`;
- p95 must be at most `0.25R`;
- candidate-window coverage must be reported, never silently reduced;
- M15 and M30 are assessed separately;
- the current quote must also be below `0.25R`; and
- a strategy that failed stability gates is not resurrected merely because
  observed spread happens to be below the historical placeholder.

At least several weeks of quotes should be collected, spanning ordinary and
event days. Demo fill slippage is added only after the order/confirmation
adapter records expected quote, submitted time, broker fill, side, size, and
exit fill. Until then, `0.5 bp` per side remains an explicit stress assumption,
not a measurement.
