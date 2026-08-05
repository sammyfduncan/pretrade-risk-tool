# Executable quote-path measurement

**Status:** offline measurement and outcome construction implemented; OANDA
stream recording smoke validated, with no completed standardized trade outcome
**Purpose:** measure realised paper-trade path risk without midpoint or candle
substitution

## Measurement rule

The input is a sequence of timestamped top-of-book `bid` and `ask` observations
covering the actual fill-to-exit interval. It is not an OHLC candle file.

- Long positions are liquidated on bids. Long MAE is the distance from actual
  entry to the minimum observed bid; MFE uses the maximum bid.
- Short positions are liquidated on asks. Short MAE is the distance from actual
  entry to the maximum observed ask; MFE uses the minimum ask.
- Entry and exit spread are measured separately from their boundary quotes.
- Entry adverse slippage compares the actual fill with the recorded expected
  executable-side price. Exit adverse slippage compares the actual exit with
  the final executable-side quote.

The implementation performs no interpolation, resampling, forward-fill or
midpoint substitution. A post-exit quote is never included in excursion.

## Data-quality refusal rules

`src/commodity_ml/intraday/quote_path.py` refuses:

- naive, unsorted or duplicate timestamps;
- missing, non-positive, non-finite or crossed bid/ask values;
- candle columns in place of top-of-book `bid` and `ask`;
- fewer than two in-trade observations;
- no first quote within five seconds after fill;
- no last quote within five seconds before exit; or
- any internal observation gap over 120 seconds.

The 5/120-second defaults are operational coverage gates, not optimized alpha
parameters. Both are written into the audit and may be overridden explicitly;
changing them during a prospective evaluation must be treated as a data-policy
version change.

## Outcome construction

Populate `examples/pretrade_execution.template.json` from actual broker/demo
records. `expected_entry_price` means the executable-side price recorded before
submission—not a midpoint. Broker P&L, fees and peak margin remain observed
facts and are never replaced by a price-path estimate.

Given a recorded quote file:

```bash
python3 scripts/build_pretrade_outcome.py \
  examples/pretrade_execution.json \
  data/runtime/oanda_quotes_for_intent.parquet \
  --output data/runtime/pretrade_outcome.json
```

The command produces:

- a ledger-compatible outcome JSON containing measured MAE/MFE, boundary
  spreads and adverse slippage; and
- a separate audit JSON containing file hashes, path boundaries, observation
  count, maximum gap, executable side and the difference between broker P&L
  and a simple price-unit P&L reconstruction.

The P&L difference is diagnostic because broker financing, currency conversion
or product semantics may differ. It is not silently forced to zero.

## Remaining live validation

The sampled-pricing recorder and outage-aware path exporter are implemented in
`scripts/record_oanda_quotes.py` and `scripts/export_oanda_quote_path.py`; see
`docs/oanda_stream_recorder.md`. Authenticated practice runs have verified
streaming, storage and reconnect recording. Every exported trade interval must
still pass its own continuity checks. The full assessment-to-future-path
workflow has not yet been exercised on the standardized validation design.
