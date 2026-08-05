# CME gold-futures data boundary

Databento `GLBX.MDP3` / `GC.FUT` supplies exchange-derived COMEX gold-futures
definitions, one-minute OHLCV, and statistics. It is a new market-information
source for later research; OANDA remains the intended spot-metals execution broker.

`scripts/import_cme_gc.py` verifies every batch file against its Databento
manifest, requires matching ranges across the three schemas, maps numeric
instrument IDs through the definition history as of each record timestamp,
filters `instrument_class=F` outright
futures, rejects malformed OHLCV, and writes checksummed normalized artifacts.
For a split delivery, OHLCV and statistics files are converted one at a time
into corresponding Parquet partitions; the full history is never duplicated
in memory.

The temporal join is required because Databento may reuse an `instrument_id`
for a different symbol later in a multi-year delivery. The importer retains
each identity-change boundary and refuses any record that has no earlier or
same-time definition; it never fills an old record from a future definition.

The production importer fences `2025-07-01` at the file level before opening
DBN records. A wholly post-cutoff monthly/daily file is checksum-verified but
excluded and named in the output manifest. A file straddling the cutoff is
refused rather than silently row-sliced.

Databento condition codes are also retained. `missing` or `pending` dates stop
the import. `degraded` dates are available but have vendor-identified quality
risk; they are written to the manifest and must be excluded, together with
dependent lag/rolling windows, by the later preregistered feature builder.

The importer intentionally does **not**:

- select a front contract;
- use same-day final volume to choose that day's contract;
- back-adjust prices;
- fill minutes with no reported trade;
- turn final open interest into information available earlier that day; or
- create a feature, signal, candidate, or strategy result.

## TBBO extension

The separate `scripts/import_cme_gc_tbbo.py` command handles trade-space BBO
data for the registered order-flow experiment. It requires schema `tbbo` with
the same `GLBX.MDP3`, `GC.FUT`, `instrument_id`, DBN, and Zstandard contract.
It validates aggressor side, trade price/size, and the best bid/offer immediately
before each trade; maps identities through the existing temporal definitions;
and retains only outright futures. Incomplete, locked, or crossed pre-trade
books are retained with `book_valid=false` for audit and excluded from feature
aggregation rather than repaired.

Start with the validation range specified in
`docs/archive/intraday_order_flow_design.md`. The continuous-history quote was rejected
after validation. Development TBBO must instead follow the checksummed sparse
plan produced by `scripts/plan_cme_gc_tbbo_windows.py`: exact 30-minute causal
windows and the causally selected raw contract only. Strategy evaluation was
blocked until every planned window was present and validated. That acquisition
subsequently completed under a $7.24 ceiling. The sparse normalizer retained
3.47 million outright trades, after which the frozen OF-GC gate failed and the
family was retired. See the design document §8 for the result.

Those choices require a separate preregistration after the complete
development download is validated. Data at or after `2025-07-01` remains
outside feature design and development evaluation.

## ZN/6E macro-driver extension

P4 uses the same temporal-identity safeguards for `ZN.FUT` and `6E.FUT`, but
needs only `ohlcv-1m` and `definition`. Its complete development request is
frozen before acquisition in `docs/archive/intraday_macro_impulse_design.md`.

Quote it without purchasing data:

```bash
python scripts/plan_cme_macro_drivers.py --estimate-cost
```

That command can call only Databento's metadata cost and dataset-condition
methods. The completed quote is `$18.939715769142`; it found no unavailable
dates and six degraded dates that the feature builder excludes. A separate
downloader requires an explicit ceiling equal to or above the recorded quote:

```bash
python scripts/download_cme_macro_drivers.py --authorized-max-usd <approved-cost>
python scripts/normalize_cme_macro_drivers.py
python scripts/build_cme_macro_state.py
python scripts/evaluate_intraday_macro_impulse.py
```

The first three stages attest that no gold data, label, return, or strategy
outcome was read. The final command runs exactly one observed P4 rule and its 20
registered joint time shifts; it does not train ML or open confirmation data.

The acquisition and evaluation subsequently completed. Normalization retained
2.19 million outright ZN and 2.48 million outright 6E minute bars; 55
pre-definition ZN rows and all registered degraded dates were excluded without
imputation. P4 failed with PF `0.851`, mean R `-0.096`, one profitable block,
double-cost PF `0.695`, and maximum placebo PF `0.913`. The family is retired.

Validate a delivery without writing output:

```bash
python scripts/import_cme_gc.py \
  --ohlcv-job /path/to/ohlcv-job \
  --statistics-job /path/to/statistics-job \
  --definition-job /path/to/definition-job \
  --validate-only
```

Remove `--validate-only` only for the complete `2020-01-01` through
`2025-07-01` development delivery. The default normalized location is
`data/raw/cme_gc/`, which is gitignored.
