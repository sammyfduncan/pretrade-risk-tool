# Read-only IG pre-trade snapshot

**Status:** implemented and synthetic-tested; live account validation pending  
**Authority:** read-only risk assessment; no order creation, amendment or close

## What this adds

`scripts/evaluate_ig_pretrade_risk.py` translates current IG account/market
reads into the version-one quantitative risk contract. It reads:

- account equity, available funds, currency, type and status;
- the exact configured epic's bid, ask, timestamp, delay and market status;
- account-specific minimum size, instrument type, expiry and percentage margin;
- a bounded M30 bid/ask history for causal, closed, same-session spread
  quantiles; and
- current open positions, including their stops and liquidation-side quotes.

The default bounded history request is 250 M30 price points and counts against
IG's historical-price allowance; change it with `--points`. Ordinary account,
market and position reads do not consume that historical point count.

The adapter records a redacted account suffix and field provenance. Its output
sets `order_authority=false`. There is no call to create, update or close a
position anywhere in the adapter or CLI.

## Facts it refuses to guess

The following remain user/account/runtime facts rather than research defaults:

- `data.ig.epic`: exact regional spot spread-bet epic;
- `data.ig.size_increment`: confirmed order-ticket size step;
- `data.ig.value_per_price_unit`: account-currency P&L for size 1 and a 1.0
  price move;
- day-start and week-start equity plus entries already made today; and
- the current mechanical scheduled-event blackout state.

Unknown values flow into the report as `INCOMPLETE`. A historical event file
ending in 2025 is not silently treated as “no event” in 2026.

## Margin and portfolio safeguards

IG can publish tiered `marginDepositBands`. The adapter uses the first
percentage band only when it begins at size zero, and passes the band maximum
to the sizing engine as a hard scope cap. It never applies a low first-tier
rate to a larger position. A configured fallback is accepted only as an
explicitly labelled fallback.

For one existing position, `equity - available funds` is conservatively
assigned as its margin. With multiple positions the broker's aggregate margin
cannot be allocated safely from the available response, so the assessment is
`INCOMPLETE`. A missing protective stop is also incomplete, not zero risk.

## Input and command

Start from `examples/ig_pretrade_intent.template.json`. Runtime values can be
inside its `runtime` object or supplied on the command line:

```bash
python3 scripts/evaluate_ig_pretrade_risk.py \
  examples/ig_pretrade_intent.json \
  --config config/gold.yaml \
  --portfolio-config config/silver.yaml \
  --day-start-equity 10000 \
  --week-start-equity 10000 \
  --entries-today 0 \
  --event-state clear
```

`--event-state clear` is an assertion about a checked mechanical calendar, not
a default. Omitting it leaves the state unknown and therefore incomplete.
`--portfolio-config` supplies the confirmed epic and point value needed to
value an existing position in the other metal; it does not fetch that market
or generate another trade.

Render the resulting JSON as a standalone local report:

```bash
python3 scripts/render_pretrade_risk_report.py \
  reports/ig_pretrade_risk.json \
  --output reports/ig_pretrade_risk.html
```

The HTML is static, self-contained, print-friendly and escapes report text. It
performs no broker or network action and repeats that order authority is
disabled.

## Live validation still required

Before the tool can issue a complete current assessment:

1. discover and manually confirm the exact account epic;
2. confirm the size increment and P&L-per-price-unit on the demo order ticket;
3. extend/operate the scheduled-event calendar for the current date;
4. persist day/week equity anchors and entry counts from the paper runtime;
5. run the bounded IG read and inspect quote freshness, spread samples, margin
   source and product type; and
6. shadow-log assessments versus realised spread, slippage and excursion.

The development calibration profile currently ends on 2025-06-30 and is
refused as stale for current decisions. It remains a research baseline until a
causal fresh profile update is defined and verified.

## Broker references

- [IG REST market details (`GET /markets/{epic}`)](https://labs.ig.com/reference/markets-epic.html)
- [IG REST open positions (`GET /positions`)](https://labs.ig.com/reference/positions.html)
- [IG REST versus streaming price guidance](https://labs.ig.com/trading-basics.html)
