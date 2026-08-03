# Quantitative pre-trade risk engine — version 1 contract

**Status:** offline foundation in implementation  
**Decision date:** 2026-08-03  
**Assets:** spot gold (`XAUUSD`) and spot silver (`XAGUSD`) spread bets  
**Scope:** technical and mathematical risk assessment, not directional or
fundamental forecasting

## 1. Purpose and boundary

The engine evaluates an intended trade before an order exists. It answers:

> If this trade is entered with the supplied broker, account, market and
> portfolio facts, what can be lost under the planned stop and conservative
> stresses, what size fits the policy, and is the request complete and safe
> enough to proceed?

It does not decide whether gold or silver should rise or fall. It does not read
news, infer geopolitics, use an LLM, or assign an unsupported probability of
profit. Scheduled CPI, NFP and FOMC windows may enter as deterministic blackout
flags because they alter observable execution and volatility risk; their
content is never interpreted.

The engine is decision support. `ALLOW` never means profitable or safe from
loss. A normal stop is not a guaranteed maximum loss: gaps, slippage, broker
delay and market closure can produce a larger loss.

## 2. Version 1 inputs

Inputs are point-in-time snapshots. All monetary values are in the account
currency, and every timestamp is timezone-aware.

### 2.1 Trade intent

| Field | Meaning |
|---|---|
| `intent_id` | Stable id used for audit and later idempotency |
| `instrument` | `XAUUSD` or `XAGUSD` only |
| `side` | `long` or `short` |
| `order_type` | `market`, `limit`, or `stop` |
| `decision_time` | Time at which the assessment is made |
| `entry_price` | Intended midpoint/trigger reference price |
| `stop_price` | Intended protective-stop trigger |
| `target_price` | Intended profit target |
| `requested_size` | Optional stake/size; when absent the engine recommends a cap |
| `holding_period_minutes` | Maximum intended holding time |
| `allow_overnight` | Must be false under version 1 |
| `max_slippage_points_per_side` | Explicit adverse execution allowance |

The geometry must be coherent: a long stop is below entry and target above it;
a short stop is above entry and target below it.

### 2.2 Account snapshot

| Field | Meaning |
|---|---|
| `currency` | Trading-account currency |
| `equity` | Balance plus open P&L |
| `available_funds` | Broker-reported funds available for new margin |
| `day_start_equity` | Equity at the London trading-day boundary |
| `week_start_equity` | Equity at the London week boundary |
| `entries_today` | New entries already made during the London day |

### 2.3 Broker instrument facts

| Field | Meaning |
|---|---|
| `instrument` | Must exactly match the trade |
| `epic` | Exact regional IG market identifier for these product facts |
| `account_currency` | Currency in which P&L and margin facts are expressed |
| `min_size` | Smallest permitted stake |
| `size_increment` | Permitted stake step |
| `value_per_price_unit` | Account-currency P&L for size 1 and a 1.0 price move |
| `margin_rate` | Initial margin divided by gross economic exposure |
| `commission_per_size_per_side` | Explicit per-side commission, normally zero for a spread bet |

These are account- and epic-specific facts. Missing values produce
`INCOMPLETE`; the engine never guesses them.

### 2.4 Quantitative market snapshot

| Field | Meaning |
|---|---|
| `epic` | Must match the broker-instrument epic exactly |
| `as_of` | Quote timestamp |
| `bid`, `ask` | Current executable quote |
| `market_open` | Whether the broker says the market is tradeable |
| `median_spread_points` | Causal recent same-session median, when measured |
| `p95_spread_points` | Causal recent same-session 95th percentile, when measured |
| `forecast_range_points` | Optional forecast move over the supplied horizon |
| `forecast_horizon_minutes` | Horizon matching the forecast range |
| `volatility_percentile` | Optional causal percentile in `[0, 1]` |
| `volatility_artifact_approved` | Whether that forecast passed its release controls |
| `scheduled_event_blackout_active` | Mechanical version-1 event-window flag |

Forecast fields can enrich and conservatively reduce the assessment. They may
never increase size. Until a forecast artifact is approved for this use, it is
reported as unavailable and ignored by sizing.

### 2.5 Existing positions

Each position supplies instrument, side, size, current price, stop price,
account-currency value per price unit, and current margin requirement. This is
enough to calculate existing stop risk, gross and signed one-percent exposure,
and the proposed trade's incremental portfolio impact. A position without a
known stop is treated as incomplete/unbounded rather than assigned a fictional
loss cap.

## 3. Frozen conservative policy

Version 1 reuses the existing project mandate:

- maximum planned stop risk: `0.25%` of current equity;
- daily loss gate: `0.75%` from day-start equity;
- weekly loss gate: `1.50%` from week-start equity;
- maximum three entries per London day;
- maximum one open precious-metals position portfolio-wide;
- maximum gross economic exposure: `100%` of equity;
- no routine overnight holding;
- eligible entry time: `[07:00, 19:30)` Europe/London;
- forced-flat cutoff: `21:30` Europe/London;
- current/recent median cost no more than `0.10R` and p95 stressed cost no more
  than `0.25R` when those measurements exist; and
- maximum total planned stop risk: `0.50%` of equity as a dormant portfolio
  backstop. The one-position limit binds first in version 1.

Margin is additionally capped at the lower of broker-available funds and `30%`
of equity. This is a conservative operational limit, not a claim that broker
liquidation occurs at 30%.

An approved volatility forecast applies only a downside ladder: below p80 uses
the normal risk cap, p80–p95 halves it, and p95 or above blocks a new entry.

## 4. Calculations and scenarios

For size `q`, value per price unit `v`, spread `s`, slippage allowance `l`, and
per-side commission `c`:

- base round-trip cost = `q * (v * (s + 2l) + 2c)`;
- planned stop loss = `q * (v * abs(entry - stop) + base cost per size)`;
- net target reward = `q * (v * abs(target - entry) - base cost per size)`;
- gross exposure proxy = `q * entry * v`;
- initial margin = `gross exposure * margin_rate`.

The engine reports planned-stop loss plus hypothetical 1.5x and 2x spread
stresses. If an approved horizon-matched forecast range is supplied, it also
reports adverse one-, two- and three-range moves and a gap-through-stop stress
of half one forecast range beyond the stop. These scenarios have no assigned
probability and are not value-at-risk.

## 5. Decisions

| Decision | Meaning |
|---|---|
| `ALLOW` | Complete inputs and the proposed/recommended size fits every hard gate |
| `REDUCE` | The thesis is assessable, but the requested size must be reduced to the reported cap |
| `BLOCK` | A non-size policy rule prevents entry, or no broker-permitted safe size exists |
| `INCOMPLETE` | A required account, broker, quote, event or position fact is missing |

Every decision includes stable reason codes, warnings, the exact policy
version and a JSON-serializable calculation record.

## 6. Delivery order

1. Offline schemas, deterministic calculations, scenarios and JSON report.
2. JSON CLI and fail-closed examples/tests.
3. Historical calibration of range scenarios and spread regimes without
   assigning invented probabilities.
4. Read-only IG snapshot adapter for account, epic, quote and margin facts.
5. Gold/silver portfolio aggregation and report UI.
6. Prospective paper logging: predicted risk versus realised excursion,
   slippage, cost and loss.
7. Only after operational validation, connect `ALLOW` decisions to the existing
   dry-run paper-intent path. Order submission remains a separate authority.
