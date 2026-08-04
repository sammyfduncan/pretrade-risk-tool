# Matched volatility-risk economic experiment

**Framework frozen:** 2026-07-31 before applying the volatility policy to any
retained strategy outcome  
**Registered strategy:** none — no current gold primary strategy passed  
**Forecast:** joint B0 plus GVZ plus calendar, development gate passed  
**Status:** implementation may be tested synthetically; real evaluation is
fail-closed until a future primary strategy is named here before its risk-policy
outcome is read

## 1. Question

For one independently retained M30 gold strategy, does the already-frozen
`1.00x / 0.50x / 0.00x` volatility-risk ladder reduce drawdown without
destroying the strategy's positive expectancy?

The constant-risk control and volatility-risk arm use identical candidates,
prices, costs, stop/target outcomes, and chronological range. The only change
is the risk multiplier known at the candidate decision time.

## 2. Preconditions

The real-data command must refuse to load trade outcomes unless all conditions
hold:

1. this document names exactly one strategy rather than `none`;
2. that strategy's frozen primary-gate report exists and says `passed=true`;
3. the report and candidate ledger end before `2025-07-01`;
4. the joint forecast report says `passed=true`, selection is
   `b0_iv_calendar`, and `paper_enabled=false`;
5. the candidate clock is M30 and its decision time is the signal-bar open plus
   30 minutes; and
6. at least 500 strictly earlier OOF forecasts exist before a policy decision.

P1, P2, P3, their V2 variants, B1, and the registered ML filters are ineligible
because they failed. Changing a failed report or naming one here would not make
it retained.

## 3. Matched construction

Use the joint model's development OOF forecast series only. At each candidate
decision time, calculate the 80th and 95th percentiles from all finite OOF
forecasts with timestamps strictly earlier than that decision, using NumPy's
`higher` method. The first candidates without 500 prior forecasts are a fixed
warm-up exclusion from both arms.

After that warm-up, at least 90% of otherwise eligible candidates must match an
OOF forecast at the exact decision timestamp. Lower coverage is a data-contract
failure, not permission to evaluate the convenient subset.

At the configured one-basis-point-per-side base cost, convert each candidate's
observed gross return to net R using its pre-trade `initial_r_pct`:

`net_R = (gross_return - 2 * cost_per_side) / initial_r_pct`.

- Constant arm return contribution: `0.0025 * net_R`.
- Policy arm return contribution: `0.0025 * multiplier * net_R`.

A `0.00x` decision remains in the matched ledger as a blocked opportunity with
zero return. This prevents the arms from silently using different samples.
The research calculation ignores broker size increments and notional limits;
those are operational constraints tested later with real OANDA metadata.

## 4. Reported metrics and frozen gate

Report matched opportunities, multiplier counts, compounded return, profit
factor, mean contribution, annualized trade-sequence Sharpe, maximum drawdown,
worst contribution, and the same metrics in five equal chronological blocks.
Also report the constant and policy arms at double historical cost.

The policy passes economic development only if every condition holds:

1. forecast-match coverage is at least 90%, with at least 200 matched
   opportunities and at least 30 in every block;
2. policy compounded return is positive and at least 75% of constant return;
3. policy maximum drawdown magnitude is at least 10% smaller than constant;
4. policy trade-sequence Sharpe is strictly above constant;
5. policy profit factor is strictly above constant;
6. policy compounded return is positive in at least four of five blocks; and
7. policy compounded return remains positive at double cost.

These thresholds may not be changed after a strategy is named. A pass permits
the policy to enter operational shadow paper trading with
`paper_enabled=false` still retained in the research artifact; it does not
open confirmation or authorize live orders. A fail keeps constant risk and
retires this mapping for that strategy version.

## 5. Activation boundary and look ledger

The implementation command is intentionally blocked while the registered
strategy is `none`. Naming a future strategy requires updating this document
and the harness constant in one committed preregistration before running it.

| Look | Date | Item | Outcome available? |
|---:|---|---|---|
| 0 | 2026-07-31 | Generic matched framework frozen; all existing primary strategies confirmed ineligible | no risk-policy P&L read |
