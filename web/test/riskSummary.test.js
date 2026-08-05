import test from "node:test";
import assert from "node:assert/strict";

import { buildRiskSummary, formatMoney, translateNotice } from "../src/riskSummary.js";

test("summarizes an allow decision without calling the trade safe", () => {
  const summary = buildRiskSummary({
    decision: "ALLOW",
    warnings: [
      "volatility_forecast_not_approved_for_sizing",
      "historical_range_calibration_unavailable_or_stale",
    ],
    sizing: {
      assessed_size: 2,
      recommended_max_size: 20.3,
      effective_risk_budget: 250,
    },
    cash_risk: {
      account_currency: "GBP",
      proposed: {
        planned_stop_loss: 24.58,
        planned_stop_loss_fraction: 0.0002458,
      },
    },
    costs: { current_cost_r: 0.0553, median_cost_r: 0.05 },
    market_risk: { historical_calibration: null },
    portfolio: {
      existing_position_count: 0,
      proposed_total_gross_exposure: 6208.74,
      proposed_total_margin: 310.44,
    },
  });

  assert.equal(summary.decisionTitle, "Within configured limits");
  assert.equal(summary.evidenceLevel, "LIMITED");
  assert.ok(Math.abs(summary.accountEquity - 100000) < 1e-9);
  assert.ok(Math.abs(summary.budgetUsage - 0.09832) < 1e-12);
  assert.match(summary.decisionMeaning, /does not mean the trade is safe/i);
});

test("translates internal codes and formats account currency", () => {
  assert.match(translateNotice("broker_reports_hedging_disabled"), /hedging disabled/i);
  assert.match(translateNotice("missing:trade_intent.stop_price"), /stop price/i);
  assert.match(formatMoney(24.58, "GBP"), /24\.58/);
});
