const DECISION_TITLES = {
  ALLOW: "Within configured limits",
  REDUCE: "Reduce the proposed size",
  BLOCK: "Outside configured limits",
  INCOMPLETE: "Assessment incomplete",
};

const NOTICE_TEXT = {
  volatility_forecast_not_approved_for_sizing:
    "Current volatility was not approved for position sizing, so no volatility-based size reduction was applied.",
  historical_range_calibration_unavailable_or_stale:
    "Current calibrated range and adverse-movement evidence is unavailable, so the tool cannot judge how the stop compares with recent typical movement.",
  account_type_confirmed_in_config_not_exposed_by_oanda_v20:
    "OANDA does not expose the account product through this API; the configured spread-bet account type could not be independently confirmed.",
  broker_reports_hedging_disabled:
    "OANDA reports hedging disabled for this account. The current one-position policy does not require hedging.",
  "median_cost_above_0.10R":
    "Typical same-session execution cost is above 10% of the planned trade risk.",
  "current_cost_above_0.25R":
    "Current execution cost is above 25% of the planned trade risk.",
  "p95_cost_above_0.25R":
    "Stressed historical execution cost is above 25% of the planned trade risk.",
  quote_timestamp_after_decision:
    "The market quote arrived after the recorded decision time, so the timing evidence is invalid.",
  stale_quote: "The quote is too old for a new assessment.",
  scheduled_event_blackout: "The trade falls inside a configured scheduled-event blackout.",
  market_not_open: "OANDA reports that the market is not currently tradeable.",
  daily_loss_limit_reached: "The configured daily loss limit has been reached.",
  weekly_loss_limit_reached: "The configured weekly loss limit has been reached.",
  maximum_entries_reached: "The configured daily entry limit has been reached.",
  outside_entry_session: "The proposed entry is outside the configured trading session.",
};

export function buildRiskSummary(report = {}, adapterIssues = []) {
  const decision = String(report.decision || "INCOMPLETE").toUpperCase();
  const cash = report.cash_risk?.proposed || {};
  const sizing = report.sizing || {};
  const costs = report.costs || {};
  const market = report.market_risk || {};
  const portfolio = report.portfolio || {};
  const warnings = Array.isArray(report.warnings) ? report.warnings : [];
  const reasons = Array.isArray(report.reasons) ? report.reasons : [];
  const limitations = [];

  if (warnings.includes("volatility_forecast_not_approved_for_sizing")) {
    limitations.push("No approved current-volatility adjustment was applied to size.");
  }
  if (warnings.includes("historical_range_calibration_unavailable_or_stale")) {
    limitations.push("No current calibrated range or adverse-movement comparison was available.");
  }

  const plannedLoss = finite(cash.planned_stop_loss);
  const lossFraction = finite(cash.planned_stop_loss_fraction);
  const riskBudget = finite(sizing.effective_risk_budget);
  const accountEquity =
    plannedLoss !== null && lossFraction !== null && lossFraction > 0
      ? plannedLoss / lossFraction
      : null;
  const budgetUsage =
    plannedLoss !== null && riskBudget !== null && riskBudget > 0
      ? plannedLoss / riskBudget
      : null;
  const grossExposure = finite(portfolio.proposed_total_gross_exposure);
  const margin = finite(portfolio.proposed_total_margin);
  const grossExposureFraction =
    grossExposure !== null && accountEquity ? grossExposure / accountEquity : null;
  const marginFraction = margin !== null && accountEquity ? margin / accountEquity : null;
  const sizePressure = ratio(finite(sizing.assessed_size), finite(sizing.recommended_max_size));
  const pressureComponents = [
    ["Risk allowance", budgetUsage],
    ["Size cap", sizePressure],
    ["Current cost", ratio(finite(costs.current_cost_r), 0.25)],
    ["Typical cost", ratio(finite(costs.median_cost_r), 0.10)],
    ["Stressed cost", ratio(finite(costs.p95_cost_r), 0.25)],
    ["Exposure", grossExposureFraction],
    ["Margin", marginFraction === null ? null : marginFraction / 0.30],
  ].filter((item) => item[1] !== null);
  const dominantPressure = pressureComponents.length
    ? pressureComponents.reduce((highest, item) => (item[1] > highest[1] ? item : highest))
    : null;
  const riskScore = scorePressure(decision, dominantPressure?.[1] ?? null);

  return {
    decision,
    decisionTitle: DECISION_TITLES[decision] || "Assessment result",
    decisionMeaning: decisionMeaning(decision),
    evidenceLevel:
      limitations.length >= 2 ? "LIMITED" : limitations.length === 1 ? "PARTIAL" : "COMPLETE",
    evidenceScore: decision === "INCOMPLETE" ? 0 : Math.max(1, 3 - limitations.length),
    riskScore,
    riskDriver: dominantPressure?.[0] || "Unavailable",
    pressureRatio: dominantPressure?.[1] ?? null,
    currency: report.cash_risk?.account_currency || "",
    assessedSize: finite(sizing.assessed_size),
    requestedSize: finite(sizing.requested_size),
    recommendedMaxSize: finite(sizing.recommended_max_size),
    plannedLoss,
    lossFraction,
    accountEquity,
    riskBudget,
    budgetUsage,
    currentCostR: finite(costs.current_cost_r),
    medianCostR: finite(costs.median_cost_r),
    grossExposure,
    grossExposureFraction,
    margin,
    marginFraction,
    openPositions: finite(portfolio.existing_position_count),
    limitations,
    translatedReasons: reasons.map(translateNotice),
    translatedWarnings: warnings.map(translateNotice),
    translatedAdapterIssues: (Array.isArray(adapterIssues) ? adapterIssues : []).map(translateNotice),
    calibratedMovementIncluded: Boolean(market.historical_calibration),
  };
}

function scorePressure(decision, pressure) {
  if (decision === "INCOMPLETE" || pressure === null) return null;
  if (decision === "BLOCK" || pressure > 1) return 7;
  if (pressure <= 0.15) return 1;
  if (pressure <= 0.30) return 2;
  if (pressure <= 0.45) return 3;
  if (pressure <= 0.60) return 4;
  if (pressure <= 0.75) return 5;
  return 6;
}

export function translateNotice(code) {
  const text = String(code || "");
  if (NOTICE_TEXT[text]) return NOTICE_TEXT[text];
  if (text.startsWith("missing:")) {
    return `Required information is missing: ${humanize(text.slice("missing:".length))}.`;
  }
  return `${humanize(text)}.`;
}

export function formatMoney(value, currency) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "—";
  try {
    return new Intl.NumberFormat(undefined, {
      style: "currency",
      currency: currency || "GBP",
      maximumFractionDigits: 2,
    }).format(Number(value));
  } catch {
    return `${currency || ""} ${Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`.trim();
  }
}

function decisionMeaning(decision) {
  if (decision === "ALLOW") {
    return "The supplied size fits the account, cost and policy limits that could be evaluated. This does not mean the trade is safe or likely to profit.";
  }
  if (decision === "REDUCE") {
    return "The trade can be evaluated, but the supplied size exceeds the framework's safe broker-valid maximum.";
  }
  if (decision === "BLOCK") {
    return "At least one non-negotiable account, market, cost or policy rule prevents this trade under the supplied assumptions.";
  }
  return "One or more required facts are missing, so the framework refuses to give an allow or reduce decision.";
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function ratio(numerator, denominator) {
  if (numerator === null || denominator === null || denominator <= 0) return null;
  return numerator / denominator;
}

function humanize(value) {
  return String(value)
    .replaceAll(".", " ")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
}
