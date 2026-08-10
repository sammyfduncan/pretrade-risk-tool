const DECISION_TITLES = {
  ALLOW: "Within limits",
  REDUCE: "Reduce the proposed size",
  BLOCK: "Outside limits",
  INCOMPLETE: "Assessment incomplete",
};

const NOTICE_TEXT = {
  volatility_forecast_not_approved_for_sizing:
    "Current volatility not approved for position size, so no volatility based size reduction applied",
  historical_range_calibration_unavailable_or_stale:
    "Current calibrated range and negative movement evidence not found, no comparison",
  account_type_confirmed_in_config_not_exposed_by_oanda_v20:
    "OANDA doesn't expose account type in this API",
  broker_reports_hedging_disabled:
    "OANDA reports hedging disabled for this account, current policy doesn't require hedging",
  manual_account_facts_not_broker_verified:
    "External account values not verified through broker API",
  external_margin_rate_user_supplied:
    "external broker margin requirement entered manually",
  external_existing_positions_limited_to_explicit_none_confirmation:
    "Portfolio exposure relies on confirmation that no other positions open",
  "median_cost_above_0.10R":
    "same session execution cost above 10% of the planned trade risk",
  "current_cost_above_0.25R":
    "Current execution cost above 25% of the planned trade risk",
  "p95_cost_above_0.25R":
    "Stressed historical execution cost is above 25% of the planned trade risk",
  quote_timestamp_after_decision:
    "Market quote arrived after the recorded decision time, timing evidence invalid",
  stale_quote: "Quote is too old for a new assessment",
  scheduled_event_blackout: "Trade falls inside a configured scheduled event blackout",
  market_not_open: "OANDA reports market closed",
  daily_loss_limit_reached: "Configured daily loss limit reached",
  weekly_loss_limit_reached: "Configured weekly loss limit reached",
  maximum_entries_reached: "TConfigured daily entry limit reached",
  outside_entry_session: "Proposed entry outside trading session",
};

export function buildRiskSummary(report = {}, adapterIssues = []) {
  const decision = String(report.decision || "INCOMPLETE").toUpperCase();
  const cash = report.cash_risk?.proposed || {};
  const sizing = report.sizing || {};
  const costs = report.costs || {};
  const market = report.market_risk || {};
  const portfolio = report.portfolio || {};
  const assessment = report.assessment || {};
  const warnings = Array.isArray(report.warnings) ? report.warnings : [];
  const reasons = Array.isArray(report.reasons) ? report.reasons : [];
  const limitations = [];

  if (warnings.includes("volatility_forecast_not_approved_for_sizing")) {
    limitations.push("No volatility adjustment applied to size.");
  }
  if (warnings.includes("historical_range_calibration_unavailable_or_stale")) {
    limitations.push("No calibrated range or negative movement comparison available.");
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
  const riskScore = finite(assessment.policy_pressure_score);
  const evidenceScore = finite(assessment.evidence_score);

  return {
    decision,
    decisionTitle: DECISION_TITLES[decision] || "Assessment result",
    decisionMeaning: decisionMeaning(decision),
    evidenceLevel:
      limitations.length >= 2 ? "LIMITED" : limitations.length === 1 ? "PARTIAL" : "COMPLETE",
    evidenceScore: evidenceScore ?? 0,
    riskScore,
    riskDriver: humanize(assessment.policy_pressure_driver || "Unavailable"),
    pressureRatio: finite(assessment.maximum_pressure_ratio),
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

export function translateNotice(code) {
  const text = String(code || "");
  if (NOTICE_TEXT[text]) return NOTICE_TEXT[text];
  if (text.startsWith("missing:")) {
    return `Required info missing: ${humanize(text.slice("missing:".length))}.`;
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
    return "Supplied size fits account, cost and policy limits ";
  }
  if (decision === "REDUCE") {
    return "Trade evaluated, but size exceeds the framework's safe validated maximum ";
  }
  if (decision === "BLOCK") {
    return "Account, market, cost or policy rule preventing trade under supplied params ";
  }
  return "One or more required facts are missing, so the framework refuses to give an allow or reduce decision ";
}

function finite(value) {
  if (value === null || value === undefined || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function humanize(value) {
  return String(value)
    .replaceAll(".", " ")
    .replaceAll("_", " ")
    .replace(/\s+/g, " ")
    .trim();
}
