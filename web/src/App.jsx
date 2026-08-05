import { useState } from "react";
import { buildRiskSummary, formatMoney } from "./riskSummary.js";

const initialForm = {
  instrument: "XAUUSD",
  side: "long",
  order_type: "market",
  entry_price: "",
  stop_price: "",
  target_price: "",
  requested_size: "",
  holding_period_minutes: "120",
  max_slippage_points_per_side: "",
  allow_overnight: false,
};

const optionalNumber = (value) => (value === "" ? null : Number(value));
const show = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
};
const percent = (value) =>
  value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(2)}%`;

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const update = (event) => {
    const { name, value, type, checked } = event.target;
    setForm((current) => ({ ...current, [name]: type === "checkbox" ? checked : value }));
  };

  const evaluate = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    const trade_intent = {
      ...form,
      holding_period_minutes: Number(form.holding_period_minutes),
      entry_price: optionalNumber(form.entry_price),
      stop_price: optionalNumber(form.stop_price),
      target_price: optionalNumber(form.target_price),
      requested_size: optionalNumber(form.requested_size),
      max_slippage_points_per_side: optionalNumber(form.max_slippage_points_per_side),
    };
    try {
      const response = await fetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trade_intent }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Assessment failed");
      setResult(data);
    } catch (reason) {
      setError(reason.message || "Assessment failed");
    } finally {
      setLoading(false);
    }
  };

  const downloadJson = () => downloadBlob(
    JSON.stringify(result, null, 2),
    "commodity-risk-assessment.json",
    "application/json",
  );

  const downloadHtml = async () => {
    setError("");
    try {
      const response = await fetch("/api/report-html", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      });
      if (!response.ok) throw new Error("Could not create the HTML report");
      downloadBlob(await response.text(), "commodity-risk-assessment.html", "text/html");
    } catch (reason) {
      setError(reason.message || "Could not create the HTML report");
    }
  };

  return (
    <main>
      <header>
        <p className="kicker">Read-only decision support</p>
        <h1>Commodity risk assessment</h1>
        <p>Describe a possible gold or silver trade. The tool checks it against your OANDA demo account, current market conditions and the quantitative risk policy.</p>
        <p className="safe">This app cannot place, change or close orders.</p>
      </header>

      <form onSubmit={evaluate}>
        <h2>Trade idea</h2>
        <div className="fields">
          <Field label="Instrument">
            <select name="instrument" value={form.instrument} onChange={update}>
              <option value="XAUUSD">Gold (XAU/USD)</option>
              <option value="XAGUSD">Silver (XAG/USD)</option>
            </select>
          </Field>
          <Field label="Direction">
            <select name="side" value={form.side} onChange={update}>
              <option value="long">Long</option>
              <option value="short">Short</option>
            </select>
          </Field>
          <Field label="Order type">
            <select name="order_type" value={form.order_type} onChange={update}>
              <option value="market">Market</option>
              <option value="limit">Limit</option>
              <option value="stop">Stop</option>
            </select>
          </Field>
          <Field label="Planned holding time">
            <select name="holding_period_minutes" value={form.holding_period_minutes} onChange={update}>
              <option value="60">1 hour</option>
              <option value="120">2 hours</option>
              <option value="180">3 hours</option>
              <option value="240">4 hours</option>
            </select>
          </Field>
          {form.order_type !== "market" && <NumberField label="Entry price" name="entry_price" value={form.entry_price} onChange={update} required />}
          <NumberField label="Stop price" name="stop_price" value={form.stop_price} onChange={update} required />
          <NumberField label="Target price" name="target_price" value={form.target_price} onChange={update} required />
          <NumberField label="Requested size (optional)" name="requested_size" value={form.requested_size} onChange={update} />
          <NumberField
            label="Maximum slippage per side"
            help="Enter quoted-price units, not ticks. Around a gold price of 4,185, 0.21 is approximately 0.5 basis points per side."
            name="max_slippage_points_per_side"
            value={form.max_slippage_points_per_side}
            onChange={update}
            required
          />
        </div>
        <label className="check">
          <input type="checkbox" name="allow_overnight" checked={form.allow_overnight} onChange={update} />
          Allow the intended trade to remain open overnight
        </label>
        <button type="submit" disabled={loading}>{loading ? "Assessing…" : "Assess trade risk"}</button>
      </form>

      {error && <section className="error" role="alert"><strong>Assessment could not be completed.</strong><p>{error}</p></section>}
      {result && <Assessment document={result} onJson={downloadJson} onHtml={downloadHtml} />}
    </main>
  );
}

function Assessment({ document, onJson, onHtml }) {
  const report = document.risk_report || {};
  const sizing = report.sizing || {};
  const cash = report.cash_risk?.proposed || {};
  const costs = report.costs || {};
  const market = report.market_risk || {};
  const portfolio = report.portfolio || {};
  const scenarios = Array.isArray(report.scenarios) ? report.scenarios : [];
  const limitations = Array.isArray(report.limitations) ? report.limitations : [];
  const summary = buildRiskSummary(report, document.adapter_issues);
  const notices = [
    ...summary.translatedReasons.map((text) => ["Decision reason", text]),
    ...summary.translatedWarnings.map((text) => ["Evidence gap", text]),
    ...summary.translatedAdapterIssues.map((text) => ["Broker note", text]),
  ];
  return (
    <section className="assessment">
      <div className="result-head">
        <div>
          <p className="kicker">Assessment result</p>
          <h2 className="result-title">{summary.decisionTitle}</h2>
          <span className={`decision ${summary.decision.toLowerCase()}`}>{summary.decision}</span>
        </div>
        <div className="actions"><button type="button" className="secondary" onClick={onJson}>Download JSON</button><button type="button" className="secondary" onClick={onHtml}>Download HTML</button></div>
      </div>
      <PlainLanguageSummary summary={summary} />
      {notices.length > 0 && <div className="notices">{notices.map(([label, text], index) => <p key={`${label}-${index}`}><strong>{label}:</strong> {text}</p>)}</div>}
      <div className="cards">
        <Card title="Sizing" rows={[
          ["Requested", show(sizing.requested_size, 4)],
          ["Assessed", show(sizing.assessed_size, 4)],
          ["Recommended maximum", show(sizing.recommended_max_size, 4)],
          ["Binding constraint", sizing.binding_constraint || "—"],
        ]} />
        <Card title="Trade risk" rows={[
          ["Planned stop loss", formatMoney(cash.planned_stop_loss, summary.currency)],
          ["Stop loss / equity", percent(cash.planned_stop_loss_fraction)],
          ["Net target reward", formatMoney(cash.net_target_reward, summary.currency)],
          ["Net reward / risk", show(cash.net_reward_risk)],
        ]} />
        <Card title="Market and cost" rows={[
          ["Current spread", show(costs.current_spread_points, 4)],
          ["Current cost / R", show(costs.current_cost_r, 4)],
          ["Quote age (seconds)", show(market.quote_age_seconds, 1)],
          ["Volatility state", market.volatility_state || "—"],
        ]} />
        <Card title="Portfolio after trade" rows={[
          ["OANDA positions detected", show(portfolio.existing_position_count, 0)],
          ["Total stop risk", formatMoney(portfolio.proposed_total_stop_risk, summary.currency)],
          ["Total margin", formatMoney(portfolio.proposed_total_margin, summary.currency)],
          ["Gross exposure", formatMoney(portfolio.proposed_total_gross_exposure, summary.currency)],
        ]} />
      </div>
      {scenarios.length > 0 && <div className="table-wrap">
        <h3>Stress scenarios</h3>
        <table>
          <thead><tr><th>Scenario</th><th>Cash loss</th><th>Equity loss</th></tr></thead>
          <tbody>{scenarios.map((scenario, index) => <tr key={`${scenario.name}-${index}`}><td>{scenarioName(scenario.name)}</td><td>{formatMoney(scenario.cash_loss, summary.currency)}</td><td>{percent(scenario.equity_fraction)}</td></tr>)}</tbody>
        </table>
      </div>}
      <div className="calibration"><strong>Current calibrated range evidence:</strong> {market.historical_calibration ? "included" : "unavailable or not admitted"}</div>
      {limitations.length > 0 && <details><summary>Assessment limitations</summary><ul>{limitations.map((text, index) => <li key={index}>{text}</li>)}</ul></details>}
      <p className="footnote">Quantitative risk assessment only. It does not predict direction or profitability, and an ALLOW result is not a recommendation to trade.</p>
    </section>
  );
}

function PlainLanguageSummary({ summary }) {
  return (
    <section className="plain-summary" aria-labelledby="plain-summary-title">
      <div className="summary-heading">
        <h3 id="plain-summary-title">What this means</h3>
        <span className={`evidence evidence-${summary.evidenceLevel.toLowerCase()}`}>Evidence: {summary.evidenceLevel}</span>
      </div>
      <p className="summary-lead">{summary.decisionMeaning}</p>
      <div className="summary-metrics">
        <SummaryMetric
          label="Estimated loss if stop executes"
          value={formatMoney(summary.plannedLoss, summary.currency)}
          detail={`${percent(summary.lossFraction)} of assessed account equity`}
        />
        <SummaryMetric
          label="Per-trade risk allowance used"
          value={percent(summary.budgetUsage)}
          detail={`${formatMoney(summary.plannedLoss, summary.currency)} of ${formatMoney(summary.riskBudget, summary.currency)}`}
        />
        <SummaryMetric
          label="Account equity assessed"
          value={formatMoney(summary.accountEquity, summary.currency)}
          detail="Taken from the connected OANDA account"
        />
      </div>
      <ul className="summary-points">
        <li>
          The assessed size is <strong>{show(summary.assessedSize, 4)}</strong>; the maximum under the current policy is <strong>{show(summary.recommendedMaxSize, 4)}</strong>.
        </li>
        <li>
          Current execution cost is approximately <strong>{percent(summary.currentCostR)}</strong> of one unit of planned risk{summary.medianCostR !== null ? <>; the same-session median is <strong>{percent(summary.medianCostR)}</strong></> : null}.
        </li>
        <li>
          Gross exposure is <strong>{formatMoney(summary.grossExposure, summary.currency)}</strong> ({percent(summary.grossExposureFraction)} of assessed equity), with <strong>{formatMoney(summary.margin, summary.currency)}</strong> margin ({percent(summary.marginFraction)}).
        </li>
        {summary.openPositions === 0 && <li>No existing OANDA positions were included in this assessment.</li>}
      </ul>
      {summary.limitations.length > 0 && <div className="summary-limits">
        <strong>What this assessment could not determine</strong>
        <ul>{summary.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
        <p>It therefore cannot estimate how likely the stop is to be reached or describe the trade as generally low risk.</p>
      </div>}
    </section>
  );
}

function SummaryMetric({ label, value, detail }) {
  return <div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}

function Field({ label, help, children }) {
  return <label><span>{label}</span>{children}{help && <small className="field-help">{help}</small>}</label>;
}

function NumberField({ label, help, ...props }) {
  return <Field label={label} help={help}><input type="number" step="any" min="0" {...props} /></Field>;
}

function Card({ title, rows }) {
  return <section className="card"><h3>{title}</h3><dl>{rows.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl></section>;
}

function downloadBlob(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function scenarioName(value) {
  const names = {
    planned_stop: "Planned stop",
    planned_stop_spread_1_5x: "Planned stop with spread at 1.5× current",
    planned_stop_spread_2x: "Planned stop with spread at 2× current",
  };
  return names[value] || String(value || "Unnamed").replaceAll("_", " ");
}
