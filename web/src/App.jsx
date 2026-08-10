import { useEffect, useState } from "react";
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

const initialManualAccount = {
  equity: "",
  available_funds: "",
  day_start_equity: "",
  week_start_equity: "",
  entries_today: "0",
  margin_rate_percent: "",
  commission_per_stake_per_side: "0",
  kill_switch_active: false,
  no_other_metals_positions_confirmed: false,
};

const optionalNumber = (value) => (value === "" ? null : Number(value));
const show = (value, digits = 2) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
};
const percent = (value) =>
  value === null || value === undefined ? "—" : `${(Number(value) * 100).toFixed(2)}%`;

export default function App() {
  const [mode, setMode] = useState("oanda");
  const [form, setForm] = useState(initialForm);
  const [manualAccount, setManualAccount] = useState(initialManualAccount);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [marketDefaults, setMarketDefaults] = useState(null);
  const [defaultsError, setDefaultsError] = useState("");
  const [defaultsLoading, setDefaultsLoading] = useState(true);
  const [defaultsRefresh, setDefaultsRefresh] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    const loadDefaults = async () => {
      setDefaultsLoading(true);
      setDefaultsError("");
      setMarketDefaults(null);
      try {
        const query = new URLSearchParams({
          instrument: form.instrument,
          side: form.side,
        });
        const response = await fetch(`/api/market-defaults?${query}`, {
          signal: controller.signal,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Current price unavailable");
        setMarketDefaults(data);
        setForm((current) => ({
          ...current,
          entry_price: String(data.suggested_entry_price),
          stop_price: String(data.suggested_stop_price),
          target_price: String(data.suggested_target_price),
          max_slippage_points_per_side: String(
            data.suggested_slippage_points_per_side,
          ),
        }));
      } catch (reason) {
        if (reason.name !== "AbortError") {
          setDefaultsError(reason.message || "Current price unavailable");
        }
      } finally {
        if (!controller.signal.aborted) setDefaultsLoading(false);
      }
    };
    loadDefaults();
    return () => controller.abort();
  }, [form.instrument, form.side, defaultsRefresh]);

  const update = (event) => {
    const { name, value, type, checked } = event.target;
    setForm((current) => ({ ...current, [name]: type === "checkbox" ? checked : value }));
  };

  const updateManual = (event) => {
    const { name, value, type, checked } = event.target;
    setManualAccount((current) => ({
      ...current,
      [name]: type === "checkbox" ? checked : value,
    }));
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
        body: JSON.stringify({
          mode,
          trade_intent,
          ...(mode === "manual" ? {
            manual_account: {
              equity: optionalNumber(manualAccount.equity),
              available_funds: optionalNumber(manualAccount.available_funds),
              day_start_equity: optionalNumber(manualAccount.day_start_equity),
              week_start_equity: optionalNumber(manualAccount.week_start_equity),
              entries_today: Number(manualAccount.entries_today),
              margin_rate: Number(manualAccount.margin_rate_percent) / 100,
              commission_per_stake_per_side: optionalNumber(
                manualAccount.commission_per_stake_per_side,
              ),
              kill_switch_active: manualAccount.kill_switch_active,
              no_other_metals_positions_confirmed:
                manualAccount.no_other_metals_positions_confirmed,
            },
          } : {}),
        }),
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
        <p>Describe a possible gold or silver trade. The tool checks it against the selected account facts, current OANDA market data and the quantitative risk policy.</p>
        <p className="safe">This app cannot place, change or close orders.</p>
      </header>

      <form onSubmit={evaluate}>
        <h2>Trade idea</h2>
        <div className="market-defaults" aria-live="polite">
          <div>
            <span>Current midpoint</span>
            <strong>{defaultsLoading ? "Loading…" : marketDefaults ? show(marketDefaults.mid, 4) : "—"}</strong>
            {marketDefaults && <small>Bid {show(marketDefaults.bid, 4)} · Ask {show(marketDefaults.ask, 4)}</small>}
            {defaultsError && <small className="defaults-error">{defaultsError}</small>}
          </div>
          <button
            type="button"
            className="secondary compact-button"
            onClick={() => setDefaultsRefresh((value) => value + 1)}
            disabled={defaultsLoading}
          >
            Refresh defaults
          </button>
        </div>
        <div className="fields">
          <Field label="Account source">
            <select value={mode} onChange={(event) => setMode(event.target.value)}>
              <option value="oanda">OANDA demo account</option>
              <option value="manual">External GBP spread-bet account</option>
            </select>
          </Field>
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
          {form.order_type !== "market" && <NumberField label="Entry price" help="Current midpoint supplied as an editable starting value." name="entry_price" value={form.entry_price} onChange={update} required />}
          <NumberField label="Stop price" help="Editable example: 0.5% from the current midpoint." name="stop_price" value={form.stop_price} onChange={update} required />
          <NumberField label="Target price" help="Editable example: 1.0% from the current midpoint (2:1 before costs)." name="target_price" value={form.target_price} onChange={update} required />
          <NumberField
            label={mode === "manual" ? "Stake (£ per price unit)" : "Requested OANDA units (optional)"}
            name="requested_size"
            value={form.requested_size}
            onChange={update}
            placeholder={mode === "manual" ? "e.g. 1" : "e.g. 10"}
            required={mode === "manual"}
          />
          <NumberField
            label="Maximum slippage per side"
            help="Editable example: 0.5 basis points of the current midpoint per side."
            name="max_slippage_points_per_side"
            value={form.max_slippage_points_per_side}
            onChange={update}
            required
          />
        </div>
        {mode === "manual" && <>
          <h2>External account</h2>
          <div className="fields">
            <NumberField label="Account equity (£)" name="equity" value={manualAccount.equity} onChange={updateManual} placeholder="e.g. 10000" required />
            <NumberField label="Available funds (£)" name="available_funds" value={manualAccount.available_funds} onChange={updateManual} placeholder="e.g. 9000" required />
            <NumberField label="Equity at start of day (£)" name="day_start_equity" value={manualAccount.day_start_equity} onChange={updateManual} placeholder="e.g. 10000" required />
            <NumberField label="Equity at start of week (£)" name="week_start_equity" value={manualAccount.week_start_equity} onChange={updateManual} placeholder="e.g. 10000" required />
            <NumberField label="Entries today" name="entries_today" value={manualAccount.entries_today} onChange={updateManual} placeholder="0" required />
            <NumberField label="Margin requirement (%)" name="margin_rate_percent" value={manualAccount.margin_rate_percent} onChange={updateManual} placeholder="e.g. 5" required />
            <NumberField label="Commission per £ stake, per side" name="commission_per_stake_per_side" value={manualAccount.commission_per_stake_per_side} onChange={updateManual} placeholder="0" required />
          </div>
          <label className="check">
            <input type="checkbox" name="kill_switch_active" checked={manualAccount.kill_switch_active} onChange={updateManual} />
            Account kill switch is active
          </label>
          <label className="check">
            <input type="checkbox" name="no_other_metals_positions_confirmed" checked={manualAccount.no_other_metals_positions_confirmed} onChange={updateManual} required />
            I confirm there are no other open gold or silver positions
          </label>
        </>}
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
      <CompactRiskSummary summary={summary} cash={cash} />
      {notices.length > 0 && <details className="result-details">
        <summary>Notes <span>{notices.length}</span></summary>
        <ul>{notices.map(([label, text], index) => <li key={`${label}-${index}`}><strong>{label}</strong> {text}</li>)}</ul>
      </details>}
      <details className="result-details">
        <summary>All figures</summary>
        <div className="cards">
          <Card title="Sizing" rows={[
            ["Requested", show(sizing.requested_size, 4)],
            ["Assessed", show(sizing.assessed_size, 4)],
            ["Maximum", show(sizing.recommended_max_size, 4)],
            ["Constraint", sizing.binding_constraint || "—"],
          ]} />
          <Card title="Trade" rows={[
            ["Stop loss", formatMoney(cash.planned_stop_loss, summary.currency)],
            ["Stop / equity", percent(cash.planned_stop_loss_fraction)],
            ["Target reward", formatMoney(cash.net_target_reward, summary.currency)],
            ["Reward / risk", show(cash.net_reward_risk)],
          ]} />
          <Card title="Cost" rows={[
            ["Spread", show(costs.current_spread_points, 4)],
            ["Cost / R", percent(costs.current_cost_r)],
            ["Quote age", show(market.quote_age_seconds, 1)],
            ["Volatility", market.volatility_state || "—"],
          ]} />
          <Card title="Portfolio" rows={[
            ["Positions", show(portfolio.existing_position_count, 0)],
            ["Stop risk", formatMoney(portfolio.proposed_total_stop_risk, summary.currency)],
            ["Margin", formatMoney(portfolio.proposed_total_margin, summary.currency)],
            ["Exposure", formatMoney(portfolio.proposed_total_gross_exposure, summary.currency)],
          ]} />
        </div>
        {scenarios.length > 0 && <div className="table-wrap">
          <table>
            <thead><tr><th>Scenario</th><th>Loss</th><th>Equity</th></tr></thead>
            <tbody>{scenarios.map((scenario, index) => <tr key={`${scenario.name}-${index}`}><td>{scenarioName(scenario.name)}</td><td>{formatMoney(scenario.cash_loss, summary.currency)}</td><td>{percent(scenario.equity_fraction)}</td></tr>)}</tbody>
          </table>
        </div>}
      </details>
    </section>
  );
}

function CompactRiskSummary({ summary, cash }) {
  const score = summary.riskScore ?? "—";
  const tone = score === "—" ? "unknown" : score <= 2 ? "low" : score <= 4 ? "medium" : "high";
  return (
    <section className="compact-summary" aria-label="Risk summary">
      <div className={`score score-${tone}`}>
        <strong>{score}<small>/7</small></strong>
        <span>Policy pressure</span>
        <small>{summary.riskDriver}</small>
      </div>
      <div className="number-grid">
        <NumberTile label="Stop loss" value={formatMoney(summary.plannedLoss, summary.currency)} />
        <NumberTile label="Equity" value={percent(summary.lossFraction)} />
        <NumberTile label="Risk budget" value={percent(summary.budgetUsage)} />
        <NumberTile label="Cost / R" value={percent(summary.currentCostR)} />
        <NumberTile label="Exposure" value={percent(summary.grossExposureFraction)} />
        <NumberTile label="Margin" value={percent(summary.marginFraction)} />
        <NumberTile label="Reward / risk" value={show(cash.net_reward_risk)} />
        <NumberTile label="Evidence" value={`${summary.evidenceScore}/3`} />
      </div>
    </section>
  );
}

function NumberTile({ label, value }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
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
