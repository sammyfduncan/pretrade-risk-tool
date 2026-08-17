"""Dependency-free HTML rendering for version-one pre-trade risk reports."""

from __future__ import annotations

from html import escape
import math
from typing import Any, Mapping, Sequence


DECISIONS = {"ALLOW", "REDUCE", "BLOCK", "INCOMPLETE"}


def render_pretrade_report_html(
    document: Mapping[str, Any],
    *,
    title: str = "Quantitative pre-trade risk assessment",
) -> str:
    """Render a risk report or broker wrapper as standalone static HTML."""

    report, wrapper = _extract_report(document)
    decision = str(report.get("decision") or "INCOMPLETE").upper()
    if decision not in DECISIONS:
        raise ValueError(f"unsupported pre-trade decision: {decision!r}")
    trade = _mapping(report.get("trade"))
    sizing = _mapping(report.get("sizing"))
    cash = _mapping(report.get("cash_risk"))
    proposed = _mapping(cash.get("proposed"))
    costs = _mapping(report.get("costs"))
    market = _mapping(report.get("market_risk"))
    portfolio = _mapping(report.get("portfolio"))
    assessment = _mapping(report.get("assessment"))
    provenance = _mapping(wrapper.get("provenance"))
    reasons = _strings(report.get("reasons"))
    warnings = _strings(report.get("warnings"))
    adapter_issues = _strings(wrapper.get("adapter_issues"))
    limitations = _strings(report.get("limitations"))
    scenarios = report.get("scenarios")
    scenario_rows = scenarios if isinstance(scenarios, (list, tuple)) else []

    subtitle = " · ".join(
        value
        for value in (
            _display(trade.get("instrument")),
            _display(trade.get("side")),
            _display(trade.get("order_type")),
            _display(trade.get("decision_time")),
        )
        if value != "—"
    )
    body = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f"<style>{_CSS}</style></head><body>",
        '<main class="page">',
        '<header class="masthead">',
        '<p class="eyebrow">Precious-metals decision support</p>',
        f"<h1>{escape(title)}</h1>",
        f'<p class="subtitle">{escape(subtitle or "Incomplete trade context")}</p>',
        f'<div class="decision decision-{decision.lower()}">{escape(decision)}</div>',
        "</header>",
        _notice_block(reasons, warnings, adapter_issues),
        '<section class="grid">',
        _card(
            "Policy pressure",
            (
                ("Score", _score(assessment.get("policy_pressure_score"), 7)),
                ("Primary constraint", _display(assessment.get("policy_pressure_driver"))),
                ("Maximum pressure ratio", _number(assessment.get("maximum_pressure_ratio"))),
                ("Evidence", _score(assessment.get("evidence_score"), assessment.get("evidence_maximum"))),
                ("Loss probability", "no"),
                ("Assessment version", _display(assessment.get("version"))),
            ),
        ),
        _card(
            "Sizing",
            (
                ("Requested", _number(sizing.get("requested_size"))),
                ("Assessed", _number(sizing.get("assessed_size"))),
                ("Recommended maximum", _number(sizing.get("recommended_max_size"))),
                ("Recommended action", _number(sizing.get("recommended_action_size"))),
                ("Binding constraint", _display(sizing.get("binding_constraint"))),
            ),
        ),
        _card(
            "Cash risk",
            (
                ("Currency", _display(cash.get("account_currency"))),
                ("Planned stop loss", _money(proposed.get("planned_stop_loss"))),
                ("Stop loss / equity", _percent(proposed.get("planned_stop_loss_fraction"))),
                ("Net target reward", _money(proposed.get("net_target_reward"))),
                ("Net reward / risk", _number(proposed.get("net_reward_risk"))),
                ("Initial margin", _money(proposed.get("initial_margin"))),
            ),
        ),
        _card(
            "Execution cost",
            (
                ("Current spread", _points(costs.get("current_spread_points"))),
                ("Round-trip cost", _money(proposed.get("round_trip_cost"))),
                ("Current cost / R", _number(costs.get("current_cost_r"))),
                ("Median cost / R", _number(costs.get("median_cost_r"))),
                ("P95 cost / R", _number(costs.get("p95_cost_r"))),
                ("Quote age", _seconds(market.get("quote_age_seconds"))),
            ),
        ),
        _card(
            "Portfolio after proposed trade",
            (
                ("Open positions", _number(portfolio.get("existing_position_count"), 0)),
                ("Total planned stop risk", _money(portfolio.get("proposed_total_stop_risk"))),
                ("Total margin", _money(portfolio.get("proposed_total_margin"))),
                ("Gross exposure", _money(portfolio.get("proposed_total_gross_exposure"))),
                ("Signed 1% move", _money(portfolio.get("combined_signed_one_pct_move"))),
                ("Volatility state", _display(market.get("volatility_state"))),
            ),
        ),
        "</section>",
        _scenario_table(scenario_rows, cash.get("account_currency")),
        _provenance_table(report, wrapper, provenance),
        _list_section("Limitations", limitations, "No limitations were recorded."),
        '<footer><p>Decision support only. ALLOW does not predict profitability or guarantee maximum loss.</p></footer>',
        "</main></body></html>\n",
    ]
    return "".join(body)


def _extract_report(
    document: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    wrapper = _mapping(document)
    nested = wrapper.get("risk_report")
    report = _mapping(nested) if nested is not None else wrapper
    if not report or "decision" not in report:
        raise ValueError("document does not contain a pre-trade risk report")
    return report, wrapper if nested is not None else {}


def _notice_block(
    reasons: Sequence[str], warnings: Sequence[str], adapter_issues: Sequence[str]
) -> str:
    blocks = []
    for label, values, kind in (
        ("Decision reasons", reasons, "reason"),
        ("Warnings", warnings, "warning"),
        ("Adapter audit", adapter_issues, "audit"),
    ):
        if values:
            items = "".join(f"<li>{escape(value)}</li>" for value in values)
            blocks.append(
                f'<section class="notice notice-{kind}"><h2>{label}</h2><ul>{items}</ul></section>'
            )
    return "".join(blocks)


def _card(title: str, rows: Sequence[tuple[str, str]]) -> str:
    entries = "".join(
        f"<dt>{escape(label)}</dt><dd>{escape(value)}</dd>" for label, value in rows
    )
    return f'<section class="card"><h2>{escape(title)}</h2><dl>{entries}</dl></section>'


def _scenario_table(rows: Sequence[Any], currency: Any) -> str:
    if not rows:
        return _list_section("Stress scenarios", (), "Unavailable for an incomplete assessment.")
    body = []
    for value in rows:
        row = _mapping(value)
        body.append(
            "<tr>"
            f"<td>{escape(_display(row.get('name')))}</td>"
            f"<td>{escape(_money(row.get('cash_loss')))}</td>"
            f"<td>{escape(_percent(row.get('equity_fraction')))}</td>"
            f"<td>{'yes' if row.get('probability_assigned') is True else 'no'}</td>"
            "</tr>"
        )
    return (
        '<section class="table-section"><h2>Stress scenarios</h2>'
        f'<p>Cash loss is shown in {escape(_display(currency))}; no scenario probability is inferred.</p>'
        '<div class="table-wrap"><table><thead><tr><th>Scenario</th>'
        '<th>Cash loss</th><th>Equity loss</th><th>Probability assigned</th></tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></section>"
    )


def _provenance_table(
    report: Mapping[str, Any],
    wrapper: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> str:
    rows = (
        ("Policy version", _display(report.get("policy_version"))),
        ("Report schema", _display(report.get("schema_version"))),
        ("Source mode", _display(wrapper.get("mode") or provenance.get("mode") or "offline JSON")),
        ("Order authority", "enabled" if wrapper.get("order_authority") is True else "disabled"),
        (
            "Broker instrument",
            _display(
                provenance.get("broker_instrument")
                or _mapping(report.get("trade")).get("broker_instrument")
            ),
        ),
        ("Quote timestamp", _display(provenance.get("quote_timestamp"))),
        ("Margin source", _display(provenance.get("margin_source"))),
        ("Spread observations", _number(provenance.get("spread_sample_rows"), 0)),
    )
    return '<section class="table-section"><h2>Audit and provenance</h2>' + _card("", rows) + "</section>"


def _list_section(title: str, values: Sequence[str], empty: str) -> str:
    if values:
        content = "<ul>" + "".join(f"<li>{escape(value)}</li>" for value in values) + "</ul>"
    else:
        content = f"<p>{escape(empty)}</p>"
    return f'<section class="text-section"><h2>{escape(title)}</h2>{content}</section>'


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def _display(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _number(value: Any, decimals: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def _score(value: Any, maximum: Any) -> str:
    score = _number(value, 0)
    limit = _number(maximum, 0)
    if score == "—" or limit == "—":
        return "—"
    return f"{score}/{limit}"


def _money(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.2f}" if math.isfinite(number) else "—"


def _percent(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.3%}" if math.isfinite(number) else "—"


def _points(value: Any) -> str:
    rendered = _number(value)
    return f"{rendered} pts" if rendered != "—" else rendered


def _seconds(value: Any) -> str:
    rendered = _number(value, 1)
    return f"{rendered} s" if rendered != "—" else rendered


_CSS = """
:root{color-scheme:light;--ink:#182229;--muted:#627078;--paper:#f5f2ea;--card:#fffdf7;
--line:#d8d4ca;--green:#216e4e;--amber:#9a5b12;--red:#a33834;--blue:#315b78}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;line-height:1.45}
.page{max-width:1120px;margin:0 auto;padding:48px 24px 72px}.masthead{border-bottom:1px solid var(--line);
padding-bottom:24px;margin-bottom:24px;position:relative}.eyebrow{text-transform:uppercase;letter-spacing:.12em;
font-size:.75rem;font-weight:700;color:var(--muted);margin:0 0 8px}h1{font-family:Georgia,serif;
font-size:clamp(2rem,5vw,3.5rem);line-height:1.05;margin:0 0 12px}.subtitle{color:var(--muted);margin:0;
max-width:75ch}.decision{display:inline-block;margin-top:20px;padding:8px 14px;border:1px solid currentColor;
font-weight:800;letter-spacing:.08em}.decision-allow{color:var(--green)}.decision-reduce{color:var(--amber)}
.decision-block{color:var(--red)}.decision-incomplete{color:var(--blue)}.notice{padding:16px 20px;
border-left:4px solid var(--line);background:var(--card);margin:16px 0}.notice h2{margin-top:0}
.notice-reason{border-color:var(--red)}.notice-warning{border-color:var(--amber)}.notice-audit{border-color:var(--blue)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:24px 0}.card{background:var(--card);
border:1px solid var(--line);padding:20px}.card h2,.table-section h2,.text-section h2,.notice h2{font-family:Georgia,serif;
font-size:1.25rem}.card h2:empty{display:none}dl{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px 20px;margin:0}
dt{color:var(--muted)}dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;font-weight:650}
.table-section,.text-section{margin-top:28px}.table-wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line)}
table{width:100%;border-collapse:collapse}th,td{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line);
white-space:nowrap}th{font-size:.75rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}
tbody tr:last-child td{border-bottom:0}li+li{margin-top:5px}footer{border-top:1px solid var(--line);margin-top:36px;
padding-top:16px;color:var(--muted);font-size:.9rem}@media(max-width:720px){.page{padding:28px 16px 48px}
.grid{grid-template-columns:1fr}dl{grid-template-columns:1fr}dd{text-align:left;margin-bottom:8px}}
@media print{body{background:#fff}.page{max-width:none;padding:20px}.card,.table-wrap,.notice{break-inside:avoid}}
"""
