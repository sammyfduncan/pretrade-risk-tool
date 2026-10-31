# Quantitative precious metals pretrade risk tool

Framework quantifying technical risk of a prospective intraday XAU/XAG trade (gold/Silver).

Select a proposed direction or separately validated strategy; entry, stop,
target, holding period and optional size. The framework combines this with 
account state, current executable market prices (as supplied via OANDA API).
broker contract facts, existing metals exposure, scheduled event controls and 
historically calibrated range and adverse excursion evidence.

It returns one of four decisions:

- `ALLOW`: fits the configured risk limits;
- `REDUCE`: size must be reduced;
- `BLOCK`: policy, market, cost or portfolio prevents trade entry;
- `INCOMPLETE`: a fact is absent and framework won't guess.

`ALLOW` does not predict profitability or guarantee a maximum loss.

## Specs

- covers spot gold (`XAUUSD`) and silver (`XAGUSD`);
- uses quantitative data only;
- accepts market, limit and stop trades;
- calculates cash at stop risk, cost adjusted reward + risk, margin, notional and
  portfolio concentration;
- applies London session, daily/weekly loss, entry count, scheduled event and
  kill switch controls;
- reports deterministic spread and gap stresses;
- blocks on missing, delayed, inconsistent or unreconciled data.

## Instructions

Create venv/install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Offline assessment 

Copy and complete `examples/pretrade_risk_input.template.json`, then run:

```bash
.venv/bin/python scripts/evaluate_pretrade_risk.py \
  examples/pretrade_risk_input.json \
  --output reports/pretrade_risk.json
```

Render the JSON as a standalone report:

```bash
.venv/bin/python scripts/render_pretrade_risk_report.py \
  reports/pretrade_risk.json \
  --output reports/pretrade_risk.html
```

### OANDA assessment

Local `.env` needs to supply:

```text
OANDA_API_KEY=...
OANDA_ACCOUNT_ID=...
OANDA_ENV=practice
```

Initialise local risk anchors only when the authenticated account is flat:

```bash
.venv/bin/python scripts/initialize_oanda_paper_state.py \
  --config config/gold.yaml
```

Copy `examples/oanda_pretrade_intent.template.json` then run:

```bash
.venv/bin/python scripts/evaluate_oanda_pretrade_risk.py \
  examples/oanda_pretrade_intent.json \
  --config config/gold.yaml \
  --portfolio-config config/silver.yaml \
  --paper-state data/runtime/oanda_paper_state.json \
  --output reports/oanda_pretrade_risk.json
```

### Browser UI

Install and build the local React interface:

```bash
npm ci --prefix web
npm run build --prefix web
```

Start local server:

```bash
.venv/bin/python scripts/run_risk_ui.py
```

### Manual refresh (calibration)

```bash
.venv/bin/python scripts/refresh_current_risk_calibration.py
```

### Verification

Run the complete offline test and build checks:

```bash
.venv/bin/python -m pytest
npm test --prefix web
npm run build --prefix web
```

The application is decision support only. It contains no broker order endpoint
and is not approved for unattended execution or real-money reliance.
