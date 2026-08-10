# Basic local risk interface

**Status:** implemented

The browser interface is a deliberately simple React form over the existing
Python risk engine. It is a convenience layer, not a second calculation
implementation. The Python report now owns the versioned pressure score; the
React app only displays it. Both account modes use OANDA only for read-only
market prices, spreads and recent candles.

The default result is deliberately numeric. It shows a 1–7 **policy-pressure**
scale, stop loss, equity fraction, risk-budget use, cost/R, exposure, margin,
reward/risk and a separate 0–3 evidence score. `ALLOW` remains an audit code;
the scale does not estimate the probability of loss.

The pressure score is the highest utilisation among the binding size cap,
per-trade risk allowance, current/median/p95 execution-cost gates, gross
exposure and margin gates. Scores 1–6 divide utilisation up to the applicable
limit; exceeding a limit or a hard `BLOCK` is 7. Evidence is separate because
missing volatility or calibrated movement data must not be disguised as a
precise risk probability. Explanations and translated internal codes remain
available only under the collapsed `Notes` and `All figures` controls and in
downloadable JSON.

The current score contract is
`policy-pressure-v1-2026-08-10`, embedded with the component ratios in every
JSON and HTML report. That makes later changes auditable instead of silently
changing how an old assessment should be interpreted.

## What the user supplies

- gold or silver;
- long or short;
- market, limit or stop entry;
- entry price for pending orders;
- stop, target and optional requested size;
- intended 60, 120, 180 or 240-minute holding period;
- maximum adverse slippage per side; and
- whether overnight holding is intended.

Current price, spread, contract rules, currency conversion, account equity,
margin, open precious-metals positions and recent M30 candles come from the
configured OANDA practice account. The browser never receives the API key or
account ID.

Alternatively, **External GBP spread-bet account** accepts account equity,
available funds, day/week starting equity, entries today, the external
provider's margin percentage, commission and stake in GBP per quoted-price
unit. It requires an explicit confirmation that no other gold or silver
positions are open. Those account facts are not broker-verified; current
bid/ask and spread history still come from OANDA. This mode is suitable for
assessing a proposed trade held with another provider, but not for maintaining
an automatically reconciled external portfolio.

If a complete current-calibration release has status `ADMITTED`, the backend
selects the newest pair and the risk engine revalidates its hashes, context and
freshness before use. Rejected, partial or malformed releases are ignored.

## Run it

Build once, or after changing the React source:

```bash
cd web
npm install
npm run build
cd ..
```

Start the local server:

```bash
.venv/bin/python scripts/run_risk_ui.py
```

Open `http://127.0.0.1:8765`.

For frontend development, keep the Python server on port 8765 and run
`npm run dev` from `web`; Vite serves `http://127.0.0.1:5173` and proxies only
the `/api` path to the local backend.

## Safety boundary

- The server binds only to `127.0.0.1` or `localhost`.
- It loads credentials server-side from the ignored `.env`.
- The assessment endpoint invokes either the read-only OANDA-account adapter or
  the manual-account/read-only-market adapter.
- The only POST routes evaluate risk and render an existing result as HTML.
- There is no endpoint for creating, amending or closing orders.
- The backend rejects unsupported assets, directions, order types, horizons,
  oversized request bodies and any engine response claiming order authority.
- Responses include no-store, content-type, referrer and content-security
  headers.

This is intentionally a local single-user tool. Hosting it publicly would
require authentication, transport security, rate limiting and a separate
threat review; none are necessary for the current offline-first scope.
