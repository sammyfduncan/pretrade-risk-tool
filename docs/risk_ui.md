# Basic local risk interface

**Status:** implemented

The browser interface is a deliberately simple React form over the existing
Python risk engine. It is a convenience layer, not a second calculation
implementation. The established read-only OANDA command still constructs and
evaluates the versioned risk payload.

The result begins with a plain-language interpretation: cost-inclusive stop
loss, percentage of assessed account equity, use of the per-trade risk
allowance, exposure, margin and evidence completeness. `ALLOW` is retained as
an audit code but displayed as “Within configured limits”; it is never
described as proof that a trade is safe or likely to profit. Internal reason,
warning and adapter codes are translated for the browser while remaining
unchanged in downloadable JSON.

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
- The assessment endpoint invokes only the existing read-only OANDA evaluator.
- The only POST routes evaluate risk and render an existing result as HTML.
- There is no endpoint for creating, amending or closing orders.
- The backend rejects unsupported assets, directions, order types, horizons,
  oversized request bodies and any engine response claiming order authority.
- Responses include no-store, content-type, referrer and content-security
  headers.

This is intentionally a local single-user tool. Hosting it publicly would
require authentication, transport security, rate limiting and a separate
threat review; none are necessary for the current offline-first scope.
