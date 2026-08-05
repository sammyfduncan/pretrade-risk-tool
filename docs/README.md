# Documentation map

The repository contains two bodies of work. Active product documents define
the quantitative pre-trade risk framework. Historical research documents
preserve earlier directional-strategy and ML experiments, including negative
results. Historical documents are evidence, not the current build queue.

## Active product documentation

Read these in order:

1. [`risk_framework_roadmap.md`](risk_framework_roadmap.md) — current objective,
   status, remaining work and completion criteria.
2. [`pretrade_risk_engine.md`](pretrade_risk_engine.md) — versioned inputs,
   calculations, decisions and policy boundary.
3. [`pretrade_risk_calibration.md`](pretrade_risk_calibration.md) — causal
   historical range and adverse-excursion calibration.
4. [`pretrade_calibration_refresh_policy.md`](pretrade_calibration_refresh_policy.md)
   — frozen admission rules for a current rolling profile.
5. [`pretrade_oanda_snapshot.md`](pretrade_oanda_snapshot.md) — authenticated
   read-only broker facts.
6. [`paper_risk_runtime.md`](paper_risk_runtime.md) — restart-safe account risk
   anchors and reconciliation.
7. [`pretrade_event_calendar.md`](pretrade_event_calendar.md) — deterministic
   scheduled-event controls.
8. [`pretrade_paper_observations.md`](pretrade_paper_observations.md) and
   [`pretrade_quote_path.md`](pretrade_quote_path.md) — immutable prospective
   assessments and executable-path outcomes.

Supporting operational diagnostics:

- [`oanda_stream_recorder.md`](oanda_stream_recorder.md)
- [`oanda_shadow_runner.md`](oanda_shadow_runner.md)

These diagnostics are useful but do not establish profitability and are not
required to finish the offline risk product.

## Historical research records

The following documents describe completed or retired research programs:

- `step1_freeze_record.md`, `step2_design.md`, `step2b_vol_design.md`;
- `intraday_ml_retraining.md`, `intraday_filter_gate.md`;
- `intraday_v2_design.md`, `intraday_m15_execution_design.md`;
- `intraday_market_structure_research.md`, `intraday_cost_measurement.md`;
- `intraday_risk_design.md`, `intraday_joint_risk_design.md`,
  `intraday_risk_policy.md`, `intraday_risk_economic_design.md`;
- `cme_futures_data.md`, `intraday_futures_design.md`,
  `intraday_order_flow_design.md`;
- `intraday_relative_value_design.md`, `intraday_macro_impulse_design.md`; and
- `intraday_rebuild_plan.md`, the chronological implementation/research ledger.

Do not interpret a historical statement such as “paper trading remains
disabled” or “a primary strategy is required” as a blocker for the standalone
pre-trade risk evaluator. It describes the experiment in which it was written.
