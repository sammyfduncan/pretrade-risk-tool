# Documentation map

The repository contains two bodies of work. Active product documents define
the quantitative pre-trade risk framework. Historical research documents
preserve earlier directional-strategy and ML experiments, including negative
results. Historical documents are evidence, not the current build queue.

## Active product documentation

Read these in order:

1. [`final_validation_report.md`](final_validation_report.md) — final product
   classification, evidence, limitations and supported claims.
2. [`risk_framework_roadmap.md`](risk_framework_roadmap.md) — objective,
   completed scope and explicit closure decisions.
3. [`pretrade_risk_engine.md`](pretrade_risk_engine.md) — versioned inputs,
   calculations, decisions and policy boundary.
4. [`pretrade_risk_calibration.md`](pretrade_risk_calibration.md) — causal
   historical range and adverse-excursion calibration.
5. [`pretrade_calibration_refresh_policy.md`](pretrade_calibration_refresh_policy.md)
   — frozen admission rules for a current rolling profile.
6. [`pretrade_oanda_snapshot.md`](pretrade_oanda_snapshot.md) — authenticated
   read-only broker facts.
7. [`paper_risk_runtime.md`](paper_risk_runtime.md) — restart-safe account risk
   anchors and reconciliation.
8. [`pretrade_event_calendar.md`](pretrade_event_calendar.md) — deterministic
   scheduled-event controls.
9. [`pretrade_paper_observations.md`](pretrade_paper_observations.md) and
   [`pretrade_quote_path.md`](pretrade_quote_path.md) — immutable prospective
   assessments and executable-path outcomes.
10. [`prospective_calibration_validation.md`](prospective_calibration_validation.md)
   — research-only, direction-balanced validation of current range forecasts.
11. [`risk_ui.md`](risk_ui.md) — basic localhost React workflow and its strict
    read-only security boundary.

Supporting operational diagnostics are grouped separately:

- [`operations/oanda_stream_recorder.md`](operations/oanda_stream_recorder.md)
- [`operations/oanda_shadow_runner.md`](operations/oanda_shadow_runner.md)

These diagnostics are useful but do not establish profitability and are not
required to finish the offline risk product.

## Archived research records

Completed and retired research programs live under [`archive/`](archive/).
They remain versioned because their preregistrations, negative results and
data-acquisition decisions are unique audit evidence, but they are not part of
the active product reading path.

The archive index explains why each document was retained:
[`archive/README.md`](archive/README.md).

Do not interpret a historical statement such as “paper trading remains
disabled” or “a primary strategy is required” as a blocker for the standalone
pre-trade risk evaluator. It describes the experiment in which it was written.
