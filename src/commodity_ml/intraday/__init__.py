"""Intraday research and execution contracts.

The historical H4 pipeline remains in the existing modules. New intraday work
lives here so correctness changes cannot silently alter the frozen results.
"""

from commodity_ml.intraday.data import attach_m15_context, build_intraday_frame
from commodity_ml.intraday.execution import (
    CandidateRejected,
    EntryBlackout,
    IntradayExecutionConfig,
    SessionCutoffOverride,
    TradeCandidate,
    TradeOutcome,
    labels_from_ledger,
    resolve_trade,
    simulate_candidates,
)
from commodity_ml.intraday.ml_dataset import (
    CandidateDatasetResult,
    build_candidate_dataset,
    candidate_feature_columns,
    load_candidate_bundle,
    write_candidate_bundle,
)
from commodity_ml.intraday.market_data import (
    SpreadAssessment,
    assess_spread_history,
    closed_quote_window,
)
from commodity_ml.intraday.paper import (
    PaperCandidate,
    PaperDecision,
    PaperOrderIntent,
    PaperRuntimeState,
    build_paper_intent,
)
from commodity_ml.intraday.risk import (
    InstrumentConstraints,
    RiskDecision,
    RiskLimits,
    RiskState,
    SizingDecision,
    assess_risk,
    size_for_stop,
)
from commodity_ml.intraday.strategies import (
    generate_all_candidates,
    generate_candidates,
)
from commodity_ml.intraday.state import (
    IntentRecord,
    PaperStateError,
    PaperStateStore,
    PersistentPaperState,
)

__all__ = [
    "CandidateRejected",
    "CandidateDatasetResult",
    "EntryBlackout",
    "IntradayExecutionConfig",
    "SessionCutoffOverride",
    "TradeCandidate",
    "TradeOutcome",
    "PaperCandidate",
    "PaperDecision",
    "PaperOrderIntent",
    "PaperRuntimeState",
    "PaperStateError",
    "PaperStateStore",
    "PersistentPaperState",
    "IntentRecord",
    "SpreadAssessment",
    "InstrumentConstraints",
    "RiskDecision",
    "RiskLimits",
    "RiskState",
    "SizingDecision",
    "assess_risk",
    "assess_spread_history",
    "attach_m15_context",
    "build_candidate_dataset",
    "build_intraday_frame",
    "build_paper_intent",
    "candidate_feature_columns",
    "closed_quote_window",
    "labels_from_ledger",
    "generate_all_candidates",
    "generate_candidates",
    "load_candidate_bundle",
    "resolve_trade",
    "simulate_candidates",
    "size_for_stop",
    "write_candidate_bundle",
]
