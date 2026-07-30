"""Intraday research and execution contracts.

The historical H4 pipeline remains in the existing modules. New intraday work
lives here so correctness changes cannot silently alter the frozen results.
"""

from commodity_ml.intraday.data import build_intraday_frame
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
    "SpreadAssessment",
    "InstrumentConstraints",
    "RiskDecision",
    "RiskLimits",
    "RiskState",
    "SizingDecision",
    "assess_risk",
    "assess_spread_history",
    "build_candidate_dataset",
    "build_intraday_frame",
    "build_paper_intent",
    "candidate_feature_columns",
    "closed_quote_window",
    "labels_from_ledger",
    "generate_all_candidates",
    "generate_candidates",
    "resolve_trade",
    "simulate_candidates",
    "size_for_stop",
]
