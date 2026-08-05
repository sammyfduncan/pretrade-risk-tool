"""Combined readiness diagnostics for OANDA shadow and quote collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import pandas as pd

from commodity_ml.intraday.quote_quality import QuoteCollectionQuality


LONDON = "Europe/London"
ENTRY_SLOTS_PER_DAY = 25
ALLOWED_STATUSES = {
    "NO_SIGNAL",
    "ASSESSED",
    "ALREADY_ASSESSED",
    "SUPPRESSED",
    "ERROR",
}


@dataclass(frozen=True)
class OperationalReadinessPolicy:
    minimum_london_session_days: int = 10
    minimum_entry_slots_per_pair: int = 200
    minimum_entry_slot_coverage: float = 0.80
    minimum_quotes_per_instrument: int = 10_000

    def __post_init__(self) -> None:
        for name in (
            "minimum_london_session_days",
            "minimum_entry_slots_per_pair",
            "minimum_quotes_per_instrument",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 < self.minimum_entry_slot_coverage <= 1:
            raise ValueError("minimum_entry_slot_coverage must be in (0, 1]")


def build_operational_readiness_report(
    checks: pd.DataFrame,
    quote_quality: QuoteCollectionQuality,
    *,
    instrument_map: Mapping[str, str],
    strategies: Sequence[str],
    policy: OperationalReadinessPolicy | None = None,
) -> dict[str, object]:
    """Build an operational readiness result without a performance claim."""

    policy = policy or OperationalReadinessPolicy()
    strategies = tuple(dict.fromkeys(str(value) for value in strategies))
    if not instrument_map or any(not key or not value for key, value in instrument_map.items()):
        raise ValueError("instrument_map must contain non-empty symbol mappings")
    if not strategies or any(not value for value in strategies):
        raise ValueError("strategies must be non-empty")
    required = {
        "check_id",
        "instrument",
        "strategy",
        "signal_time",
        "status",
        "decision",
        "reasons",
    }
    if missing := sorted(required - set(checks)):
        raise ValueError(f"shadow checks missing columns: {missing}")
    frame = checks.copy()
    frame["signal_time"] = pd.to_datetime(frame["signal_time"], utc=True, errors="raise")
    if frame["check_id"].duplicated().any():
        raise ValueError("shadow checks contain duplicate check IDs")
    unknown_status = sorted(set(frame["status"].dropna()) - ALLOWED_STATUSES)
    if unknown_status:
        raise ValueError(f"shadow checks contain unknown statuses: {unknown_status}")

    decision_local = (frame["signal_time"] + pd.Timedelta(minutes=30)).dt.tz_convert(LONDON)
    entry_minute = decision_local.dt.hour * 60 + decision_local.dt.minute
    entry_frame = frame.loc[(entry_minute >= 7 * 60) & (entry_minute < 19 * 60 + 30)].copy()
    entry_frame["london_date"] = decision_local.loc[entry_frame.index].dt.date.astype(str)
    london_dates = sorted(entry_frame["london_date"].unique())
    expected_slots = len(london_dates) * ENTRY_SLOTS_PER_DAY

    pair_coverage: dict[str, dict[str, object]] = {}
    collection_issues: list[str] = []
    for symbol in instrument_map:
        for strategy in strategies:
            key = f"{symbol}:{strategy}"
            pair = entry_frame.loc[
                (entry_frame["instrument"] == symbol)
                & (entry_frame["strategy"] == strategy)
            ]
            observed = int(pair["signal_time"].nunique())
            coverage = observed / expected_slots if expected_slots else 0.0
            pair_coverage[key] = {
                "observed_entry_slots": observed,
                "expected_slots_across_observed_dates": expected_slots,
                "coverage": coverage,
            }
            if observed < policy.minimum_entry_slots_per_pair:
                collection_issues.append(f"{key}:insufficient_entry_slots")
            if coverage < policy.minimum_entry_slot_coverage:
                collection_issues.append(f"{key}:entry_slot_coverage_below_policy")

    if len(london_dates) < policy.minimum_london_session_days:
        collection_issues.append("insufficient_london_session_days")

    quotes_by_instrument = {
        item.instrument: item for item in quote_quality.instruments
    }
    integrity_issues: list[str] = []
    for symbol, broker_instrument in instrument_map.items():
        quality = quotes_by_instrument.get(broker_instrument)
        if quality is None:
            collection_issues.append(f"{symbol}:missing_quote_collection")
            continue
        if quality.quotes < policy.minimum_quotes_per_instrument:
            collection_issues.append(f"{symbol}:insufficient_quotes")
        if quality.crossed_quotes:
            integrity_issues.append(f"{symbol}:crossed_quotes")
        if quality.duplicate_broker_times:
            integrity_issues.append(f"{symbol}:duplicate_broker_times")
    if quote_quality.unresolved_invalidating_events:
        integrity_issues.append("unresolved_stream_invalidating_events")
    if (frame["status"] == "ERROR").any():
        integrity_issues.append("shadow_error_checks_present")

    status_counts = _counts(frame["status"])
    decision_counts = _counts(frame["decision"].dropna())
    reason_counts: Counter[str] = Counter()
    for reasons in frame["reasons"]:
        if not isinstance(reasons, (list, tuple)):
            raise ValueError("shadow check reasons must be a list or tuple")
        reason_counts.update(str(reason) for reason in reasons)
    signal_statuses = {"ASSESSED", "ALREADY_ASSESSED", "SUPPRESSED"}
    signal_count = int(frame["status"].isin(signal_statuses).sum())
    eligible_checks = len(entry_frame)

    if integrity_issues:
        readiness = "REVIEW"
    elif collection_issues:
        readiness = "COLLECTING"
    else:
        readiness = "OPERATIONALLY_READY"
    return {
        "schema_version": 1,
        "readiness": readiness,
        "policy": asdict(policy),
        "shadow": {
            "total_checks": len(frame),
            "entry_session_checks": eligible_checks,
            "london_session_days": len(london_dates),
            "first_london_date": london_dates[0] if london_dates else None,
            "last_london_date": london_dates[-1] if london_dates else None,
            "signal_count": signal_count,
            "signal_rate_per_check": signal_count / len(frame) if len(frame) else None,
            "status_counts": status_counts,
            "decision_counts": decision_counts,
            "reason_counts": dict(sorted(reason_counts.items())),
            "pair_coverage": pair_coverage,
        },
        "quotes": quote_quality.to_dict(),
        "integrity_issues": integrity_issues,
        "collection_issues": collection_issues,
        "limitations": [
            "Operational readiness checks data collection and lifecycle integrity only.",
            "It does not establish forecast skill, strategy profitability, or permission to trade.",
            "Observed-date slot coverage is descriptive and does not substitute for exchange holiday review.",
        ],
    }


def _counts(values: pd.Series) -> dict[str, int]:
    counts = values.astype(str).value_counts().sort_index()
    return {str(key): int(value) for key, value in counts.items()}
