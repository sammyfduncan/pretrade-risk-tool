"""Record immutable prospective paper assessments and realised outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commodity_ml.intraday.pretrade_observations import (
    ObservationLedgerError,
    PaperTradeOutcome,
    PretradeObservationStore,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/runtime/pretrade_observations.sqlite"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    assessment = subparsers.add_parser("assessment")
    assessment.add_argument("input", type=Path, help="Risk report or IG wrapper JSON")
    outcome = subparsers.add_parser("outcome")
    outcome.add_argument("input", type=Path, help="Realised paper outcome JSON")
    subparsers.add_parser("summary")
    args = parser.parse_args()

    store = PretradeObservationStore(args.database)
    try:
        if args.command == "assessment":
            document = json.loads(args.input.read_text(encoding="utf-8"))
            created = store.record_assessment(document)
            print("Assessment recorded." if created else "Assessment already recorded exactly.")
        elif args.command == "outcome":
            document = json.loads(args.input.read_text(encoding="utf-8"))
            recorded = store.record_outcome(PaperTradeOutcome.from_mapping(document))
            print("Outcome recorded." if recorded else "Outcome already recorded exactly.")
        else:
            print(json.dumps(store.summary(), indent=2, sort_keys=True, allow_nan=False))
    except (
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        ObservationLedgerError,
    ) as exc:
        raise SystemExit(f"Observation ledger refused: {exc}") from exc
    print(f"Ledger -> {args.database.resolve()}")
    print("No broker or order action was performed.")


if __name__ == "__main__":
    main()
