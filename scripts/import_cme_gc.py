"""Verify and normalize three Databento GC batch jobs without creating features."""

import argparse
import json

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.data.cme import process_gc_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ohlcv-job", required=True)
    parser.add_argument("--statistics-job", required=True)
    parser.add_argument("--definition-job", required=True)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc"),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Read and validate everything but write no normalized artifacts",
    )
    args = parser.parse_args()

    report, manifest = process_gc_jobs(
        ohlcv_job=args.ohlcv_job,
        statistics_job=args.statistics_job,
        definition_job=args.definition_job,
        output=None if args.validate_only else args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.validate_only:
        print("Validation only: no output was written.")
        return
    print(f"Normalized CME GC import -> {manifest}")
    print("No front contract, roll, back-adjustment, feature, or strategy was created.")


if __name__ == "__main__":
    main()
