"""Verify and normalize a Databento GC TBBO batch delivery."""

import argparse
import json

from commodity_ml.config import PROJECT_ROOT
from commodity_ml.data.cme_microstructure import process_gc_tbbo_job


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tbbo-job", required=True)
    parser.add_argument(
        "--definitions",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc" / "definitions.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "raw" / "cme_gc_tbbo"),
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    report, manifest = process_gc_tbbo_job(
        tbbo_job=args.tbbo_job,
        definitions_path=args.definitions,
        output=None if args.validate_only else args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.validate_only:
        print("Validation only: no output was written and no strategy was evaluated.")
    else:
        print(f"Normalized GC TBBO import -> {manifest}")
        print("No feature, label, model, or strategy outcome was created.")


if __name__ == "__main__":
    main()
