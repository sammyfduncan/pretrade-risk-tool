"""Evaluate an offline gold/silver trade-intent JSON; never submit an order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commodity_ml.intraday.pretrade_risk import evaluate_pretrade_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Version-1 input JSON")
    parser.add_argument("--output", type=Path, help="Optional report JSON path")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        report = evaluate_pretrade_payload(payload)
        encoded = json.dumps(
            report.to_dict(), indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Pre-trade assessment refused: {exc}") from exc

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")

    print(f"=== PRE-TRADE RISK: {report.decision} ===")
    if report.reasons:
        print(f"Reasons: {', '.join(report.reasons)}")
    if report.warnings:
        print(f"Warnings: {', '.join(report.warnings)}")
    if report.sizing:
        print(
            "Size: assessed "
            f"{report.sizing['assessed_size']:g}; recommended maximum "
            f"{report.sizing['recommended_max_size']:g}; action size "
            f"{report.sizing['recommended_action_size']:g}"
        )
    if report.cash_risk:
        proposed = report.cash_risk["proposed"]
        print(
            f"Planned stop loss: {proposed['planned_stop_loss']:,.2f} "
            f"{report.cash_risk['account_currency']} "
            f"({proposed['planned_stop_loss_fraction']:.3%} of equity)"
        )
    if args.output:
        print(f"Report -> {args.output.resolve()}")
    else:
        print(encoded, end="")
    print("No order was created or submitted.")


if __name__ == "__main__":
    main()
