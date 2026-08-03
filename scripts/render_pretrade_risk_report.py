"""Render a pre-trade risk JSON document as a standalone static HTML report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from commodity_ml.intraday.pretrade_report import render_pretrade_report_html


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Risk report or IG wrapper JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML file")
    parser.add_argument("--title", default="Quantitative pre-trade risk assessment")
    args = parser.parse_args()

    try:
        document = json.loads(args.input.read_text(encoding="utf-8"))
        html = render_pretrade_report_html(document, title=args.title)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Report rendering refused: {exc}") from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"HTML report -> {args.output.resolve()}")
    print("Rendering performs no broker or network action.")


if __name__ == "__main__":
    main()
