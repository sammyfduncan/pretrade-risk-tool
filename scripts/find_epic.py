"""Look up IG "epics" — IG's market identifiers.

IG doesn't use tickers like XAUUSD; every market has an epic such as
CS.D.USCGC.TODAY.IP, and they differ between account regions and product
types (spot vs futures vs mini). So the epic must be looked up on YOUR
account rather than copied from anywhere.

Usage:
    python scripts/find_epic.py --search "spot gold"
    python scripts/find_epic.py --search "silver"

Copy the epic you want into data.ig.epic in config/<asset>.yaml.
Costs one ordinary API request, not historical-data allowance.
"""

import argparse

from dotenv import load_dotenv

from commodity_ml.data.ig import get_service, search_epics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", required=True, help='e.g. "spot gold"')
    args = parser.parse_args()

    load_dotenv()
    results = search_epics(get_service(), args.search)

    if results.empty:
        print(f"No markets matched {args.search!r}.")
        return

    print(f"Markets matching {args.search!r}:\n")
    print(results.to_string(index=False))
    print("\nPick the spot/cash market (usually instrumentType CURRENCIES or COMMODITIES "
          "with expiry '-' or 'DFB') and set it as data.ig.epic in your config.")


if __name__ == "__main__":
    main()
