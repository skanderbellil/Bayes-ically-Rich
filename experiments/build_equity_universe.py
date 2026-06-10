#!/usr/bin/env python3
"""
Build / refresh the large liquid US equity universe.

Mirrors build_etf_universe.py over financedatabase equities: discovers US
large/mega-cap names + instrument info, downloads adjusted history + volume
from yfinance, screens for coverage and dollar-volume liquidity, and writes:

    datasets/equity_universe_prices.csv.gz   (days × top_n adjusted closes)
    datasets/equity_universe_info.csv        (fd info for those names + median ADV)

Requires network access.  Run once; studies then load offline via
posterioralpha.data.load_equity_universe_*.

    python experiments/build_equity_universe.py --start 2010-01-01 --top-n 500
"""
import argparse
import logging
import sys

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.data.universe import build_equity_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-n", type=int, default=500, help="most-liquid names to keep")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument(
        "--include-midcap", action="store_true",
        help="also include Mid Cap names (default: Mega + Large Cap only)",
    )
    args = ap.parse_args()

    caps = ("Mega Cap", "Large Cap", "Mid Cap") if args.include_midcap else ("Mega Cap", "Large Cap")
    prices, info = build_equity_universe(
        start=args.start, end=args.end, top_n=args.top_n,
        min_coverage=args.min_coverage, market_caps=caps,
    )

    print(f"\nUniverse: {prices.shape[1]} equities × {prices.shape[0]} days "
          f"[{prices.index.min().date()} → {prices.index.max().date()}]")
    print("\nTop 15 by median daily $ volume:")
    cols = [c for c in ("name", "sector", "market_cap", "median_adv_usd") if c in info.columns]
    print(info[cols].head(15).to_string())
    if "sector" in info.columns:
        print("\nBy sector:")
        print(info["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
