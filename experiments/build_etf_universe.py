#!/usr/bin/env python3
"""
Build / refresh the large liquid ETF universe used across the studies.

Discovers the universe + instrument info from financedatabase, downloads
adjusted price history from yfinance, screens for coverage and dollar-volume
liquidity, and writes:

    datasets/etf_universe_prices.csv   (days × top_n adjusted closes)
    datasets/etf_universe_info.csv     (fd info for those names + median ADV)

Requires network access (yfinance + financedatabase).  Run once; the studies
then load the cached panel offline via posterioralpha.data.load_etf_universe_*.

    python experiments/build_etf_universe.py --start 2010-01-01 --top-n 250
"""
import argparse
import logging
import sys

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.data.universe import build_etf_universe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--top-n", type=int, default=250, help="most-liquid names to keep")
    ap.add_argument("--min-coverage", type=float, default=0.9)
    ap.add_argument(
        "--include-leveraged", action="store_true",
        help="keep leveraged/inverse ETFs (excluded by default)",
    )
    args = ap.parse_args()

    prices, info = build_etf_universe(
        start=args.start,
        end=args.end,
        top_n=args.top_n,
        min_coverage=args.min_coverage,
        exclude_leveraged=not args.include_leveraged,
    )

    print(f"\nUniverse: {prices.shape[1]} ETFs × {prices.shape[0]} days "
          f"[{prices.index.min().date()} → {prices.index.max().date()}]")
    print("\nTop 15 by median daily $ volume:")
    cols = [c for c in ("name", "category_group", "family", "median_adv_usd") if c in info.columns]
    print(info[cols].head(15).to_string())
    print("\nBy category group:")
    if "category_group" in info.columns:
        print(info["category_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
