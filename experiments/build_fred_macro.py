#!/usr/bin/env python3
"""
Build the curated FRED macro panel → ``datasets/fred_macro.csv``.

14 series across rates & curve (DFF, DGS2, DGS10, T10Y2Y, T10Y3M), credit
(HY & IG OAS), risk/conditions (VIX, NFCI, STLFSI4), inflation expectations
(5y/10y breakevens), the broad dollar (DTWEXBGS) and jobless claims (ICSA).
Each series is shifted by its publication lag before forward-filling onto a
daily business-day index, so ``panel.loc[t]`` is information available at t.

Requires a (free) FRED API key in the environment or in ``<repo>/.env``::

    FRED_API_KEY=...

Network; run once (or whenever you want to refresh).  Also refreshes
``datasets/net_liquidity.csv`` with ``--with-net-liquidity``.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging

from posterioralpha.data import build_fred_macro, build_net_liquidity
from posterioralpha.data.macro import MACRO_SERIES

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser(description="Build the FRED macro panel")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--with-net-liquidity", action="store_true",
                    help="also refresh datasets/net_liquidity.csv")
    args = ap.parse_args()

    panel = build_fred_macro(start=args.start)
    print(f"\nfred_macro.csv: {panel.shape[0]} days × {panel.shape[1]} series "
          f"({panel.index[0].date()} → {panel.index[-1].date()})")
    print(f"{'series':<14} {'lag':>3}  {'first':>10}  {'last':>10}  description")
    for sid, (lag, desc) in MACRO_SERIES.items():
        s = panel[sid].dropna()
        print(f"{sid:<14} {lag:>3}  {str(s.index[0].date()):>10}  "
              f"{str(s.index[-1].date()):>10}  {desc}")

    if args.with_net_liquidity:
        nl = build_net_liquidity(start=args.start)
        print(f"\nnet_liquidity.csv refreshed: {nl.shape[0]} days "
              f"(→ {nl.index[-1].date()})")


if __name__ == "__main__":
    main()
