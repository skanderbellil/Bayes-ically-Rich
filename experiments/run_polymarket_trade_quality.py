#!/usr/bin/env python3
"""
Polymarket trade quality — which behavioural tells mark an informed fill?
=========================================================================

A trade-level study (one row per fill, not per position). For every fill of the
non-MM candidate pool we measure four behavioural tells and ask whether each
predicts that the trade was *right* — that the token drifted in the trade's
direction over the next few days, and that it matched the eventual resolution:

  aggressiveness — did they pay *up* through the CLOB mid (urgency / conviction)?
  entry vs add   — is this the wallet's first fill in the market (initiation)?
  conviction     — is the fill large vs the wallet's own typical size (z>1)?
  patience       — do long-holding wallets forecast better than flippers?

Output: per-tell buckets with mean forward directional drift (+ t-stat) and
resolution hit-rate. Validated on a deliberately **large** universe.

Usage
-----
  python experiments/run_polymarket_trade_quality.py
  python experiments/run_polymarket_trade_quality.py --fwd 5 --refresh
"""
import argparse
import logging

import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    build_token_panel,
    build_trade_features,
    candidate_universe,
    fetch_many_traders,
    fetch_volume_wallets,
    screen_market_makers,
    select_universe,
    summarize_quality,
)
from posterioralpha.polymarket.paths import ensure_dirs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=120)
    ap.add_argument("--top-tokens", type=int, default=600)
    ap.add_argument("--max-trades", type=int, default=3000)
    ap.add_argument("--min-obs",    type=int, default=10)
    ap.add_argument("--fwd",        type=int, default=3, help="forward drift horizon (days)")
    ap.add_argument("--refresh",    action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   trade-quality tells           ║
║   what marks an informed fill: urgency / entry / size / patience
╚══════════════════════════════════════════════════════════╝""")

    pool = candidate_universe(("1d", "7d", "30d", "all"), per_window=args.per_window)
    wallets = list(dict.fromkeys(pool["wallet"].tolist()
                                 + sorted(fetch_volume_wallets("all", 150))))
    traders = fetch_many_traders(wallets, max_trades=args.max_trades, refresh=args.refresh)
    if not traders:
        raise SystemExit("No trader histories returned.")
    universe = select_universe(traders, top_k=args.top_tokens)
    panel = build_token_panel(universe, min_obs=args.min_obs, use_cache=not args.refresh)
    universe = [t for t in universe if t in panel.columns]
    is_mm = screen_market_makers(traders, panel, set(fetch_volume_wallets("all", 150)))
    logger.info("Panel %d×%d, pool %d (%d MM)", panel.shape[0], panel.shape[1],
                len(traders), sum(is_mm.values()))

    feat = build_trade_features(traders, panel, is_mm, fwd=args.fwd)
    if feat.empty:
        raise SystemExit("No fills with priced forward drift.")
    logger.info("Trade features: %d fills, %d wallets, %d tokens",
                len(feat), feat["wallet"].nunique(), feat["asset"].nunique())

    tables = summarize_quality(feat)
    base_m = feat["dir_fwd"].mean()
    base_h = feat["resolved_correct"].dropna().mean()

    print("\n" + "═" * 72)
    print(f"  POLYMARKET TRADE QUALITY  ·  {len(feat):,} non-MM fills, "
          f"fwd={args.fwd}d directional drift")
    print(f"  baseline: dir_fwd {base_m:+.4f}   resolution hit-rate {base_h:.3f}")
    print("═" * 72)
    for tell, df in tables.items():
        show = df.copy()
        show["dir_fwd"] = show["dir_fwd"].map(lambda v: f"{v:+.4f}")
        show["t"] = show["t"].map(lambda v: f"{v:+.1f}")
        show["hit_rate"] = show["hit_rate"].map(lambda v: f"{v:.3f}")
        print(f"\n  ▸ {tell}")
        print(tabulate(show.values, headers=["bucket", "n", "dir_fwd", "t", "hit"],
                       tablefmt="rounded_grid"))

    print("\n✓  Polymarket trade-quality study complete.")


if __name__ == "__main__":
    main()
