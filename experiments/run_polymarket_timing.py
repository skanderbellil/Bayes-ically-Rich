#!/usr/bin/env python3
"""
Polymarket pre-resolution timing — is informed timing a *persistent* wallet skill?
=================================================================================

Some wallets buy right before favourable moves. The question that decides whether
that is exploitable (vs luck, or vs leaked information you can't legally act on)
is **persistence**: does a wallet that timed well in the first half of its history
keep timing well in the second half?

We score each wallet's **lead** — the mean forward directional drift of its fills
(how reliably its trades *precede* moves in their favour) — then run a strict
split-half test: split each wallet's fills at its own median trade time, score the
lead in each half, and correlate across wallets. A positive cross-half correlation
is the signature of a real, repeatable timing skill; ~zero means the leaderboard
of "great timers" is mostly noise.

Validated on the **large** universe. Reuses the per-fill forward-drift from the
trade-quality study.

Usage
-----
  python experiments/run_polymarket_timing.py
  python experiments/run_polymarket_timing.py --fwd 5 --refresh
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
    timing_persistence,
    wallet_lead_scores,
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
    ap.add_argument("--fwd",        type=int, default=5, help="forward drift horizon (days)")
    ap.add_argument("--min-fills",  type=int, default=30, help="min fills to rank a wallet")
    ap.add_argument("--refresh",    action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   pre-resolution timing         ║
║   is informed timing a persistent, exploitable skill?      ║
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

    feat = build_trade_features(traders, panel, is_mm, fwd=args.fwd)
    if feat.empty:
        raise SystemExit("No fills with priced forward drift.")
    logger.info("Trade features: %d fills, %d wallets", len(feat), feat["wallet"].nunique())

    lead = wallet_lead_scores(feat, min_fills=args.min_fills)
    persist = timing_persistence(feat, min_each=max(10, args.min_fills // 2))

    print("\n" + "═" * 70)
    print(f"  POLYMARKET PRE-RESOLUTION TIMING  ·  fwd={args.fwd}d, {len(feat):,} fills")
    print("═" * 70)

    print(f"\n  Top timers by lead score (mean forward {args.fwd}d drift, n≥{args.min_fills}):")
    top = lead.head(8).copy()
    top["wallet"] = top["wallet"].str[:12]
    top["lead"] = top["lead"].map(lambda v: f"{v:+.4f}")
    top["t"] = top["t"].map(lambda v: f"{v:+.1f}")
    top["hit"] = top["hit"].map(lambda v: f"{v:.3f}")
    print(tabulate(top.values, headers=["wallet", "n", "lead", "t", "hit"], tablefmt="rounded_grid"))

    print("\n  ── PERSISTENCE (the decisive test) ──")
    print(f"  wallets with both halves scorable: {persist.get('n_wallets', 0)}")
    if "pearson" in persist:
        print(f"  early→late lead correlation:  Pearson {persist['pearson']:+.3f}   "
              f"Spearman {persist['spearman']:+.3f}")
        print(f"  late lead | top-half-early timers:  {persist['late_given_top_early']:+.4f}")
        print(f"  late lead | bottom-half-early:      {persist['late_given_bot_early']:+.4f}")
        gap = persist["late_given_top_early"] - persist["late_given_bot_early"]
        # judge on the rank-based Spearman — Pearson is inflated by survivorship
        # outliers (a few wallets stuck in tokens that all resolved Yes).
        verdict = ("MODESTLY PERSISTENT (Spearman) — informed timing repeats, but weakly"
                   if persist["spearman"] > 0.15 and gap > 0
                   else "NOT persistent — 'great timers' are mostly luck/regime")
        print(f"  → {verdict}  (note: Pearson is outlier-inflated; trust Spearman)")
    else:
        print("  too few wallets with two scorable halves to test persistence.")

    print("\n✓  Polymarket pre-resolution timing study complete.")


if __name__ == "__main__":
    main()
