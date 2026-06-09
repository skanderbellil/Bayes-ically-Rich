#!/usr/bin/env python3
"""
Dealer gamma exposure (GEX) and the effect on equities.

Computes a current dealer-GEX snapshot for an index ETF from the live options
chain (Black-Scholes gamma × open interest, calls long / puts short):

  - total GEX and the long/short-gamma regime,
  - the zero-gamma flip level (where dealer hedging flips from stabilising to
    destabilising) and the spot's distance to it,
  - the gamma profile by strike.

⚠️  This is a SNAPSHOT.  A GEX *backtest* needs historical open-interest-by-
strike, which yfinance does not provide — see the note at the end for how to
extend it with a historical options dataset via ``research.gamma.chain_gex``.

    python experiments/run_gamma_exposure.py --ticker SPY     # network (live chain)
"""
import argparse
import logging
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.research.gamma import fetch_gamma_snapshot

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--max-expiries", type=int, default=12)
    ap.add_argument("--moneyness", type=float, default=0.25)
    args = ap.parse_args()

    logger.info("Fetching %s options chain and computing dealer GEX …", args.ticker)
    snap = fetch_gamma_snapshot(
        args.ticker, max_expiries=args.max_expiries, moneyness=args.moneyness
    )

    gex_bn = snap.total_gex / 1e9
    print("\n" + "=" * 70)
    print(f"  DEALER GAMMA EXPOSURE  ·  {args.ticker}")
    print("=" * 70)
    print(f"  spot                : {snap.spot:,.2f}")
    print(f"  total GEX           : ${gex_bn:+.2f}bn  per 1% move")
    print(f"  regime              : {snap.regime}")
    if snap.flip_level is not None:
        dist = (snap.spot / snap.flip_level - 1) * 100
        print(f"  zero-gamma flip     : {snap.flip_level:,.2f}  "
              f"(spot is {dist:+.1f}% {'above' if dist >= 0 else 'below'} it)")
    else:
        print("  zero-gamma flip     : not found within ±20% band")
    print(f"  contracts in window : {snap.n_contracts}")

    print("\n  Interpretation:")
    if snap.total_gex >= 0:
        print("    Long gamma — dealers sell strength / buy weakness → vol suppression,")
        print("    mean-reverting tape.  A break below the flip level would flip this.")
    else:
        print("    Short gamma — dealers buy strength / sell weakness → vol amplification,")
        print("    trend-prone / fragile tape.  Reclaiming the flip level would calm it.")

    # ── Plot the gamma profile by strike ─────────────────────────────────────
    bs = snap.by_strike
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = np.where(bs["gex"].values >= 0, "tab:green", "tab:red")
    ax.bar(bs.index, bs["gex"].values / 1e9, width=(bs.index[1] - bs.index[0]) * 0.9
           if len(bs) > 1 else 1.0, color=colors, alpha=0.7)
    ax.axvline(snap.spot, color="black", lw=1.2, label=f"spot {snap.spot:,.0f}")
    if snap.flip_level is not None:
        ax.axvline(snap.flip_level, color="tab:blue", ls="--", lw=1.2,
                   label=f"flip {snap.flip_level:,.0f}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xlabel("strike"); ax.set_ylabel("dealer GEX ($bn / 1% move)")
    ax.set_title(f"{args.ticker} dealer gamma profile  ·  total {gex_bn:+.2f}bn "
                 f"({snap.regime.split()[0]})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / f"gamma_exposure_{args.ticker}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)

    print("\n  Note: this is a point-in-time snapshot. To backtest GEX → equity")
    print("  dynamics, feed a history of option chains (strike, OI, IV, expiry)")
    print("  into research.gamma.chain_gex to build a daily GEX series, then")
    print("  study next-day realised vol / returns conditioned on the regime.")
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
