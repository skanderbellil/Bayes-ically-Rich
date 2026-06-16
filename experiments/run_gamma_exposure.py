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

    python experiments/run_gamma_exposure.py                  # SPX via CBOE CDN (no key)
    python experiments/run_gamma_exposure.py --ticker SPY --source yfinance
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
    ap.add_argument("--ticker", default="SPX", help="SPX (index, drives market GEX) or an ETF like SPY")
    ap.add_argument("--source", default="cboe", choices=["cboe", "yfinance"],
                    help="cboe = exchange greeks+OI (no key); yfinance = fallback")
    ap.add_argument("--max-expiries", type=int, default=12, help="(yfinance only)")
    ap.add_argument("--moneyness", type=float, default=0.10)
    ap.add_argument("--log", action="store_true",
                    help="append this snapshot to datasets/gex_snapshots.csv "
                         "(run daily to accumulate a backtestable GEX history)")
    args = ap.parse_args()

    logger.info("Fetching %s options chain (%s) and computing dealer GEX …",
                args.ticker, args.source)
    snap = fetch_gamma_snapshot(
        args.ticker, source=args.source,
        max_expiries=args.max_expiries, moneyness=args.moneyness,
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

    if args.log:
        # Free historical GEX isn't easily available, so the practical path is
        # to log one snapshot per (trading) day and let a history accumulate.
        import pandas as pd
        hist_csv = _bootstrap.ROOT / "datasets" / "gex_snapshots.csv"
        row = pd.DataFrame([{
            "date": pd.Timestamp.utcnow().normalize().tz_localize(None),
            "ticker": args.ticker, "source": args.source, "spot": snap.spot,
            "total_gex": snap.total_gex, "flip_level": snap.flip_level,
            "regime": snap.regime.split()[0], "n_contracts": snap.n_contracts,
        }])
        if hist_csv.exists():
            prev = pd.read_csv(hist_csv)
            # one row per ticker per day — overwrite a same-day re-run
            prev = prev[~((prev["date"] == str(row["date"].iloc[0])) &
                          (prev["ticker"] == args.ticker))]
            row = pd.concat([prev, row], ignore_index=True)
        row.to_csv(hist_csv, index=False)
        logger.info("Logged snapshot → %s (%d rows)", hist_csv, len(row))

    print("\n  Note: this is a point-in-time snapshot — free *historical* options")
    print("  data is scarce. Two ways to get a backtestable GEX series:")
    print("   • run this daily with --log to accumulate datasets/gex_snapshots.csv;")
    print("   • or load a historical chain dataset (OptionsDX / CBOE EOD) and call")
    print("     research.gamma.gex_time_series to build a daily GEX series, then")
    print("     condition next-day vol / returns on the gamma regime.")
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
