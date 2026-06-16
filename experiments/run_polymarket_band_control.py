#!/usr/bin/env python3
"""
Polymarket — is the sharp-move edge real, or just a price-level artifact?
=========================================================================

The hold-to-resolution edge (`EVENT_TRADE_HARVEST.md`) entered around price 0.27,
and `upside_vol` is only high for markets that are alive — i.e. away from 0.01 —
so high-vol episodes carry a higher mean price. That raises the obvious objection:
maybe markets near 0.2–0.4 simply underprice Yes (favorite–longshot calibration),
and "upside vol" is just a proxy for "this market has a tradable price". This study
runs the decisive control: **within each price band**, hold price roughly fixed and
compare *high* upside-vol vs *low* upside-vol episodes.

  hi_minus_lo = Yes-rate(high vol) − Yes-rate(low vol),  at the same price band.

A positive, consistent `hi_minus_lo` ⇒ a genuine vol signal beyond price level.
A flat/zero one ⇒ the "edge" was the band's calibration all along.

Usage
-----
  python experiments/run_polymarket_band_control.py --refresh
  python experiments/run_polymarket_band_control.py --metric sharp_up
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import (
    band_matched_calibration,
    build_event_table,
    build_price_panel,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=250)
    ap.add_argument("--min-volume", type=float, default=50_000)
    ap.add_argument("--metric",     type=str,   default="upside_vol",
                    choices=["upside_vol", "sharp_up", "vol_skew"])
    ap.add_argument("--hi-quantile",type=float, default=0.70, help="within-band high-vol cutoff")
    ap.add_argument("--vol-window", type=int,   default=10)
    ap.add_argument("--fwd-horizon",type=int,   default=10)
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   level-controlled vol edge       ║
║   within each price band: high vol vs low vol → Yes-rate    ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    events = build_event_table(panel, outcomes, vol_window=args.vol_window,
                               fwd_horizon=args.fwd_horizon)
    if events.empty:
        logger.error("No episodes — try --refresh / larger --n-markets.")
        return

    tbl = band_matched_calibration(events, metric=args.metric, hi_quantile=args.hi_quantile)
    if tbl.empty:
        logger.error("No price band had enough episodes.")
        return

    rows = [[
        r["band"], int(r["n"]), int(r["n_hi"]),
        f"{r['price_hi']:>5.2f}/{r['price_lo']:<5.2f}",
        f"{r['yes_hi']:>6.1%}", f"{r['yes_lo']:>6.1%}",
        f"{r['hi_minus_lo']:>+6.1%}",
        f"{r['calib_hi']:>+6.1%}", f"{r['calib_lo']:>+6.1%}",
        f"{r['fwd_hi']:>+6.3f}", f"{r['fwd_lo']:>+6.3f}",
    ] for _, r in tbl.iterrows()]

    print("\n" + "═" * 96)
    print(f"  LEVEL-CONTROLLED VOL EDGE  ·  metric={args.metric}, "
          f"within-band high-vol cutoff = top {1 - args.hi_quantile:.0%}")
    print("═" * 96)
    print(tabulate(rows, headers=[
        "price band", "n", "n_hi", "price hi/lo", "yes hi", "yes lo",
        "hi−lo", "calib hi", "calib lo", "fwd hi", "fwd lo"], tablefmt="rounded_grid"))
    n_tot   = int(tbl["n"].sum())
    w_lift  = float(np.average(tbl["hi_minus_lo"], weights=tbl["n"]))
    print(f"  hi−lo = within-band Yes-rate lift of high vol over low vol (level-controlled signal)")
    print(f"  fwd = forward {args.fwd_horizon}d log-odds drift (+ momentum / − reversion)")
    print(f"\n  ➤ episode-weighted mean within-band lift: {w_lift:+.2%}  "
          f"({sum(tbl['hi_minus_lo'] > 0)}/{len(tbl)} bands positive, {n_tot} episodes)")
    if w_lift > 0.01:
        print("  ➤ the upside-vol edge SURVIVES the price-level control — a real vol signal.")
    elif w_lift < -0.01:
        print("  ➤ high vol resolves Yes LESS within band — the raw edge was a level artifact.")
    else:
        print("  ➤ within-band lift ≈ 0 — the raw edge was mostly the band's calibration.")

    if not args.no_plots:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
        x = np.arange(len(tbl))
        axL.bar(x - 0.2, tbl["yes_lo"], 0.4, label="low vol", color="tab:gray")
        axL.bar(x + 0.2, tbl["yes_hi"], 0.4, label="high vol", color="tab:green")
        axL.plot(x, (tbl["price_hi"] + tbl["price_lo"]) / 2, "k--o", lw=1, ms=4, label="mean price")
        axL.set_xticks(x); axL.set_xticklabels(tbl["band"], rotation=20, fontsize=8)
        axL.set_title(f"Yes-rate by price band: high vs low {args.metric}")
        axL.set_ylabel("Yes-rate"); axL.legend(fontsize=8); axL.grid(alpha=0.25, axis="y")

        axR.axhline(0, color="k", lw=0.6, alpha=0.5)
        axR.bar(x, tbl["hi_minus_lo"], color=["tab:green" if v > 0 else "tab:red"
                                              for v in tbl["hi_minus_lo"]])
        axR.set_xticks(x); axR.set_xticklabels(tbl["band"], rotation=20, fontsize=8)
        axR.set_title("within-band Yes-rate lift (high − low vol)")
        axR.set_ylabel("Δ Yes-rate"); axR.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        out = RESULTS_DIR / f"band_control_{args.metric}.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Level-controlled vol-edge study complete.")


if __name__ == "__main__":
    main()
