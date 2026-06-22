#!/usr/bin/env python3
"""
Polymarket — winner price-path trajectory analysis
===================================================

For the mid-priced YES strategy (band [0.10,0.50], horizon 5d):
do winning markets (resolve YES) tend to drift up monotonically, or do many
dip below entry before recovering?  This matters for stop-loss design.

Key outputs
-----------
  1. Average normalised price path  : winners vs losers, day-by-day (entry=100%)
  2. Max-intra-period-drawdown dist : histogram for winners only
  3. Dip statistics                  : % of winners that ever go below entry,
                                       below 90%, 80%, 70% of entry
  4. "When is a winner obvious?"     : CDF of first day winner stays above entry
  5. Heatmap                         : distribution of per-day prices for winners

Usage
-----
  python experiments/run_winner_trajectory.py
  python experiments/run_winner_trajectory.py --n-markets 300 --horizon 5
"""
import argparse
import logging
import time

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.polymarket.fetch import fetch_markets, fetch_token_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def build_paths(markets, H, lo, hi, sleep):
    paths = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        s = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(s) < 3:
            continue
        t_res = s.index.max()
        t_entry = t_res - pd.Timedelta(days=H)
        grid = pd.date_range(t_entry, t_res, freq="D")
        sp = s.reindex(s.index.union(grid)).sort_index().ffill().reindex(grid)
        sp = sp.dropna()
        if len(sp) < 3:
            continue
        p0 = float(sp.iloc[0])
        if not (lo <= p0 < hi):
            continue
        paths.append({
            "path": sp.values.astype(float),
            "norm": sp.values.astype(float) / p0,  # normalised: entry=1.0
            "outcome": float(m.outcome),
            "entry_price": p0,
        })
        if i % 50 == 0:
            logger.info("  built %d/%d (%d in band)", i, len(markets), len(paths))
        if sleep and i % 5 == 0:
            time.sleep(sleep)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=300)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.50])
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out-csv", type=str, default="data/paper_trade/winner_trajectory.csv")
    ap.add_argument("--out-plot", type=str, default="data/paper_trade/winner_trajectory.png")
    args = ap.parse_args()

    lo, hi = args.band
    H = args.horizon

    print("""
╔══════════════════════════════════════════════════════════╗
║  WINNER TRAJECTORY ANALYSIS                               ║
║  do winners dip before resolving YES?                     ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  horizon {H}d · band [{lo:.2f},{hi:.2f}) · markets≤{args.n_markets}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets loaded\n")

    paths = build_paths(markets, H, lo, hi, args.sleep)
    if not paths:
        print("No qualifying paths — aborting.")
        return

    winners = [p for p in paths if p["outcome"] == 1.0]
    losers  = [p for p in paths if p["outcome"] == 0.0]
    n_w, n_l = len(winners), len(losers)
    print(f"  {len(paths)} markets in band  →  {n_w} winners ({n_w/len(paths):.0%})  "
          f"/ {n_l} losers ({n_l/len(paths):.0%})\n")

    # ── 1. Normalised average paths ───────────────────────────────────────────
    # Pad all paths to length H+1 by repeating the last value (resolution)
    def stack_norms(group, L):
        arr = np.full((len(group), L), np.nan)
        for i, p in enumerate(group):
            n = p["norm"]
            arr[i, :len(n)] = n[:L]
            if len(n) < L:
                arr[i, len(n):] = n[-1]
        return arr

    L = H + 1
    w_arr = stack_norms(winners, L)
    l_arr = stack_norms(losers,  L)

    # ── 2. Intra-period drawdown for winners ──────────────────────────────────
    # Max drawdown from entry: min(norm[1:]) - 1.0  (entry=day0)
    w_min  = np.array([p["norm"][1:].min() for p in winners])  # lowest ratio seen
    w_maxdd = w_min - 1.0   # negative = dropped below entry

    dip_any    = (w_min < 1.00).mean()
    dip_90     = (w_min < 0.90).mean()
    dip_80     = (w_min < 0.80).mean()
    dip_70     = (w_min < 0.70).mean()
    dip_50     = (w_min < 0.50).mean()
    med_drop   = float(np.median(w_maxdd))
    mean_drop  = float(np.mean(w_maxdd))

    print("  ── Intra-period dip statistics (WINNERS only) ──")
    print(f"    Ever below entry           : {dip_any:.0%} of winners")
    print(f"    Below 90% of entry (−10%)  : {dip_90:.0%}")
    print(f"    Below 80% of entry (−20%)  : {dip_80:.0%}")
    print(f"    Below 70% of entry (−30%)  : {dip_70:.0%}")
    print(f"    Below 50% of entry (−50%)  : {dip_50:.0%}")
    print(f"    Median worst dip           : {med_drop:+.1%}")
    print(f"    Mean   worst dip           : {mean_drop:+.1%}")

    # ── 3. "When is the winner obvious?" ─────────────────────────────────────
    # For each winner, first day (after day-0) where price never falls below entry again
    # Simpler proxy: first day price > entry
    days_first_above = []
    for p in winners:
        n = p["norm"]
        above = np.where(n[1:] > 1.0)[0]
        days_first_above.append(int(above[0]) + 1 if len(above) else H)
    days_first_above = np.array(days_first_above)

    print(f"\n  ── When does the winner's price first exceed entry? ──")
    for d in range(1, H + 1):
        pct = (days_first_above <= d).mean()
        bar = "█" * int(pct * 30)
        print(f"    By day {d}: {pct:5.0%}  {bar}")

    # ── Save per-market summary ───────────────────────────────────────────────
    rows = []
    for p in paths:
        n = p["norm"]
        min_norm = n[1:].min() if len(n) > 1 else n[0]
        max_norm = n[1:].max() if len(n) > 1 else n[0]
        rows.append({"outcome": p["outcome"], "entry_price": round(p["entry_price"], 4),
                     "min_norm": round(float(min_norm), 4), "max_norm": round(float(max_norm), 4),
                     "worst_dip": round(float(min_norm - 1.0), 4),
                     "dipped_below_entry": int(min_norm < 1.0)})
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    days = np.arange(L)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Winner trajectory analysis — long YES @ {H}d, band [{lo:.2f},{hi:.2f})",
                 fontsize=13, fontweight="bold")

    # Panel 1: average normalised path
    ax = axes[0, 0]
    w_mean = np.nanmean(w_arr, axis=0)
    w_p25  = np.nanpercentile(w_arr, 25, axis=0)
    w_p75  = np.nanpercentile(w_arr, 75, axis=0)
    l_mean = np.nanmean(l_arr, axis=0)
    l_p25  = np.nanpercentile(l_arr, 25, axis=0)
    l_p75  = np.nanpercentile(l_arr, 75, axis=0)
    ax.fill_between(days, w_p25, w_p75, alpha=0.18, color="#22c55e")
    ax.fill_between(days, l_p25, l_p75, alpha=0.18, color="#ef4444")
    ax.plot(days, w_mean, color="#22c55e", lw=2.2, label=f"Winners (n={n_w})")
    ax.plot(days, l_mean, color="#ef4444", lw=2.2, label=f"Losers  (n={n_l})")
    ax.axhline(1.0, color="grey", lw=0.9, ls="--", label="entry price")
    ax.set_title("Average normalised price path (entry = 1.0)")
    ax.set_xlabel("Days since entry"); ax.set_ylabel("Price / entry price")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 2: drawdown distribution for winners
    ax = axes[0, 1]
    ax.hist(w_maxdd * 100, bins=30, color="#06b6d4", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="grey", lw=1.2, ls="--")
    ax.axvline(med_drop * 100, color="#f59e0b", lw=1.8, label=f"Median {med_drop:.1%}")
    ax.set_title("Winners — worst intra-period dip from entry")
    ax.set_xlabel("Worst dip (%)"); ax.set_ylabel("# markets")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: heatmap of winner path distribution (day × price-bin)
    ax = axes[1, 0]
    bins = np.linspace(0, 2.5, 51)
    hmap = np.zeros((len(bins) - 1, L))
    for d in range(L):
        col = w_arr[:, d]
        col = col[~np.isnan(col)]
        if len(col):
            hmap[:, d], _ = np.histogram(col, bins=bins)
            hmap[:, d] /= hmap[:, d].sum()   # normalise to density
    im = ax.imshow(hmap, origin="lower", aspect="auto",
                   extent=[0, L - 1, bins[0] * 100, bins[-1] * 100],
                   cmap="YlOrRd", vmin=0, vmax=hmap.max())
    ax.axhline(100, color="white", lw=1.2, ls="--", label="entry = 100%")
    ax.set_title("Winners — price distribution by day (heatmap)")
    ax.set_xlabel("Days since entry"); ax.set_ylabel("Price (% of entry)")
    ax.legend(fontsize=9)
    plt.colorbar(im, ax=ax, label="density")

    # Panel 4: CDF of first day above entry
    ax = axes[1, 1]
    cdf = [(days_first_above <= d).mean() for d in range(0, H + 1)]
    ax.step(range(H + 1), cdf, where="post", color="#8b5cf6", lw=2.2)
    ax.fill_between(range(H + 1), cdf, step="post", alpha=0.15, color="#8b5cf6")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.set_title("Winners — CDF of first day price exceeds entry")
    ax.set_xlabel("Day"); ax.set_ylabel("Fraction of winners")
    ax.set_xticks(range(H + 1))
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=130, bbox_inches="tight")

    print(f"\n  Plot    → {args.out_plot}")
    print(f"  Summary → {args.out_csv}")
    print("✓  Winner trajectory analysis complete.")


if __name__ == "__main__":
    main()
