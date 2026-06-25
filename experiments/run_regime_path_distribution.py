#!/usr/bin/env python3
"""
Geo-calm sleeve — distribution of price paths, winners vs losers
================================================================

Plots the intra-trade price PATHS of every market inside the strategy bounds
(geopolitics · YES · price<=--max-price · GPR-calm · hold 2-45d), split by how
they resolved (WON -> $1 vs LOST -> $0). This is the picture behind every exit
finding: it shows whether winners and losers are separable while the trade is
live, and where in the path that separation happens.

Construction
------------
Each trade's daily path (committed cache) is placed on a NORMALISED time axis,
0 = entry (decision_date), 1 = settlement (res_date), so paths of different
durations overlay. The true settlement value (outcome) is appended at t=1 so
each path ends at its real $1/$0. Per group we draw the individual paths
(spaghetti) plus the median and 25-75% / 10-90% quantile bands — the
"distribution of paths".

Usage:
  python experiments/run_regime_path_distribution.py
  python experiments/run_regime_path_distribution.py --max-price 0.35 --grid 25
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book
from run_regime_take_profit import load_paths, PANEL, OUT


def normalised_path(seg, dec, res, outcome, grid):
    """Resample one daily path onto a common 0..1 time grid; settle at outcome."""
    tn = (seg["date"] - dec) / max((res - dec), pd.Timedelta(days=1))
    tn = tn.dt.total_seconds() / (24 * 3600) if hasattr(tn, "dt") else tn
    df = pd.DataFrame({"t": np.asarray(tn, float), "p": seg["p"].values})
    df = df[df["t"] < 1.0]                                  # drop the pre-settlement mark at t=1
    df = pd.concat([df, pd.DataFrame({"t": [1.0], "p": [float(outcome)]})], ignore_index=True)
    df = df.drop_duplicates("t").sort_values("t")
    if len(df) < 2:
        return None
    return np.interp(grid, df["t"].values, df["p"].values)


def band(mat):
    return (np.nanmedian(mat, 0), np.nanpercentile(mat, 25, 0), np.nanpercentile(mat, 75, 0),
            np.nanpercentile(mat, 10, 0), np.nanpercentile(mat, 90, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    ap.add_argument("--grid", type=int, default=25, help="number of normalised-time points")
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    paths = load_paths(book["token"].astype(str).tolist())
    grid = np.linspace(0.0, 1.0, args.grid)

    won, lost = [], []
    for _, r in book.iterrows():
        h = paths.get(str(r["token"]))
        if h is None:
            continue
        seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])]
        gp = normalised_path(seg, r["decision_date"], r["res_date"], r["outcome"], grid)
        if gp is None:
            continue
        (won if r["outcome"] == 1 else lost).append(gp)
    W, L = np.array(won), np.array(lost)

    print("=" * 84)
    print("GEO-CALM SLEEVE — PRICE-PATH DISTRIBUTION (winners vs losers)")
    print("  in-bounds book: geopolitics · YES · price<=%.2f · GPR-calm[%s] · hold 2-45d"
          % (args.max_price, args.gate))
    print("  paths: %d won, %d lost (normalised entry=0 -> settlement=1)" % (len(W), len(L)))
    print("=" * 84)
    wmed = band(W)[0]; lmed = band(L)[0]
    print("\n  normalised-time median price:")
    print("  %-6s %8s %8s" % ("t", "WON", "LOST"))
    for q in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        i = int(round(q * (args.grid - 1)))
        print("  %-6.2f %8.3f %8.3f" % (grid[i], wmed[i], lmed[i]))
    # separation diagnostics
    ever_mid_loser = np.mean([(L[k][:-1] >= 0.5).any() for k in range(len(L))]) if len(L) else np.nan
    ever_mid_winner = np.mean([(W[k][:-1] >= 0.5).any() for k in range(len(W))]) if len(W) else np.nan
    halfway_gap = wmed[args.grid // 2] - lmed[args.grid // 2]
    print("\n  separation:")
    print("   losers that EVER touch >=0.50 before settle: %.0f%%   (=> few stalls for TP to save)" % (100 * ever_mid_loser))
    print("   winners that touch >=0.50 before settle:     %.0f%%" % (100 * ever_mid_winner))
    print("   median price gap (won-lost) at halfway point: %+.3f" % halfway_gap)

    # ---- plot ----
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
    GW, GL = "#1f9d55", "#c0392b"
    # panel A: spaghetti + medians
    for k in range(len(W)):
        ax[0].plot(grid, W[k], color=GW, lw=0.7, alpha=0.28)
    for k in range(len(L)):
        ax[0].plot(grid, L[k], color=GL, lw=0.7, alpha=0.28)
    ax[0].plot(grid, wmed, color=GW, lw=2.6, label="WON median (n=%d)" % len(W))
    ax[0].plot(grid, lmed, color=GL, lw=2.6, label="LOST median (n=%d)" % len(L))
    ax[0].axhline(0.5, color="grey", ls=":", lw=.9); ax[0].axhline(args.max_price, color="grey", ls="--", lw=.7)
    ax[0].text(0.01, args.max_price + 0.01, "entry cap %.2f" % args.max_price, fontsize=7, color="grey")
    ax[0].set_title("All paths (entry=0 → settlement=1)"); ax[0].set_xlabel("fraction of holding period")
    ax[0].set_ylabel("price (P[YES])"); ax[0].set_ylim(-0.02, 1.02); ax[0].legend(fontsize=8, loc="center left")
    # panel B: quantile bands
    for G, col, lbl in [(W, GW, "WON"), (L, GL, "LOST")]:
        if not len(G):
            continue
        med, q25, q75, q10, q90 = band(G)
        ax[1].fill_between(grid, q10, q90, color=col, alpha=0.12)
        ax[1].fill_between(grid, q25, q75, color=col, alpha=0.25)
        ax[1].plot(grid, med, color=col, lw=2.6, label="%s median" % lbl)
    ax[1].axhline(0.5, color="grey", ls=":", lw=.9)
    ax[1].set_title("Path distribution: median, 25–75% & 10–90% bands"); ax[1].set_xlabel("fraction of holding period")
    ax[1].set_ylabel("price (P[YES])"); ax[1].set_ylim(-0.02, 1.02); ax[1].legend(fontsize=8, loc="center left")
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_path_distribution.png", dpi=120)
    pd.DataFrame({"t": grid, "won_median": wmed, "lost_median": lmed,
                  "won_q25": band(W)[1], "won_q75": band(W)[2],
                  "lost_q25": band(L)[1], "lost_q75": band(L)[2]}).to_csv(
        f"{OUT}/regime_path_distribution.csv", index=False)
    print("\nSaved: regime_path_distribution.png , regime_path_distribution.csv")


if __name__ == "__main__":
    main()
