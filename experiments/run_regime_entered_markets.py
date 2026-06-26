#!/usr/bin/env python3
"""
Geo-calm sleeve — every market we entered, as small multiples
=============================================================

Plots the real daily price path of EACH market the strategy entered (the
geo-calm book: geopolitics · YES · price<=--max-price · GPR-calm · hold 2-45d),
one panel per market, from entry (day 0) to settlement. Line is green if it
resolved YES (won) and red if NO (lost); the entry price is marked, and the
0.50 "mids" / entry-cap guides are drawn. A companion overlay of all paths on a
normalised axis is saved too.

Usage:
  python experiments/run_regime_entered_markets.py
  python experiments/run_regime_entered_markets.py --sort outcome
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book
from run_regime_take_profit import load_paths, PANEL, OUT


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    ap.add_argument("--sort", choices=["date", "outcome", "price"], default="date")
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    if args.sort == "date":
        book = book.sort_values("decision_date")
    elif args.sort == "outcome":
        book = book.sort_values(["outcome", "decision_date"], ascending=[False, True])
    else:
        book = book.sort_values("price")
    book = book.reset_index(drop=True)
    paths = load_paths(book["token"].astype(str).tolist())

    n = len(book)
    ncol = 5
    nrow = math.ceil(n / ncol)
    GW, GL = "#1f9d55", "#c0392b"

    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.0, nrow * 1.9), squeeze=False)
    won = lost = 0
    for k, (_, r) in enumerate(book.iterrows()):
        ax = axes[k // ncol][k % ncol]
        h = paths.get(str(r["token"]))
        col = GW if r["outcome"] == 1 else GL
        won += int(r["outcome"] == 1); lost += int(r["outcome"] == 0)
        if h is not None:
            seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])].copy()
            if len(seg):
                days = (seg["date"] - r["decision_date"]).dt.days
                ax.plot(days, seg["p"], color=col, lw=1.6)
                ax.scatter([0], [r["price"]], color="black", s=14, zorder=5)
                # settlement marker
                ax.scatter([days.iloc[-1]], [r["outcome"]], color=col, marker="*", s=40, zorder=5)
        ax.axhline(0.5, color="grey", ls=":", lw=.7)
        ax.axhline(args.max_price, color="grey", ls="--", lw=.5)
        ax.set_ylim(-0.05, 1.05); ax.set_xlim(left=-0.5)
        q = str(r["question"])[:34]
        ax.set_title("%s\nentry %.2f · %s · %dd" % (q, r["price"], "WON" if r["outcome"] == 1 else "LOST",
                     (r["res_date"] - r["decision_date"]).days), fontsize=6.5)
        ax.tick_params(labelsize=6)
        if k % ncol == 0:
            ax.set_ylabel("P[YES]", fontsize=7)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    fig.suptitle("Every market the geo-calm sleeve entered (n=%d · %d won / %d lost) — green=won, red=lost"
                 % (n, won, lost), fontsize=11, y=1.0)
    fig.supxlabel("days since entry", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(f"{OUT}/regime_entered_markets.png", dpi=120, bbox_inches="tight")
    print("saved small-multiples: regime_entered_markets.png  (%d markets, %d won / %d lost)" % (n, won, lost))

    # companion: all paths overlaid on normalised time
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for _, r in book.iterrows():
        h = paths.get(str(r["token"]))
        if h is None:
            continue
        seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])]
        if len(seg) < 2:
            continue
        span = max((r["res_date"] - r["decision_date"]).total_seconds(), 86400.0)
        tn = (seg["date"] - r["decision_date"]).dt.total_seconds() / span
        ax2.plot(tn, seg["p"], color=GW if r["outcome"] == 1 else GL, lw=1.0, alpha=0.5)
    ax2.axhline(0.5, color="grey", ls=":", lw=.8); ax2.axhline(args.max_price, color="grey", ls="--", lw=.6)
    ax2.set_ylim(-0.03, 1.03); ax2.set_xlabel("fraction of holding period (entry=0 → settle=1)")
    ax2.set_ylabel("price (P[YES])")
    ax2.set_title("All entered markets overlaid (green=won, red=lost, n=%d)" % n)
    fig2.tight_layout(); fig2.savefig(f"{OUT}/regime_entered_markets_overlay.png", dpi=120)
    print("saved overlay: regime_entered_markets_overlay.png")


if __name__ == "__main__":
    main()
