#!/usr/bin/env python3
"""
Polymarket — how informative are prices, and which domains are (in)efficient?
=============================================================================

A descriptive study of price *information efficiency* rather than a hunt for a
trade. Using the Brier score (mean squared error of the price as a probability
forecast), it asks:

  1. Do prices converge — does Brier fall as resolution approaches?
  2. Which topics are efficient? Brier + Brier-skill-vs-base-rate per category.
  3. Where is the price barely better than knowing the base rate (skill ≈ 0)?

skill = 1 − brier_price / brier_baseline, baseline = predict the group's Yes-rate.
skill → 1 means the price is highly informative; skill ≈ 0 means it adds nothing
over the base rate. Computed over each market's full life (no forward-horizon drop).

Usage
-----
  python experiments/run_polymarket_efficiency.py
  python experiments/run_polymarket_efficiency.py --refresh --n-markets 600
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import (
    brier_skill,
    build_brier_frame,
    build_price_panel,
    classify_markets,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DBINS = [0, 3, 7, 14, 30, 60, 120, 9999]
DLAB = ["0-3", "3-7", "7-14", "14-30", "30-60", "60-120", "120+"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=600)
    ap.add_argument("--min-volume", type=float, default=20_000)
    ap.add_argument("--horizon",    type=int,   default=30, help="days-out snapshot for the ranking")
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   price information efficiency    ║
║   Brier convergence + which domains are (in)efficient       ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    cat = classify_markets(meta)
    fr = build_brier_frame(panel, outcomes)
    fr["category"] = fr["market"].map(cat.to_dict())
    logger.info("Brier frame: %d market-days across %d markets", len(fr), fr["market"].nunique())

    # ── 1. convergence: Brier vs days-to-resolution ───────────────────────
    fr["dband"] = pd.cut(fr["days_left"], DBINS, labels=DLAB)
    conv = fr.groupby("dband", observed=True).agg(
        n=("brier", "size"), brier=("brier", "mean"),
        decided=("price", lambda s: ((s > 0.9) | (s < 0.1)).mean()))
    print("\n" + "═" * 58)
    print("  CONVERGENCE  ·  Brier vs days-to-resolution (lower = more informative)")
    print("═" * 58)
    print(tabulate([[i, int(r["n"]), f"{r['brier']:.3f}", f"{r['decided']:.0%}"]
                    for i, r in conv.iterrows()],
                   headers=["days left", "n", "Brier", "% decided"], tablefmt="rounded_grid"))

    # ── 2. efficiency by category at a fixed horizon ──────────────────────
    snap = fr[(fr["days_left"] > args.horizon) & (fr["days_left"] <= args.horizon * 2)]
    rows = []
    for c in cat.value_counts().index:
        sub = snap[snap["category"] == c]
        if sub["market"].nunique() < 5:
            continue
        sk = brier_skill(sub)
        rows.append([c, sk["n_markets"], f"{sk['base_rate']:.2f}",
                     f"{sk['brier_price']:.3f}", f"{sk['brier_base']:.3f}",
                     f"{sk['skill']:+.1%}"])
    rows.sort(key=lambda r: -float(r[5].rstrip("%").replace("+", "")))
    print("\n" + "═" * 74)
    print(f"  EFFICIENCY BY DOMAIN  ·  snapshot {args.horizon}-{args.horizon*2}d before resolution")
    print("═" * 74)
    print(tabulate(rows, headers=["category", "n_mkt", "base Yes", "Brier(price)",
                                  "Brier(base)", "skill"], tablefmt="rounded_grid"))
    print("  skill = 1 − Brier(price)/Brier(base)  ·  →100% perfectly informative, ≈0% no better than base rate")

    # ── plot: convergence curves per category ─────────────────────────────
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        centers = [1.5, 5, 10.5, 22, 45, 90, 160]
        for c in ["sports", "macro", "politics", "crypto", "geopolitics"]:
            sub = fr[fr["category"] == c]
            if sub["market"].nunique() < 5:
                continue
            b = sub.groupby("dband", observed=True)["brier"].mean().reindex(DLAB)
            ax.plot(centers, b.values, "o-", lw=1.8, label=c)
        ax.set_xscale("log")
        ax.set_xlabel("days to resolution (log)"); ax.set_ylabel("Brier score (lower = more informative)")
        ax.set_title("Price information by domain & time-to-resolution")
        ax.legend(); ax.grid(alpha=0.25, which="both")
        ax.invert_xaxis()
        fig.tight_layout()
        out = RESULTS_DIR / "efficiency.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Information-efficiency study complete.")


if __name__ == "__main__":
    main()
