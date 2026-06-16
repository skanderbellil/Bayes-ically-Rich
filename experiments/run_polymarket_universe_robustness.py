#!/usr/bin/env python3
"""
Polymarket — is the [0.15,0.30] mispricing real, or a 2024-election artifact?
=============================================================================

The only robust positive PnL we found (`BAND_CONTROL.md`) was buying Yes in the
[0.15,0.30] price band and holding to resolution — the favorite–longshot
*calibration* of mid-low-probability markets. But the top-volume resolved universe
is dominated by the 2024 US-election complex, so that "structural edge" might be a
single-event artifact. This study pulls a larger, more diverse universe, splits it
by topic (politics / sports / crypto / macro / geopolitics / culture / other), and
re-runs the band calibration and the hold-to-resolution trade **per category**.

If the [0.15,0.30] Yes-underpricing survives with politics *removed*, it's a real
structural feature of prediction-market pricing. If it collapses, it was 2024.

Usage
-----
  python experiments/run_polymarket_universe_robustness.py --refresh --n-markets 600
  python experiments/run_polymarket_universe_robustness.py --min-volume 20000
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
    build_event_table,
    build_price_panel,
    classify_markets,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.polymarket.trades import TradeParams, simulate_event_trades

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BAND = (0.15, 0.30)   # the band where the calibration edge lived


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=600)
    ap.add_argument("--min-volume", type=float, default=20_000)
    ap.add_argument("--slippage",   type=float, default=0.01)
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   universe robustness            ║
║   does the [0.15,0.30] mispricing survive outside politics? ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    cat = classify_markets(meta)                     # indexed by market id
    events = build_event_table(panel, outcomes)
    events["category"] = events["market"].map(cat.to_dict())
    lo, hi = BAND
    band_ev = events[(events["price"] >= lo) & (events["price"] < hi)]

    # ── universe composition ──────────────────────────────────────────────
    comp = []
    for c in cat.value_counts().index:
        ids = cat.index[cat == c]
        labelled = [i for i in ids if i in outcomes.index]
        yes = outcomes.reindex(labelled).mean() if labelled else float("nan")
        comp.append([c, int((cat == c).sum()), len(labelled),
                     f"{yes:.1%}" if labelled else "—",
                     int((band_ev["category"] == c).sum())])
    print("\n" + "═" * 70)
    print(f"  UNIVERSE COMPOSITION  ({panel.shape[1]} markets, {len(outcomes)} labelled)")
    print("═" * 70)
    print(tabulate(comp, headers=["category", "markets", "labelled",
                                  "base Yes", f"[{lo},{hi}) episodes"],
                   tablefmt="rounded_grid"))

    # ── headline: [0.15,0.30] calibration + hold-to-resolution PnL by group ─
    groups = {
        "ALL":            cat.index,
        "politics":       cat.index[cat == "politics"],
        "non-politics":   cat.index[cat != "politics"],
    }
    rows = []
    for name, ids in groups.items():
        ids = [i for i in ids if i in panel.columns]
        ev = band_ev[band_ev["market"].isin(ids)]
        if len(ev) < 10:
            rows.append([name, len(ev), "—", "—", "—", "—", "—"]); continue
        mean_p   = ev["price"].mean()
        yes_rate = ev["outcome"].mean()
        resid    = yes_rate - mean_p
        # the actual trade: buy Yes in-band, all vol, hold to resolution
        sub = panel[ids]
        tr = simulate_event_trades(sub, outcomes, TradeParams(
            entry_quantile=0.0, side="yes_only", exit="resolution",
            slippage=args.slippage, min_price=lo, max_price=hi)).summary
        rows.append([
            name, int(len(ev)), f"{mean_p:.2f}", f"{yes_rate:.1%}", f"{resid:+.1%}",
            tr.get("n_trades", 0),
            f"{tr.get('avg_pnl', float('nan')):+.3f}" if tr.get("n_trades") else "—",
        ])

    print("\n" + "═" * 78)
    print(f"  [{lo},{hi}] CALIBRATION & HOLD-TO-RESOLUTION PnL  ·  by group (slippage {args.slippage:.0%})")
    print("═" * 78)
    print(tabulate(rows, headers=["group", "episodes", "mean price", "yes rate",
                                  "calib resid", "n trades", "avg PnL/trade"],
                   tablefmt="rounded_grid"))
    print(f"  calib resid = yes_rate − mean_price (the favorite–longshot underpricing)")
    print(f"  avg PnL/trade > 0 ⇒ the structural edge is tradable in that group")

    # per-category calibration residual in the band (for the chart / detail)
    detail = []
    for c in cat.value_counts().index:
        ev = band_ev[band_ev["category"] == c]
        if len(ev) < 20:
            continue
        detail.append((c, len(ev), ev["price"].mean(), ev["outcome"].mean(),
                       ev["outcome"].mean() - ev["price"].mean()))

    if not args.no_plots and detail:
        cats = [d[0] for d in detail]
        resid = [d[4] for d in detail]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax.bar(cats, resid, color=["tab:green" if r > 0 else "tab:red" for r in resid])
        for i, d in enumerate(detail):
            ax.text(i, d[4], f"n={d[1]}", ha="center",
                    va="bottom" if d[4] >= 0 else "top", fontsize=8)
        ax.set_title(f"[{lo},{hi}] calibration residual (yes_rate − price) by category")
        ax.set_ylabel("calibration residual"); ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        out = RESULTS_DIR / "universe_robustness.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Universe robustness study complete.")


if __name__ == "__main__":
    main()
