#!/usr/bin/env python3
"""
Polymarket — can the sharp-move edge be *harvested*? Two exit policies.
======================================================================

The vol→outcome study found that after a sharp upside move a market (1) underprices
Yes — a positive *calibration* edge realised only at settlement — but (2) overreacts
short-term — negative forward drift. So the *exit policy* should decide everything:

  Idea 1 — detect Q4/Q5, take the side, HOLD TO RESOLUTION.  Monetises the
           calibration edge; the short-term overreaction is irrelevant because only
           the 0/1 settlement matters.
  Idea 2 — ride the DRIFT and bank winners early (take-profit / fixed horizon).
           Directly exposed to the overreaction, so expected to fare worse on the
           very Q5 spikes that revert.

Every trade pays realistic frictions: causal trigger, fill one bar AFTER the move
(at the elevated price), slippage on each book crossing, one position per market,
re-entry cooldown. PnL is per Yes share on $1 notional (probability units).

Usage
-----
  python experiments/run_polymarket_event_trades.py --refresh
  python experiments/run_polymarket_event_trades.py --metric sharp_up --slippage 0.02
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import build_price_panel, resolution_outcomes
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.polymarket.trades import TradeParams, simulate_event_trades

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=250)
    ap.add_argument("--min-volume", type=float, default=50_000)
    ap.add_argument("--metric",     type=str,   default="upside_vol",
                    choices=["upside_vol", "sharp_up", "vol_skew"])
    ap.add_argument("--slippage",   type=float, default=0.01, help="half-spread per crossing")
    ap.add_argument("--take-profit",type=float, default=0.10)
    ap.add_argument("--stop-loss",  type=float, default=0.25)
    ap.add_argument("--horizon",    type=int,   default=10)
    ap.add_argument("--min-price",  type=float, default=0.05, help="entry price band floor")
    ap.add_argument("--max-price",  type=float, default=0.95, help="entry price band ceiling")
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   harvesting the sharp-move edge  ║
║   Idea 1: hold to resolution   vs   Idea 2: trade the drift ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    logger.info("Panel %d×%d · %d labelled markets (base Yes %.1f%%)",
                *panel.shape, len(outcomes), 100 * outcomes.mean())

    m, slip = args.metric, args.slippage
    band = dict(min_price=args.min_price, max_price=args.max_price)
    configs = {
        # ── Idea 1: hold to resolution ──────────────────────────────────
        "res · Q5 · Yes":   TradeParams(m, 0.8, "yes_only",    "resolution", slippage=slip, **band),
        "res · Q4Q5 · Yes": TradeParams(m, 0.6, "yes_only",    "resolution", slippage=slip, **band),
        "res · Q5 · dir":   TradeParams(m, 0.8, "directional", "resolution", slippage=slip, **band),
        # ── Idea 2: trade the drift ─────────────────────────────────────
        "TP · Q5 · Yes":    TradeParams(m, 0.8, "yes_only", "take_profit", slippage=slip,
                                        take_profit=args.take_profit, stop_loss=args.stop_loss, **band),
        f"horizon{args.horizon} · Q5 · Yes":
                            TradeParams(m, 0.8, "yes_only", "horizon", slippage=slip,
                                        max_hold=args.horizon, **band),
    }

    results, rows = {}, []
    for name, p in configs.items():
        r = simulate_event_trades(panel, outcomes, p)
        results[name] = r
        s = r.summary
        if s.get("n_trades", 0) == 0:
            rows.append([name, 0, "—", "—", "—", "—", "—", "—"]); continue
        rows.append([
            name, s["n_trades"], f"{s['hit_rate']:>6.1%}",
            f"{s['avg_pnl']:>+7.3f}", f"{s['total_pnl']:>+7.2f}",
            f"{s['pnl_sharpe']:>6.3f}", f"{s['avg_entry']:>5.2f}",
            f"{s['yes_rate']:>6.1%}", f"{s['avg_hold']:>5.0f}",
        ])

    print("\n" + "═" * 92)
    print(f"  EVENT-TRADE HARVEST  ·  metric={m}, slippage={slip:.0%}/crossing, "
          f"PnL in prob-units per $1 notional")
    print("═" * 92)
    print(tabulate(rows,
                   headers=["config", "n", "hit", "avg PnL", "tot PnL", "trade\nSharpe",
                            "avg\nentry", "yes\nrate", "avg\nhold"],
                   tablefmt="rounded_grid"))
    print("  res = hold to resolution (Idea 1)   ·   TP/horizon = trade the drift (Idea 2)")
    print("  avg PnL > 0 ⇒ edge per trade survives entry slippage; trade-Sharpe = mean/std per trade")

    # ── Cumulative PnL by entry date ──────────────────────────────────────
    if not args.no_plots:
        fig, ax = plt.subplots(figsize=(11, 6))
        for name, r in results.items():
            if r.trades.empty:
                continue
            t = r.trades.sort_values("entry_date")
            ax.plot(t["entry_date"].values, t["pnl"].cumsum().values, lw=1.6, label=name)
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_title(f"Cumulative event-trade PnL — {m} (slippage {slip:.0%}/crossing)")
        ax.set_ylabel("cumulative PnL ($/​$1 notional)")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)
        fig.tight_layout()
        out = RESULTS_DIR / f"event_trades_{m}.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Event-trade harvest study complete.")


if __name__ == "__main__":
    main()
