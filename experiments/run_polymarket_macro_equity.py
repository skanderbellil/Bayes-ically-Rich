#!/usr/bin/env python3
"""
Polymarket — macro buy-leader strategy, $1,000 equity curve
===========================================================

Visualises the one surviving directional candidate from the whole Polymarket thread:
in **macro** multi-outcome fields the leader is materially under-priced (wins 93% at
mean price 0.73; +19.8¢/event, event-clustered t≈3 — see FIELD_SHAPE.md /
STRATEGY_SYNTHESIS.md). Here we hold the leader to resolution and compound a $1,000
bankroll, chronologically by resolution date, under several sizing schemes.

The bet pays **win ≈ +(1/p − 1) of stake, lose = −100% of stake**, so full-bankroll
betting is ruin on a single loss; fractional sizing is mandatory. Full-Kelly (~0.76)
is an artefact of a 93% win rate over 15 in-sample, one-regime events — the 10–20%
band is the honest zone, and even that trusts a win-rate that will not fully persist.

Because there is only one losing event, fixed-fraction outcomes are **order-invariant**
(multiplication commutes) and the worst single-event drawdown is exactly the bet
fraction — so the curve's shape is cosmetic but its endpoints are robust.

Usage
-----
  python experiments/run_polymarket_macro_equity.py [--start 1000]
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    build_price_panel,
    classify_markets,
    event_panels,
    field_shape_events,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def macro_fields(panel, meta, evp, df):
    """Subset the field table to macro events, with entry price, outcome and resolution date."""
    cat = classify_markets(meta)
    def ev_category(ev):
        cs = [cat.get(m) for m in evp[ev].columns if m in cat.index]
        cs = [c for c in cs if c]
        return pd.Series(cs).mode().iloc[0] if cs else "other"
    df = df.copy()
    df["cat"] = [ev_category(ev) for ev in df["event"]]
    mac = df[df["cat"] == "macro"].copy()
    mac["res_date"] = [evp[ev][evp[ev].mean().idxmax()].dropna().index[-1] for ev in mac["event"]]
    return mac.sort_values("res_date").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=2500)
    ap.add_argument("--min-volume", type=float, default=3_000)
    ap.add_argument("--start",      type=float, default=1000.0)
    ap.add_argument("--slip",       type=float, default=0.01)
    ap.add_argument("--refresh",    action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    out = resolution_outcomes(meta, panel)
    evp = event_panels(panel, meta, min_k=3)
    df = field_shape_events(evp, out)
    mac = macro_fields(panel, meta, evp, df)

    mac["ret"] = mac["fav_win"] / (mac["p_fav"] + args.slip) - 1.0
    logger.info("%d macro fields | win-rate %.2f | mean entry %.3f | mean per-stake return %+.3f",
                len(mac), mac.fav_win.mean(), mac.p_fav.mean(), mac.ret.mean())
    print(mac[["res_date", "p_fav", "fav_win", "ret"]].to_string(index=False))

    q, p = mac.fav_win.mean(), mac.p_fav.mean()
    kelly = max(0.0, q - (1 - q) / ((1 - p) / p))

    def curve(frac):
        bank, eq = args.start, [args.start]
        for r in mac["ret"]:
            bank *= (1 + frac * r); eq.append(bank)
        return np.array(eq)

    schemes = [("10% of bankroll", 0.10, "#2a9d8f"),
               ("20% of bankroll", 0.20, "#264653"),
               (f"quarter-Kelly ({0.25*kelly:.0%})", 0.25 * kelly, "#e9c46a")]

    x = [mac["res_date"].iloc[0] - pd.Timedelta(days=20)] + list(mac["res_date"])
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, f, c in schemes:
        eq = curve(f)
        dd = (eq / np.maximum.accumulate(eq) - 1).min()
        ax.plot(x, eq, marker="o", ms=4, lw=2, color=c, label=f"{name}  →  ${eq[-1]:,.0f}  (DD {dd:+.0%})")
    for _, r in mac[mac["fav_win"] < 0.5].iterrows():
        ax.axvline(r["res_date"], color="#e76f51", ls="--", lw=1, alpha=0.7)
        ax.text(r["res_date"], args.start * 0.55, " loss\n (favorite\n resolved No)",
                color="#e76f51", fontsize=8, va="top")
    ax.axhline(args.start, color="grey", lw=0.8, ls=":")
    ax.set_title(f"Macro buy-leader strategy — ${args.start:,.0f} starting bankroll\n"
                 f"{len(mac)} macro multi-outcome fields, leader held to resolution "
                 f"({args.slip:.0%} slippage), compounded by resolution date", fontsize=11)
    ax.set_ylabel("bankroll ($)"); ax.set_xlabel("resolution date")
    ax.legend(title="sizing (fraction of bankroll per bet)", fontsize=9)
    ax.grid(alpha=0.25); fig.autofmt_xdate(); fig.tight_layout()
    outp = RESULTS_DIR / "macro_equity.png"
    fig.savefig(outp, dpi=140)
    logger.info("Saved chart → %s  (full-Kelly≈%.2f — do NOT use; 15 in-sample events)", outp, kelly)
    print("\n✓  Macro equity curve complete.")


if __name__ == "__main__":
    main()
