#!/usr/bin/env python3
"""
Polymarket — dip-averaging strategy backtest
=============================================

The trajectory analysis showed ~53% of winners dip below entry before
resolving YES.  Can we exploit this?

Two questions answered here:

  Q1. Conditional probability: given a market in band has dipped X% by day D,
      is P(WIN) higher or lower than the unconditional win rate?
      → If dips are equally common in winners & losers, adding on dip is neutral.
      → If dips are more common in winners → average-down is alpha.
      → If dips are more common in losers  → dips are actually a SELL signal.

  Q2. Backtest of "add on dip" sizing vs flat hold:
      Base: $1 at entry (5d out), hold to resolution.
      Dip+:  $0.50 at entry + $0.50 more if price falls ≥ DIP_THRESH below
             entry at any point in days 1–3 (buy the dip).

Usage
-----
  python experiments/run_dip_strategy.py
  python experiments/run_dip_strategy.py --n-markets 800 --dip 0.10 0.15 0.20 0.25 0.30
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
            "norm": sp.values.astype(float) / p0,
            "outcome": float(m.outcome),
            "entry_price": p0,
        })
        if i % 50 == 0:
            logger.info("  built %d/%d (%d in band)", i, len(markets), len(paths))
        if sleep and i % 5 == 0:
            time.sleep(sleep)
    return paths


def sim_dip_add(path, outcome, entry_price, haircut, dip_thresh, add_day_max=3):
    """
    Simulate flat-hold vs dip-add for one market.

    flat: spend $1 at entry, hold to resolution.
    dip:  spend $0.50 at entry; if price falls ≥ dip_thresh below entry
          on any day 1..add_day_max, spend another $0.50 at that price.

    Returns (flat_pnl, dip_pnl, did_add).
    """
    entry = min(entry_price + haircut, 0.99)
    flat_pnl = outcome / entry - 1.0

    # dip strategy
    cost = 0.50 * entry
    units = 0.50 / entry
    did_add = False
    for d in range(1, min(add_day_max + 1, len(path))):
        px = path[d]
        if px <= entry_price * (1.0 - dip_thresh):
            add_px = min(px + haircut, 0.99)
            cost  += 0.50 * add_px
            units += 0.50 / add_px
            did_add = True
            break
    # if no dip triggered, still spent $0.50 (half position, hold to res)
    dip_pnl = units * outcome - cost   # net $PnL on $1 bankroll allocated

    return flat_pnl, dip_pnl, did_add


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=800)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.50])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--dip", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30])
    ap.add_argument("--add-day-max", type=int, default=3,
                    help="only add on dip within first N days after entry")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/paper_trade/dip_strategy.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/dip_strategy.png")
    args = ap.parse_args()

    lo, hi = args.band
    H = args.horizon

    print("""
╔══════════════════════════════════════════════════════════╗
║  DIP-AVERAGING STRATEGY BACKTEST                          ║
║  does buying the dip improve returns?                     ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  horizon {H}d · band [{lo:.2f},{hi:.2f}) · dip thresholds: "
          f"{[f'{d:.0%}' for d in args.dip]} · add window days 1–{args.add_day_max}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets\n")

    paths = build_paths(markets, H, lo, hi, args.sleep)
    if not paths:
        print("No qualifying paths — aborting.")
        return

    n_all = len(paths)
    n_win = sum(1 for p in paths if p["outcome"] == 1.0)
    base_wr = n_win / n_all
    print(f"  {n_all} markets in band  →  {n_win} winners ({base_wr:.0%} base win rate)\n")

    # ── Q1: conditional P(WIN) given dip by each day ─────────────────────────
    print("  ── Q1: Does a dip predict win or loss? ──")
    print(f"  {'dip≥':>6} | {'day 1':>8} | {'day 2':>8} | {'day 3':>8} | {'day 4':>8} | {'day 5':>8}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")

    cond_rows = []
    for thresh in args.dip:
        row = {"thresh": thresh}
        parts = []
        for day in range(1, H + 1):
            dipped = [p for p in paths if len(p["norm"]) > day and p["norm"][day] <= 1.0 - thresh]
            n_d = len(dipped)
            if n_d == 0:
                parts.append("  —  ")
                row[f"day{day}"] = float("nan")
            else:
                wr_d = sum(1 for p in dipped if p["outcome"] == 1.0) / n_d
                row[f"day{day}"] = round(wr_d, 3)
                delta = wr_d - base_wr
                parts.append(f"{wr_d:.0%}({delta:+.0%})")
        row["base_wr"] = round(base_wr, 3)
        cond_rows.append(row)
        print(f"  {thresh:.0%}    | " + " | ".join(f"{p:>8}" for p in parts))

    pd.DataFrame(cond_rows).to_csv(args.out, index=False)

    # ── Q2: flat vs dip-add backtest ─────────────────────────────────────────
    print(f"\n  ── Q2: Flat hold vs dip-add returns (add if dipped within days 1–{args.add_day_max}) ──")
    print(f"  {'thresh':>7} | {'flat mean':>10} | {'dip mean':>10} | {'flat Sh':>8} | "
          f"{'dip Sh':>8} | {'flat PnL':>9} | {'dip PnL':>9} | {'%added':>7}")
    print(f"  {'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")

    summary_rows = []
    for thresh in args.dip:
        flat_rets, dip_rets, added = [], [], []
        for p in paths:
            fr, dr, did = sim_dip_add(p["path"], p["outcome"], p["entry_price"],
                                       args.haircut, thresh, args.add_day_max)
            flat_rets.append(fr); dip_rets.append(dr); added.append(int(did))
        flat_rets = np.array(flat_rets)
        dip_rets  = np.array(dip_rets)
        f_sh = flat_rets.mean() / flat_rets.std(ddof=1) if flat_rets.std(ddof=1) > 0 else float("nan")
        d_sh = dip_rets.mean()  / dip_rets.std(ddof=1)  if dip_rets.std(ddof=1)  > 0 else float("nan")
        pct_added = sum(added) / len(added)
        print(f"  {thresh:.0%}     | {flat_rets.mean():>+9.1%} | {dip_rets.mean():>+9.1%} | "
              f"{f_sh:>8.2f} | {d_sh:>8.2f} | {flat_rets.sum():>+9.2f} | "
              f"{dip_rets.sum():>+9.2f} | {pct_added:>6.0%}")
        summary_rows.append({
            "thresh": thresh, "n": n_all, "base_wr": round(base_wr, 3),
            "flat_mean": round(float(flat_rets.mean()), 4),
            "dip_mean": round(float(dip_rets.mean()), 4),
            "flat_sharpe": round(float(f_sh), 3), "dip_sharpe": round(float(d_sh), 3),
            "flat_pnl": round(float(flat_rets.sum()), 2),
            "dip_pnl": round(float(dip_rets.sum()), 2),
            "pct_added": round(pct_added, 3),
        })

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Dip-averaging — long YES @ {H}d, band [{lo:.2f},{hi:.2f})",
                 fontsize=12, fontweight="bold")

    # Panel 1: conditional win rate by dip depth at day 1
    ax = axes[0]
    for i, p in enumerate(paths):
        norm = p["norm"]
        color = "#22c55e" if p["outcome"] == 1.0 else "#ef4444"
        alpha = 0.3 if len(paths) > 50 else 0.5
        ax.plot(range(len(norm)), norm, color=color, lw=0.8, alpha=alpha)
    ax.axhline(1.0, color="grey", lw=1.2, ls="--")
    ax.set_title("All price paths (green=win, red=loss)")
    ax.set_xlabel("Days since entry"); ax.set_ylabel("Price / entry")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color="#22c55e",lw=1.5,label=f"Win (n={n_win})"),
                        Line2D([0],[0],color="#ef4444",lw=1.5,label=f"Loss (n={n_all-n_win})")],
              fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: conditional P(WIN) given dip at day 1 vs day 3
    ax = axes[1]
    threshs = [t * 100 for t in args.dip]
    d1_wr, d3_wr = [], []
    for thresh in args.dip:
        d1_dip = [p for p in paths if len(p["norm"]) > 1 and p["norm"][1] <= 1.0 - thresh]
        d3_dip = [p for p in paths if len(p["norm"]) > 3 and
                  min(p["norm"][1:4]) <= 1.0 - thresh]
        d1_wr.append(sum(1 for p in d1_dip if p["outcome"]==1.0)/len(d1_dip) if d1_dip else float("nan"))
        d3_wr.append(sum(1 for p in d3_dip if p["outcome"]==1.0)/len(d3_dip) if d3_dip else float("nan"))
    ax.axhline(base_wr, color="grey", lw=1.5, ls="--", label=f"Base win rate {base_wr:.0%}")
    ax.plot(threshs, d1_wr, "o-", color="#3b82f6", lw=2, label="Dip by day 1")
    ax.plot(threshs, d3_wr, "s-", color="#f59e0b", lw=2, label="Dip by day 3")
    ax.set_title("P(WIN) conditional on dip depth")
    ax.set_xlabel("Dip threshold (%)"); ax.set_ylabel("Win rate")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    # Panel 3: flat vs dip Sharpe by threshold
    ax = axes[2]
    threshs = [r["thresh"] * 100 for r in summary_rows]
    flat_sh = [r["flat_sharpe"] for r in summary_rows]
    dip_sh  = [r["dip_sharpe"]  for r in summary_rows]
    x = np.arange(len(threshs))
    w = 0.35
    ax.bar(x - w/2, flat_sh, w, color="#94a3b8", label="Flat hold")
    ax.bar(x + w/2, dip_sh,  w, color="#06b6d4", label="Dip-add")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{t:.0f}%" for t in threshs])
    ax.set_title(f"Sharpe: flat hold vs dip-add (days 1–{args.add_day_max})")
    ax.set_xlabel("Dip threshold"); ax.set_ylabel("Sharpe ratio")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130, bbox_inches="tight")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out}")
    print("✓  Dip-averaging backtest complete.")


if __name__ == "__main__":
    main()
