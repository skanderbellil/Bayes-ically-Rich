#!/usr/bin/env python3
"""
Polymarket — dip strategies: average-down vs persistent-dip exit
================================================================

From the trajectory analysis: ~53% of winners dip below entry, but a day-1
dip barely changes P(WIN) while a dip that PERSISTS to day 2-3 collapses
win rate to ~20-27%.  Two strategies follow:

  dip_add   : enter half ($0.50) at day 0; if price drops ≥ DIP_THRESH in
              days 1–3, add the other half at that lower price.  Remaining
              cash earns 0 (no re-deployment if dip never triggers).
              Per $1 bankroll: winner+dip → more tokens at better price;
              no-dip → only half deployed (reduces both wins and losses).

  pers_exit : enter full $1 at day 0; if price is still below EXIT_FLOOR
              of entry at day CHECK_DAY (default day 2), exit at bid
              (price − haircut). Cuts persistent dippers, which are
              3:1 losers per Q1.

Comparison baseline: flat hold ($1 at entry, hold to resolution).

Usage
-----
  python experiments/run_dip_strategy.py
  python experiments/run_dip_strategy.py --n-markets 800 --check-day 2
  python experiments/run_dip_strategy.py --n-markets 800 --check-day 3
"""
import argparse
import logging
import time

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from posterioralpha.polymarket.fetch import fetch_markets, fetch_token_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DIP_THRESHOLDS  = [0.10, 0.15, 0.20, 0.25, 0.30]
EXIT_FLOORS     = [1.00, 0.90, 0.80, 0.70]   # fraction of entry; exit if price < floor*entry


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


def sim_flat(path, outcome, entry_price, haircut):
    """$1 at entry, hold to resolution."""
    entry = min(entry_price + haircut, 0.99)
    return outcome / entry - 1.0


def sim_dip_add(path, outcome, entry_price, haircut, dip_thresh, add_day_max=3):
    """
    $0.50 at entry; add $0.50 if price drops ≥ dip_thresh within days 1..add_day_max.
    Cash not deployed earns 0.  PnL on the full $1 bankroll.

    No add:  tokens=0.50/entry held to res.  Other $0.50 in cash.
             PnL = tokens × outcome + 0.50_cash - 1.0
                 = 0.50/entry × outcome + 0.50 - 1.0
                 = 0.50 × (outcome/entry - 1)

    With add at add_px:
             total_tokens = 0.50/entry + 0.50/add_px  (all $1 deployed)
             PnL = total_tokens × outcome - 1.0
    """
    entry = min(entry_price + haircut, 0.99)
    for d in range(1, min(add_day_max + 1, len(path))):
        if path[d] <= entry_price * (1.0 - dip_thresh):
            add_px = min(path[d] + haircut, 0.99)
            total_tokens = 0.50 / entry + 0.50 / add_px
            return total_tokens * outcome - 1.0, True
    # no dip triggered — half in cash
    return 0.50 * (outcome / entry - 1.0), False


def sim_pers_exit(path, outcome, entry_price, haircut, exit_floor, check_day):
    """
    $1 at entry.  On check_day, if price < exit_floor × entry_price → exit at bid.
    Otherwise hold to resolution.
    """
    entry = min(entry_price + haircut, 0.99)
    if len(path) > check_day:
        px = path[check_day]
        if px < exit_floor * entry_price:
            bid = max(px - haircut, 0.0)
            return bid / entry - 1.0, True
    return outcome / entry - 1.0, False


def sharpe(rets):
    r = np.array(rets)
    s = r.std(ddof=1)
    return float(r.mean() / s) if s > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=800)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.50])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--add-day-max", type=int, default=3)
    ap.add_argument("--check-day", type=int, default=2,
                    help="day on which persistent-dip exit is evaluated")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/paper_trade/dip_strategy.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/dip_strategy.png")
    args = ap.parse_args()

    lo, hi = args.band
    H = args.horizon

    print("""
╔══════════════════════════════════════════════════════════╗
║  DIP STRATEGIES: average-down vs persistent-dip exit     ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  horizon {H}d · band [{lo:.2f},{hi:.2f}) · add window days 1–{args.add_day_max} "
          f"· exit check day {args.check_day}\n")

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
    flat_rets = np.array([sim_flat(p["path"], p["outcome"], p["entry_price"], args.haircut)
                          for p in paths])
    flat_sh = sharpe(flat_rets)

    print(f"  {n_all} markets  |  {n_win} winners ({base_wr:.0%})  |  "
          f"flat Sharpe {flat_sh:.2f}  mean {flat_rets.mean():+.1%}  PnL {flat_rets.sum():+.2f}\n")

    # ── Q1: conditional P(WIN) by dip depth and day ───────────────────────────
    print("  ── Q1: P(WIN) conditional on dip (vs base {:.0%}) ──".format(base_wr))
    print(f"  {'dip≥':>5} | {'day 1':>10} | {'day 2':>10} | {'day 3':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for thresh in DIP_THRESHOLDS:
        parts = []
        for day in range(1, min(4, H + 1)):
            dipped = [p for p in paths if len(p["norm"]) > day
                      and p["norm"][day] <= 1.0 - thresh]
            if not dipped:
                parts.append("  —  ")
            else:
                wr_d = sum(1 for p in dipped if p["outcome"] == 1.0) / len(dipped)
                delta = wr_d - base_wr
                parts.append(f"{wr_d:.0%} ({delta:+.0%}) n={len(dipped)}")
        print(f"  {thresh:.0%}   | " + " | ".join(f"{p:>10}" for p in parts))

    # ── Q2: dip-add vs flat ───────────────────────────────────────────────────
    print(f"\n  ── Q2: Dip-add ($0.50 entry + $0.50 on dip, days 1–{args.add_day_max}) ──")
    print(f"  {'dip≥':>5} | {'mean ret':>9} | {'vs flat':>8} | {'Sharpe':>7} | "
          f"{'vs flat':>7} | {'PnL':>8} | {'%added':>7}")
    print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")

    dip_rows = []
    for thresh in DIP_THRESHOLDS:
        rets, added = [], []
        for p in paths:
            r, did = sim_dip_add(p["path"], p["outcome"], p["entry_price"],
                                 args.haircut, thresh, args.add_day_max)
            rets.append(r); added.append(int(did))
        rets = np.array(rets)
        sh = sharpe(rets)
        pct = sum(added) / len(added)
        print(f"  {thresh:.0%}   | {rets.mean():>+8.1%} | {rets.mean()-flat_rets.mean():>+7.1%} | "
              f"{sh:>7.2f} | {sh-flat_sh:>+6.2f} | {rets.sum():>+8.2f} | {pct:>6.0%}")
        dip_rows.append({"strategy": f"dip_add_{thresh:.0%}", "thresh": thresh,
                         "mean_ret": round(float(rets.mean()), 4),
                         "sharpe": round(sh, 3), "tot_pnl": round(float(rets.sum()), 2),
                         "pct_added": round(pct, 3)})

    # ── Q3: persistent-dip exit vs flat ──────────────────────────────────────
    print(f"\n  ── Q3: Persistent-dip exit (exit at day {args.check_day} if price < floor × entry) ──")
    print(f"  {'floor':>6} | {'mean ret':>9} | {'vs flat':>8} | {'Sharpe':>7} | "
          f"{'vs flat':>7} | {'PnL':>8} | {'%exited':>8}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}")

    exit_rows = []
    for floor in EXIT_FLOORS:
        rets, exited = [], []
        for p in paths:
            r, did = sim_pers_exit(p["path"], p["outcome"], p["entry_price"],
                                   args.haircut, floor, args.check_day)
            rets.append(r); exited.append(int(did))
        rets = np.array(rets)
        sh = sharpe(rets)
        pct = sum(exited) / len(exited)
        print(f"  {floor:.0%}   | {rets.mean():>+8.1%} | {rets.mean()-flat_rets.mean():>+7.1%} | "
              f"{sh:>7.2f} | {sh-flat_sh:>+6.2f} | {rets.sum():>+8.2f} | {pct:>7.0%}")
        exit_rows.append({"strategy": f"pers_exit_d{args.check_day}_{floor:.0%}",
                          "floor": floor, "check_day": args.check_day,
                          "mean_ret": round(float(rets.mean()), 4),
                          "sharpe": round(sh, 3), "tot_pnl": round(float(rets.sum()), 2),
                          "pct_exited": round(pct, 3)})

    # Save
    all_rows = (
        [{"strategy": "flat_hold", "mean_ret": round(float(flat_rets.mean()), 4),
          "sharpe": round(flat_sh, 3), "tot_pnl": round(float(flat_rets.sum()), 2)}]
        + dip_rows + exit_rows
    )
    pd.DataFrame(all_rows).to_csv(args.out, index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Dip strategies — long YES @ {H}d, band [{lo:.2f},{hi:.2f})",
                 fontsize=12, fontweight="bold")

    # Panel 1: all price paths
    ax = axes[0]
    for p in paths:
        c = "#22c55e" if p["outcome"] == 1.0 else "#ef4444"
        ax.plot(range(len(p["norm"])), p["norm"], color=c, lw=0.8, alpha=0.35)
    ax.axhline(1.0, color="grey", lw=1.2, ls="--", label="entry")
    ax.set_title("All price paths (green=win, red=loss)")
    ax.set_xlabel("Days since entry"); ax.set_ylabel("Price / entry")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color="#22c55e",lw=1.5,label=f"Win n={n_win}"),
                        Line2D([0],[0],color="#ef4444",lw=1.5,label=f"Loss n={n_all-n_win})")],
              fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: dip-add Sharpe vs flat
    ax = axes[1]
    x = [t * 100 for t in DIP_THRESHOLDS]
    dip_sharpes = [r["sharpe"] for r in dip_rows]
    ax.plot(x, dip_sharpes, "o-", color="#06b6d4", lw=2, label="Dip-add")
    ax.axhline(flat_sh, color="#94a3b8", lw=1.8, ls="--", label=f"Flat hold ({flat_sh:.2f})")
    ax.set_title(f"Dip-add Sharpe by add threshold\n(add if dip ≥ thresh in days 1–{args.add_day_max})")
    ax.set_xlabel("Add threshold (% below entry)"); ax.set_ylabel("Sharpe")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel 3: persistent-exit Sharpe vs flat
    ax = axes[2]
    x2 = [(1.0 - f) * 100 for f in EXIT_FLOORS]  # convert floor to dip depth
    exit_sharpes = [r["sharpe"] for r in exit_rows]
    exit_pcts    = [r["pct_exited"] for r in exit_rows]
    ax.plot(x2, exit_sharpes, "s-", color="#f59e0b", lw=2, label=f"Pers-exit (day {args.check_day})")
    ax.axhline(flat_sh, color="#94a3b8", lw=1.8, ls="--", label=f"Flat hold ({flat_sh:.2f})")
    for xi, sh_i, pct in zip(x2, exit_sharpes, exit_pcts):
        ax.annotate(f"{pct:.0%} exit", (xi, sh_i), textcoords="offset points",
                    xytext=(4, 6), fontsize=8)
    ax.set_title(f"Persistent-exit Sharpe by floor\n(exit on day {args.check_day} if price < floor × entry)")
    ax.set_xlabel("Dip depth at exit (% below entry)"); ax.set_ylabel("Sharpe")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130, bbox_inches="tight")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out}")
    print("✓  Dip strategy comparison complete.")


if __name__ == "__main__":
    main()
