#!/usr/bin/env python3
"""
Polymarket — dynamic-exit long-YES backtest
============================================

Buy-and-hold mid-priced YES has a fat left tail: most cheap bets ride to zero
(median −100%), and only the winners pay.  This tests whether a *dynamic exit* —
cutting positions that fail to converge upward toward resolution — improves the
risk/return by chopping the doomed losers early.

For every market we reconstruct the daily YES price path from the entry horizon
(default 7d) to resolution and simulate several exit rules:

  hold        : ride to resolution (baseline)
  stop_30/50  : exit if price falls 30% / 50% below entry (fixed stop)
  trail_25    : exit if price falls 25% below its running peak since entry
  below_d3    : exit if price is below entry after 3 days (no early progress)
  decay       : DECAYING-HORIZON stop — the floor starts loose and ratchets up
                toward entry as resolution nears, so the position must
                increasingly be converging our way or it is cut.
                floor(t) = entry * (1 − s0 · days_left / total_days)

Entry pays +haircut (cross the spread), path exits pay −haircut (sell the bid);
resolution pays face.  Equal $1 per trade.

Usage
-----
  python experiments/run_dynamic_yes_backtest.py --n-markets 250 --horizon 7
  python experiments/run_dynamic_yes_backtest.py --band 0.10 0.50   # cheap end only
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


def simulate(path: np.ndarray, outcome: float, entry: float, haircut: float,
             rule: str, total_days: int) -> tuple[float, str]:
    """Return (pnl_per_$1, exit_kind) for one market under one exit rule.

    path[0] is the entry-day price; path[-1] is the last pre-resolution mark.
    Resolution payoff uses `outcome` (0/1). Path exits sell at price − haircut.
    exit_kind: "won" | "lost" (held to resolution) | "cut" (stop fired) | "tp" (take-profit fired)
    """
    peak = entry
    n = len(path)
    for d in range(1, n):
        px = path[d]
        peak = max(peak, px)
        days_left = total_days - d
        cut = tp = False

        if rule == "hold":
            pass
        # ── downside stops ────────────────────────────────────────────────
        elif rule == "stop_30":
            cut = px <= entry * 0.70
        elif rule == "stop_50":
            cut = px <= entry * 0.50
        elif rule == "trail_25":
            cut = px <= peak * 0.75
        elif rule == "below_d3":
            cut = (d >= 3) and (px < entry)
        elif rule == "decay":
            floor = entry * (1.0 - 0.5 * days_left / total_days)
            cut = px < floor
        # ── take-profit only ──────────────────────────────────────────────
        elif rule == "tp_2x":     # exit when price doubles from entry
            tp = px >= entry * 2.0
        elif rule == "tp_3x":     # exit when price triples from entry
            tp = px >= entry * 3.0
        elif rule == "tp_80pct":  # exit when market implies ≥80% YES
            tp = px >= 0.80
        elif rule == "tp_90pct":  # exit when market implies ≥90% YES
            tp = px >= 0.90
        # ── stop + take-profit combos ─────────────────────────────────────
        elif rule == "s30_tp2x":
            cut = px <= entry * 0.70
            tp  = px >= entry * 2.0
        elif rule == "s30_tp3x":
            cut = px <= entry * 0.70
            tp  = px >= entry * 3.0
        elif rule == "s30_tp80":
            cut = px <= entry * 0.70
            tp  = px >= 0.80
        elif rule == "s30_tp90":
            cut = px <= entry * 0.70
            tp  = px >= 0.90

        if cut or tp:
            exit_px = max(px - haircut, 0.0)
            return exit_px / entry - 1.0, "tp" if tp else "cut"

    # never triggered → hold to resolution
    return outcome / entry - 1.0, "won" if outcome == 1 else "lost"


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=250)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, default=7, help="entry days before resolution")
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.90])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/paper_trade/dynamic_yes.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/dynamic_yes.png")
    args = ap.parse_args()

    lo, hi = args.band
    H = args.horizon
    RULES = [
        "hold",
        # ── downside stops ──────────────────────────────────────────
        "stop_30", "stop_50", "trail_25", "below_d3", "decay",
        # ── take-profit only ────────────────────────────────────────
        "tp_2x", "tp_3x", "tp_80pct", "tp_90pct",
        # ── stop + take-profit combos ────────────────────────────────
        "s30_tp2x", "s30_tp3x", "s30_tp80", "s30_tp90",
    ]

    print("""
╔══════════════════════════════════════════════════════════╗
║  DYNAMIC-EXIT LONG-YES BACKTEST                           ║
║  cut positions that don't converge toward resolution      ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  entry {H}d out · band [{lo:.2f},{hi:.2f}] · haircut {args.haircut:.2f} "
          f"· markets≤{args.n_markets}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets\n")

    # Build per-market daily path over the entry window
    paths = []   # dict per qualifying market
    for i, m in enumerate(markets.itertuples(index=False), 1):
        s = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(s) < 3:
            continue
        t_res = s.index.max()
        t_entry = t_res - pd.Timedelta(days=H)
        # daily grid entry→resolution, forward-filled
        grid = pd.date_range(t_entry, t_res, freq="D")
        sp = s.reindex(s.index.union(grid)).sort_index().ffill().reindex(grid)
        sp = sp.dropna()
        if len(sp) < 3:
            continue
        p0 = float(sp.iloc[0])
        if not (lo <= p0 < hi):
            continue
        paths.append({"path": sp.values.astype(float), "outcome": float(m.outcome),
                      "t_res": t_res, "entry_price": p0})
        if i % 50 == 0:
            logger.info("  built %d/%d (%d in band)", i, len(markets), len(paths))
        time.sleep(args.sleep if i % 5 == 0 else 0)

    if not paths:
        print("No qualifying paths — aborting.")
        return
    paths.sort(key=lambda r: r["t_res"])
    print(f"  {len(paths)} markets entered in band\n")

    # Simulate each rule across all markets
    rows = []
    rule_curves = {}
    for rule in RULES:
        rets, kinds = [], []
        for pm in paths:
            entry = min(pm["entry_price"] + args.haircut, 0.99)
            r, kind = simulate(pm["path"], pm["outcome"], entry, args.haircut,
                               rule, total_days=H)
            rets.append(r); kinds.append(kind)
        rets = np.array(rets)
        kinds = np.array(kinds)
        eq = np.cumsum(rets)
        rule_curves[rule] = eq
        n_cut = int((kinds == "cut").sum())
        n_tp  = int((kinds == "tp").sum())
        held_outcomes = np.array([pm["outcome"] for pm in paths])
        cut_mask = kinds == "cut"
        cut_would_win = int(held_outcomes[cut_mask].sum()) if cut_mask.any() else 0
        rows.append({
            "rule": rule, "n": len(rets),
            "mean_ret": rets.mean(), "median_ret": float(np.median(rets)),
            "sharpe": rets.mean() / rets.std(ddof=1) if rets.std(ddof=1) > 0 else float("nan"),
            "tot_pnl": rets.sum(), "maxdd": max_drawdown(eq),
            "n_cut": n_cut, "n_tp": n_tp, "cut_would_have_won": cut_would_win,
        })

    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)

    print(f"{'═'*92}")
    print(f"  EXIT-RULE COMPARISON — {len(paths)} markets, entry {H}d out, band [{lo:.2f},{hi:.2f}]")
    print(f"{'═'*92}")
    print(f"  {'rule':>10} | {'mean ret':>9} | {'median':>8} | {'Sharpe':>7} | "
          f"{'tot PnL':>8} | {'maxDD':>7} | {'#cut':>5} | {'#tp':>5} | {'cut→win':>7}")
    print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*8}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}")
    for _, r in res.iterrows():
        print(f"  {r['rule']:>10} | {r['mean_ret']:>+8.1%} | {r['median_ret']:>+7.1%} | "
              f"{r['sharpe']:>7.2f} | {r['tot_pnl']:>+7.2f} | {r['maxdd']:>7.2f} | "
              f"{int(r['n_cut']):>5} | {int(r['n_tp']):>5} | {int(r['cut_would_have_won']):>7}")

    hold = res[res["rule"] == "hold"].iloc[0]
    print(f"\n  Baseline (hold): Sharpe {hold['sharpe']:.2f}, PnL {hold['tot_pnl']:+.1f}, "
          f"maxDD {hold['maxdd']:.1f}")
    print("  '#cut'=stop exits  '#tp'=take-profit exits  'cut→win'=how many stops would have resolved YES")

    # ── Plot equity curves ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    palette = {
        "hold":     "#636363",
        # stops (orange/red/purple spectrum)
        "stop_30":  "#fdae6b", "stop_50":  "#e6550d",
        "trail_25": "#756bb1", "below_d3": "#31a354", "decay": "#c51b8a",
        # take-profits only (blue spectrum)
        "tp_2x":    "#2196f3", "tp_3x":    "#0d47a1",
        "tp_80pct": "#00bcd4", "tp_90pct": "#006064",
        # combos (green spectrum)
        "s30_tp2x": "#66bb6a", "s30_tp3x": "#2e7d32",
        "s30_tp80": "#a5d6a7", "s30_tp90": "#1b5e20",
    }
    for rule in RULES:
        eq = rule_curves[rule]
        rr = res[res["rule"] == rule].iloc[0]
        color = palette.get(rule, "#999999")
        lw = 2.5 if rule == "hold" else 1.6
        ax.plot(range(len(eq)), eq, lw=lw, color=color,
                label=f"{rule:<10}  Sh {rr['sharpe']:.2f}  PnL {rr['tot_pnl']:+.1f}  DD {rr['maxdd']:.1f}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"Dynamic exits — long YES @ {H}d, band [{lo:.2f},{hi:.2f}], $1/trade")
    ax.set_xlabel("trade # (ordered by resolution date)")
    ax.set_ylabel("cumulative PnL ($)")
    ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120, bbox_inches="tight")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out}")
    print("✓  Dynamic-exit backtest complete.")


if __name__ == "__main__":
    main()
