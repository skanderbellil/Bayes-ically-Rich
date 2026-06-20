#!/usr/bin/env python3
"""
Polymarket — entry-horizon × exit-horizon grid for long-YES
===========================================================

Buy the YES token at entry horizon E (days before resolution) for markets in
the price band, then SELL at a later horizon X (X < E days left), booking the
price move.  X = 0 means hold all the way to resolution (binary 0/1 payoff).

This maps where the YES underpricing drift is actually monetizable:

  * Holding to resolution (X=0) takes the full binary −100% / +Nx tail.
  * Selling early (X>0) caps both tails — you sell winners below 1.0 but also
    sell losers above 0.0 — which can lift Sharpe even if mean return falls.

Entry pays +haircut; a timed sale pays −haircut; resolution pays face. Band
membership is judged at the ENTRY horizon. Equal $1 per trade.

Outputs mean-return and Sharpe matrices (entry rows × exit cols) and heatmaps.

Usage
-----
  python experiments/run_horizon_grid_backtest.py --n-markets 250
  python experiments/run_horizon_grid_backtest.py --band 0.10 0.50
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

ENTRIES = [7, 5, 3, 2]        # days before resolution to BUY
EXITS   = [5, 3, 2, 1, 0]     # days before resolution to SELL (0 = hold to resolution)


def price_asof(series: pd.Series, when: pd.Timestamp) -> float | None:
    s = series[series.index <= when]
    return float(s.iloc[-1]) if not s.empty else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=250)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.90])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/paper_trade/horizon_grid.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/horizon_grid.png")
    args = ap.parse_args()

    lo, hi = args.band
    print("""
╔══════════════════════════════════════════════════════════╗
║  ENTRY × EXIT HORIZON GRID — long YES                    ║
║  buy at E days, sell at X days (0 = hold to resolution)   ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  band [{lo:.2f},{hi:.2f}]  haircut {args.haircut:.2f}  markets≤{args.n_markets}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets\n")

    # Cache each market's daily series + resolution time
    series_list = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        s = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(s) < 3:
            continue
        series_list.append({"s": s, "t_res": s.index.max(), "outcome": float(m.outcome)})
        if i % 50 == 0:
            logger.info("  loaded %d/%d", i, len(markets))
        time.sleep(args.sleep if i % 5 == 0 else 0)

    rows = []
    for E in ENTRIES:
        for X in EXITS:
            if X >= E:
                continue
            rets = []
            for sm in series_list:
                s, t_res, outcome = sm["s"], sm["t_res"], sm["outcome"]
                p_entry = price_asof(s, t_res - pd.Timedelta(days=E))
                if p_entry is None or not (lo <= p_entry < hi):
                    continue
                entry = min(p_entry + args.haircut, 0.99)
                if X == 0:
                    ret = outcome / entry - 1.0
                else:
                    p_exit = price_asof(s, t_res - pd.Timedelta(days=X))
                    if p_exit is None:
                        continue
                    exit_px = max(p_exit - args.haircut, 0.0)
                    ret = exit_px / entry - 1.0
                rets.append(ret)
            if len(rets) < 5:
                continue
            rets = np.array(rets)
            rows.append({
                "entry_d": E, "exit_d": X, "n": len(rets),
                "mean_ret": rets.mean(), "median_ret": float(np.median(rets)),
                "win_rate": float((rets > 0).mean()),
                "sharpe": rets.mean() / rets.std(ddof=1) if rets.std(ddof=1) > 0 else np.nan,
                "tot_pnl": rets.sum(),
            })

    res = pd.DataFrame(rows)
    if res.empty:
        print("No grid cells with enough data — aborting.")
        return
    res.to_csv(args.out, index=False)

    # ── Matrices ──────────────────────────────────────────────────────────
    def matrix(metric, fmt):
        print(f"\n  {metric.upper()}  (rows = entry day, cols = exit day; 0=resolution)")
        hdr = "  entry\\exit |" + "".join(f"{('res' if x==0 else str(x)+'d'):>9}" for x in EXITS)
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for E in ENTRIES:
            cells = []
            for X in EXITS:
                r = res[(res["entry_d"] == E) & (res["exit_d"] == X)]
                cells.append(fmt.format(r.iloc[0][metric]) if not r.empty else "    —   ")
            print(f"  {E:>4}d      |" + "".join(f"{c:>9}" for c in cells))

    matrix("mean_ret", "{:+.1%}")
    matrix("sharpe",   "{:.2f}")
    matrix("win_rate", "{:.0%}")

    # Best by Sharpe
    best = res.sort_values("sharpe", ascending=False).head(6)
    print(f"\n{'═'*72}")
    print("  TOP CELLS BY SHARPE")
    print(f"{'═'*72}")
    for _, r in best.iterrows():
        xlbl = "resolution" if r["exit_d"] == 0 else f"{int(r['exit_d'])}d"
        print(f"  buy {int(r['entry_d'])}d → sell {xlbl:>10} : n={int(r['n'])}  "
              f"mean {r['mean_ret']:+.1%}  win {r['win_rate']:.0%}  "
              f"Sharpe {r['sharpe']:.2f}  PnL {r['tot_pnl']:+.1f}")

    # ── Heatmaps ──────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, title, cmap in [
        (ax1, "mean_ret", "Mean return per trade", "RdYlGn"),
        (ax2, "sharpe",   "Per-trade Sharpe",      "RdYlGn"),
    ]:
        M = np.full((len(ENTRIES), len(EXITS)), np.nan)
        for ei, E in enumerate(ENTRIES):
            for xi, X in enumerate(EXITS):
                r = res[(res["entry_d"] == E) & (res["exit_d"] == X)]
                if not r.empty:
                    M[ei, xi] = r.iloc[0][metric]
        im = ax.imshow(M, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(EXITS)))
        ax.set_xticklabels(["res" if x == 0 else f"{x}d" for x in EXITS])
        ax.set_yticks(range(len(ENTRIES)))
        ax.set_yticklabels([f"{e}d" for e in ENTRIES])
        ax.set_xlabel("exit horizon"); ax.set_ylabel("entry horizon")
        ax.set_title(title)
        for ei in range(len(ENTRIES)):
            for xi in range(len(EXITS)):
                if not np.isnan(M[ei, xi]):
                    val = f"{M[ei,xi]:+.0%}" if metric == "mean_ret" else f"{M[ei,xi]:.2f}"
                    ax.text(xi, ei, val, ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Buy@entry, sell@exit — long YES band [{lo:.2f},{hi:.2f}]", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120, bbox_inches="tight")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out}")
    print("✓  Horizon-grid backtest complete.")


if __name__ == "__main__":
    main()
