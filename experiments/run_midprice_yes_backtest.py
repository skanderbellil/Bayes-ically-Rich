#!/usr/bin/env python3
"""
Polymarket — "long mid-priced YES, hold to resolution" backtest
===============================================================

The price-convergence deep dive found that markets trading between ~0.10 and
~0.90 a week before resolution resolve YES *more often than priced* — a
systematic upward drift in "live" markets.  This backtests that directly:

  At a fixed horizon before resolution, BUY the YES token of every market whose
  price sits inside a bracket [lo, hi], pay a spread haircut, hold to
  resolution, and book PnL = outcome/entry_cost − 1 per $1 staked.

It sweeps a grid of brackets within [0.10, 0.90] and several entry horizons,
ranks them by mean return / win-rate / Sharpe, and draws time-ordered equity
curves for the headline configurations.

Realism knobs
-------------
--haircut   cents added to the quoted price at entry (spread/slippage proxy)
--horizon   days before resolution to enter (can pass several)

Usage
-----
  python experiments/run_midprice_yes_backtest.py --n-markets 250
  python experiments/run_midprice_yes_backtest.py --haircut 0.03 --horizon 7 3 1
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

# Bracket grid to sweep (all within the 0.10–0.90 band the deep dive flagged)
BRACKETS = [
    (0.10, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90),   # quartile slices
    (0.10, 0.50), (0.50, 0.90),                               # halves
    (0.40, 0.70), (0.50, 0.80), (0.60, 0.90),                 # shifted windows
    (0.30, 0.90), (0.10, 0.90),                               # wide
]


def price_at(series: pd.Series, when: pd.Timestamp) -> float | None:
    s = series[series.index <= when]
    return float(s.iloc[-1]) if not s.empty else None


def backtest_bracket(obs: pd.DataFrame, lo: float, hi: float, haircut: float) -> dict:
    """Equal-$1 YES backtest for one bracket at one horizon.

    obs: DataFrame[price, outcome, t_res] already filtered to one horizon.
    Returns summary stats + the per-trade frame (time-ordered) for equity curves.
    """
    sel = obs[(obs["price"] >= lo) & (obs["price"] < hi)].copy()
    if sel.empty:
        return {"n": 0}
    sel["entry"] = np.minimum(sel["price"] + haircut, 0.99)
    sel["ret"]   = sel["outcome"] / sel["entry"] - 1.0     # per $1 staked
    sel = sel.sort_values("t_res")
    rets = sel["ret"].values
    win  = (sel["outcome"] == 1).mean()
    mean = rets.mean()
    std  = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    return {
        "n":        len(sel),
        "win_rate": win,
        "mean_ret": mean,
        "median_ret": float(np.median(rets)),
        "std":      std,
        "sharpe":   mean / std if std and std > 0 else float("nan"),
        "total_pnl": rets.sum(),               # additive, $1 per trade
        "avg_entry": sel["entry"].mean(),
        "frame":    sel[["t_res", "ret", "price", "outcome"]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=250)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, nargs="+", default=[7, 3, 1],
                    help="entry horizon(s) in days before resolution")
    ap.add_argument("--haircut", type=float, default=0.02,
                    help="cents added to quoted price at entry (spread proxy)")
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/paper_trade/midprice_backtest.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/midprice_backtest.png")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  LONG MID-PRICED YES → HOLD TO RESOLUTION  ·  backtest    ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  horizons={args.horizon}d  haircut={args.haircut:.2f}  "
          f"markets≤{args.n_markets}  vol≥${args.min_volume:,.0f}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets with clean 0/1 outcome\n")

    # Build (market × horizon) price observations
    recs = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        prices = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(prices) < 3:
            continue
        t_res = prices.index.max()
        for d in args.horizon:
            p = price_at(prices, t_res - pd.Timedelta(days=d))
            if p is None or not (0.0 < p < 1.0):
                continue
            recs.append({"horizon_d": d, "price": p, "outcome": float(m.outcome),
                         "t_res": t_res, "question": str(m.question)[:45]})
        if i % 50 == 0:
            logger.info("  priced %d/%d markets (%d obs)", i, len(markets), len(recs))
        time.sleep(args.sleep if i % 5 == 0 else 0)

    obs_all = pd.DataFrame(recs)
    if obs_all.empty:
        print("No observations — aborting.")
        return
    obs_all.to_csv(args.out, index=False)

    # ── Bracket sweep per horizon ──────────────────────────────────────────
    results = []
    for d in args.horizon:
        obs = obs_all[obs_all["horizon_d"] == d]
        print(f"\n{'═'*86}")
        print(f"  ENTRY HORIZON: {d} DAY(S) BEFORE RESOLUTION   ({len(obs)} markets in band)")
        print(f"{'═'*86}")
        print(f"  {'bracket':>13} | {'n':>4} | {'win%':>6} | {'mean ret':>9} | "
              f"{'median':>7} | {'Sharpe':>7} | {'tot PnL($1ea)':>13}")
        print(f"  {'-'*13}-+-{'-'*4}-+-{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*7}-+-{'-'*13}")
        for lo, hi in BRACKETS:
            r = backtest_bracket(obs, lo, hi, args.haircut)
            if r["n"] == 0:
                continue
            results.append({"horizon_d": d, "lo": lo, "hi": hi, **{k: v for k, v in r.items() if k != "frame"}})
            print(f"  [{lo:.2f}, {hi:.2f}) | {r['n']:>4} | {r['win_rate']*100:>5.1f} | "
                  f"{r['mean_ret']:>+8.1%} | {r['median_ret']:>+6.1%} | "
                  f"{r['sharpe']:>7.2f} | {r['total_pnl']:>+12.2f}")

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out.replace(".csv", "_summary.csv"), index=False)

    # ── Headline equity curves (7d horizon, key brackets) ─────────────────
    base_h = args.horizon[0]
    obs0 = obs_all[obs_all["horizon_d"] == base_h]
    key_brackets = [(0.10, 0.90), (0.50, 0.90), (0.50, 0.70), (0.70, 0.90)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for lo, hi in key_brackets:
        r = backtest_bracket(obs0, lo, hi, args.haircut)
        if r["n"] == 0:
            continue
        f = r["frame"].reset_index(drop=True)
        eq = f["ret"].cumsum()
        ax1.plot(range(len(eq)), eq, lw=1.8,
                 label=f"[{lo:.2f},{hi:.2f})  n={r['n']}  PnL {r['total_pnl']:+.1f}")
    ax1.axhline(0, color="grey", lw=0.8)
    ax1.set_title(f"Equity curve — $1/trade, YES @ {base_h}d, hold to resolution")
    ax1.set_xlabel("trade # (ordered by resolution date)")
    ax1.set_ylabel("cumulative PnL ($)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    # mean return by bracket bar (base horizon)
    bars = [(f"[{lo:.1f},{hi:.1f})", backtest_bracket(obs0, lo, hi, args.haircut))
            for lo, hi in BRACKETS]
    bars = [(lbl, r) for lbl, r in bars if r["n"] > 0]
    labels = [lbl for lbl, _ in bars]
    means  = [r["mean_ret"] * 100 for _, r in bars]
    colors = ["#31a354" if mr > 0 else "#e34a33" for mr in means]
    ax2.bar(range(len(labels)), means, color=colors)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax2.axhline(0, color="grey", lw=0.8)
    ax2.set_title(f"Mean return per trade by bracket (@ {base_h}d, haircut {args.haircut:.2f})")
    ax2.set_ylabel("mean return (%)")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Long mid-priced YES, held to resolution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120, bbox_inches="tight")

    # ── Best config callout ───────────────────────────────────────────────
    if not res_df.empty:
        ranked = res_df[res_df["n"] >= 10].sort_values("mean_ret", ascending=False)
        print(f"\n{'═'*86}")
        print("  TOP CONFIGS (n≥10, by mean return)")
        print(f"{'═'*86}")
        for _, r in ranked.head(6).iterrows():
            print(f"  [{r['lo']:.2f},{r['hi']:.2f}) @ {int(r['horizon_d'])}d : "
                  f"n={int(r['n'])}  win {r['win_rate']*100:.0f}%  "
                  f"mean {r['mean_ret']:+.1%}  Sharpe {r['sharpe']:.2f}  "
                  f"PnL {r['total_pnl']:+.1f}")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out.replace('.csv', '_summary.csv')}")
    print("✓  Backtest complete.")


if __name__ == "__main__":
    main()
