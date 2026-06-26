#!/usr/bin/env python3
"""
Geo-calm sleeve — staleness exit (price range-bound within a band over days)
============================================================================

Generalises the "reached the mids and stayed" idea: instead of triggering on a
fixed 0.50 level, trigger when the price has gone STALE — confined to a tight
band over a rolling window of days (max-min over the last W days <= B). A stale
price means the move (up or down) has stopped.

Two economically distinct uses of the same staleness flag:
  * TAKE-PROFIT (profit-gated): when a position is stale AND in profit (p > entry),
    bank it — the run has stalled. (Like the trend-exit, but level-agnostic.)
  * CUT-DEAD (loss/flat-gated): when a position is stale AND NOT in profit, cut it
    and recycle the capital into fresh calm-cheap entries — a dead longshot that
    isn't moving ties up capital to settlement for nothing.
  * BOTH: exit on any staleness.

Swept over window W (days) and band B (price width). Compared to HOLD. Uses the
committed DAILY paths (staleness is a multi-day concept; all 35 book trades).

Usage:
  python experiments/run_regime_staleness_exit.py
  python experiments/run_regime_staleness_exit.py --mode cut_dead --window 3 --band 0.05
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book
from run_regime_take_profit import load_paths, sim, summarize, apply_tp, PANEL, OUT


def apply_staleness(book, paths, W, B, mode, entry_cost, exit_cost):
    """Exit when trailing-W-day price range <= B. mode: take_profit (only if in
    profit), cut_dead (only if not in profit), both (any stale)."""
    rows = []
    for _, r in book.iterrows():
        ep = min(r["price"] + entry_cost, 0.99)
        h = paths.get(str(r["token"]))
        seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])].reset_index(drop=True) if h is not None else None
        exit_date, exit_val, exited, reason = r["res_date"], float(r["outcome"]), False, "settle"
        if seg is not None and len(seg) > 1:
            p = seg["p"].values; dts = seg["date"].values
            for t in range(W, len(seg)):                       # need W days of history
                win = p[t - W + 1:t + 1]
                if (win.max() - win.min()) <= B:               # stale: range-bound
                    in_profit = p[t] > ep
                    fire = (mode == "both" or (mode == "take_profit" and in_profit)
                            or (mode == "cut_dead" and not in_profit))
                    if fire:
                        exit_date = pd.Timestamp(dts[t]); exit_val = max(p[t] - exit_cost, 0.0)
                        exited, reason = True, ("tp" if in_profit else "cut"); break
        rows.append(dict(token=r["token"], decision_date=r["decision_date"], res_date=exit_date,
                         settle_date=r["res_date"], price=r["price"], ep=ep, volume=r["volume"],
                         outcome=r["outcome"], exit_value=exit_val, exited_early=exited, reason=reason,
                         saved=exited and r["outcome"] == 0, capped=exited and r["outcome"] == 1,
                         ret=exit_val / ep - 1.0, hold=(exit_date - r["decision_date"]).days))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--mode", choices=["take_profit", "cut_dead", "both"], default="take_profit")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--entry-cost", type=float, default=0.03)
    ap.add_argument("--exit-cost", type=float, default=0.03)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    paths = load_paths(book["token"].astype(str).tolist())

    hold = apply_tp(book, paths, tp=1.01, entry_cost=args.entry_cost, exit_cost=args.exit_cost)
    hrow, heq = summarize(hold, "HOLD")

    print("=" * 100)
    print("GEO-CALM SLEEVE — STALENESS EXIT (price range-bound within band B over W days)")
    print("  n=%d · entry %.0f¢ / exit %.0f¢ · HOLD: edge/$1 %+.4f · CAGR %+.0f%% · Sharpe %.2f"
          % (len(book), args.entry_cost * 100, args.exit_cost * 100, hrow["edge"], 100 * hrow["cagr"], hrow["sharpe"]))
    print("=" * 100)

    for mode in (["take_profit", "cut_dead", "both"] if args.mode == args.mode else [args.mode]):
        pass
    modes = ["take_profit", "cut_dead", "both"]
    for mode in modes:
        print("\n" + "-" * 100)
        print("MODE = %s   (sweep window W x band B; cell = edge/$1 | CAGR%% | #exits[cap/cut])" % mode.upper())
        print("-" * 100)
        print("  %-8s" % "W\\B" + "".join("%-26s" % ("B=%.2f" % b) for b in (0.03, 0.05, 0.08)))
        for W in (2, 3, 5):
            cells = []
            for B in (0.03, 0.05, 0.08):
                df = apply_staleness(book, paths, W, B, mode, args.entry_cost, args.exit_cost)
                row, _ = summarize(df, "x")
                cells.append("%+.3f|%+4.0f%%|%d[%d/%d]" % (row["edge"], 100 * row["cagr"],
                             int(df.exited_early.sum()), int(df.capped.sum()), int(df.saved.sum())))
            print("  %-8s" % ("W=%d" % W) + "".join("%-26s" % c for c in cells))
        print("  HOLD baseline: edge/$1 %+.4f | CAGR %+.0f%%" % (hrow["edge"], 100 * hrow["cagr"]))

    # focused decomposition at the chosen params
    df = apply_staleness(book, paths, args.window, args.band, args.mode, args.entry_cost, args.exit_cost)
    row, eq = summarize(df, "stale")
    fired = df[df.exited_early]
    print("\n" + "=" * 100)
    print("DETAIL  mode=%s  W=%d  B=%.2f" % (args.mode, args.window, args.band))
    print("=" * 100)
    print("  edge/$1 %+.4f (vs hold %+.4f)  · CAGR %+.0f%% (vs %+.0f%%) · Sharpe %.2f (vs %.2f) · avg hold %.1fd (vs %.1f)"
          % (row["edge"], hrow["edge"], 100 * row["cagr"], 100 * hrow["cagr"], row["sharpe"], hrow["sharpe"],
             row["hold"], hrow["hold"]))
    print("  fired on %d/%d trades: %d capped winners, %d cut losers" %
          (len(fired), len(df), int(df.capped.sum()), int(df.saved.sum())))
    for _, r in fired.iterrows():
        print("   token …%s entry %.3f exit %.3f hold %dd -> %s (%s)"
              % (str(r["token"])[-6:], r["price"], r["exit_value"] + args.exit_cost, r["hold"],
                 "WIN" if r["outcome"] == 1 else "LOSS", r["reason"]))

    # plot equity for the three modes at chosen W,B vs hold
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax[0].plot(heq.index, heq.values, color="#333", lw=1.8, label="HOLD")
    for mode, col in [("take_profit", "#1f9d55"), ("cut_dead", "#2980b9"), ("both", "#c0392b")]:
        d = apply_staleness(book, paths, args.window, args.band, mode, args.entry_cost, args.exit_cost)
        _, e = summarize(d, mode)
        if len(e) > 1:
            ax[0].plot(e.index, e.values, color=col, lw=1.4, label="stale-%s" % mode)
    ax[0].axhline(1000, color="grey", ls="--", lw=.7); ax[0].legend(fontsize=8)
    ax[0].set_title("Equity by staleness mode (W=%d,B=%.2f)" % (args.window, args.band)); ax[0].set_ylabel("equity $")
    # edge vs band for take_profit & cut_dead at W=window
    bs = [0.02, 0.03, 0.05, 0.08, 0.12]
    for mode, col in [("take_profit", "#1f9d55"), ("cut_dead", "#2980b9")]:
        es = []
        for B in bs:
            d = apply_staleness(book, paths, args.window, B, mode, args.entry_cost, args.exit_cost)
            r2, _ = summarize(d, "x"); es.append(r2["edge"])
        ax[1].plot(bs, es, "o-", color=col, label=mode)
    ax[1].axhline(hrow["edge"], color="#333", ls="--", lw=1, label="HOLD")
    ax[1].set_xlabel("staleness band B (W=%d)" % args.window); ax[1].set_ylabel("net edge/$1")
    ax[1].set_title("Edge vs staleness band"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_staleness_exit.png", dpi=120)
    df.to_csv(f"{OUT}/regime_staleness_exit.csv", index=False)
    print("\nSaved: regime_staleness_exit.png , regime_staleness_exit.csv")


if __name__ == "__main__":
    main()
