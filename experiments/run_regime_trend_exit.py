#!/usr/bin/env python3
"""
Geo-calm sleeve — trend-conditional take-profit ("bank the stall, ride the trend")
==================================================================================

run_regime_take_profit.py showed a DUMB fixed take-profit loses: every contract
that reached the mids went on to settle YES, so a flat TP only capped winners.
This tests the user's refinement: don't exit just because price hit the mids —
exit only if it reaches the mids AND STALLS there; if it's still TRENDING UP,
keep holding (let the convex winner run).

The rule
--------
Walk each trade's daily path from entry. Once price enters the mids zone
(p >= --arm), evaluate short-horizon momentum each day:

    recent_gain = p[t] - p[t - w]         (w = --trend-window days)
    p[t] >= arm  and  recent_gain <= eps  ->  STALLED: take profit at p[t]-exit_cost
    p[t] >= arm  and  recent_gain  > eps  ->  still climbing: HOLD, re-check tomorrow

If it never stalls in-zone before settlement, it rides to $1/$0 (full hold).
So the rule only ever diverges from HOLD on a position that has plateaued in the
profit zone — exactly the "reached the mids and stayed there" case.

Baselines compared: HOLD-to-settlement, and the dumb fixed TP @ --arm.

Caveat: 35 trades, short daily paths (~9d holds) — this is a behavioural probe,
not a tuned parameter. The honest read is the DECOMP (how often it fires, and
whether the fires were would-be winners or losers), not a 3rd-decimal CAGR.

Usage:
  python experiments/run_regime_trend_exit.py
  python experiments/run_regime_trend_exit.py --arm 0.50 --trend-window 2 --trend-eps 0.0
"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book
from run_regime_take_profit import load_paths, sim, summarize, PANEL, OUT, apply_tp


def apply_trend_exit(book, paths, arm, w, eps, entry_cost, exit_cost):
    """Trend-conditional exit: in the mids zone, bank profit only when momentum
    over the last `w` days has died (gain <= eps); otherwise ride the trend."""
    rows = []
    for _, r in book.iterrows():
        ep = min(r["price"] + entry_cost, 0.99)
        h = paths.get(str(r["token"]))
        seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])].reset_index(drop=True) if h is not None else None
        exit_date, exit_val, exited, reason = r["res_date"], float(r["outcome"]), False, "settle"
        if seg is not None and len(seg) > 1:
            p = seg["p"].values; dts = seg["date"].values
            for t in range(1, len(seg)):
                if p[t] >= arm:
                    j = max(0, t - w)
                    recent_gain = p[t] - p[j]
                    if recent_gain <= eps:                 # stalled / rolling over in the zone
                        exit_date = pd.Timestamp(dts[t]); exit_val = max(p[t] - exit_cost, 0.0)
                        exited, reason = True, "stall"; break
        rows.append(dict(token=r["token"], decision_date=r["decision_date"], res_date=exit_date,
                         settle_date=r["res_date"], price=r["price"], ep=ep, volume=r["volume"],
                         outcome=r["outcome"], exit_value=exit_val, exited_early=exited, reason=reason,
                         saved=exited and r["outcome"] == 0, capped=exited and r["outcome"] == 1,
                         ret=exit_val / ep - 1.0, hold=(exit_date - r["decision_date"]).days))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--entry-max", type=float, default=None)
    ap.add_argument("--arm", type=float, default=0.50, help="mids zone that arms the stall check")
    ap.add_argument("--trend-window", type=int, default=2, help="days over which to measure momentum")
    ap.add_argument("--trend-eps", type=float, default=0.0, help="min gain over window to count as 'still trending up'")
    ap.add_argument("--entry-cost", type=float, default=0.03)
    ap.add_argument("--exit-cost", type=float, default=0.03)
    ap.add_argument("--stake", type=float, default=0.02)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    if args.entry_max is not None:
        book = book[book["price"] <= args.entry_max].reset_index(drop=True)
    paths = load_paths(book["token"].astype(str).tolist())

    print("=" * 100)
    print("GEO-CALM SLEEVE — TREND-CONDITIONAL TAKE-PROFIT (bank the stall, ride the trend)")
    print("  arm=%.2f · trend-window=%dd · trend-eps=%.2f · entry %.0f¢ / exit %.0f¢ · stake %.0f%%%s"
          % (args.arm, args.trend_window, args.trend_eps, args.entry_cost * 100, args.exit_cost * 100,
             args.stake * 100, "" if args.entry_max is None else " · entry<=%.2f" % args.entry_max))
    print("  n=%d trades · real daily paths" % len(book))
    print("=" * 100)

    hold = apply_tp(book, paths, tp=1.01, entry_cost=args.entry_cost, exit_cost=args.exit_cost)
    dumb = apply_tp(book, paths, tp=args.arm, entry_cost=args.entry_cost, exit_cost=args.exit_cost)
    trend = apply_trend_exit(book, paths, args.arm, args.trend_window, args.trend_eps, args.entry_cost, args.exit_cost)

    print("\n  %-22s %4s %7s %9s %8s %8s %7s %7s %8s %7s %7s" %
          ("rule", "n", "pos%", "edge/$1", "t", "hold-d", "early%", "saved", "capped", "CAGR", "Sharpe"))
    labels = []
    for df, name in [(hold, "HOLD (settle)"),
                     (dumb, "dumb TP @ %.2f" % args.arm),
                     (trend, "trend-exit @ %.2f" % args.arm)]:
        row, eq = summarize(df, name)
        labels.append((row, eq, df))
        print("  %-22s %4d %6.0f%% %+9.4f %+8.2f %8.1f %6.0f%% %7d %8d %+7.0f%% %7.2f" %
              (name, row["n"], 100 * row["pos"], row["edge"], row["t"], row["hold"],
               100 * row["early"], row["saved"], row["capped"], 100 * row["cagr"], row["sharpe"]))

    hold_row, dumb_row, trend_row = labels[0][0], labels[1][0], labels[2][0]

    # what did the trend rule actually do?
    fired = trend[trend["reason"] == "stall"]
    print("\n[DECOMP] trend-exit fired on %d / %d trades (stalled in the mids zone):" % (len(fired), len(trend)))
    if len(fired):
        for _, r in fired.iterrows():
            print("   token …%s  entry %.3f  exit %.3f  would-settle %s  -> %s"
                  % (str(r["token"])[-6:], r["price"], r["exit_value"] + args.exit_cost,
                     "WIN" if r["outcome"] == 1 else "LOSS", "saved a loser" if r["outcome"] == 0 else "capped a winner"))
    else:
        print("   (never fired — no position stalled in the zone; identical to HOLD on this book)")
    rode = trend[(trend["exit_value"].isin([0.0, 1.0])) & (trend["reason"] == "settle")]
    winners_rode = int(((trend["reason"] == "settle") & (trend["outcome"] == 1)).sum())
    print("   winners ridden to $1 (not capped): %d   | dumb TP capped %d winners"
          % (winners_rode, dumb_row["capped"]))

    print("\n[VERDICT]")
    print("   HOLD        edge %+.4f · CAGR %+.0f%% · Sharpe %.2f" % (hold_row["edge"], 100 * hold_row["cagr"], hold_row["sharpe"]))
    print("   dumb TP     edge %+.4f · CAGR %+.0f%% · Sharpe %.2f" % (dumb_row["edge"], 100 * dumb_row["cagr"], dumb_row["sharpe"]))
    print("   trend-exit  edge %+.4f · CAGR %+.0f%% · Sharpe %.2f" % (trend_row["edge"], 100 * trend_row["cagr"], trend_row["sharpe"]))
    d_hold = trend_row["edge"] - hold_row["edge"]
    if abs(d_hold) < 1e-6:
        print("   -> trend-exit == HOLD here: nothing stalled, so it correctly rode every winner.")
        print("      It only diverges when a position plateaus in the mids — a free insurance option")
        print("      that costs nothing in-sample and would bank a stall if one ever appears.")
    elif d_hold > 0:
        print("   -> trend-exit BEATS hold by %+.4f edge/$1: the stalls it banked were genuine risk." % d_hold)
    else:
        print("   -> trend-exit trails hold by %+.4f edge/$1: even the stalls recovered, so still ride." % d_hold)

    # sensitivity over arm x trend-window
    print("\n[SENSITIVITY] trend-exit edge/$1 (rows arm, cols trend-window days), eps=%.2f" % args.trend_eps)
    print("  %-7s" % "arm\\w" + "".join("%9d" % w for w in (1, 2, 3)))
    for arm in (0.40, 0.50, 0.60, 0.70):
        cells = []
        for w in (1, 2, 3):
            df = apply_trend_exit(book, paths, arm, w, args.trend_eps, args.entry_cost, args.exit_cost)
            row, _ = summarize(df, "x")
            cells.append("%+9.4f" % row["edge"])
        print("  %-7.2f" % arm + "".join(cells))
    print("  (HOLD baseline edge/$1 = %+.4f)" % hold_row["edge"])

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for (row, eq, _), col in zip(labels, ["#333", "#c0392b", "#1f9d55"]):
        if len(eq) > 1:
            ax[0].plot(eq.index, eq.values, lw=1.7, color=col, label=row["label"])
    ax[0].axhline(1000, color="grey", ls="--", lw=.8); ax[0].legend(fontsize=8)
    ax[0].set_title("Equity: hold vs dumb-TP vs trend-exit (stake %.0f%%)" % (args.stake * 100)); ax[0].set_ylabel("equity $")
    bars = ["HOLD", "dumb TP", "trend-exit"]
    ax[1].bar(bars, [hold_row["edge"], dumb_row["edge"], trend_row["edge"]],
              color=["#333", "#c0392b", "#1f9d55"])
    ax[1].set_title("Net edge/$1 by exit rule"); ax[1].set_ylabel("net edge/$1")
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_trend_exit.png", dpi=120)
    pd.DataFrame([hold_row, dumb_row, trend_row]).to_csv(f"{OUT}/regime_trend_exit.csv", index=False)
    print("\nSaved: regime_trend_exit.png , regime_trend_exit.csv")


if __name__ == "__main__":
    main()
