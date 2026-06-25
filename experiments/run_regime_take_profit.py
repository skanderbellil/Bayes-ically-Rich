#!/usr/bin/env python3
"""
Geo-calm sleeve — take-profit vs hold-to-settlement
===================================================

The validated sleeve buys the cheap YES leg in GPR-calm geopolitics and HOLDS to
settlement (winners auto-settle at $1, losers at $0; the only cost is the entry
spread). This script tests the user's idea: *enter very low, and when the price
runs up to "the mids", TAKE PROFIT instead of holding* — does locking in an early
mark beat riding every position to the end?

The trade-off
-------------
A take-profit at threshold TP does two opposite things:
  * SAVES some would-be LOSERS — a contract that spikes to TP and later collapses
    to $0 is sold green instead of going to zero, and
  * CAPS the WINNERS — a contract that would settle at $1 is sold at TP < $1.
Plus it pays the spread a SECOND time (exit sells into the bid: proceeds = TP -
exit_cost), whereas holding pays the spread only once. It also FREES capital
earlier, which compounds faster. The net of those four effects is an empirical
question — answered here on the real daily price paths of the 35 book trades.

Model
-----
  entry  = mid + entry_cost           (buy the ask, as in the validated backtest)
  TP exit: first day AFTER entry with p >= TP -> sell at  TP - exit_cost
           (a resting limit at TP; assumes you fill at your price, flagged)
  else   : hold to settlement -> $1 (win) or $0 (loss), no exit cost
  capital recycles at the (earlier) exit date, so TP compounding is credited.

Data: data/polymarket_calibration_snapshot/token_history_cache_*.tar.gz
      (committed daily price paths; all 35 book tokens covered, entry day = panel price).

Usage:
  python experiments/run_regime_take_profit.py
  python experiments/run_regime_take_profit.py --entry-max 0.15 --exit-cost 0.03
"""
from __future__ import annotations
import argparse, glob, math, tarfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
OUT = "data/polymarket_calibration_snapshot"
CACHE = sorted(glob.glob(f"{OUT}/token_history_cache_*.tar.gz"))[-1] if glob.glob(f"{OUT}/token_history_cache_*.tar.gz") else None


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def load_paths(tokens):
    """token -> sorted daily price path DataFrame(date,p), from the committed tar."""
    paths = {}
    with tarfile.open(CACHE) as tf:
        names = set(tf.getnames())
        for tok in tokens:
            nm = "polymarket/token_%s.csv" % tok
            if nm in names:
                h = pd.read_csv(tf.extractfile(tf.getmember(nm)), parse_dates=["date"])
                paths[tok] = h.dropna(subset=["p"]).sort_values("date").reset_index(drop=True)
    return paths


def apply_tp(book, paths, tp, entry_cost, exit_cost):
    """Return a per-trade frame with exit_date, exit_value (per-share settlement $),
    hold-time, and tags (saved loser / capped winner) under take-profit at `tp`."""
    rows = []
    for _, r in book.iterrows():
        ep = min(r["price"] + entry_cost, 0.99)
        h = paths.get(str(r["token"]))
        seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])] if h is not None else None
        exit_date, exit_val, exited = r["res_date"], float(r["outcome"]), False
        if seg is not None and len(seg) > 1:
            after = seg.iloc[1:]                       # strictly after entry day
            hit = after[after["p"] >= tp]
            if not hit.empty:
                exit_date = hit.iloc[0]["date"]
                exit_val = max(tp - exit_cost, 0.0)    # sell into the bid
                exited = True
        rows.append(dict(token=r["token"], decision_date=r["decision_date"], res_date=exit_date,
                         settle_date=r["res_date"], price=r["price"], ep=ep, volume=r["volume"],
                         outcome=r["outcome"], exit_value=exit_val, exited_early=exited,
                         saved=exited and r["outcome"] == 0, capped=exited and r["outcome"] == 1,
                         ret=exit_val / ep - 1.0, hold=(exit_date - r["decision_date"]).days))
    return pd.DataFrame(rows)


def sim(df, cap0=1000.0, stake=0.02):
    """Capital sim recycling at exit_date; settlement value per share = exit_value."""
    d = df.sort_values("decision_date").copy()
    ev = []
    for i, r in d.iterrows():
        ev.append((r["decision_date"], 1, i)); ev.append((r["res_date"], 0, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    cash, open_pos, curve = cap0, {}, []
    for ts, kind, i in ev:
        if kind == 0:
            p = open_pos.pop(i, None)
            if p:
                cash += p["shares"] * float(d.at[i, "exit_value"])
        else:
            eq = cash + sum(v["cost"] for v in open_pos.values())
            s = min(stake * eq, cash); px = float(d.at[i, "ep"])
            if s > 0 and px > 0:
                open_pos[i] = {"shares": s / px, "cost": s}; cash -= s
        curve.append((ts, cash + sum(v["cost"] for v in open_pos.values())))
    eq = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    if len(eq) < 2:
        return eq, {}
    daily = eq.resample("D").ffill(); rets = daily.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    cagr = (eq.iloc[-1] / cap0) ** (1 / yrs) - 1
    down = rets[rets < 0].std(ddof=1)
    return eq, {"final": eq.iloc[-1], "ret": eq.iloc[-1] / cap0 - 1, "cagr": cagr,
                "sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(252) if rets.std() > 0 else np.nan,
                "sortino": rets.mean() / down * np.sqrt(252) if down > 0 else np.nan,
                "maxdd": dd, "calmar": cagr / abs(dd) if dd < 0 else np.nan}


def summarize(df, label):
    pos = (df["ret"] > 0).mean()
    edge = df["exit_value"] - df["ep"]
    eq, m = sim(df)
    return dict(label=label, n=len(df), pos=pos, mean_ret=df["ret"].mean(), edge=edge.mean(),
                t=tstat(edge), hold=df["hold"].mean(), early=df["exited_early"].mean(),
                saved=int(df["saved"].sum()), capped=int(df["capped"].sum()), **m), eq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--entry-max", type=float, default=None, help="restrict to 'enter very low': only trades with price <= this")
    ap.add_argument("--entry-cost", type=float, default=0.03)
    ap.add_argument("--exit-cost", type=float, default=0.03, help="spread paid again on the TP sell")
    ap.add_argument("--stake", type=float, default=0.02)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    args = ap.parse_args()
    if CACHE is None:
        print("no token_history_cache_*.tar.gz in", OUT); return

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    if args.entry_max is not None:
        book = book[book["price"] <= args.entry_max].reset_index(drop=True)
    paths = load_paths(book["token"].astype(str).tolist())

    print("=" * 100)
    print("GEO-CALM SLEEVE — TAKE-PROFIT vs HOLD-TO-SETTLEMENT")
    print("  geopolitics · YES · price<=%.2f%s · GPR-calm[%s] · entry %.0f¢ / exit %.0f¢ spread · stake %.0f%%"
          % (args.max_price, ("" if args.entry_max is None else " (entry<=%.2f)" % args.entry_max),
             args.gate, args.entry_cost * 100, args.exit_cost * 100, args.stake * 100))
    print("  n=%d trades · price cache %s · all paths entry-aligned to panel price"
          % (len(book), CACHE.split("/")[-1]))
    print("=" * 100)

    # baseline = hold (TP=1.0 never fires)
    hold = apply_tp(book, paths, tp=1.01, entry_cost=args.entry_cost, exit_cost=args.exit_cost)
    base_row, base_eq = summarize(hold, "HOLD (settle)")

    tps = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    print("\n[SWEEP] take-profit threshold TP  (sell at TP - %.0f¢; else hold to $1/$0)" % (args.exit_cost * 100))
    print("  %-14s %4s %7s %8s %9s %7s %7s %7s %7s %8s %7s %7s" %
          ("rule", "n", "pos%", "edge/$1", "t", "hold-d", "early%", "saved", "capped", "CAGR", "Sharpe", "maxDD"))
    print("  %-14s %4d %6.0f%% %+8.4f %+9.2f %7.1f %6s %7s %8s %+7.0f%% %7.2f %6.0f%%" %
          (base_row["label"], base_row["n"], 100 * base_row["pos"], base_row["edge"], base_row["t"],
           base_row["hold"], "-", "-", "-", 100 * base_row["cagr"], base_row["sharpe"], 100 * base_row["maxdd"]))
    rows, eqs = [base_row], {"HOLD": base_eq}
    for tp in tps:
        df = apply_tp(book, paths, tp=tp, entry_cost=args.entry_cost, exit_cost=args.exit_cost)
        row, eq = summarize(df, "TP @ %.2f" % tp)
        rows.append(row); eqs["TP %.2f" % tp] = eq
        print("  %-14s %4d %6.0f%% %+8.4f %+9.2f %7.1f %5.0f%% %7d %8d %+7.0f%% %7.2f %6.0f%%" %
              (row["label"], row["n"], 100 * row["pos"], row["edge"], row["t"], row["hold"],
               100 * row["early"], row["saved"], row["capped"], 100 * row["cagr"], row["sharpe"], 100 * row["maxdd"]))

    R = pd.DataFrame(rows)
    best_edge = R.iloc[1:].sort_values("edge", ascending=False).iloc[0]
    best_cagr = R.iloc[1:].sort_values("cagr", ascending=False).iloc[0]
    print("\n[VERDICT vs HOLD]  hold: edge/$1 %+.4f · CAGR %+.0f%% · Sharpe %.2f · maxDD %.0f%%"
          % (base_row["edge"], 100 * base_row["cagr"], base_row["sharpe"], 100 * base_row["maxdd"]))
    print("  best edge : %-9s edge %+.4f (%+.4f vs hold) · CAGR %+.0f%% · Sharpe %.2f"
          % (best_edge["label"], best_edge["edge"], best_edge["edge"] - base_row["edge"],
             100 * best_edge["cagr"], best_edge["sharpe"]))
    print("  best CAGR : %-9s CAGR %+.0f%% (%+.0f%% vs hold) · Sharpe %.2f · maxDD %.0f%%"
          % (best_cagr["label"], 100 * best_cagr["cagr"], 100 * (best_cagr["cagr"] - base_row["cagr"]),
             best_cagr["sharpe"], 100 * best_cagr["maxdd"]))
    verdict = "TAKE-PROFIT helps" if best_edge["edge"] > base_row["edge"] + 1e-9 else "HOLDING wins — TP caps more than it saves"
    print("  -> %s" % verdict)

    # decomposition at TP=0.50 (the 'mids')
    mid = apply_tp(book, paths, tp=0.50, entry_cost=args.entry_cost, exit_cost=args.exit_cost)
    early = mid[mid["exited_early"]]
    print("\n[DECOMP @ TP=0.50]  of %d early exits: %d were would-be WINNERS (capped at 0.50 vs $1),"
          % (len(early), int(mid["capped"].sum())))
    print("                     %d were would-be LOSERS (saved at 0.50 vs $0)." % int(mid["saved"].sum()))
    if int(mid["capped"].sum()):
        give = early[mid.loc[early.index, "capped"]]
        lost_upside = ((1.0 - args.exit_cost) - (0.50 - args.exit_cost)) / give["ep"]
        print("                     upside given up per capped winner ≈ %.1fx the stake (sold 0.50 not ~1.0)" % lost_upside.mean())
    if int(mid["saved"].sum()):
        sv = early[mid.loc[early.index, "saved"]]
        print("                     gain captured per saved loser ≈ %.1fx the stake (0.50 not 0)" % ((0.50 - args.exit_cost) / sv["ep"] - 1.0).mean())

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for lbl, col in [("HOLD", "#333"), ("TP 0.50", "#1f9d55"), ("TP 0.70", "#2980b9"), ("TP 0.30", "#c0392b")]:
        e = eqs.get(lbl)
        if e is not None and len(e) > 1:
            ax[0].plot(e.index, e.values, lw=1.6, color=col, label=lbl)
    ax[0].axhline(1000, color="grey", ls="--", lw=.8); ax[0].legend(fontsize=8)
    ax[0].set_title("Equity: hold vs take-profit (stake %.0f%%)" % (args.stake * 100)); ax[0].set_ylabel("equity $")
    x = ["HOLD"] + ["%.2f" % t for t in tps]
    ax[1].axhline(base_row["edge"], color="#333", ls="--", lw=1, label="hold edge/$1")
    ax[1].plot(range(len(R)), R["edge"], "o-", color="#1f9d55")
    ax[1].set_xticks(range(len(R))); ax[1].set_xticklabels(x, rotation=45, fontsize=8)
    ax[1].set_title("Net edge/$1 vs exit rule"); ax[1].set_ylabel("net edge/$1"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_take_profit.png", dpi=120)
    R.to_csv(f"{OUT}/regime_take_profit.csv", index=False)
    print("\nSaved: regime_take_profit.png , regime_take_profit.csv")


if __name__ == "__main__":
    main()
