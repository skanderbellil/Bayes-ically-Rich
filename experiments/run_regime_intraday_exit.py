#!/usr/bin/env python3
"""
Geo-calm sleeve — intraday mid-touch & take-profit re-test (10-min paths)
========================================================================

The daily-close exit study (run_regime_take_profit.py) found 0% of losers ever
touched the mids — but that was at the DAILY CLOSE. A contract can spike to 0.50
intraday and fade back the same day, invisible to a daily mark. This re-tests on
the true 10-minute CLOB paths for the geo-calm book trades that have intraday
coverage (2026 markets; older ones predate CLOB intraday retention).

Two questions:
  1. MID-TOUCH: how many trades (and crucially, how many would-be LOSERS) touch
     a take-profit level INTRADAY that they never reached at the daily close?
  2. TAKE-PROFIT: does an intraday-fillable limit at TP beat hold-to-settlement
     on these trades, once the daily-vs-intraday touch difference is accounted?

Honest scope: only the intraday-covered subset (~15 trades). A probe of whether
the daily-close conclusion survives finer data, NOT a fresh verdict.

Usage:
  python experiments/run_regime_intraday_exit.py
  python experiments/run_regime_intraday_exit.py --tp 0.50
"""
from __future__ import annotations
import argparse, glob, tarfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book
from run_regime_take_profit import load_paths, PANEL, OUT

INTRA = sorted(glob.glob(f"{OUT}/token_intraday_cache_*.tar.gz"))[-1]


def load_intraday(tokens):
    out = {}
    with tarfile.open(INTRA) as tf:
        names = set(tf.getnames())
        for tok in tokens:
            nm = "polymarket/tokenhr_%s.csv" % tok
            if nm in names:
                s = pd.read_csv(tf.extractfile(tf.getmember(nm)), index_col=0, parse_dates=True)
                out[tok] = s.iloc[:, 0].sort_index()
    return out


def first_touch(series, dec_date, res_date, tp):
    """First timestamp at/after entry day where price >= tp; returns (ts, peak) or (None, peak)."""
    s = series[(series.index >= pd.Timestamp(dec_date, tz="UTC")) & (series.index <= pd.Timestamp(res_date, tz="UTC") + pd.Timedelta(days=1))]
    if s.empty:
        return None, np.nan
    hit = s[s >= tp]
    return (hit.index[0] if len(hit) else None), float(s.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--tp", type=float, default=0.50)
    ap.add_argument("--entry-cost", type=float, default=0.03)
    ap.add_argument("--exit-cost", type=float, default=0.03)
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price)
    intr = load_intraday(book["token"].astype(str).tolist())
    daily = load_paths(book["token"].astype(str).tolist())
    cov = book[book["token"].astype(str).isin(intr.keys())].copy().reset_index(drop=True)

    print("=" * 96)
    print("GEO-CALM SLEEVE — INTRADAY MID-TOUCH & TAKE-PROFIT (10-min paths)")
    print("  intraday-covered book trades: %d / %d  ·  TP=%.2f  ·  entry %.0f¢ / exit %.0f¢"
          % (len(cov), len(book), args.tp, args.entry_cost * 100, args.exit_cost * 100))
    print("=" * 96)

    rows = []
    for _, r in cov.iterrows():
        tok = str(r["token"])
        # daily peak (after entry day) vs intraday peak
        h = daily.get(tok)
        seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])] if h is not None else None
        daily_peak = float(seg["p"].iloc[1:].max()) if seg is not None and len(seg) > 1 else float("nan")
        ts, intra_peak = first_touch(intr[tok], r["decision_date"], r["res_date"], args.tp)
        daily_touch = daily_peak >= args.tp
        intra_touch = ts is not None
        rows.append(dict(token=tok, q=str(r["question"])[:40], price=r["price"], outcome=r["outcome"],
                         daily_peak=daily_peak, intra_peak=intra_peak,
                         daily_touch=daily_touch, intra_touch=intra_touch,
                         hidden=intra_touch and not daily_touch))
    D = pd.DataFrame(rows)

    print("\n[MID-TOUCH @ %.2f]  daily-close view vs intraday view" % args.tp)
    print("  trades touching TP at daily close : %d" % D["daily_touch"].sum())
    print("  trades touching TP intraday       : %d   (+%d the daily close MISSED)"
          % (D["intra_touch"].sum(), D["hidden"].sum()))
    losers = D[D["outcome"] == 0]
    print("  would-be LOSERS touching TP intraday: %d / %d  <-- the key number"
          % (losers["intra_touch"].sum(), len(losers)))
    if D["hidden"].any():
        print("\n  trades that wicked to %.2f intraday but NOT at any daily close:" % args.tp)
        for _, x in D[D["hidden"]].iterrows():
            print("   %-40s entry %.2f  intraday-peak %.2f  -> %s"
                  % (x["q"], x["price"], x["intra_peak"], "WIN" if x["outcome"] == 1 else "LOSS"))

    # take-profit on intraday fills vs hold, on the covered subset
    def econ(use_intraday):
        pnl_hold, pnl_tp = [], []
        for _, r in cov.iterrows():
            tok = str(r["token"]); ep = min(r["price"] + args.entry_cost, 0.99)
            pnl_hold.append(r["outcome"] / ep - 1.0)
            if use_intraday:
                ts, _ = first_touch(intr[tok], r["decision_date"], r["res_date"], args.tp)
                hit = ts is not None
            else:
                h = daily.get(tok); seg = h[(h.date >= r["decision_date"]) & (h.date <= r["res_date"])]
                hit = len(seg) > 1 and seg["p"].iloc[1:].max() >= args.tp
            exitv = (args.tp - args.exit_cost) if hit else r["outcome"]
            pnl_tp.append(exitv / ep - 1.0)
        return np.mean(pnl_hold), np.mean(pnl_tp)

    h_ret, tp_daily = econ(False)
    _, tp_intra = econ(True)
    print("\n[TAKE-PROFIT econ on covered subset, mean per-$1 return]")
    print("  HOLD to settlement            : %+.4f" % h_ret)
    print("  TP@%.2f, DAILY-close fills     : %+.4f" % (args.tp, tp_daily))
    print("  TP@%.2f, INTRADAY 10-min fills : %+.4f" % (args.tp, tp_intra))
    print("  -> intraday %s hold" % ("still trails" if tp_intra < h_ret else "now BEATS"))

    # plot: daily peak vs intraday peak per trade
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#1f9d55" if o == 1 else "#c0392b" for o in D["outcome"]]
    ax[0].scatter(D["daily_peak"], D["intra_peak"], c=colors, s=40)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8); ax[0].axhline(args.tp, color="grey", ls=":", lw=.8)
    ax[0].axvline(args.tp, color="grey", ls=":", lw=.8)
    ax[0].set_xlabel("daily-close peak"); ax[0].set_ylabel("intraday peak")
    ax[0].set_title("Peak price: daily vs intraday (green=won, red=lost)"); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    bars = ["daily touch", "intraday touch", "losers intraday"]
    vals = [D["daily_touch"].sum(), D["intra_touch"].sum(), losers["intra_touch"].sum()]
    ax[1].bar(bars, vals, color=["#888", "#2980b9", "#c0392b"])
    ax[1].set_title("Touches of TP=%.2f (n=%d covered)" % (args.tp, len(cov))); ax[1].set_ylabel("trades")
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_intraday_exit.png", dpi=120)
    D.to_csv(f"{OUT}/regime_intraday_exit.csv", index=False)
    print("\nSaved: regime_intraday_exit.png , regime_intraday_exit.csv")


if __name__ == "__main__":
    main()
