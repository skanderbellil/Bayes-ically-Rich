#!/usr/bin/env python3
"""
Geo-calm sleeve — price-cap sensitivity
=======================================

The sleeve only buys YES legs priced <= a cap (default 0.35, "the cheap tail"
where the calm-underpricing bias is strongest). This sweeps that cap to show the
trade-off: a higher cap admits MORE trades (bigger n) but reaches into
better-priced contracts where the edge is thinner. Reports, per cap, the book
size, win rate, net per-$1 edge (+ t and week-clustered bootstrap CI), breakeven
margin (win rate - avg entry price), and a flat-stake capital sim (CAGR/Sharpe/
maxDD). Marginal-bucket view shows the edge of the trades each cap step *adds*.

Usage:
  python experiments/run_regime_pricecap_sensitivity.py
  python experiments/run_regime_pricecap_sensitivity.py --panel results/polymarket/regime_calibration_panel.csv
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book, tilt_weights, sim, weighted_edge

DEFAULT_PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
OUT = "data/polymarket_calibration_snapshot"


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def boot_ci(df, cost, n=5000):
    if len(df) < 3:
        return np.nan, np.nan
    g = df.assign(rwk=df.res_date.dt.to_period("W").astype(str),
                  e=df.outcome - (df.price + cost).clip(lower=0.01))
    wk = [x for _, x in g.groupby("rwk")]
    wm = np.array([x["e"].mean() for x in wk]); ww = np.array([len(x) for x in wk])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(wk), size=(n, len(wk)))
    blk = (wm[idx] * ww[idx]).sum(1) / ww[idx].sum(1)
    return np.percentile(blk, 5), np.percentile(blk, 95)


def metrics_at(P, cap, cost, stake, gate):
    book = gpr_calm_book(P, max_price=cap, gate=gate)
    if not len(book):
        return None, None
    ep = (book["price"] + cost).clip(0.01)
    edge = book["outcome"] - ep
    w = tilt_weights(book["price"].values, 0.0)            # flat
    _, m, _ = sim(book, w, stake=stake, cost=cost)
    lo, hi = boot_ci(book, cost)
    return book, dict(cap=cap, n=len(book), win=book.outcome.mean(), avg_price=book.price.mean(),
                      edge=edge.mean(), t=tstat(edge), lo=lo, hi=hi,
                      be_margin=book.outcome.mean() - book.price.mean(),
                      cagr=m.get("cagr", np.nan), sharpe=m.get("sharpe", np.nan),
                      maxdd=m.get("maxdd", np.nan), final=m.get("final", np.nan))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--base-cost", type=float, default=0.03)
    ap.add_argument("--stake", type=float, default=0.02)
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()
    P = pd.read_csv(args.panel, parse_dates=["decision_date", "res_date"])
    c = args.base_cost
    caps = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]

    print("=" * 104)
    print("GEO-CALM SLEEVE — PRICE-CAP SENSITIVITY   (panel: %s)" % args.panel)
    print("  geopolitics · YES · GPR-calm[%s] · hold 2-45d · entry %.0f¢ · flat %.0f%% stake"
          % (args.gate, c * 100, args.stake * 100))
    print("=" * 104)
    print("  %-6s %4s %6s %8s %9s %8s %9s %10s %7s %7s %7s" %
          ("cap", "n", "win", "avgP", "edge/$1", "t", "BE-marg", "90%CI", "CAGR", "Sharpe", "maxDD"))
    rows = []
    books = {}
    for cap in caps:
        book, m = metrics_at(P, cap, c, args.stake, args.gate)
        if m is None:
            continue
        rows.append(m); books[cap] = book
        ci = "[%+.2f,%+.2f]" % (m["lo"], m["hi"]) if not np.isnan(m["lo"]) else "   n/a"
        tag = "  <-- base" if abs(cap - 0.35) < 1e-9 else ""
        print("  %-6.2f %4d %6.2f %8.3f %+9.4f %+8.2f %+9.3f %10s %+6.0f%% %7.2f %6.0f%%%s" %
              (cap, m["n"], m["win"], m["avg_price"], m["edge"], m["t"], m["be_margin"], ci,
               100 * m["cagr"], m["sharpe"], 100 * m["maxdd"], tag))
    R = pd.DataFrame(rows)

    # marginal buckets: edge of the trades each cap-step ADDS
    print("\n[MARGINAL] edge of the trades ADDED by each cap step (price in (prev, cap])")
    print("  %-14s %4s %7s %9s %8s" % ("added band", "n", "win", "edge/$1", "t"))
    prev = 0.0
    fullbook = gpr_calm_book(P, max_price=max(caps), gate=args.gate)
    for cap in caps:
        band = fullbook[(fullbook.price > prev) & (fullbook.price <= cap)]
        if len(band):
            ep = (band.price + c).clip(0.01); e = band.outcome - ep
            print("  (%.2f,%.2f]%5s %4d %7.2f %+9.4f %+8.2f"
                  % (prev, cap, "", len(band), band.outcome.mean(), e.mean(), tstat(e)))
        prev = cap

    print("\n[READ] higher cap -> more n but watch edge/$1, the breakeven margin, and the marginal")
    print("  bands: if added bands carry weak/negative edge, the cap is buying noise, not signal.")

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    ax2 = ax[0].twinx()
    ax[0].plot(R["cap"], R["edge"], "o-", color="#1f9d55", label="net edge/$1 (left)")
    ax[0].fill_between(R["cap"], R["lo"], R["hi"], color="#1f9d55", alpha=0.15)
    ax2.plot(R["cap"], R["n"], "s--", color="#2980b9", label="n trades (right)")
    ax[0].axvline(0.35, color="grey", ls=":", lw=1); ax[0].axhline(0, color="grey", lw=.6)
    ax[0].set_xlabel("price cap"); ax[0].set_ylabel("net edge/$1", color="#1f9d55"); ax2.set_ylabel("n trades", color="#2980b9")
    ax[0].set_title("Edge/$1 (90% CI) and n vs price cap")
    ax[1].plot(R["cap"], 100 * R["cagr"], "o-", color="#8e44ad", label="CAGR %")
    ax[1].plot(R["cap"], 100 * R["be_margin"], "^-", color="#c0392b", label="breakeven margin %")
    ax[1].axvline(0.35, color="grey", ls=":", lw=1); ax[1].axhline(0, color="grey", lw=.6)
    ax[1].set_xlabel("price cap"); ax[1].set_title("CAGR & breakeven margin vs price cap"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    suf = ("_" + args.tag) if args.tag else ""
    fig.savefig(f"{OUT}/regime_pricecap_sensitivity{suf}.png", dpi=120)
    R.to_csv(f"{OUT}/regime_pricecap_sensitivity{suf}.csv", index=False)
    print("\nSaved: regime_pricecap_sensitivity%s.png , .csv" % suf)


if __name__ == "__main__":
    main()
