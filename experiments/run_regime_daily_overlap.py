#!/usr/bin/env python3
"""
Geo-calm sleeve — daily-overlapping observations
================================================

The validated book uses ONE decision per market (the mid 7 days before
resolution). That throws away most of the information: a market that is cheap and
in a calm regime is an entry opportunity on EVERY such day it is open. This builds
the daily-overlapping panel — one observation per (market, day) where, using only
causal data, the YES mid is <= cap, the GPR regime is calm, and the market
resolves in [2,45] days — and measures the edge across all of them.

Why it doesn't just multiply your sample: same-market days share one settlement,
so the observations are heavily autocorrelated. The honest significance comes
from a MARKET-CLUSTERED bootstrap (resample whole markets), whose effective n is
closer to the number of distinct markets than to the raw day count. The value is
robustness — does the edge survive being measured at every entry day, not just
the hand-picked one — plus a fairer t-stat.

Usage:
  python experiments/run_regime_daily_overlap.py
  python experiments/run_regime_daily_overlap.py --max-price 0.35 --panel results/polymarket/regime_calibration_panel.csv
"""
from __future__ import annotations
import argparse, glob, math, tarfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

DEFAULT_PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
GPR = "datasets/gpr_daily.csv.gz"
OUT = "data/polymarket_calibration_snapshot"
CACHE = sorted(glob.glob(f"{OUT}/token_history_cache_*.tar.gz"))[-1]


def daily_calm():
    """Causal daily GPR-calm series (level_or_vol) + calm-depth z, on a D calendar."""
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index().asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean()
    lvl_med = lvl.rolling(365, min_periods=180).median()
    lvl_std = lvl.rolling(365, min_periods=180).std()
    lvl_calm = lvl <= lvl_med
    vol = g.diff().rolling(45, min_periods=15).std()
    vol_calm = vol <= vol.rolling(365, min_periods=180).median()
    calm = lvl_calm | vol_calm
    z = (lvl_med - lvl) / lvl_std
    return pd.DataFrame({"date": g.index, "calm": calm.values, "z": z.values}).dropna(subset=["calm"])


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def clustered_boot(df, cost, n=5000):
    """Resample whole markets (clusters); CI/effective-t for mean net edge."""
    df = df.assign(e=df.outcome - (df.price + cost).clip(lower=0.01))
    groups = [x["e"].values for _, x in df.groupby("token")]
    means = np.array([gg.mean() for gg in groups]); ws = np.array([len(gg) for gg in groups])
    rng = np.random.default_rng(0); K = len(groups)
    idx = rng.integers(0, K, size=(n, K))
    blk = (means[idx] * ws[idx]).sum(1) / ws[idx].sum(1)
    obs = (means * ws).sum() / ws.sum()
    return obs, np.percentile(blk, 5), np.percentile(blk, 95), float((blk <= 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--base-cost", type=float, default=0.03)
    ap.add_argument("--days-min", type=int, default=2)
    ap.add_argument("--days-max", type=int, default=45)
    args = ap.parse_args()
    c = args.base_cost

    P = pd.read_csv(args.panel, parse_dates=["decision_date", "res_date"])
    geo = P[(P.domain == "geopolitics") & (P.leg == "yes")].copy()
    calm = daily_calm().set_index("date")
    tf = tarfile.open(CACHE); names = set(tf.getnames())

    rows = []
    for _, r in geo.iterrows():
        nm = "polymarket/token_%s.csv" % r["token"]
        if nm not in names:
            continue
        h = pd.read_csv(tf.extractfile(tf.getmember(nm)), parse_dates=["date"]).dropna(subset=["p"])
        for _, d in h.iterrows():
            dt = d["date"]
            dtr = (r["res_date"] - dt).days
            if not (args.days_min <= dtr <= args.days_max):
                continue
            if d["p"] > args.max_price or d["p"] < 0.02:
                continue
            if dt not in calm.index or not bool(calm.at[dt, "calm"]):
                continue
            rows.append(dict(token=r["token"], date=dt, price=d["p"], outcome=r["outcome"],
                             res_date=r["res_date"], dtr=dtr, z=calm.at[dt, "z"]))
    D = pd.DataFrame(rows)

    print("=" * 92)
    print("GEO-CALM SLEEVE — DAILY-OVERLAPPING PANEL   (panel: %s)" % args.panel)
    print("  geopolitics · YES · price<=%.2f · GPR-calm · resolves in [%d,%d]d · entry %.0f¢"
          % (args.max_price, args.days_min, args.days_max, c * 100))
    print("=" * 92)
    if D.empty:
        print("no overlapping observations"); return
    nmk = D["token"].nunique()
    ep = (D.price + c).clip(0.01); edge = D.outcome - ep
    obs, lo, hi, p0 = clustered_boot(D, c)
    print("  raw observations (market-days) : %d" % len(D))
    print("  distinct markets               : %d   (avg %.1f entry-days/market)" % (nmk, len(D) / nmk))
    print("  pooled win rate / avg price     : %.2f / %.3f" % (D.outcome.mean(), D.price.mean()))
    print("  net edge / $1 (pooled)          : %+.4f   naive t %+.2f  (IGNORES clustering)" % (edge.mean(), tstat(edge)))
    print("  market-clustered 90%% CI         : [%+.4f, %+.4f]   P(edge<=0)=%.3f" % (lo, hi, p0))
    print("  -> clustered effective n ~ %d markets, not %d day-rows" % (nmk, len(D)))

    # edge by days-to-resolution (does the entry-timing matter?)
    print("\n[BY HORIZON] net edge/$1 by days-to-resolution at entry")
    D["hb"] = pd.cut(D.dtr, [1, 7, 14, 30, 45], labels=["2-7d", "8-14d", "15-30d", "31-45d"])
    for hb, s in D.groupby("hb", observed=True):
        e = s.outcome - (s.price + c).clip(0.01)
        print("   %-7s n=%4d  markets=%2d  win=%.2f  edge=%+.4f (naive t %+.2f)"
              % (hb, len(s), s.token.nunique(), s.outcome.mean(), e.mean(), tstat(e)))

    # compare to the single-decision book
    from run_regime_sizing_sensitivity import gpr_calm_book
    book = gpr_calm_book(P, max_price=args.max_price)
    be = book.outcome - (book.price + c).clip(0.01)
    print("\n[VS single-decision book] book n=%d  edge=%+.4f (t %+.2f)  |  daily-overlap pooled edge=%+.4f"
          % (len(book), be.mean(), tstat(be), edge.mean()))

    # plot: edge vs price level on the overlapping panel (calibration of entry price)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.7))
    bins = np.arange(0, args.max_price + 0.05, 0.05)
    D["pb"] = pd.cut(D.price, bins)
    gb = D.groupby("pb", observed=True).agg(n=("outcome", "size"), win=("outcome", "mean"),
                                            price=("price", "mean"))
    ax[0].bar([str(i) for i in gb.index], gb["win"], color="#1f9d55", alpha=.8, label="realised win rate")
    ax[0].plot([str(i) for i in gb.index], gb["price"], "ks--", label="avg entry price")
    ax[0].set_title("Daily-overlap: win rate vs entry price (gap = edge)"); ax[0].tick_params(axis="x", rotation=45, labelsize=7)
    ax[0].legend(fontsize=8); ax[0].set_ylabel("probability")
    ax[1].hist(D["dtr"], bins=20, color="#2980b9", alpha=.8)
    ax[1].set_title("Entry opportunities by days-to-resolution"); ax[1].set_xlabel("days to resolution"); ax[1].set_ylabel("market-days")
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_daily_overlap.png", dpi=120)
    D.to_csv(f"{OUT}/regime_daily_overlap.csv", index=False)
    print("\nSaved: regime_daily_overlap.png , regime_daily_overlap.csv")


if __name__ == "__main__":
    main()
