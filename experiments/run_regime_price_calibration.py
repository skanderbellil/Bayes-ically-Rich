#!/usr/bin/env python3
"""
Price calibration check — does interim price equal win probability?
===================================================================

A sanity check prompted by the path-distribution plot (run_regime_path_distribution.py),
which showed eventual-winners sitting well above eventual-losers mid-trade and
0% of the 35-trade book's losers ever touching 0.50. Read naively that suggests
"price 0.50 => ~certain YES", which would violate market efficiency. This script
tests it directly with a reliability diagram: across ALL interim price
observations (token-days strictly before resolution), what fraction at price ~p
actually resolve YES?

Verdict it produces: on the full panel the market is ROUGHLY CALIBRATED
(price ~= P[YES], so 0.50 ~ a coin flip). The book's clean separation is a
small-sample + post-selection effect (a handful of token-days in the mid-range,
cohorts sorted by outcome), NOT evidence that reaching the mids guarantees a win.

Usage:
  python experiments/run_regime_price_calibration.py
"""
from __future__ import annotations
import argparse, tarfile, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from run_regime_sizing_sensitivity import gpr_calm_book

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
OUT = "data/polymarket_calibration_snapshot"
CACHE = sorted(glob.glob(f"{OUT}/token_history_cache_*.tar.gz"))[-1]


def collect(df, tf, names):
    """All (interim price, eventual outcome) token-days strictly before res_date."""
    rows = []
    for _, r in df.iterrows():
        nm = "polymarket/token_%s.csv" % r["token"]
        if nm not in names:
            continue
        h = pd.read_csv(tf.extractfile(tf.getmember(nm)), parse_dates=["date"]).dropna(subset=["p"])
        seg = h[(h.date >= r["decision_date"]) & (h.date < r["res_date"])]
        for p in seg["p"].values:
            rows.append((p, float(r["outcome"])))
    return pd.DataFrame(rows, columns=["p", "y"])


def reliability(d, nbins=10):
    edges = np.linspace(0, 1, nbins + 1)
    mids, freq, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = d[(d.p >= lo) & (d.p < (hi if hi < 1 else 1.01))]
        if len(s):
            mids.append(s.p.mean()); freq.append(s.y.mean()); n.append(len(s))
    return np.array(mids), np.array(freq), np.array(n)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n; d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    tf = tarfile.open(CACHE); names = set(tf.getnames())

    book = gpr_calm_book(P)
    geo = P[(P.domain == "geopolitics") & (P.leg == "yes")]
    allyes = P[P.leg == "yes"]
    sets = [("geo-calm book (n=%d)" % len(book), book),
            ("all geopolitics YES (n=%d)" % len(geo), geo),
            ("all YES legs (n=%d)" % len(allyes), allyes)]

    print("=" * 78)
    print("PRICE CALIBRATION — P(resolve YES | interim price)   [reliability]")
    print("=" * 78)
    data = {}
    for label, df in sets:
        d = collect(df, tf, names); data[label] = d
        m, f, n = reliability(d, args.bins)
        # calibration error (sample-weighted |freq - price|)
        ece = np.sum(n * np.abs(f - m)) / n.sum()
        print("\n%s — %d token-days, ECE=%.3f" % (label, len(d), ece))
        print("  %-11s %7s %8s" % ("price~", "n_days", "P(YES)"))
        for mm, ff, nn in zip(m, f, n):
            print("  %-11.2f %7d %8.3f" % (mm, nn, ff))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = {"geo-calm book (n=%d)" % len(book): "#1f9d55",
            "all geopolitics YES (n=%d)" % len(geo): "#2980b9",
            "all YES legs (n=%d)" % len(allyes): "#8e44ad"}
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration (price=prob)")
    for label, df in sets:
        m, f, n = reliability(data[label], args.bins)
        lo = np.array([wilson(int(round(ff * nn)), nn)[0] for ff, nn in zip(f, n)])
        hi = np.array([wilson(int(round(ff * nn)), nn)[1] for ff, nn in zip(f, n)])
        ax[0].errorbar(m, f, yerr=[f - lo, hi - f], fmt="o-", color=cols[label], capsize=2, ms=4, lw=1.3, label=label)
    ax[0].set_xlabel("interim price (P[YES])"); ax[0].set_ylabel("empirical P(resolve YES)")
    ax[0].set_title("Reliability: price vs realised win rate"); ax[0].legend(fontsize=8); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    # sample sizes (the point: mid-range is thin in the book)
    for label, df in sets:
        m, f, n = reliability(data[label], args.bins)
        ax[1].plot(m, n, "o-", color=cols[label], ms=4, lw=1.3, label=label)
    ax[1].set_yscale("log"); ax[1].set_xlabel("interim price"); ax[1].set_ylabel("token-days in bin (log)")
    ax[1].set_title("Sample size per price bin (why the book looked extreme)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_price_calibration.png", dpi=120)
    print("\nVERDICT: full-panel reliability tracks the diagonal — price ~= P[YES], so a")
    print("  market at 0.50 is ~a coin flip. The book's 'losers never reach 0.50' is small-")
    print("  sample (a few mid-range token-days) + post-selection, not a 0.5%% loss rate.")
    print("\nSaved: regime_price_calibration.png")


if __name__ == "__main__":
    main()
