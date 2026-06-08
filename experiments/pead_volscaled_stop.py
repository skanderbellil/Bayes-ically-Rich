#!/usr/bin/env python3
"""Per-name volatility-scaled stops vs fixed stops for tradeable PEAD market-neutral.

A fixed 6% stop is ~8 daily sigmas for a sleepy utility but ~2 sigmas for a volatile
growth name — so a fixed stop knocks out high-vol names on noise and barely protects
low-vol ones. A vol-scaled stop sets each position's stop to k x its own pre-announcement
volatility (here k x the 20-day / "monthly" sigma), equalising the stop-out probability
across names. This script asks whether that principled scaling beats a fixed stop at a
matched average level. Entry t+1, hold-to-next-earnings (63d), net of Alpaca costs.

Requires data/raw/pead/ (run experiments/run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths, signals
from experiments.pead_dynamic_exit import COST, _stats

HOLD = 63
VOL_WIN = 60          # trading days of pre-announcement daily returns for sigma
STOP_FLOOR, STOP_CAP = 2.0, 25.0   # sane bounds on a vol-scaled stop (%)


def build_paths_vol():
    """Like build_paths but also returns each position's pre-announcement monthly sigma (%)."""
    panel = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
    bucket = panel.drop_duplicates("ticker").set_index("ticker")["market_cap"].to_dict()
    names = [t for t, b in bucket.items() if b in ("Mega Cap", "Large Cap")]

    rows, meta = [], []
    for t in names:
        e = signals.load_earnings_signal(paths.RAW_DIR / f"{t}__earnings.csv")
        px = signals.load_prices(paths.RAW_DIR / f"{t}__prices.csv")
        if e is None or px is None:
            continue
        px = px.sort_index()
        dret = np.log(px["Close"]).diff()
        for ann, row in e.iterrows():
            sue = row.get("Surprise(%)")
            if pd.isna(sue):
                continue
            fut = px[px.index > ann]
            if len(fut) < 3:
                continue
            past = dret[dret.index <= ann].dropna()
            if len(past) < 20:
                continue
            sigma_m = past.iloc[-VOL_WIN:].std() * np.sqrt(20) * 100   # monthly sigma, %
            entry = fut["Close"].iloc[1]; cl = fut["Close"].values
            path = np.full(HOLD + 1, np.nan)
            for d in range(1, min(HOLD, len(fut) - 1) + 1):
                path[d] = np.log(cl[d] / entry) * 100
            rows.append(path); meta.append((pd.Timestamp(ann).tz_localize(None), t, sue, sigma_m))
    P = np.array(rows)
    M = pd.DataFrame(meta, columns=["ann", "ticker", "sue", "sigma_m"])
    M["season"] = M["ann"].dt.to_period("Q"); M["year"] = M["ann"].dt.year
    return P, M.reset_index(drop=True)


def _exit_ret(c, stop):
    """Signal-direction exit return given the cumret path c and a stop level (%)."""
    s = np.where(c <= -stop)[0]
    return -stop if len(s) else c[-1]


def ls_season(idx, P, M, fixed=None, k=None):
    sue = M.loc[idx, "sue"].values
    sig = M.loc[idx, "sigma_m"].values
    hi, lo = np.quantile(sue, 0.8), np.quantile(sue, 0.2)
    idx = np.array(idx)

    def leg(members, msig, sign):
        out = []
        for i, sg in zip(members, msig):
            c = sign * P[i, 1:HOLD + 1]; c = c[~np.isnan(c)]
            if len(c) == 0:
                continue
            stop = fixed if fixed is not None else np.clip(k * sg, STOP_FLOOR, STOP_CAP)
            out.append(_exit_ret(c, stop))
        return np.nanmean(out) if out else np.nan

    L = leg(idx[sue >= hi], sig[sue >= hi], +1.0)
    S = leg(idx[sue <= lo], sig[sue <= lo], -1.0)
    return L + S


def run(P, M, years, fixed=None, k=None):
    sp = [ls_season(list(g.index), P, M, fixed, k)
          for _, g in M[M["year"].isin(years)].groupby("season") if len(g) >= 8]
    return _stats(np.array(sp) / 100 - COST)


def avg_stop(P, M, years, k):
    """Mean applied stop level (%) for a vol-scaled rule, to match against fixed."""
    sub = M[M["year"].isin(years)]
    return float(np.clip(k * sub["sigma_m"].values, STOP_FLOOR, STOP_CAP).mean())


def main():
    P, M = build_paths_vol()
    oos = list(range(2014, 2027)); allyrs = list(range(2000, 2027))
    print(f"Built {len(M)} positions with pre-announcement vol "
          f"(median monthly sigma {M['sigma_m'].median():.1f}%, "
          f"IQR {M['sigma_m'].quantile(.25):.1f}-{M['sigma_m'].quantile(.75):.1f}%)\n")

    print("Fixed stops (baseline):")
    print(f"  {'stop':>6}{'OOS Ann':>9}{'OOS Shrp':>9}{'OOS MaxDD':>10}")
    for s in (4, 5, 6, 8):
        b = run(P, M, oos, fixed=s)
        print(f"  {s:>5}%{b[0]:>+9.1%}{b[1]:>9.2f}{b[2]:>+10.1%}")

    print("\nVolatility-scaled stops (stop = k x name's monthly sigma, clipped 2-25%):")
    print(f"  {'k':>5}{'avg stop':>9}{'OOS Ann':>9}{'OOS Shrp':>9}{'OOS MaxDD':>10}")
    for k in (0.5, 0.6, 0.75, 0.9, 1.1, 1.3):
        b = run(P, M, oos, k=k)
        print(f"  {k:>5.2f}{avg_stop(P, M, oos, k):>8.1f}%{b[0]:>+9.1%}{b[1]:>9.2f}{b[2]:>+10.1%}")

    # Head-to-head at a matched ~6% average stop
    k6 = 0.75
    while avg_stop(P, M, oos, k6) < 6.0:
        k6 += 0.02
    fx = run(P, M, oos, fixed=round(avg_stop(P, M, oos, k6)))
    vs = run(P, M, oos, k=k6)
    print(f"\nHead-to-head at ~{avg_stop(P,M,oos,k6):.1f}% average stop (OOS):")
    print(f"  Fixed {round(avg_stop(P,M,oos,k6))}%      : Ann {fx[0]:+.1%}  Sharpe {fx[1]:.2f}  MaxDD {fx[2]:+.1%}")
    print(f"  Vol-scaled k={k6:.2f}: Ann {vs[0]:+.1%}  Sharpe {vs[1]:.2f}  MaxDD {vs[2]:+.1%}")
    print(f"  Full-sample Sharpe — fixed {fx if False else run(P,M,allyrs,fixed=round(avg_stop(P,M,oos,k6)))[1]:.2f}"
          f"  vs vol-scaled {run(P,M,allyrs,k=k6)[1]:.2f}")


if __name__ == "__main__":
    main()
