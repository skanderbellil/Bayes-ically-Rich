#!/usr/bin/env python3
"""
Is the QLD pod worth the hustle vs just holding the bench?

Once the vehicle is levered QQQ, the benchmark is QLD itself — and since
Sharpe is scale-invariant, the fair null is stricter than buy & hold: a
CONSTANT static QLD exposure matched to the strategy's own realized risk.
Two matchings per strategy:

  vol-matched — static w = vol(strategy) / vol(QLD): same volatility, so
                any CAGR edge is pure timing value;
  DD-matched  — static w chosen so the static path has the same max
                drawdown: same worst-case pain, compare wealth.

The dial only justifies its complexity if it beats the matched static
exposure decisively (joint block bootstrap). Inputs are the saved daily
books from run_hiring_policies.py --underlying QLD [--rich-macro
--with-liquidity]; run those first.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"


def stats(r: pd.Series) -> dict:
    cum = (1 + r).cumprod()
    dd = float(((cum / cum.cummax()) - 1).min())
    cagr = float(cum.iloc[-1] ** (252 / len(r)) - 1)
    return {"cagr": round(cagr, 3),
            "vol": round(float(r.std() * np.sqrt(252)), 3),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def dd_matched_weight(bench: pd.Series, target_dd: float) -> float:
    lo, hi = 0.01, 1.0
    for _ in range(40):
        w = (lo + hi) / 2
        cum = (1 + w * bench).cumprod()
        if float(((cum / cum.cummax()) - 1).min()) < target_dd:
            hi = w
        else:
            lo = w
    return w


def main():
    qld = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)["QLD"].dropna()
    base = pd.read_csv(OUT_DIR / "daily_returns_QLD.csv",
                       index_col=0, parse_dates=True)
    rich = pd.read_csv(OUT_DIR / "daily_returns_QLD_rich_liq.csv",
                       index_col=0, parse_dates=True)
    idx = base.index
    bench = qld.pct_change().reindex(idx).fillna(0.0)
    qqq = (load_etf_universe_prices()["QQQ"].pct_change()
           .reindex(idx).fillna(0.0))

    strats = {"pod raw (base)": base["raw"], "pod sized (base)": base["sized"],
              "pod raw (rich)": rich["raw"], "pod sized (rich)": rich["sized"]}

    rows = {}
    for name, s in strats.items():
        rows[name] = stats(s)
        wv = float(s.std() / bench.std())
        rows[f"  static {wv:.2f}x QLD (vol-matched)"] = stats(wv * bench)
        wd = dd_matched_weight(bench, rows[name]["max_dd"])
        rows[f"  static {wd:.2f}x QLD (DD-matched)"] = stats(wd * bench)
    rows["QLD buy & hold"] = stats(bench)
    rows["QQQ buy & hold"] = stats(qqq)

    print(f"JUSTIFICATION vs THE BENCH ({idx[0].date()} → {idx[-1].date()})")
    print("=" * 78)
    print(pd.DataFrame(rows).T.to_string())

    # joint block bootstrap: strategy vs QLD B&H Sharpe, and terminal wealth
    # vs its (full-sample) vol-matched static weight
    rng = np.random.default_rng(7)
    X = pd.DataFrame({"b": bench, **strats}).values
    T, n_boot = len(X), 2000
    names = list(strats)
    wins_sh = np.zeros(len(names))
    wins_w = np.zeros(len(names))
    w_static = [X[:, i + 1].std() / X[:, 0].std() for i in range(len(names))]
    for _ in range(n_boot):
        ix, t = np.empty(T, dtype=int), 0
        while t < T:
            s0 = rng.integers(0, T)
            for j in range(rng.geometric(1 / 63)):
                if t >= T:
                    break
                ix[t] = (s0 + j) % T
                t += 1
        R = X[ix]
        b = R[:, 0]
        for i in range(len(names)):
            s = R[:, i + 1]
            wins_sh[i] += s.mean() / s.std() > b.mean() / b.std()
            wins_w[i] += (1 + s).prod() > (1 + w_static[i] * b).prod()

    print("\nJoint block bootstrap (2000 draws, mean block 63d):")
    for i, n in enumerate(names):
        print(f"  {n:<18} P(Sharpe > QLD B&H): {wins_sh[i] / n_boot:.2f} | "
              f"P(wealth > vol-matched static): {wins_w[i] / n_boot:.2f}")
    print("\nThe dial earns its complexity only if those probabilities are "
          "decisive (~0.9+);\n~0.6-0.7 at +1pp/yr is benchmark parity with "
          "extra moving parts.")


if __name__ == "__main__":
    main()
