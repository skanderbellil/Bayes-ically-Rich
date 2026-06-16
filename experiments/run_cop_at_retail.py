#!/usr/bin/env python3
"""
PRE-REGISTERED: can the cop_at edge be captured LONG-ONLY via real ETFs?

The agent hunt found cash-based operating profitability (cop_at) is the one
factor clearing t>3 cross-region (US t=9 gross, 6.8 net). But it is a LONG-
SHORT academic factor. This tests how much survives the honest retail
translation: long-only, real ETF screen, net of real cost, beta-adjusted.

PRE-REGISTRATION (fixed before viewing results):
  PROXY ETFs (long-only, Alpaca): COWZ (Pacer US Cash Cows 100 — explicit
    free-cash-flow / cash-profitability screen, the closest cop_at proxy) and
    QUAL (iShares MSCI USA Quality — profitability/leverage/earnings-stability).
    Secondary: DGRW, SPHQ. Benchmark: SPY. Returns are NET of expense ratios.
  METRIC: beta-adjusted appraisal IR (alpha / idiosyncratic risk) vs SPY,
    with t(alpha). This is the honest edge after stripping equity beta.
  HYPOTHESIS H1: COWZ has POSITIVE alpha vs SPY (captures part of cop_at's
    long leg). H2: but t(alpha) is FAR below the academic factor's t (the
    long-only ETF loses the short leg and adds screen noise + beta).
  BAR reminder: our discovery standard is t>3. Pre-registered prediction:
    the long-only ETF will NOT clear t>3 — it will show a positive but
    sub-bar alpha, quantifying how much of the edge survives long-only.
"""
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import _bootstrap  # noqa: F401

DATA = _bootstrap.ROOT / "datasets"
ANN, RF = 252, 0.04


def monthly(px_col):
    s = px_col.dropna()
    return (1 + s.pct_change()).resample("ME").apply(lambda x: x.prod() - 1).dropna()


def appraisal(r, b):
    d = pd.concat([r, b], axis=1, keys=["r", "b"]).dropna()
    if len(d) < 24:
        return dict(alpha=np.nan, beta=np.nan, ir=np.nan, t=np.nan, n=len(d))
    X = np.column_stack([np.ones(len(d)), d["b"].values])
    c, *_ = np.linalg.lstsq(X, d["r"].values, rcond=None)
    res = d["r"].values - X @ c
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * res.var(ddof=2))
    return dict(alpha=c[0] * 12, beta=c[1],
                ir=c[0] * 12 / (res.std() * np.sqrt(12) + 1e-12),
                t=c[0] / se[0], n=len(d))


def main():
    px = pd.read_csv(DATA / "quality_etfs.csv", index_col=0, parse_dates=True)
    spy_m = monthly(px["SPY"])
    rows = []
    for t in ["COWZ", "QUAL", "DGRW", "SPHQ", "VFLO"]:
        if t not in px:
            continue
        rm = monthly(px[t])
        b = spy_m.reindex(rm.index)
        a = appraisal(rm, b)
        # raw vs SPY (total return comparison over common window)
        idx = rm.index.intersection(spy_m.index)
        er = rm.loc[idx]; sr = spy_m.loc[idx]
        cagr_e = (1 + er).prod() ** (12 / len(er)) - 1
        cagr_s = (1 + sr).prod() ** (12 / len(sr)) - 1
        rows.append([t, f"{rm.index[0].date()}", a["n"], f"{cagr_e:+.1%}",
                     f"{cagr_s:+.1%}", f"{a['alpha']:+.1%}", f"{a['beta']:.2f}",
                     f"{a['ir']:.2f}", f"{a['t']:.2f}"])
    print("\nPRE-REGISTERED: long-only cop_at proxies vs SPY, beta-adjusted, "
          "net of ETF fees:")
    print(tabulate(rows, headers=["ETF", "from", "n_mo", "ETF CAGR", "SPY CAGR",
                                  "alpha", "beta", "apprIR", "t(alpha)"],
                   tablefmt="github"))
    print("\nVERDICT: H1 = does COWZ (closest cop_at proxy) show POSITIVE alpha? "
          "H2 = is t(alpha) far below the academic factor's t=9 (i.e. the long-"
          "only ETF loses the short leg)? Our bar is t>3 — the pre-registered "
          "prediction is the long-only ETF does NOT clear it, quantifying how "
          "much of the cop_at edge survives a retail long-only translation.")


if __name__ == "__main__":
    main()
