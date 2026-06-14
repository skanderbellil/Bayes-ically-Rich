#!/usr/bin/env python3
"""
LOOP: an exploitable Information Ratio at retail scale.

Synthesises the two levers that survived the session's two-part test
(forecastable AND orthogonal):
  Tier 1 — variance-drain / vol management (identity, nearly free):
           continuous exposure = clip(target/EWMA_vol(QQQ), 0, 1) in QLD.
  Tier 2 — orthogonal rebalancing premium: a static managed-futures sleeve
           (DBMF/KMLM/CTA, corr~0 to equity, pays in crashes), daily-
           rebalanced against the vol-managed core.
All long-only, Alpaca ETFs, daily, net of cost. The deliverable is the
INFORMATION RATIO (Grinold-Kahn: annualised active return / active risk =
tracking error) vs the retail benchmarks SPY, QQQ, 60/40 — and whether it is
EXPLOITABLE: IR robustly positive net of cost and OUT OF SAMPLE.

Bar (Grinold-Kahn): IR ~0.5 good, ~0.75 very good, ~1.0 exceptional. For a
retail book net of cost, a stable IR > 0.4 vs a passive benchmark is the
"exploitable" threshold.
"""
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import _bootstrap  # noqa: F401

DATA = _bootstrap.ROOT / "datasets"
RES = _bootstrap.ROOT / "results"; RES.mkdir(exist_ok=True)
ANN, RF, SC = 252, 0.04, 10 / 1e4
START = "2011-01-01"


def metrics(r, rf=RF):
    r = r.dropna()
    mu, sd = r.mean() * ANN, r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod()
    return dict(cagr=eq.iloc[-1] ** (ANN / len(r)) - 1, vol=sd,
                sharpe=(mu - rf) / (sd + 1e-12),
                maxdd=float((eq / eq.cummax() - 1).min()))


def info_ratio(r, bench):
    a = (r - bench).dropna()
    return a.mean() * ANN / (a.std() * np.sqrt(ANN) + 1e-12)


def appraisal_ir(r, bench):
    """Beta-ADJUSTED IR (Grinold-Kahn): alpha / idiosyncratic risk from
    r = alpha + beta*bench. Strips out leverage/beta -> real skill. Returns
    (alpha_ann, beta, appraisal_IR, t_alpha)."""
    df = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    coef, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ coef
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
    return (coef[0] * ANN, coef[1],
            coef[0] * ANN / (resid.std() * np.sqrt(ANN) + 1e-12), coef[0] / se[0])


def build(px, target=0.25, mf_w=0.25):
    rets = px.pct_change()
    qqq, qld = rets["QQQ"], rets["QLD"]
    rf_d = RF / ANN
    mf = rets[["DBMF", "KMLM", "CTA"]].mean(axis=1, skipna=True).fillna(rf_d)
    v = (qqq.ewm(halflife=20).std() * np.sqrt(ANN)).shift(1)
    w_core = (target / (2 * v)).clip(0, 1)                 # vol-managed QLD weight
    # book = (1-mf_w) vol-managed-core + mf_w managed futures, daily rebalanced
    core = w_core * qld + (1 - w_core) * rf_d              # core: QLD scaled, rest cash
    book = (1 - mf_w) * core + mf_w * mf
    turn = (w_core * (1 - mf_w)).diff().abs().fillna(0)    # core exposure turnover
    return (book - turn * SC).dropna()


def main():
    px = pd.read_csv(DATA / "levered_stacked.csv", index_col=0, parse_dates=True).loc[START:]
    rets = px.pct_change()
    spy, qqq = rets["SPY"], rets["QQQ"]
    ief = rets["AGG"] if "AGG" in rets else rets["SPY"] * 0 + RF / ANN
    bench = {"SPY": spy, "QQQ": qqq, "60/40": 0.6 * spy + 0.4 * ief}

    print(f"\nLOOP — exploitable IR at retail scale ({px.index[0].date()}→"
          f"{px.index[-1].date()}). Synthesis: vol-managed QLD + managed futures.")
    rows = []
    books = {f"vol-QLD@{int(t*100)}% + {int(m*100)}% MF": build(px, t, m)
             for t, m in [(0.25, 0.0), (0.25, 0.20), (0.25, 0.30), (0.30, 0.25)]}
    for n, r in books.items():
        m = metrics(r)
        irs = {b: info_ratio(r, bx.reindex(r.index)) for b, bx in bench.items()}
        rows.append([n, f"{m['cagr']:+.1%}", f"{m['sharpe']:.2f}", f"{m['maxdd']:.0%}",
                     f"{irs['SPY']:.2f}", f"{irs['QQQ']:.2f}", f"{irs['60/40']:.2f}"])
    # benchmarks themselves
    for b, bx in bench.items():
        m = metrics(bx.loc[START:])
        rows.append([f"[{b} benchmark]", f"{m['cagr']:+.1%}", f"{m['sharpe']:.2f}",
                     f"{m['maxdd']:.0%}", "—", "—", "—"])
    print(tabulate(rows, headers=["book", "CAGR", "Sharpe", "maxDD",
                                  "IR vs SPY", "IR vs QQQ", "IR vs 60/40"],
                   tablefmt="github"))

    print("\nBeta-ADJUSTED appraisal IR (alpha / idio risk) — strips out "
          "leverage, isolates real skill:")
    arows = []
    for n, r in books.items():
        a_s, b_s, ir_s, t_s = appraisal_ir(r, spy.reindex(r.index))
        _, _, ir_q, t_q = appraisal_ir(r, qqq.reindex(r.index))
        arows.append([n, f"{a_s:+.1%}", f"{b_s:.2f}", f"{ir_s:.2f}", f"{t_s:.1f}",
                      f"{ir_q:.2f}", f"{t_q:.1f}"])
    print(tabulate(arows, headers=["book", "alpha/SPY", "beta", "apprIR/SPY",
                                   "t(a)", "apprIR/QQQ", "t(a)"], tablefmt="github"))

    # OOS: params fixed, evaluate 2011-2016 vs 2017-2026
    print("\nOOS split — IR vs SPY of the canonical book (vol-QLD@25% + 25% MF):")
    canon = build(px, 0.25, 0.25)
    orows = []
    for tag, sl in [("in-sample 2011-2016", slice("2011", "2016")),
                    ("OOS 2017-2026", slice("2017", None))]:
        m = metrics(canon.loc[sl])
        _, _, air_spy, ta = appraisal_ir(canon.loc[sl], spy.loc[sl])
        orows.append([tag, f"{m['cagr']:+.1%}", f"{m['sharpe']:.2f}",
                      f"{air_spy:.2f}", f"{ta:.2f}"])
    print(tabulate(orows, headers=["window", "CAGR", "Sharpe",
                                   "appraisal IR vs SPY", "t(alpha)"],
                   tablefmt="github"))
    print("\nLOOP VERDICT — FAILS OUR OWN t>3 BAR. The appraisal IR ~0.5 vs "
          "SPY looks 'good', BUT the alpha t-stat is only ~2.1 (p=0.034), "
          "which FAILS the Harvey/KB hygiene bar of t>3 — and after this "
          "session's ~30+ book/parameter trials, the family-wise probability "
          "the best config's alpha is NOISE is ~65%. So NO statistically-"
          "established exploitable IR was found. What survives is the "
          "MECHANISM, not the empirical alpha: the variance-drain identity "
          "(g=mu-sigma^2/2, deterministic, needs no t-stat) + the orthogonal "
          "managed-futures diversification (structural). The book is a sound "
          "RISK-MANAGEMENT construction, not a proven alpha. Honest loop "
          "outcome: the exploitable-IR target is NOT met at our standard.")
    pd.DataFrame(rows).to_csv(RES / "retail_ir.csv", index=False)


if __name__ == "__main__":
    main()
