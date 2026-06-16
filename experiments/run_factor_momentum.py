#!/usr/bin/env python3
"""
Factor momentum on the zoo (Ehsani-Linnainmaa) vs the EW-ALL benchmark.

Idea (roadmap Idea 3, OpenAP variant)
-------------------------------------
Ehsani & Linnainmaa: factor returns are positively autocorrelated — last
year's winning factors keep winning. The half-life study (Idea 7) left a
benchmark on the table: EW of all published factors at Sharpe ~1.2 gross.
This study asks whether timing the zoo by its own trailing returns beats
just holding all of it. Five momentum portfolios, all causal (signal from
months t-11..t, held in month t+1), published factors only:

  XS winners   EW of the top 1/3 by trailing 12m return
  XS losers    EW of the bottom 1/3
  XS spread    winners − losers (the classic factor-momentum trade)
  TS winners   EW of factors with positive trailing 12m return
  1m spread    top 1/3 − bottom 1/3 by last month's return (short-horizon)

against EW ALL (every published factor, the Idea-7 byproduct). The deciding
tests: does the spread carry alpha over EW ALL, and does any pre-declared
combo (50/50 EW + vol-matched spread) beat EW ALL's Sharpe?

Data: Chen-Zimmermann OpenAP — 212 predictors' monthly L/S returns
(1926-2024) + publication years.

Honesty box
-----------
• Gross academic L/S returns: no costs, monthly rebalanced, include
  hard-to-short names. The momentum portfolios turn over factor sleeves
  (XS spread doubles leverage); costs hit them harder than EW ALL, so
  gross comparisons FLATTER factor momentum.
• Same hindsight-replication bias as Idea 7 (zoo = survivors); hits all
  portfolios on the same panel, spreads partially net it out.
• Trials: 5 momentum portfolios + 1 benchmark + 1 pre-declared combo on
  one panel = 7.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_ELIGIBLE = 30               # live once this many factors are tradable
SPLIT_YEAR = 2004
LOOKBACK = 12                   # months of trailing return for the signal


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "max_dd": np.nan, "n_months": len(r)}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    sharpe = mu / (sd + 1e-12)
    eq = (1 + r).cumprod()
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": sharpe,
            "t_stat": sharpe * np.sqrt(len(r) / 12.0),
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "n_months": len(r)}


def alpha_vs(r: pd.Series, bench: pd.Series) -> tuple[float, float, float]:
    """Monthly OLS r = a + b*bench; returns (ann alpha, t(alpha), beta)."""
    df = pd.concat([r, bench], axis=1, keys=["r", "b"]).dropna()
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    beta, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
    return beta[0] * 12, beta[0] / se[0], beta[1]


def main():
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns), ["Year"]].astype(float)
    rets = rets[meta.index]

    # causal universe: published ≥1y, with a full trailing-12m history
    age_y = rets.index.year.values[:, None] - meta.Year.values
    published = pd.DataFrame(age_y >= 1, index=rets.index, columns=rets.columns)
    mom12 = rets.rolling(LOOKBACK).sum()
    eligible = (published & mom12.notna()).shift(1).fillna(False)
    live = eligible.sum(axis=1)
    live = live[live >= MIN_ELIGIBLE].index
    logger.info(f"{rets.shape[1]} predictors; live (≥{MIN_ELIGIBLE} eligible): "
                f"{live[0].date()} → {live[-1].date()}")

    sig12 = mom12.shift(1).where(eligible)        # signal known at month start
    sig1 = rets.shift(1).where(eligible)
    rete = rets.where(eligible)

    def xs_legs(sig):
        lo = sig.quantile(1 / 3, axis=1)
        hi = sig.quantile(2 / 3, axis=1)
        win = rete.where(sig.ge(hi, axis=0)).mean(axis=1)
        los = rete.where(sig.le(lo, axis=0)).mean(axis=1)
        return win, los

    win, los = xs_legs(sig12)
    win1, los1 = xs_legs(sig1)
    ports = {
        "XS winners (top 1/3 by 12m)": win,
        "XS losers (bottom 1/3)": los,
        "XS spread (win - los)": win - los,
        "TS winners (12m > 0)": rete.where(sig12 > 0).mean(axis=1),
        "1m spread": win1 - los1,
        "EW ALL (benchmark)": rete.mean(axis=1),
    }
    ports = {k: v.loc[live] for k, v in ports.items()}

    rows, raw = [], []
    for name, r in ports.items():
        s = ann_stats(r)
        rows.append([name, f"{s['ann_ret']:+.2%}", f"{s['ann_vol']:.2%}",
                     f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}",
                     f"{s['max_dd']:.0%}"])
        raw.append({"portfolio": name, **s})
        for tag, sub in [(f"≤{SPLIT_YEAR}", r[r.index.year <= SPLIT_YEAR]),
                         (f">{SPLIT_YEAR}", r[r.index.year > SPLIT_YEAR])]:
            raw.append({"portfolio": f"{name} {tag}", **ann_stats(sub)})

    print("\nFactor momentum on the zoo (EW of factor L/S returns, gross):")
    print(tabulate(rows, headers=["portfolio", "ann ret", "vol", "Sharpe",
                                  "t", "maxDD"], tablefmt="github"))

    sub_rows = [[e["portfolio"], f"{e['ann_ret']:+.2%}",
                 f"{e['sharpe']:.2f}", f"{e['t_stat']:.2f}"]
                for e in raw if str(SPLIT_YEAR) in e["portfolio"]]
    print(f"\nSub-period stability (split {SPLIT_YEAR}):")
    print(tabulate(sub_rows, headers=["portfolio", "ann ret", "Sharpe", "t"],
                   tablefmt="github"))

    # ── the deciding tests vs EW ALL ──────────────────────────────────────
    sp, bench = ports["XS spread (win - los)"], ports["EW ALL (benchmark)"]
    corr = sp.corr(bench)
    a, t_a, b = alpha_vs(sp, bench)
    print(f"\ncorr(XS spread, EW ALL) = {corr:.2f}")
    print(f"XS spread alpha vs EW ALL: {a:+.2%}/yr  t={t_a:.2f}  beta={b:.2f}")

    combo = 0.5 * bench + 0.5 * sp * (bench.std() / sp.std())
    print("\nPre-declared combo (50/50 EW ALL + vol-matched spread):")
    for name, r in [("EW ALL", bench), ("combo 50/50 vol-matched", combo)]:
        s = ann_stats(r)
        print(f"  {name:<26} ann {s['ann_ret']:+.2%}  Sharpe {s['sharpe']:.2f}"
              f"  maxDD {s['max_dd']:.0%}")
        raw.append({"portfolio": name if name != "EW ALL" else "combo input EW",
                    **s})
    raw.append({"portfolio": "XS spread alpha vs EW ALL",
                "ann_ret": a, "t_stat": t_a, "ann_vol": b,
                "sharpe": corr, "max_dd": np.nan, "n_months": len(sp)})

    out = RESULTS_DIR / "factor_momentum.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ gross academic L/S returns; costs hit the momentum portfolios "
          "harder than EW ALL — gross comparisons flatter factor momentum.")


if __name__ == "__main__":
    main()
