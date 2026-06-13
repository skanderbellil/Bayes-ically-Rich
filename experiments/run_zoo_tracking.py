#!/usr/bin/env python3
"""
Tracking the levered zoo with a long-only, no-leverage portfolio.

Idea (user prompt)
------------------
Idea 12's absolute-return winner — EW zoo @10% vol, 3× cap, +10.9%/yr
gross — is unreachable directly (150+ academic L/S sleeves, leverage).
Question: how close can a REACHABLE portfolio get? Constraints: long-only
funds, weights ≥ 0, Σw ≤ 1 (rest in cash at RF), no leverage anywhere.
Objective: minimize tracking error to the levered-zoo stream.

Method
------
Target: the exact Idea-12 construction (EW of published OpenAP factors,
causal 12m vol-target to 10%, cap 3×), monthly, gross.
Two long-only universes:
  A. 12 FF industry portfolios + cash       (full overlap, 1998→2024)
  B. 27 curated ETFs + cash                 (2010→2024: style, defensive,
     sector, bond, gold, commodity, dollar — no levered/inverse funds)
Two fits each:
  in-sample bound   constrained min-TE on the full overlap — the FLOOR of
                    achievable TE (cheating: uses the future)
  walk-forward      trailing 60m re-fit monthly (min 36), weights applied
                    next month — the honest number
Baselines that need no optimizer: cash, SPY, 60/40.

The verdict metric: out-of-sample TE vs the target's own vol. If TE ≈
target vol, the stream is unspanned and "low tracking error" does not
exist in long-only space — knowing that is the point.

Honesty box
-----------
• Target is gross academic; the replica is real funds — if even the
  GROSS stream can't be tracked, the net one can't either (a fortiori).
• Min-TE optimizers love cash (TE to anything is capped by target vol);
  corr and capture are reported so a cash-hugging "tracker" is exposed.
• Trials: 2 universes × 2 fits + 3 baselines = 7 descriptive fits; no
  return-seeking search.
"""
import logging
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA = _bootstrap.ROOT / "datasets"

MIN_ELIGIBLE = 30
TARGET_VOL, LEV_CAP = 0.10, 3.0
FIT_WIN, FIT_MIN = 60, 36
ETFS = ["SPY", "QQQ", "IWM", "IWD", "IWF", "VTV", "VUG", "SPLV", "SPHQ",
        "RSP", "VYM", "VIG", "EFA", "EEM", "VNQ", "GDX", "GLD", "SLV",
        "DBC", "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "UUP", "BIL"]


def build_target() -> pd.Series:
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns), ["Year"]].astype(float)
    rets = rets[meta.index]
    age_y = rets.index.year.values[:, None] - meta.Year.values
    published = pd.DataFrame(age_y >= 1, index=rets.index, columns=rets.columns)
    elig = (published & rets.notna()).shift(1).fillna(False)
    live = elig.sum(axis=1)
    live = live[live >= MIN_ELIGIBLE].index
    ew = rets.where(elig).mean(axis=1).loc[live]
    lev = (TARGET_VOL / (ew.rolling(12).std() * np.sqrt(12))
           ).clip(upper=LEV_CAP).shift(1)
    return (ew * lev).dropna()


def min_te_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    res = minimize(lambda w: float(np.mean((X @ w - y) ** 2)),
                   np.full(n, 0.5 / n), bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "ineq",
                                 "fun": lambda w: 1.0 - w.sum()}],
                   method="SLSQP", options={"maxiter": 300})
    return res.x


def stats_row(name, replica, target, weights_note=""):
    df = pd.concat([replica, target], axis=1, keys=["r", "t"],
                   sort=False).dropna()
    diff = df["r"] - df["t"]
    te = diff.std() * np.sqrt(12)
    corr = df["r"].corr(df["t"])
    mu_r, mu_t = df["r"].mean() * 12, df["t"].mean() * 12
    sd_r = df["r"].std() * np.sqrt(12)
    return ([name, len(df), f"{te:.2%}", f"{corr:+.2f}", f"{mu_r:+.2%}",
             f"{mu_t:+.2%}", f"{sd_r:.2%}", weights_note],
            {"fit": name, "n_months": len(df), "te_ann": te, "corr": corr,
             "ret_replica": mu_r, "ret_target": mu_t, "vol_replica": sd_r})


def main():
    target = build_target()
    logger.info(f"target: levered EW zoo, {target.index[0].date()} → "
                f"{target.index[-1].date()}, ann vol "
                f"{target.std()*np.sqrt(12):.2%}")

    ff = pd.read_csv(DATA / "ff_factors_monthly.csv", index_col=0,
                     parse_dates=True)
    rf = ff["RF"]
    ind = pd.read_csv(DATA / "ff_industry12_monthly.csv", index_col=0,
                      parse_dates=True)
    etf_px = pd.read_csv(DATA / "etf_universe_prices.csv.gz", index_col=0,
                         parse_dates=True)[ETFS]
    etf = etf_px.resample("ME").last().pct_change().iloc[1:]
    etf.index = etf.index + pd.offsets.MonthEnd(0)

    universes = {"A: FF12 industries + cash": ind,
                 "B: 27 ETFs + cash": etf}
    rows, raw = [], []

    for uname, assets in universes.items():
        common = assets.dropna().index.intersection(target.index)
        X_all = assets.loc[common].values
        y_all = target.loc[common].values
        rf_c = rf.reindex(common).fillna(0.0).values

        # in-sample floor
        w = min_te_weights(X_all, y_all)
        repl = pd.Series(X_all @ w + (1 - w.sum()) * rf_c, index=common)
        top = ", ".join(f"{assets.columns[i]} {w[i]:.0%}"
                        for i in np.argsort(w)[::-1][:4] if w[i] >= 0.02)
        cash = 1 - w.sum()
        r, d = stats_row(f"{uname} — IN-SAMPLE floor", repl, target,
                         f"cash {cash:.0%}; {top}")
        rows.append(r); raw.append(d)

        # walk-forward
        wf = pd.Series(np.nan, index=common)
        for i in range(FIT_MIN, len(common)):
            lo = max(0, i - FIT_WIN)
            w = min_te_weights(X_all[lo:i], y_all[lo:i])
            wf.iloc[i] = X_all[i] @ w + (1 - w.sum()) * rf_c[i]
        r, d = stats_row(f"{uname} — walk-forward", wf.dropna(), target)
        rows.append(r); raw.append(d)

    # baselines
    spy = etf["SPY"]
    b6040 = 0.6 * etf["SPY"] + 0.4 * etf["IEF"]
    for name, b in [("baseline: cash", rf.reindex(target.index).fillna(0.0)),
                    ("baseline: SPY", spy), ("baseline: 60/40", b6040)]:
        r, d = stats_row(name, b, target)
        rows.append(r); raw.append(d)

    print("\nTracking the levered zoo, long-only & unlevered:")
    print(tabulate(rows, headers=["fit", "months", "TE ann", "corr",
                                  "replica ret", "target ret", "replica vol",
                                  "weights"], tablefmt="github"))
    tv = target.std() * np.sqrt(12)
    print(f"\n  yardstick: target's own vol = {tv:.2%} — a tracker with "
          f"TE ≈ that has captured nothing.")

    out = RESULTS_DIR / "zoo_tracking.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")


if __name__ == "__main__":
    main()
