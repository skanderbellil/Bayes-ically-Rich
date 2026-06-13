#!/usr/bin/env python3
"""
Factor attribution of the champion stack — the "Buffett's Alpha" regression.

Question
--------
The frozen champion (5-layer vote → 2× QQQ exposure, UUP slack, ceiling,
slow-down) posts a headline scorecard. But Frazzini-Kabiller-Pedersen's
lesson is to regress any track record on factor returns BEFORE crediting
skill: Berkshire's alpha vanished once BAB+QMJ were controlled for. How
much of the champion is liquidity timing, and how much is just levered
growth-factor beta you could buy statically?

Method
------
Daily strategy excess returns (from results/champion_stack_returns.csv —
run experiments/run_champion_stack.py first) regressed on the French
factors (datasets/ff_factors_daily.csv.gz):

    r_t − rf_t = α + β·Mkt-RF + s·SMB + h·HML + r·RMW + c·CMA + m·Mom + ε

Newey-West (HAC) standard errors with 5 lags. The verdict hangs on t(α):
  t(α) > 3   — timing alpha survives the factor model (Harvey-adjusted bar)
  t(α) ∈ 2-3 — survives classically, fails the multiple-testing bar
  t(α) < 2   — the champion is (mostly) a factor portfolio with extra steps

QQQ buy & hold is regressed alongside as the control: its α tells us how
much "alpha" this factor model hands to ANY levered-Nasdaq strategy in
this sample, which is the honest baseline for reading the champion's α.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_ff_factors

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"

RETURNS_CSV = RESULTS_DIR / "champion_stack_returns.csv"
FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]
NW_LAGS = 5
ANN = 252

STRATS = [
    "+ slow-down  (FULL STACK)",
    "FULL STACK in QLD+UUP (retail)",
    "champion (2×vote, cash slack)",
    "QQQ buy & hold",
    "SPY buy & hold",
]


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = NW_LAGS):
    """OLS with HAC (Newey-West, Bartlett kernel) standard errors."""
    T, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1.0 - u.var() / y.var()
    return beta, se, r2


def main():
    if not RETURNS_CSV.exists():
        raise FileNotFoundError(
            "results/champion_stack_returns.csv not found — run "
            "experiments/run_champion_stack.py first.")
    rets = pd.read_csv(RETURNS_CSV, parse_dates=[0], index_col=0)
    ff = load_ff_factors("daily")

    common = rets.index.intersection(ff.index)
    rets, ff = rets.loc[common], ff.loc[common]
    logger.info(f"Sample: {common[0].date()} → {common[-1].date()} "
                f"({len(common)} days; FF daily ends {ff.index[-1].date()})")

    X = np.column_stack([np.ones(len(common)), ff[FACTORS].values])
    rows, raw = [], []
    for name in STRATS:
        if name not in rets.columns:
            continue
        y = (rets[name] - ff["RF"]).values
        ok = np.isfinite(y)
        beta, se, r2 = newey_west_ols(y[ok], X[ok], lags=NW_LAGS)
        t = beta / se
        alpha_ann = beta[0] * ANN
        rows.append([name, f"{alpha_ann:+.1%}", f"{t[0]:.2f}"]
                    + [f"{b:.2f}" for b in beta[1:]]
                    + [f"{r2:.2f}"])
        raw.append({"strategy": name, "alpha_ann": alpha_ann,
                    "t_alpha": t[0], **dict(zip(FACTORS, beta[1:])),
                    **{f"t_{f}": tv for f, tv in zip(FACTORS, t[1:])},
                    "r2": r2})

    print()
    print(tabulate(rows, headers=["strategy", "α (ann)", "t(α)"]
                   + FACTORS + ["R²"], tablefmt="github"))
    print(f"\nNewey-West {NW_LAGS} lags · α bar: |t|>3 real, 2–3 fragile, "
          f"<2 = factor exposure, not timing skill.")
    print("Read the champion's α RELATIVE to QQQ B&H's α: the difference is "
          "what the vote/timing machinery adds beyond static growth beta.")

    out = RESULTS_DIR / "factor_attribution.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"Saved {out}")


if __name__ == "__main__":
    main()
