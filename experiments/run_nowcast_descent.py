#!/usr/bin/env python3
"""
Gradient descent on portfolio weights against a NOWCAST — researched from
all standpoints.

The idea splits into two objectives with opposite theoretical standing:

  MINIMIZE NOWCAST VOL — well-posed. Volatility clusters, so an EWMA
      nowcast of next-month vol has real predictive power; minimizing the
      nowcast genuinely minimizes realized vol. The repo's volmgmt overlay
      is the 1-asset special case; min_variance_weights the exact-solve one.
  MAXIMIZE NOWCAST RETURN — ill-posed by theory. Trailing-mean return
      nowcasts have ~zero predictive R²; optimizing against them maximizes
      estimation noise (Markowitz's "error maximizer", and the pod thread's
      winner's curse at weight frequency). Included to be falsified.

What gradient descent adds vs the exact solve: a small learning rate is
implicit regularization — each step shrinks toward yesterday's weights, so
turnover is bounded for free (the role the TC penalty plays in
bayesian.py) and noise in the nowcast is averaged over steps. Online
convex optimization gives this regret guarantees (Cover; Helmbold's EG).

A-priori parameters (no sweeps; multiplicity disclosed: 3 objectives × 2
online algorithms + exact QP = 7 books at a 0.90 bar ⇒ ~0.7 false
positives expected): EWMA halflife 21d, weekly steps, OGD/EG step
η = 0.05 on inf-norm-normalized gradients, mean-variance risk aversion
γ = 4, long-only simplex (no cap), Alpaca ETF costs (1.0 bp buy /
1.3 bp sell). Universe: the 13-ETF multiasset basket (SPY QQQ + 9 SPDR
sectors + TLT GLD). Baselines: equal weight, EWMA inverse-vol, SPY B&H.

Outputs: (0) the foundation table — measured nowcastability of vol vs
returns on this very data; (1) the strategy grid with realized-vol
delivery; (2) per-year returns; (3) block-bootstrap P vs both baselines.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices
from posterioralpha.research import min_variance_weights

OUT_DIR = _bootstrap.ROOT / "results" / "nowcast_descent"
BASKET = ["SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP",
          "XLI", "XLU", "XLB", "TLT", "GLD"]
HALFLIFE = 21
ETA = 0.05
GAMMA = 4.0
WARMUP = 252
BUY_BP, SELL_BP = 1.0e-4, 1.3e-4


def proj_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def gradient(objective: str, w, mu, cov):
    if objective == "minvol":
        return 2.0 * cov @ w
    if objective == "maxret":
        return -mu
    return -mu + 2.0 * GAMMA * cov @ w        # meanvar


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    years = len(r) / 252
    cagr = float(cum.iloc[-1] ** (1 / years) - 1)
    return {"cagr": round(cagr, 3),
            "real_vol": round(float(r.std() * np.sqrt(252)), 3),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2)}


def main():
    prices = load_etf_universe_prices()[BASKET].dropna()
    rets = prices.pct_change().iloc[1:]
    T, N = rets.shape

    mu_nc = rets.ewm(halflife=HALFLIFE).mean() * 252           # return nowcast
    vol_nc = rets.ewm(halflife=HALFLIFE).std() * np.sqrt(252)  # vol nowcast
    cov_nc = rets.ewm(halflife=HALFLIFE).cov() * 252           # cov nowcast

    # ── 0. Foundation: is each nowcast predictive on THIS data? ───────────
    fwd_vol = rets.rolling(21).std().shift(-21) * np.sqrt(252)
    fwd_ret = (prices.shift(-21) / prices - 1.0) * 12
    rows = []
    for a in BASKET:
        v = pd.concat([vol_nc[a], fwd_vol[a]], axis=1).dropna()
        m = pd.concat([mu_nc[a], fwd_ret[a]], axis=1).dropna()
        rows.append({"asset": a,
                     "vol_nowcast_R2": round(v.corr().iloc[0, 1] ** 2, 3),
                     "ret_nowcast_R2": round(m.corr().iloc[0, 1] ** 2, 4)})
    foundation = pd.DataFrame(rows).set_index("asset")
    print("=" * 64)
    print("0. NOWCASTABILITY (EWMA nowcast at t vs realized t+1..t+21)")
    print("=" * 64)
    print(foundation.to_string())
    print(f"\nmean R² — vol: {foundation.vol_nowcast_R2.mean():.3f} | "
          f"returns: {foundation.ret_nowcast_R2.mean():.4f}  "
          f"({foundation.vol_nowcast_R2.mean() / max(foundation.ret_nowcast_R2.mean(), 1e-9):.0f}x)")

    # ── 1. Walk the strategies weekly ──────────────────────────────────────
    week_dates = rets.index[WARMUP::5]
    strategies = [f"{o}_{a}" for o in ("minvol", "meanvar", "maxret")
                  for a in ("ogd", "eg")] + ["minvol_qp", "invvol", "equal"]
    w_state = {s: np.ones(N) / N for s in strategies}
    targets = {s: {} for s in strategies}

    for t in week_dates:
        mu = mu_nc.loc[t].values
        cov = cov_nc.loc[t].values
        sig = np.sqrt(np.clip(np.diag(cov), 1e-10, None))

        for obj in ("minvol", "meanvar", "maxret"):
            for algo in ("ogd", "eg"):
                s = f"{obj}_{algo}"
                w = w_state[s]
                g = gradient(obj, w, mu, cov)
                g = g / (np.abs(g).max() + 1e-12)
                if algo == "ogd":
                    w = proj_simplex(w - ETA * g)
                else:
                    w = w * np.exp(-ETA * g)
                    w = w / w.sum()
                w_state[s] = w
                targets[s][t] = w.copy()
        targets["minvol_qp"][t] = min_variance_weights(cov, max_weight=1.0)
        iv = 1.0 / sig
        targets["invvol"][t] = iv / iv.sum()
        targets["equal"][t] = np.ones(N) / N

    # ── 2. Books with Alpaca costs ─────────────────────────────────────────
    books, turns = {}, {}
    for s in strategies:
        tw = pd.DataFrame(targets[s]).T.reindex(rets.index).ffill()
        tw.columns = BASKET
        held = tw.shift(1)
        dw = held.diff()
        cost = (dw.clip(lower=0).sum(axis=1) * BUY_BP
                + (-dw.clip(upper=0)).sum(axis=1) * SELL_BP)
        books[s] = ((held * rets).sum(axis=1) - cost).loc[week_dates[0]:]
        turns[s] = float(dw.abs().sum(axis=1).mean() * 252)
    books["SPY_bh"] = rets["SPY"].loc[week_dates[0]:]
    turns["SPY_bh"] = 0.0

    table = pd.DataFrame({s: stats(b) for s, b in books.items()}).T
    table["turn/yr"] = pd.Series(turns).round(1)

    # ── 3. Joint block bootstrap vs both baselines ─────────────────────────
    rng = np.random.default_rng(7)
    X = np.column_stack([books[s].values for s in books])
    names = list(books)
    Tn = len(X)
    wins_eq = {s: 0 for s in names}
    wins_iv = {s: 0 for s in names}
    n_boot = 1000
    for _ in range(n_boot):
        ix, t = np.empty(Tn, dtype=int), 0
        while t < Tn:
            s0 = rng.integers(0, Tn)
            for j in range(rng.geometric(1 / 63)):
                if t >= Tn:
                    break
                ix[t] = (s0 + j) % Tn
                t += 1
        R = X[ix]
        sh = R.mean(axis=0) / R.std(axis=0)
        for i, s in enumerate(names):
            wins_eq[s] += sh[i] > sh[names.index("equal")]
            wins_iv[s] += sh[i] > sh[names.index("invvol")]
    table["P_sh>equal"] = pd.Series({s: round(w / n_boot, 2)
                                     for s, w in wins_eq.items()})
    table["P_sh>invvol"] = pd.Series({s: round(w / n_boot, 2)
                                      for s, w in wins_iv.items()})

    print("\n" + "=" * 96)
    print(f"1. STRATEGY GRID ({books['equal'].index[0].date()} → "
          f"{books['equal'].index[-1].date()}, weekly steps, Alpaca costs)")
    print("=" * 96)
    print(table.to_string())

    yr = pd.DataFrame({s: books[s].groupby(books[s].index.year)
                       .apply(lambda r: (1 + r).prod() - 1)
                       for s in ("minvol_ogd", "minvol_qp", "maxret_ogd",
                                 "invvol", "equal", "SPY_bh")}).round(3)
    print("\n2. PER-YEAR RETURNS (selected)")
    print(yr.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    foundation.to_csv(OUT_DIR / "nowcastability.csv")
    table.to_csv(OUT_DIR / "strategy_grid.csv")
    yr.to_csv(OUT_DIR / "per_year.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5),
                             gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    show = {"SPY_bh": "#bbbbbb", "equal": "#888888", "invvol": "#555555",
            "minvol_ogd": "#1a6fb5", "minvol_qp": "#7a4fb5",
            "maxret_ogd": "#b5341a", "meanvar_ogd": "#e0a020"}
    for s, c in show.items():
        ax.plot((1 + books[s]).cumprod(), color=c,
                lw=1.8 if "minvol" in s else 1.1,
                ls="--" if s == "maxret_ogd" else "-", label=s)
    ax.set_yscale("log")
    ax.set_title("Gradient descent on nowcasts — 13-ETF basket, Alpaca costs")
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.bar(np.arange(len(BASKET)) - 0.2, foundation.vol_nowcast_R2, 0.4,
           label="vol nowcast R²", color="#1a6fb5")
    ax.bar(np.arange(len(BASKET)) + 0.2, foundation.ret_nowcast_R2, 0.4,
           label="return nowcast R²", color="#b5341a")
    ax.set_xticks(range(len(BASKET)))
    ax.set_xticklabels(BASKET, rotation=60, fontsize=8)
    ax.set_title("Why one objective works and the other can't")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "nowcast_descent.png", dpi=150)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
