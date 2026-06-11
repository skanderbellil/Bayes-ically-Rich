#!/usr/bin/env python3
"""
Tilting toward MAXIMUM RETURNS, honestly.

Epilogue 8 killed return-nowcast optimization (R² 0.032). But "maximize
returns" doesn't need return forecasts: with an unpredictable mean, the
max-expected-wealth portfolio is maximal premium exposure sized by RISK —
Kelly with a CONSTANT long-run premium and the vol nowcast that actually
works (R² 0.246):

    e_t = min( μ̄ / (γ · σ̂_t²), 2 )          fractional Kelly
    e_t = min( σ* / σ̂_t,        2 )          vol targeting (its cousin)

μ̄ = 7%/yr a priori (textbook equity premium, never re-estimated);
σ̂ = EWMA(21d) vol of QQQ; cap 2× (the retail-implementable QLD ceiling).
Retail form: weight w = e/2 in QLD, rest cash, Alpaca costs, daily.

Multiplicity disclosed: σ* ∈ {15%, 20%, 25%} (a dose-response, not a
sweep — monotone CAGR-vs-target expected if the mechanism is real) and
γ ∈ {1, 2}. The bar a return-maximizer must clear: P(wealth > QLD B&H).
Context rows: QQQ B&H, QLD B&H, and the alive QLD/GLD rotation barbell.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices

OUT_DIR = _bootstrap.ROOT / "results" / "nowcast_descent"
COUNCIL = _bootstrap.ROOT / "results" / "council" / "minutes_QQQ.csv"
MU_BAR = 0.07
CAP = 2.0
BUY_BP, SELL_BP = 1.0e-4, 1.3e-4


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    years = len(r) / 252
    cagr = float(cum.iloc[-1] ** (1 / years) - 1)
    return {"cagr": round(cagr, 3), "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def qld_book(r_qld: pd.Series, exposure: pd.Series) -> pd.Series:
    """Exposure e on QQQ via w = e/2 in QLD + cash; Alpaca costs on w."""
    w = (exposure / 2.0).clip(0.0, 1.0).shift(1).fillna(0.0)
    dw = w.diff().fillna(0.0)
    cost = dw.clip(lower=0) * BUY_BP + (-dw.clip(upper=0)) * SELL_BP
    return w * r_qld - cost


def main():
    uni = load_etf_universe_prices()
    lev = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)
    qld = lev["QLD"].dropna()
    minutes = pd.read_csv(COUNCIL, index_col=0, parse_dates=True)
    idx = qld.loc[minutes.index[0]:].index

    r_qld = qld.pct_change().reindex(idx).fillna(0.0)
    r_qqq = uni["QQQ"].pct_change().reindex(idx).fillna(0.0)
    r_gld = uni["GLD"].pct_change().reindex(idx).fillna(0.0)
    sig = (uni["QQQ"].pct_change().ewm(halflife=21).std()
           * np.sqrt(252)).reindex(idx).ffill()

    books = {"QQQ B&H": r_qqq, "QLD B&H": r_qld}
    for tgt in (0.15, 0.20, 0.25):
        books[f"voltarget {int(tgt*100)}%"] = qld_book(
            r_qld, (tgt / sig).clip(0, CAP))
    for gamma in (1.0, 2.0):
        books[f"kelly γ={gamma:.0f}"] = qld_book(
            r_qld, (MU_BAR / (gamma * sig ** 2)).clip(0, CAP))

    # context: the alive barbell (council rotation, Epilogue 4/7)
    e = minutes["exposure"].reindex(qld.index).ffill().reindex(idx).fillna(0.5)
    wl, wg = e.shift(1).fillna(0.0), (1 - e).shift(1).fillna(0.0)
    turn = wl.diff().abs().fillna(0) + wg.diff().abs().fillna(0)
    books["QLD/GLD rotation"] = wl * r_qld + wg * r_gld - 2e-4 * turn

    table = pd.DataFrame({n: stats(b) for n, b in books.items()}).T

    # bootstrap: P(wealth > QLD B&H) — the return-maximizer's bar
    rng = np.random.default_rng(7)
    names = list(books)
    X = np.column_stack([books[n].values for n in names])
    T = len(X)
    wins = {n: 0 for n in names}
    for _ in range(1000):
        ix, t = np.empty(T, dtype=int), 0
        while t < T:
            s0 = rng.integers(0, T)
            for j in range(rng.geometric(1 / 63)):
                if t >= T:
                    break
                ix[t] = (s0 + j) % T
                t += 1
        R = X[ix]
        wq = (1 + R[:, names.index("QLD B&H")]).prod()
        for i, n in enumerate(names):
            wins[n] += (1 + R[:, i]).prod() > wq
    table["P_w>QLD"] = pd.Series({n: round(w / 1000, 2) for n, w in wins.items()})

    print(f"RETURN TILT ({idx[0].date()} → {idx[-1].date()}; μ̄ = 7% fixed, "
          "σ̂ = EWMA(21d), cap 2x, QLD+cash, Alpaca costs)")
    print("=" * 80)
    print(table.to_string())

    yr = pd.DataFrame({n: books[n].groupby(books[n].index.year)
                       .apply(lambda r: (1 + r).prod() - 1)
                       for n in ("QLD B&H", "voltarget 20%", "kelly γ=1",
                                 "QLD/GLD rotation")}).round(2)
    print("\nPER-YEAR")
    print(yr.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "return_tilt.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"QQQ B&H": "#bbbbbb", "QLD B&H": "#888888",
              "voltarget 20%": "#1a6fb5", "kelly γ=1": "#b5341a",
              "kelly γ=2": "#e0a020", "QLD/GLD rotation": "#2a7a2a"}
    for n, c in colors.items():
        ax.plot((1 + books[n]).cumprod(), color=c,
                lw=1.9 if n in ("kelly γ=1", "voltarget 20%") else 1.1,
                label=f"{n} (CAGR {stats(books[n])['cagr']:.0%}, "
                      f"DD {stats(books[n])['max_dd']:.0%})")
    ax.set_yscale("log")
    ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Return maximization done honestly: constant-premium Kelly / vol "
                 "targeting\nsized by the nowcast that works — no return forecasts anywhere")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "return_tilt.png", dpi=150)
    print(f"\nSaved to {OUT_DIR}/: return_tilt.csv, return_tilt.png")


if __name__ == "__main__":
    main()
