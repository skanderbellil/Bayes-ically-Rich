#!/usr/bin/env python3
"""
The barbell stack: every surviving piece, tested JOINTLY.

The hiring-policies thread ended with three pieces alive, each validated
in isolation: the strategic levered-equity/gold barbell (conviction, not
discovery), the council dial as leg ROTATION (Epilogue 4, P = 0.91), and
volatility management with a leverage dose-response (Epilogue 6 addendum,
P = 0.84/0.89/1.00 at 1x/2x-QLD/2x-SSO). Pieces that pass separately can
break jointly — the same reason the liquidity thread ran a champion stack.

Configurations per leg (QLD with the QQQ council, SSO with the SPY
council; GLD is the fixed slack):

  static     — 50/50 leg/GLD, monthly-drift-free daily weights (the base)
  rotation   — w_leg = council e, w_GLD = 1 − e   (Epilogue 4 survivor)
  volmgmt    — 50/50 × clip(median_1y(vol21)/vol21, 0, 1), cash earns 0
  STACK      — rotation × volmgmt (both surviving overlays together)

Each row faces the matched static nulls (avg-scale / vol-matched /
DD-matched constant on the 50/50 barbell) under a joint block bootstrap,
plus the leave-one-out questions that decide the stack:

  P(stack > rotation)  — does volmgmt still add once rotation governs?
  P(stack > volmgmt)   — does rotation still add once volmgmt governs?

All causal: council exposure is monthly and lagged a day, volmgmt scale is
trailing and lagged a day, Alpaca-tier 2 bps on turnover.

Run run_council_backtest.py for QQQ and SPY first (council minutes).
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"
COUNCIL_DIR = _bootstrap.ROOT / "results" / "council"
TC = 2e-4
W_LEG = 0.5
LEGS = {"QLD": "QQQ", "SSO": "SPY"}


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    years = len(r) / 252
    cagr = float(cum.iloc[-1] ** (1 / years) - 1)
    return {"cagr": round(cagr, 3),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def book(r_leg, r_gld, w_leg, scale):
    """Daily book from leg weight series and a global scale; cash earns 0."""
    wl = (w_leg * scale).shift(1).fillna(0.0)
    wg = ((1.0 - w_leg) * scale).shift(1).fillna(0.0)
    turn = wl.diff().abs().fillna(0.0) + wg.diff().abs().fillna(0.0)
    return wl * r_leg + wg * r_gld - TC * turn


def volmgmt_scale(book_proxy: pd.Series) -> pd.Series:
    vol = book_proxy.rolling(21).std()
    ref = vol.rolling(252, min_periods=126).median()
    return (ref / vol).clip(0.0, 1.0).fillna(1.0)


def matched_const(r_leg, r_gld, target, mode):
    lo, hi = 0.01, 1.0
    for _ in range(40):
        c = (lo + hi) / 2
        mix = c * (W_LEG * r_leg + W_LEG * r_gld)
        if mode == "vol":
            hit = float(mix.std()) > target
        else:
            cum = (1 + mix).cumprod()
            hit = float(((cum / cum.cummax()) - 1).min()) < target
        hi, lo = (c, lo) if hit else (hi, c)
    return c


def joint_boot(series: dict, r_leg, r_gld, consts: dict, rng, n_boot=2000):
    """Joint stationary block bootstrap; same blocks for every column."""
    names = list(series)
    X = np.column_stack([series[n].values for n in names]
                        + [r_leg.values, r_gld.values])
    T = len(X)
    wins = {}
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
        cols = {n: R[:, i] for i, n in enumerate(names)}
        leg, gld = R[:, -2], R[:, -1]
        stack = cols["stack"]
        for n in names:
            if n != "stack":
                wins.setdefault(f"P_w(stack>{n})", 0)
                wins[f"P_w(stack>{n})"] += (1 + stack).prod() > (1 + cols[n]).prod()
        for k, c in consts.items():
            m = c * (W_LEG * leg + W_LEG * gld)
            wins.setdefault(k, 0)
            wins[k] += (1 + stack).prod() > (1 + m).prod()
    return {k: round(v / n_boot, 2) for k, v in wins.items()}


def main():
    uni = load_etf_universe_prices()
    lev = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)
    rng = np.random.default_rng(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows, curves = {}, {}
    for leg_name, dial_under in LEGS.items():
        leg = lev[leg_name].dropna()
        minutes = pd.read_csv(COUNCIL_DIR / f"minutes_{dial_under}.csv",
                              index_col=0, parse_dates=True)
        e_per = minutes["exposure"]
        idx = leg.loc[e_per.index[0]:].index
        r_leg = leg.pct_change().reindex(idx).fillna(0.0)
        r_gld = uni["GLD"].pct_change().reindex(idx).fillna(0.0)
        e = e_per.reindex(leg.index).ffill().reindex(idx).fillna(0.5)

        ones = pd.Series(1.0, index=idx)
        half = pd.Series(W_LEG * 2, index=idx) * 0 + 1.0  # scale=1 for static
        static_mix = W_LEG * r_leg + W_LEG * r_gld
        vm = volmgmt_scale(static_mix)

        books = {
            "static":   book(r_leg, r_gld, pd.Series(W_LEG, index=idx), ones),
            "rotation": book(r_leg, r_gld, e, ones),
            "volmgmt":  book(r_leg, r_gld, pd.Series(W_LEG, index=idx), vm),
            "stack":    book(r_leg, r_gld, e, vm),
        }

        st = stats(books["stack"])
        consts = {
            "P_w(stack>const_avg)": float((e * vm).shift(1).mean()),
            "P_w(stack>const_volm)": matched_const(
                r_leg, r_gld, float(books["stack"].std()), "vol"),
            "P_w(stack>const_ddm)": matched_const(
                r_leg, r_gld, st["max_dd"], "dd"),
        }
        p = joint_boot(books, r_leg, r_gld, consts, rng)

        for name, b in books.items():
            row = stats(b)
            if name == "stack":
                row.update(p)
            all_rows[f"{leg_name}/GLD {name}"] = row
        curves[leg_name] = {n: (1 + b).cumprod() for n, b in books.items()}

        print(f"\n{'=' * 96}\n{leg_name}/GLD (council dial: {dial_under}, "
              f"{idx[0].date()} → {idx[-1].date()})\n{'=' * 96}")
        sub = {k.split(' ', 1)[1]: v for k, v in all_rows.items()
               if k.startswith(leg_name)}
        print(pd.DataFrame(sub).T.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    for ax, (leg_name, cs) in zip(axes, curves.items()):
        colors = {"static": "#999999", "rotation": "#1a6fb5",
                  "volmgmt": "#e0a020", "stack": "#b5341a"}
        for n, c in cs.items():
            ax.plot(c, color=colors[n], lw=1.8 if n == "stack" else 1.1,
                    label=n)
        ax.set_yscale("log")
        ax.set_title(f"{leg_name}/GLD: pieces vs the stack")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "barbell_stack.png", dpi=150)

    pd.DataFrame(all_rows).T.to_csv(OUT_DIR / "barbell_stack.csv")
    print(f"\nSaved to {OUT_DIR}/: barbell_stack.csv, barbell_stack.png")


if __name__ == "__main__":
    main()
