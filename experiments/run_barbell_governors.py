#!/usr/bin/env python3
"""
Governors for the strategic barbell — four NOVEL overlays, one shot each.

The fixed 50/50 QLD/GLD barbell (Epilogue 4/5: a strategic conviction, cf.
WisdomTree GDE) has one documented failure mode: both legs falling together.
Four overlays that might govern it, each parameterized ONCE, a priori, no
sweeps (multiplicity disclosed: 4 ideas at a 0.90 bar ⇒ expect ~0.4 false
positives; only decisive, mechanism-consistent wins count):

  volmgmt — Moreira–Muir-style volatility management: book scale =
            clip(trailing-1y median of 21d realized vol / current 21d
            realized vol, 0, 1). Self-normalizing, parameter-free, causal.
  corrgov — correlation-spike governor: when the trailing 63d QLD–GLD
            correlation exceeds +0.20 (diversification premise failing),
            halve the book. Directly targets the documented failure mode.
  vixslope — VIX term structure: scale = clip(VIX3M / VIX, 0, 1).
            Contango (calm) ⇒ full book; backwardation (crash regime)
            scales down proportionally. Continuous, parameter-free.
            (VIX slope was a conditioner in the liquidity thread — its
            use as a barbell governor is new; adjacency disclosed.)
  disagree — council disagreement: confidence-weighted dispersion of the
            six specialists' stances each month; above its expanding
            median ⇒ halve the book. Tests whether the council's
            *uncertainty* carries information beyond its vote.

Reference rows: the council exposure dial itself (Epilogue 4's survivor),
and every overlay's null — the SAME barbell statically scaled to the
overlay's average / vol-matched / DD-matched constant. An overlay only
matters if it beats the constant version of itself.

Run run_council_backtest.py (QQQ) first.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"
TC = 2e-4
W_LEG = 0.5                       # fixed strategic barbell: 50/50


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum / cum.cummax()) - 1).min())
    cagr = float(cum.iloc[-1] ** (252 / len(r)) - 1)
    return {"cagr": round(cagr, 3),
            "vol": round(float(r.std() * np.sqrt(252)), 3),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def scaled_book(r_qld, r_gld, scale):
    """Barbell with a causal daily scale on the whole book; cash earns 0."""
    held = scale.shift(1).fillna(0.0)
    w_q, w_g = W_LEG * held, W_LEG * held
    turn = w_q.diff().abs().fillna(0.0) + w_g.diff().abs().fillna(0.0)
    return w_q * r_qld + w_g * r_gld - TC * turn


def matched_scale(r_qld, r_gld, target, mode):
    lo, hi = 0.01, 1.0
    for _ in range(40):
        c = (lo + hi) / 2
        mix = c * (W_LEG * r_qld + W_LEG * r_gld)
        if mode == "vol":
            hit = float(mix.std()) > target
        else:
            cum = (1 + mix).cumprod()
            hit = float(((cum / cum.cummax()) - 1).min()) < target
        hi, lo = (c, lo) if hit else (hi, c)
    return c


def boot(s, r_qld, r_gld, consts, rng, n_boot=2000):
    X = np.column_stack([s.values, r_qld.values, r_gld.values])
    T = len(X)
    wins = {k: 0 for k in consts}
    wins["sharpe_avg"] = 0
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
        a = R[:, 0]
        for k, c in consts.items():
            m = c * (W_LEG * R[:, 1] + W_LEG * R[:, 2])
            wins[k] += (1 + a).prod() > (1 + m).prod()
            if k == "P_w_avg":
                wins["sharpe_avg"] += a.mean() / a.std() > m.mean() / m.std()
    return {k: round(v / n_boot, 2) for k, v in wins.items()}


def main():
    uni = load_etf_universe_prices()
    qld = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)["QLD"].dropna()
    vix = pd.read_csv(_bootstrap.ROOT / "datasets" / "vix_term_structure.csv",
                      index_col=0, parse_dates=True)
    minutes = pd.read_csv(_bootstrap.ROOT / "results" / "council"
                          / "minutes_QQQ.csv", index_col=0, parse_dates=True)
    rng = np.random.default_rng(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    start = minutes.index[0]
    idx = qld.loc[start:].index
    r_qld = qld.pct_change().reindex(idx).fillna(0.0)
    r_gld = uni["GLD"].pct_change().reindex(idx).fillna(0.0)
    r_qqq = uni["QQQ"].pct_change().reindex(idx).fillna(0.0)
    base = W_LEG * r_qld + W_LEG * r_gld     # static barbell, scale 1

    # ── the four governors (scales in [0, 1], all causal) ──────────────────
    book_full = base.copy()
    rv21 = book_full.rolling(21).std()
    volmgmt = (rv21.rolling(252).median() / rv21).clip(0, 1).fillna(1.0)

    corr63 = r_qld.rolling(63).corr(r_gld)
    corrgov = pd.Series(np.where(corr63 > 0.20, 0.5, 1.0), index=idx)

    vr = (vix["VIX3M"] / vix["VIX"]).reindex(idx).ffill()
    vixslope = vr.clip(0, 1).fillna(1.0)

    stances = minutes[[c for c in minutes.columns if c.endswith("_stance")]]
    confs = minutes[[c for c in minutes.columns if c.endswith("_conf")]]
    cw = confs.values / confs.values.sum(axis=1, keepdims=True)
    mu = (stances.values * cw).sum(axis=1, keepdims=True)
    disp = pd.Series(np.sqrt((cw * (stances.values - mu) ** 2).sum(axis=1)),
                     index=minutes.index)
    thresh = disp.expanding(12).median()
    dis_scale = pd.Series(np.where(disp > thresh, 0.5, 1.0),
                          index=minutes.index).reindex(idx).ffill().fillna(1.0)

    council = minutes["exposure"].reindex(qld.index).ffill().reindex(idx)

    overlays = {"volmgmt": volmgmt, "corrgov": corrgov,
                "vixslope": vixslope, "disagree": dis_scale,
                "council dial (ref)": council}

    rows = {"QQQ buy & hold": stats(r_qqq),
            "static 50/50 QLD/GLD (scale 1)": stats(base)}
    curves = {"static 50/50 QLD/GLD": base, "QQQ buy & hold": r_qqq}
    for name, sc in overlays.items():
        s = scaled_book(r_qld, r_gld, sc)
        st = stats(s)
        avg = float(sc.shift(1).mean())
        consts = {"P_w_avg": avg,
                  "P_w_volm": matched_scale(r_qld, r_gld, float(s.std()), "vol"),
                  "P_w_ddm": matched_scale(r_qld, r_gld, st["max_dd"], "dd")}
        st.update(boot(s, r_qld, r_gld, consts, rng))
        st["avg_scale"] = round(avg, 2)
        rows[f"barbell × {name}"] = st
        if name in ("vixslope", "corrgov"):
            curves[f"barbell × {name}"] = s

    df = pd.DataFrame(rows).T
    print(f"BARBELL GOVERNORS ({idx[0].date()} → {idx[-1].date()}; "
          "fixed 50/50 QLD/GLD; P_* vs the CONSTANT-scaled barbell)")
    print("=" * 110)
    print(df.to_string())
    print("\nP_w_avg / P_w_volm / P_w_ddm: bootstrap P(governor beats the "
          "same barbell at a constant scale —\naverage / vol-matched / "
          "DD-matched). sharpe_avg: P(Sharpe beats avg-scale constant). "
          "Four novel\noverlays at a 0.90 bar ⇒ ~0.4 false positives "
          "expected; only decisive wins count.")
    df.to_csv(OUT_DIR / "barbell_governors.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, r in curves.items():
        cum = (1 + r.fillna(0)).cumprod()
        style = dict(lw=1.2, ls="--", color="0.4") if "buy & hold" in name \
            else dict(lw=1.5)
        ax1.plot(cum.index, cum.values, label=name, **style)
        ax2.plot(cum.index, (cum / cum.cummax() - 1).values, **style)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Barbell governors: vol mgmt / corr spike / VIX slope / "
                  "council disagreement")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "barbell_governors.png", dpi=140)
    print(f"\nSaved to {OUT_DIR}/: barbell_governors.csv, "
          "barbell_governors.png")


if __name__ == "__main__":
    main()
