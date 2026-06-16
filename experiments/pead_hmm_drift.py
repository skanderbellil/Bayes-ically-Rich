#!/usr/bin/env python3
"""Per-position HMM drift-confirmation exit for tradeable PEAD, with SPY plots.

The exit rule the simple stop-loss approximates, done properly: for EACH position,
watch its own daily post-announcement P&L path (signed in the signal's direction —
+return for a long, -return for a short) and ask, sequentially, "is the drift
confirming?" A 2-state Gaussian HMM (fit on the cross-section of training paths,
causally) gives two states — a positive-mean "drift" state and a flat/negative
"no-drift" state. At a decision day we filter the posterior P(drift | path-so-far):

    P(drift) >= threshold  ->  CONFIRMED -> extend the hold to day 63
    P(drift) <  threshold  ->  NOT CONFIRMED -> exit now

This is richer than a level stop-loss: it rewards *persistent* drift (a steady +3%
beats a spike-then-flat +3%), and it cuts positions whose drift never materialises.

Entry is the realistic t+1 close (see pead_dynamic_exit.py for why). Market-neutral,
Mega+Large (ETB) only, net of Alpaca costs. Produces results/pead/hmm_vs_spy.png.

Requires the cached fetch under data/raw/pead/ (run experiments/run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths
from posterioralpha.backtest.alpaca_costs import AlpacaCosts
from posterioralpha.data.loaders import load_portfolio_returns
from experiments.pead_dynamic_exit import build_paths, COST, _stats

DECISION_DAY = 15        # filter the drift posterior this many days after entry
EXTEND_TO = 63           # confirmed positions are held to this day
P_THRESH = 0.50          # P(drift) needed to confirm


def fit_drift_hmm(P, M, train_years):
    """Fit a 2-state Gaussian HMM on pooled signal-direction daily returns (train only).

    Returns (model, drift_state_index). Longs contribute +daily-return, shorts -daily.
    """
    seqs, lengths = [], []
    sub = M[M["year"].isin(train_years)]
    for _, g in sub.groupby("season").groups.items():
        idx = list(g)
        sue = M.loc[idx, "sue"].values
        hi, lo = np.quantile(sue, 0.8), np.quantile(sue, 0.2)
        for i, s in zip(idx, sue):
            if s < hi and s > lo:
                continue
            sign = 1.0 if s >= hi else -1.0          # long vs short → signal direction
            path = P[i, :EXTEND_TO + 1]
            dr = np.diff(path[~np.isnan(path)]) * sign
            if len(dr) >= 5:
                seqs.append(dr.reshape(-1, 1)); lengths.append(len(dr))
    X = np.vstack(seqs)
    model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=50, random_state=0)
    model.fit(X, lengths)
    drift_state = int(np.argmax(model.means_.ravel()))   # higher-mean state = "drift"
    return model, drift_state


def hmm_exit_return(P, i, sign, model, drift_state):
    """Signal-direction exit return (%) for one position under the HMM rule."""
    path = P[i, :EXTEND_TO + 1]
    valid = path[~np.isnan(path)]
    dr = np.diff(valid) * sign
    d0 = min(DECISION_DAY, len(dr))
    if d0 < 3:
        return sign * (valid[-1] - valid[0]) if len(valid) > 1 else np.nan
    p_drift = model.predict_proba(dr[:d0].reshape(-1, 1))[-1, drift_state]
    if p_drift >= P_THRESH:                       # confirmed → hold to EXTEND_TO
        exit_cum = valid[min(EXTEND_TO, len(valid) - 1)]
    else:                                         # not confirmed → exit at decision day
        exit_cum = valid[d0]
    return sign * exit_cum                         # signed P&L in signal direction


def hmm_strategy(P, M, eval_years, model, drift_state):
    sp = []
    sub = M[M["year"].isin(eval_years)]
    for _, g in sub.groupby("season").groups.items():
        idx = list(g)
        if len(idx) < 8:
            continue
        sue = M.loc[idx, "sue"].values
        hi, lo = np.quantile(sue, 0.8), np.quantile(sue, 0.2)
        longs = [hmm_exit_return(P, i, +1.0, model, drift_state)
                 for i, s in zip(idx, sue) if s >= hi]
        shorts = [hmm_exit_return(P, i, -1.0, model, drift_state)
                  for i, s in zip(idx, sue) if s <= lo]
        if longs and shorts:
            sp.append((np.nanmean(longs) + np.nanmean(shorts)))   # both already signed +
    return np.array(sp) / 100 - COST


def fixed_or_stop(P, M, eval_years, hold, stop=None):
    from experiments.pead_dynamic_exit import _ls
    sub = M[M["year"].isin(eval_years)]
    sp = [_ls(list(g), P, M, hold, stop) for _, g in sub.groupby("season").groups.items()
          if len(g) >= 8]
    return np.array(sp) / 100 - COST


def quarterly_series(P, M, eval_years, kind, model=None, drift_state=None):
    """Return a pd.Series indexed by quarter of net per-quarter returns."""
    sub = M[M["year"].isin(eval_years)]
    out = {}
    from experiments.pead_dynamic_exit import _ls
    for s, g in sub.groupby("season"):
        idx = list(g.index)
        if len(idx) < 8:
            continue
        if kind == "hmm":
            sue = M.loc[idx, "sue"].values
            hi, lo = np.quantile(sue, 0.8), np.quantile(sue, 0.2)
            longs = [hmm_exit_return(P, i, +1.0, model, drift_state) for i, x in zip(idx, sue) if x >= hi]
            shorts = [hmm_exit_return(P, i, -1.0, model, drift_state) for i, x in zip(idx, sue) if x <= lo]
            if not (longs and shorts):
                continue
            r = (np.nanmean(longs) + np.nanmean(shorts)) / 100 - COST
        elif kind == "stop":
            r = _ls(idx, P, M, 42, 6) / 100 - COST
        else:
            r = _ls(idx, P, M, 42, None) / 100 - COST
        out[s] = r
    return pd.Series(out)


def main():
    P, M = build_paths()
    M = M.reset_index(drop=True)
    train_years = list(range(2000, 2014))
    oos_years = list(range(2014, 2027))
    allyrs = list(range(2000, 2027))

    print(f"Fitting per-position drift HMM on {len([y for y in train_years])}-yr train window...")
    model, ds = fit_drift_hmm(P, M, train_years)
    mu = model.means_.ravel()
    print(f"  HMM states (daily signal-dir mean %): drift={mu[ds]:+.3f}, no-drift={mu[1-ds]:+.3f}")
    print(f"  Decision day={DECISION_DAY}, extend-to={EXTEND_TO}, P(drift)>={P_THRESH}\n")

    print("Tradeable market-neutral (t+1 entry, net) — full & OOS≥2014:")
    print(f"  {'Strategy':<34}{'Full Ann':>9}{'Shrp':>6} | {'OOS Ann':>8}{'Shrp':>6}{'MaxDD':>7}")
    rows = [
        ("Fixed 42d", lambda yr: fixed_or_stop(P, M, yr, 42)),
        ("Exit-losers 6% stop, 42d", lambda yr: fixed_or_stop(P, M, yr, 42, 6)),
        ("Per-position HMM drift-confirm", lambda yr: hmm_strategy(P, M, yr, model, ds)),
    ]
    for lbl, fn in rows:
        a, b = _stats(fn(allyrs)), _stats(fn(oos_years))
        print(f"  {lbl:<34}{a[0]:>+9.1%}{a[1]:>6.2f} | {b[0]:>+8.1%}{b[1]:>6.2f}{b[2]:>+7.1%}")

    # ---- Plots vs SPY ----
    spy = load_portfolio_returns()["SPY"]
    spq = (1 + spy).resample("QE").prod() - 1
    spq.index = spq.index.to_period("Q")

    stop_s = quarterly_series(P, M, allyrs, "stop")
    hmm_s = quarterly_series(P, M, allyrs, "hmm", model, ds)
    common = stop_s.index.intersection(spq.index)
    spy_s = spq.loc[common]
    # The recommended deployment: 60% SPY core + 40% market-neutral sleeve (unlevered).
    blend_s = (0.6 * spy_s.add(0.0)).add(0.4 * stop_s.reindex(common).fillna(0.0))

    series = {
        "SPY buy & hold": (spy_s, "#2ca02c", "--", 1.8),
        "PEAD MN — exit-losers stop": (stop_s.reindex(common), "#1f77b4", "-", 2.0),
        "PEAD MN — HMM drift-confirm": (hmm_s.reindex(common), "#d62728", "-", 1.6),
        "60% SPY + 40% MN sleeve": (blend_s, "#000000", "-", 2.6),
    }

    def dd_curve(r):
        eq = np.cumprod(1 + np.asarray(r)); return eq / np.maximum.accumulate(eq) - 1

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    for name, (s, c, ls, lw) in series.items():
        s = s.loc[common].sort_index()
        x = s.index.to_timestamp()
        a, sh, dd, _ = _stats(s.values)
        ax1.plot(x, 100_000 * np.cumprod(1 + s.values), color=c, ls=ls, lw=lw,
                 label=f"{name}   (CAGR {a:+.1%}, Sharpe {sh:.2f}, MaxDD {dd:+.0%})")
        ax2.plot(x, dd_curve(s.values) * 100, color=c, ls=ls, lw=lw)
    ax1.set_yscale("log")
    ax1.set_title("Tradeable PEAD market-neutral vs SPY — growth of $100k (2005–2026, net of Alpaca costs)",
                  fontweight="bold")
    ax1.set_ylabel("Portfolio value ($, log)"); ax1.legend(fontsize=9, loc="upper left"); ax1.grid(alpha=0.3)
    ax2.set_title("Drawdown (underwater)", fontweight="bold")
    ax2.set_ylabel("Drawdown (%)"); ax2.set_xlabel("Year"); ax2.grid(alpha=0.3)
    ax2.axhline(0, color="k", lw=0.6)
    plt.tight_layout()
    out = paths.RESULTS_DIR / "hmm_vs_spy.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nSaved plot: {out}")
    print("Blend (60/40) is the recommended deployment: SPY's return at ~half the drawdown.")


if __name__ == "__main__":
    main()
