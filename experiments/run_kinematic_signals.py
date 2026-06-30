#!/usr/bin/env python3
"""
Kinematic state-space signals from price alone (user 2026-06-30).

Builds and HONESTLY evaluates the generative layer (price = noisy observation of
a latent [level, velocity, acceleration] state) and benchmarks it against the
integral layer (EMA/MACD smoothing). The question is not "is there a pretty
backtest" but "what is real signal vs noise-amplified garbage".

Sections (map 1:1 to the brief):
  1. Kinematic Kalman filter, Q/R by MLE on TRAIN only; innovation whiteness.
  2. Noise-amplification study — filtered vs naive finite-difference
     derivatives, on synthetic ground truth AND real data. Does the filter
     rescue acceleration, or is it differencing noise?
  3. Local-level innovation as a mean-reversion signal: IC / hit-rate / decay.
  4. Dynamic Kalman hedge ratio (SPY~QQQ) vs static OLS beta: does the adaptive
     spread survive where the static one breaks (2008)?
  5. Regime layer — 2-state causal HMM on filtered velocity gating momentum vs
     reversion.
  6. Integral baseline — EMA/MACD (the first integral of price); path-signature
     corner (flagged overfit-prone).
  RIGOR: strict causality (FILTER output only — smoother is diagnostics),
  walk-forward params, net-of-cost with turnover, PnL concentration, effective
  bets, Q/R sweep, deflated Sharpe across the variants tried.

Honest box: bundled history is 1993–2009 (SPY), 1999–2009 (SPY/QQQ overlap), so
the pair test is dominated by — and stress-tested on — the 2008 crisis. Single
liquid index ⇒ small effective breadth; read SHAPES and sign, not headline
Sharpes. Costs are charged on turnover; index ETFs are cheap (~1bp), so the
default is deliberately modest and a higher-cost column is shown.
"""
import sys
import warnings

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_fresh_prices
from posterioralpha.kinematic import (
    KinematicKalman, LocalLevelKalman, PairsKalman, VelocityRegimeHMM,
    velocity_signal, acceleration_signal, innovation_signal,
    pair_spread, spread_zscore, static_ols_spread,
    run_signal_backtest, walk_forward, ema_signal, macd_signal,
    signature_trend_signal, diagnostics as dg,
)
from posterioralpha.mining.validation import deflated_sharpe_ratio

warnings.filterwarnings("ignore")
ANN = 252
COST_BPS = 1.0          # index ETF round-trip per unit turnover (bps)
COST_BPS_HI = 5.0       # stress column

OUT = _bootstrap.ROOT / "data" / "kinematic"
OUT.mkdir(parents=True, exist_ok=True)


def _tab(df, **kw):
    print(tabulate(df, headers="keys", tablefmt="github", floatfmt=".4f",
                   showindex=kw.get("showindex", False)))


def half_life(spread: pd.Series) -> float:
    """AR(1) mean-reversion half-life (days) from Δs = a + b·s_{t-1} + e."""
    s = spread.dropna()
    s_lag = s.shift(1).dropna()
    ds = (s - s.shift(1)).dropna()
    df = pd.concat([ds, s_lag], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    b = np.polyfit(df.iloc[:, 1], df.iloc[:, 0], 1)[0]
    return float(-np.log(2) / np.log(1 + b)) if -1 < b < 0 else np.inf


# ════════════════════════════════════════════════════════════════════════════
def section1_filter(logp: pd.Series, ntr: int):
    print("\n" + "═" * 78)
    print("SECTION 1 — Kinematic Kalman filter (constant-acceleration), MLE Q/R")
    print("═" * 78)
    kf = KinematicKalman().fit_mle(logp.values[:ntr], verbose=True)
    fr = kf.filter(logp.values)
    te = slice(ntr, len(logp))
    z = fr.std_innovation[te]
    z = z[np.isfinite(z)]
    # Ljung-Box-ish: lag-1 autocorr of standardised innovations should be ~0
    ac1 = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    print(f"\nTRAIN MLE → q={kf.q:.3e}  r={kf.r:.3e}  q/r={kf.q/kf.r:.3e}")
    print(f"OOS standardised-innovation: mean={z.mean():+.3f}  std={z.std():.3f} "
          f"(≈1 if well-specified)  lag1 AC={ac1:+.3f} (≈0 if white)")
    return kf, fr


def section2_noise(logp: pd.Series, kf: KinematicKalman):
    print("\n" + "═" * 78)
    print("SECTION 2 — Noise amplification: filter vs naive finite differences")
    print("═" * 78)
    print("\n(a) SYNTHETIC ground truth (the definitive test — truth is known):")
    syn = dg.noise_amplification_synthetic(T=4000, seed=1)
    _tab(syn)
    print("\n(b) REAL SPY (RTS smoother = low-noise reference, DIAGNOSTIC ONLY):")
    real = dg.noise_amplification_real(logp, kf)
    _tab(real)
    # headline read
    vk = real.loc[real.order == "velocity · Kalman", "var_est"].iloc[0]
    vn = real.loc[real.order == "velocity · naive Δ", "var_est"].iloc[0]
    ak = real.loc[real.order == "accel · Kalman", "var_est"].iloc[0]
    an = real.loc[real.order == "accel · naive ΔΔ", "var_est"].iloc[0]
    print(f"\nVariance shrink (filter vs naive): velocity ×{vn/vk:,.0f}, "
          f"acceleration ×{an/ak:,.0f}.")
    syn.to_csv(OUT / "noise_amp_synthetic.csv", index=False)
    real.to_csv(OUT / "noise_amp_real.csv", index=False)
    return syn, real


def section3_innovation(logp: pd.Series, ret: pd.Series, ntr: int):
    print("\n" + "═" * 78)
    print("SECTION 3 — Local-level innovation as a mean-reversion signal")
    print("═" * 78)
    ll = LocalLevelKalman().fit_mle(logp.values[:ntr], verbose=True)
    sig = innovation_signal(ll.filter(logp.values), logp.index)
    te = logp.index[ntr:]
    sig_te, ret_te = sig.reindex(te), ret.reindex(te)
    fwd = ret.shift(-1).reindex(te)
    ic = dg.information_coefficient(sig_te, fwd)
    print(f"\nMLE q/r={ll.q/ll.r:.2e} (on daily closes the filter trusts the "
          f"print → innovation ≈ 1-day return).")
    print(f"OOS IC (signal vs next-day return): {ic:+.4f}")
    decay = dg.ic_decay(sig_te, ret_te)
    print("\nIC decay by horizon (OOS):")
    _tab(decay)
    bt = run_signal_backtest(sig_te, ret, cost_bps=COST_BPS)
    bt_hi = run_signal_backtest(sig_te, ret, cost_bps=COST_BPS_HI)
    print(f"\nReversion backtest (OOS): Sharpe gross={bt.stats['sharpe_gross']:.2f}  "
          f"net@{COST_BPS:.0f}bp={bt.stats['sharpe_net']:.2f}  "
          f"net@{COST_BPS_HI:.0f}bp={bt_hi.stats['sharpe_net']:.2f}  "
          f"ann turnover={bt.stats['ann_turnover']:.0f}x")
    decay.to_csv(OUT / "innovation_ic_decay.csv", index=False)
    return sig, bt


def section4_pair(ntr_frac=0.5):
    print("\n" + "═" * 78)
    print("SECTION 4 — Dynamic Kalman hedge ratio vs static OLS beta (SPY~QQQ)")
    print("═" * 78)
    px = load_fresh_prices(["SPY", "QQQ"]).dropna()
    la, lb = np.log(px["SPY"]), np.log(px["QQQ"])
    n = len(px); ntr = int(ntr_frac * n)
    print(f"Pair sample: {px.index[0].date()} → {px.index[-1].date()} ({n} days)")

    # adaptive: MLE on train, filter full (causal beta_t)
    pk = PairsKalman().fit_mle(la.values[:ntr], lb.values[:ntr], verbose=True)
    pf = pair_spread(la, lb, pk)
    # static: OLS beta/alpha on TRAIN only
    beta_ols, alpha_ols = np.polyfit(lb.values[:ntr], la.values[:ntr], 1)
    static_sp = static_ols_spread(la, lb, beta_ols, alpha_ols)
    print(f"Static OLS (train): beta={beta_ols:.3f}  |  "
          f"adaptive beta_t: {pf['beta'].iloc[0]:.3f} → {pf['beta'].iloc[-1]:.3f} "
          f"(mean OOS {pf['beta'].iloc[ntr:].mean():.3f})")

    te = px.index[ntr:]
    # Stationarity of each hedge's spread. The static spread uses the
    # contemporaneous residual; the adaptive uses the one-step PREDICTIVE
    # residual (innov_spread) — the honest causal "surprise" (the filtered
    # contemporaneous residual is shrunk to ~0 by construction and must NOT be
    # traded, so we never report it as a tradeable edge).
    rows = []
    for name, sp in [("static OLS", static_sp), ("adaptive Kalman (innov)", pf["innov_spread"])]:
        sp_te = sp.reindex(te)
        rows.append({
            "spread": name,
            "OOS std": float(sp_te.std()),
            "OOS |mean|": float(abs(sp_te.mean())),
            "half-life(d)": half_life(sp_te),
        })
    print("\nSpread stationarity (OOS): a hedge that still holds has a small, "
          "non-drifting mean and a short half-life. The static spread drifts as "
          "beta moves from 0.32 toward ~0.62 — the linear hedge breaks.")
    _tab(pd.DataFrame(rows))

    # Causal pair reversion: signal known at close t, P&L from the NEXT-day
    # hedged return formed with beta known at t (filtered beta_t, lagged into
    # the realised return). No contemporaneous-fit lookahead.
    dlogA, dlogB = la.diff(), lb.diff()
    hr_static = dlogA - beta_ols * dlogB
    hr_adapt = dlogA - pf["beta"].shift(1) * dlogB
    print("\nSpread-reversion backtest (OOS), causal, net of cost:")
    bt_rows = []
    curves = {}
    for name, sp, hr in [("static OLS", static_sp, hr_static),
                         ("adaptive Kalman", pf["innov_spread"], hr_adapt)]:
        sig = spread_zscore(sp, window=60).reindex(te)
        bt = run_signal_backtest(sig, hr.reindex(te), cost_bps=COST_BPS)
        bt_hi = run_signal_backtest(sig, hr.reindex(te), cost_bps=COST_BPS_HI)
        bt_rows.append({"spread": name, "Sharpe gross": bt.stats["sharpe_gross"],
                        f"net@{COST_BPS:.0f}bp": bt.stats["sharpe_net"],
                        f"net@{COST_BPS_HI:.0f}bp": bt_hi.stats["sharpe_net"],
                        "ann turnover": bt.stats["ann_turnover"]})
        curves[name] = bt.net
    _tab(pd.DataFrame(bt_rows))

    print("\nPnL by year (net) — the static hedge dislocates worst in 2008:")
    yr = pd.DataFrame({k: dg.pnl_by_year(v) for k, v in curves.items()})
    _tab(yr.round(4), showindex=True)
    yr.to_csv(OUT / "pair_pnl_by_year.csv")
    pf.to_csv(OUT / "pair_filter.csv")
    return pf, curves, float(beta_ols)


def _kalman_velocity_signal(train_lp: pd.Series, full_lp: pd.Series) -> pd.Series:
    """fit_signal_fn for walk_forward: MLE on train, causal velocity z-score."""
    kf = KinematicKalman().fit_mle(train_lp.values)
    return velocity_signal(kf.filter(full_lp.values), full_lp.index)


def section5_regime(logp: pd.Series, ret: pd.Series, ntr: int):
    print("\n" + "═" * 78)
    print("SECTION 5 — Regime gate: 2-state causal HMM on filtered velocity")
    print("═" * 78)
    kf = KinematicKalman().fit_mle(logp.values[:ntr])
    fr = kf.filter(logp.values)
    vel = fr.velocity
    hmm = VelocityRegimeHMM().fit(vel[:ntr])
    info = hmm.summary()
    print(f"Regime emission means (velocity): trend={info.get('trend_mean_vel',0):+.2e} "
          f"revert={info.get('revert_mean_vel',0):+.2e}; "
          f"persistence trend={info.get('trend_persist',0):.2f} "
          f"revert={info.get('revert_persist',0):.2f}")
    p_trend = pd.Series(hmm.trend_prob(vel), index=logp.index)

    mom = velocity_signal(fr, logp.index)                 # trend tilt
    rev = innovation_signal(
        LocalLevelKalman().fit_mle(logp.values[:ntr]).filter(logp.values), logp.index)
    te = logp.index[ntr:]
    pt = p_trend.reindex(te)
    gated = (pt * mom.reindex(te) + (1 - pt) * rev.reindex(te))

    rows = []
    for name, s in [("momentum only", mom), ("reversion only", rev),
                    ("regime-gated", gated)]:
        bt = run_signal_backtest(s.reindex(te), ret, cost_bps=COST_BPS)
        rows.append({"strategy": name, "Sharpe gross": bt.stats["sharpe_gross"],
                     "Sharpe net": bt.stats["sharpe_net"],
                     "ann turnover": bt.stats["ann_turnover"]})
    print("\nOOS net-of-cost, does the gate beat either leg alone?")
    _tab(pd.DataFrame(rows))
    print(f"Mean OOS P(trend) = {pt.mean():.2f}")
    return gated


def section6_baselines(logp: pd.Series, price: pd.Series, ret: pd.Series, ntr: int):
    print("\n" + "═" * 78)
    print("SECTION 6 — Integral baseline (EMA/MACD) vs Kalman velocity; signatures")
    print("═" * 78)
    te = logp.index[ntr:]
    kf = KinematicKalman().fit_mle(logp.values[:ntr])
    kvel = velocity_signal(kf.filter(logp.values), logp.index)

    variants = {
        "Kalman velocity": kvel,
        "EMA(50)": ema_signal(price, span=50),
        "MACD(12,26,9)": macd_signal(price),
        "path-signature ⚠": signature_trend_signal(logp, window=40, train_frac=ntr/len(logp)),
    }
    rows = {}
    for name, s in variants.items():
        s_te = s.reindex(te)
        fwd = ret.shift(-1).reindex(te)
        bt = run_signal_backtest(s_te, ret, cost_bps=COST_BPS)
        rows[name] = {
            "IC": dg.information_coefficient(s_te, fwd),
            "Sharpe gross": bt.stats["sharpe_gross"],
            "Sharpe net": bt.stats["sharpe_net"],
            "ann turnover": bt.stats["ann_turnover"],
            "net": bt.net,
        }
    tab = pd.DataFrame({k: {kk: vv for kk, vv in v.items() if kk != "net"}
                        for k, v in rows.items()}).T
    print("\nTrend signals head-to-head (OOS). The honest question: does the "
          "Kalman velocity beat a plain EMA/MACD, net?")
    _tab(tab.reset_index().rename(columns={"index": "signal"}))
    print("⚠ path-signature is the flagged overfit corner: rich features, short "
          "single-asset sample. Treat any edge as a warning, not a result.")
    return {k: v["net"] for k, v in rows.items()}


def section_rigor(logp: pd.Series, ret: pd.Series, ntr: int, all_nets: dict,
                  headline_net: pd.Series, headline_name: str,
                  extra_nets: dict):
    print("\n" + "═" * 78)
    print("RIGOR — walk-forward, PnL concentration, effective bets, Q/R sweep, DSR")
    print("═" * 78)

    # Walk-forward Kalman velocity (params refit per fold, fully OOS)
    wf_sig = walk_forward(logp, _kalman_velocity_signal, n_folds=5)
    bt_wf = run_signal_backtest(wf_sig, ret, cost_bps=COST_BPS)
    bt_wf_hi = run_signal_backtest(wf_sig, ret, cost_bps=COST_BPS_HI)
    print(f"\nWalk-forward (5 folds, Q/R refit each fold) Kalman-velocity:")
    print(f"  Sharpe gross={bt_wf.stats['sharpe_gross']:.2f}  "
          f"net@{COST_BPS:.0f}bp={bt_wf.stats['sharpe_net']:.2f}  "
          f"net@{COST_BPS_HI:.0f}bp={bt_wf_hi.stats['sharpe_net']:.2f}  "
          f"ann turnover={bt_wf.stats['ann_turnover']:.0f}x")

    # Concentration / breadth on the HEADLINE POSITIVE result (the only place
    # these are meaningful — a losing strategy's 'concentration' is just noise).
    net = headline_net.dropna()
    conc = dg.pnl_concentration(net)
    print(f"\nPnL concentration — {headline_name} (the headline positive): "
          f"top5={conc['top5_share']:.1%}  top10={conc['top10_share']:.1%}  "
          f"top20={conc['top20_share']:.1%}  Gini|pnl|={conc['gini_abs']:.2f}")
    print(f"Effective bets: participation N_eff={dg.effective_bets(net):.0f} of "
          f"{len(net)} days; autocorr-adjusted independent bets="
          f"{dg.autocorr_adjusted_breadth(net):.0f}")
    print(f"\nPnL by year ({headline_name}, net):")
    _tab(dg.pnl_by_year(net).round(4).to_frame("net_pnl"), showindex=True)

    # Q/R sweep — is the velocity edge curve-fit to one noise ratio?
    print("\nQ/R sensitivity sweep (net Sharpe of velocity signal vs q/r ratio):")

    def build(kf):
        return velocity_signal(kf.filter(logp.values), logp.index).reindex(logp.index[ntr:])

    def evals(sig):
        return run_signal_backtest(sig, ret, cost_bps=COST_BPS).stats["sharpe_net"]

    sweep = dg.qr_sweep(logp, build, evals, r_fixed=1.4e-4)
    _tab(sweep)
    sweep.to_csv(OUT / "qr_sweep.csv", index=False)

    # Deflated Sharpe across every variant tried (count them honestly). The
    # family includes the trend signals on SPY, the walk-forward velocity, the
    # innovation reversion and the adaptive pair — every distinct bet this study
    # placed, so the deflation count is not understated.
    trials = {**all_nets, "walk-forward velocity": bt_wf.net.dropna(), **extra_nets}
    daily_srs = [float(v.dropna().mean() / (v.dropna().std() + 1e-12)) for v in trials.values()]
    sr_var = float(np.var(daily_srs, ddof=1))
    print(f"\nDeflated Sharpe (n_trials={len(trials)} variants tried this study):")
    drows = []
    for name, v in trials.items():
        v = v.dropna()
        drows.append({"variant": name,
                      "Sharpe(ann)": float(v.mean()/(v.std()+1e-12))*np.sqrt(ANN),
                      "DSR": deflated_sharpe_ratio(v.values, len(trials), sr_var)})
    _tab(pd.DataFrame(drows).sort_values("DSR", ascending=False))
    print("DSR > 0.95 is the conventional 'real after data-mining' bar.")


def make_plots(logp, kf, pf, beta_ols):
    """Diagnostic figures (the smoother lives HERE — plots only, never traded)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"(plots skipped: {exc})")
        return
    fr = kf.filter(logp.values)
    sm = kf.smooth(logp.values)              # DIAGNOSTIC ONLY
    idx = logp.index
    sl = slice(-500, None)                    # last ~2y for legibility

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ax[0, 0].plot(idx[sl], logp.values[sl], ".", ms=2, alpha=.4, label="log price (obs)")
    ax[0, 0].plot(idx[sl], fr.level[sl], lw=1.2, label="filtered level (causal)")
    ax[0, 0].plot(idx[sl], sm[sl, 0], lw=1.0, ls="--", label="smoothed level (diagnostic)")
    ax[0, 0].set_title("Level: causal filter vs hindsight smoother"); ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(idx[sl], fr.velocity[sl], lw=1.0, label="filtered velocity (causal)")
    ax[0, 1].plot(idx[sl], sm[sl, 1], lw=1.0, ls="--", label="smoothed velocity")
    ax[0, 1].plot(idx[sl], np.r_[np.nan, np.diff(logp.values)][sl], lw=.6, alpha=.5,
                  label="naive Δ (1-bar)")
    ax[0, 1].axhline(0, color="k", lw=.5); ax[0, 1].set_title("Velocity: filter tames the noise")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(pf.index, pf["beta"], lw=1.2, label="adaptive Kalman beta_t (causal)")
    ax[1, 0].axhline(beta_ols, color="r", ls="--", label=f"static OLS beta={beta_ols:.2f}")
    ax[1, 0].set_title("SPY~QQQ hedge ratio: adaptive vs static"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].plot(idx[sl], fr.acceleration[sl], lw=1.0, label="filtered accel (causal)")
    ax[1, 1].plot(idx[sl], sm[sl, 2], lw=1.0, ls="--", label="smoothed accel")
    ax[1, 1].axhline(0, color="k", lw=.5)
    ax[1, 1].set_title("Acceleration: noisy even after filtering (real prices ≠ const-accel)")
    ax[1, 1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "kinematic_diagnostics.png", dpi=110)
    print(f"\nFigure → {(OUT / 'kinematic_diagnostics.png').relative_to(_bootstrap.ROOT)}")


def main():
    px = load_fresh_prices("SPY").dropna()
    price = px["SPY"]
    logp = np.log(price)
    ret = price.pct_change()
    n = len(price); ntr = n // 2
    print(f"SPY: {price.index[0].date()} → {price.index[-1].date()} ({n} days); "
          f"train/test split at {price.index[ntr].date()}")

    kf, fr = section1_filter(logp, ntr)
    section2_noise(logp, kf)
    _, bt_innov = section3_innovation(logp, ret, ntr)
    pf, pair_curves, beta_ols = section4_pair()
    gated = section5_regime(logp, ret, ntr)
    nets = section6_baselines(logp, price, ret, ntr)
    # headline positive = adaptive Kalman pair reversion (best DSR-eligible net)
    section_rigor(logp, ret, ntr, nets,
                  headline_net=pair_curves["adaptive Kalman"],
                  headline_name="adaptive Kalman pair",
                  extra_nets={"innovation reversion": bt_innov.net.dropna(),
                              "adaptive Kalman pair": pair_curves["adaptive Kalman"].dropna()})
    make_plots(logp, kf, pf, beta_ols)

    print("\n" + "═" * 78)
    print(f"Artifacts written to {OUT.relative_to(_bootstrap.ROOT)}/")
    print("═" * 78)


if __name__ == "__main__":
    main()
