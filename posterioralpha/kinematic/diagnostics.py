"""
Honest diagnostics for the kinematic layer (the part the brief actually cares
about): noise amplification, signal decay, PnL concentration, effective number
of independent bets, and a Q/R sensitivity sweep.

The guiding principle is to make a NEGATIVE result legible rather than hide it.
The acceleration state in particular is expected to be mostly differencing
noise; these tools are built to show exactly how much (if any) the filter
rescues, against a ground-truth synthetic where the answer is knowable.
"""
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from posterioralpha.kinematic.filter import KinematicKalman

ANN = 252


# ─────────────────────────────────────────────────────────────────────────────
# 1. Noise amplification — synthetic ground truth (the definitive test)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_kinematic(
    T: int = 3000, q: float = 1e-6, r: float = 4e-4,
    drift0: float = 5e-4, p0: float = np.log(100.0), seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Simulate a constant-acceleration latent state driven by white-noise jerk,
    observed with Gaussian noise. Returns the TRUE level/velocity/acceleration
    and the noisy observation — so estimator quality can be scored against truth.
    """
    rng = np.random.default_rng(seed)
    F, Qj = KinematicKalman.F, KinematicKalman.Qj
    L = np.linalg.cholesky(q * Qj + 1e-18 * np.eye(3))
    x = np.array([p0, drift0, 0.0])
    states = np.empty((T, 3))
    obs = np.empty(T)
    for t in range(T):
        x = F @ x + L @ rng.standard_normal(3)
        states[t] = x
        obs[t] = x[0] + rng.normal(0.0, np.sqrt(r))
    return {"level": states[:, 0], "velocity": states[:, 1],
            "acceleration": states[:, 2], "obs": obs}


def _snr_vs_truth(est: np.ndarray, truth: np.ndarray) -> float:
    """SNR ≡ Var(truth) / Var(est − truth): how much of the true signal variance
    survives the estimation error. >1 means the estimate carries more signal
    than error; ≲1 means it is dominated by noise."""
    est, truth = np.asarray(est), np.asarray(truth)
    m = np.isfinite(est) & np.isfinite(truth)
    err = est[m] - truth[m]
    return float(np.var(truth[m]) / (np.var(err) + 1e-300))


def noise_amplification_synthetic(
    T: int = 3000, q: float = 1e-6, r: float = 4e-4, seed: int = 0,
    train_frac: float = 0.5,
) -> pd.DataFrame:
    """
    Filter vs naive finite-difference derivatives on a KNOWN process. For
    velocity and acceleration, reports correlation-to-truth, RMSE and SNR for
    (a) the causal Kalman filter and (b) naive differencing. (q, r) for the
    filter are MLE-fit on the train half only.
    """
    sim = simulate_kinematic(T=T, q=q, r=r, seed=seed)
    obs = sim["obs"]
    ntr = int(train_frac * T)
    kf = KinematicKalman().fit_mle(obs[:ntr])
    fr = kf.filter(obs)
    te = slice(ntr, T)

    naive_v = np.r_[np.nan, np.diff(obs)]
    naive_a = np.r_[np.nan, np.nan, np.diff(np.diff(obs))]

    def row(name, est, truth):
        est, truth = est[te], truth[te]
        m = np.isfinite(est) & np.isfinite(truth)
        return {
            "order": name,
            "corr_truth": float(np.corrcoef(est[m], truth[m])[0, 1]),
            "rmse": float(np.sqrt(np.mean((est[m] - truth[m]) ** 2))),
            "snr": _snr_vs_truth(est, truth),
            "var_est": float(np.var(est[m])),
        }

    rows = [
        row("velocity · Kalman", fr.velocity, sim["velocity"]),
        row("velocity · naive Δ", naive_v, sim["velocity"]),
        row("accel · Kalman", fr.acceleration, sim["acceleration"]),
        row("accel · naive ΔΔ", naive_a, sim["acceleration"]),
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise amplification — real data (smoother as low-noise reference)
# ─────────────────────────────────────────────────────────────────────────────

def noise_amplification_real(
    log_price: pd.Series, kf: KinematicKalman, train_frac: float = 0.5,
) -> pd.DataFrame:
    """
    On real price there is no ground truth, so we use the RTS smoother (full
    sample — DIAGNOSTIC ONLY, never traded) as the low-noise reference for what
    the latent derivative "should" look like, and measure how close each causal
    estimator gets on the OOS half.

    Reports, per derivative order: variance of the estimator (the noise-
    amplification headline — naive ΔΔ should explode), correlation to the
    smoother reference, an SNR proxy (Var(reference)/Var(est−reference)), and the
    lag-1 autocorrelation (differencing noise shows up as strong negative AC).
    """
    y = log_price.values
    n = len(y)
    te = slice(int(train_frac * n), n)
    fr = kf.filter(y)
    sm = kf.smooth(y)                                   # reference only

    naive_v = np.r_[np.nan, np.diff(y)]
    naive_a = np.r_[np.nan, np.nan, np.diff(np.diff(y))]

    def lag1(x):
        x = x[te]; x = x[np.isfinite(x)]
        if len(x) < 3:
            return np.nan
        return float(np.corrcoef(x[:-1], x[1:])[0, 1])

    def row(name, est, ref):
        e, rf = est[te], ref[te]
        m = np.isfinite(e) & np.isfinite(rf)
        return {
            "order": name,
            "var_est": float(np.var(e[m])),
            "corr_smoother": float(np.corrcoef(e[m], rf[m])[0, 1]),
            "snr_vs_smoother": _snr_vs_truth(e, rf),
            "lag1_autocorr": lag1(est),
        }

    rows = [
        row("velocity · Kalman", fr.velocity, sm[:, 1]),
        row("velocity · naive Δ", naive_v, sm[:, 1]),
        row("accel · Kalman", fr.acceleration, sm[:, 2]),
        row("accel · naive ΔΔ", naive_a, sm[:, 2]),
    ]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Signal quality: IC, hit-rate, decay
# ─────────────────────────────────────────────────────────────────────────────

def information_coefficient(
    signal: pd.Series, fwd_ret: pd.Series, method: str = "spearman",
) -> float:
    df = pd.concat([signal, fwd_ret], axis=1).dropna()
    if len(df) < 30:
        return np.nan
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method=method))


def ic_decay(
    signal: pd.Series, asset_ret: pd.Series, horizons=(1, 2, 3, 5, 10, 20),
) -> pd.DataFrame:
    """IC of the signal against forward returns at increasing horizons — how
    fast the edge decays. Also reports the hit-rate of sign agreement."""
    rows = []
    for h in horizons:
        fwd = asset_ret.shift(-1).rolling(h).sum().shift(-(h - 1))  # cum fwd ret over next h
        df = pd.concat([signal, fwd], axis=1).dropna()
        if len(df) < 30:
            rows.append({"horizon": h, "IC": np.nan, "hit_rate": np.nan, "n": len(df)})
            continue
        ic = df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")
        hit = float((np.sign(df.iloc[:, 0]) == np.sign(df.iloc[:, 1])).mean())
        rows.append({"horizon": h, "IC": float(ic), "hit_rate": hit, "n": len(df)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PnL concentration & effective number of bets
# ─────────────────────────────────────────────────────────────────────────────

def pnl_concentration(pnl: pd.Series, top_k=(5, 10, 20)) -> Dict[str, float]:
    """
    Where does PnL live? Share of total positive-direction PnL from the top-k
    days, and a Gini coefficient of |daily PnL|. High concentration → the result
    rides a handful of days and is fragile.
    """
    p = pnl.dropna()
    total = p.sum()
    out: Dict[str, float] = {}
    ordered = p.reindex(p.abs().sort_values(ascending=False).index)
    for k in top_k:
        out[f"top{k}_share"] = float(ordered.iloc[:k].sum() / total) if total != 0 else np.nan
    # Gini of |pnl|
    a = np.sort(np.abs(p.values))
    nn = len(a)
    if nn > 0 and a.sum() > 0:
        out["gini_abs"] = float((2 * np.arange(1, nn + 1) - nn - 1).dot(a) / (nn * a.sum()))
    else:
        out["gini_abs"] = np.nan
    return out


def pnl_by_year(pnl: pd.Series) -> pd.Series:
    """Calendar-year sum of daily PnL — which years carry the result."""
    return pnl.dropna().groupby(pnl.dropna().index.year).sum()


def effective_bets(pnl: pd.Series) -> float:
    """
    Participation ratio of daily PnL, N_eff = (Σ|p|)² / Σ p². Loosely 'how many
    days actually contributed'. Far below the sample length → a few days do the
    work. (Reported alongside concentration, not as a Sharpe substitute.)
    """
    p = pnl.dropna().values
    s2 = np.sum(p ** 2)
    return float(np.sum(np.abs(p)) ** 2 / s2) if s2 > 0 else np.nan


def autocorr_adjusted_breadth(pnl: pd.Series) -> float:
    """
    Effective number of INDEPENDENT bets ≈ N · (1 − ρ₁)/(1 + ρ₁) using the lag-1
    autocorrelation of daily PnL (a standard variance-inflation correction).
    Positive autocorrelation shrinks the independent-bet count.
    """
    p = pnl.dropna().values
    if len(p) < 30:
        return np.nan
    rho = np.corrcoef(p[:-1], p[1:])[0, 1]
    rho = np.clip(rho, -0.999, 0.999)
    return float(len(p) * (1 - rho) / (1 + rho))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Q/R sensitivity sweep — is the result curve-fit to one noise ratio?
# ─────────────────────────────────────────────────────────────────────────────

def qr_sweep(
    log_price: pd.Series,
    build_signal: Callable[[KinematicKalman], pd.Series],
    eval_fn: Callable[[pd.Series], float],
    ratios=np.logspace(-5, 0, 11),
    r_fixed: float = 4e-4,
) -> pd.DataFrame:
    """
    Sweep the process/observation-noise ratio q/r across orders of magnitude,
    rebuild the signal with each (r held fixed, q = ratio·r), and re-evaluate
    via `eval_fn` (e.g. net Sharpe or IC). A result that only exists at one
    ratio is curve-fit; a robust one is a plateau across the sweep.

    `build_signal(kf)` must return a causal signal from a configured filter;
    `eval_fn(signal)` must return a scalar score.
    """
    rows = []
    for ratio in ratios:
        kf = KinematicKalman(q=ratio * r_fixed, r=r_fixed)
        score = eval_fn(build_signal(kf))
        rows.append({"q_over_r": float(ratio), "score": float(score)})
    return pd.DataFrame(rows)
