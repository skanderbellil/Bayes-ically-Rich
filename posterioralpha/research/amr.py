"""
Asymmetric Minimum-Risk (AMR) strategy primitives (strategy-research layer).

This module holds the *definitions* the AMR backtest engine consumes:
portfolio optimisers, the BOCPD-driven continuous-λ calibration, and the
volatility-targeting overlay. The time-stepping engine that strings these
together lives in ``posterioralpha.backtest.amr``.

Design principles (as specified):
  • No expected-return estimation — avoids μ estimation error entirely
  • Objective: minimize downside_risk − λ · upside_potential
      downside_risk    = sqrt( mean( min(r_port, 0)² ) )
      upside_potential = mean( max(r_port, 0) )
  • Volatility targeting overlay: scale weights so portfolio targets 10% ann. vol
  • Transaction costs handled by the engine on effective (scaled) weights

Optimisers provided
-------------------
  amr_weights         — core asymmetric min-risk (semi-deviation objective)
  amr_cvar_weights    — CVaR (Expected Shortfall) objective + low-vol tilt
  hrp_weights         — Hierarchical Risk Parity (Lopez de Prado 2016)
  inverse_vol_weights — inverse-volatility baseline

Signals / overlays
------------------
  compute_continuous_lam      — data-derived λ via EWMA Omega + ERL credibility
  vol_target_scale            — fixed-cap volatility targeting scale factor
  vol_target_scale_adaptive   — regime-adaptive (BOCPD-driven) vol targeting
"""
import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

_EPS = 1e-8
_ANN = 252


# ---------------------------------------------------------------------------
# Continuous λ calibration
# ---------------------------------------------------------------------------

def compute_continuous_lam(
    window_arr: np.ndarray,
    cp: float,
    spy_idx: int = 0,
    erl: Optional[float] = None,
    rf_daily: float = 0.04 / 252,
    lam_min: float = 0.10,
    lam_max: float = 0.80,
    amplify: float = 2.5,
    ewma_halflife: float = 42.0,
    erl_halflife: float = 63.0,
) -> float:
    """
    Data-derived continuous λ via the EWMA Omega ratio + ERL credibility discount.

    Two improvements over the original flat-window version:

    1. EWMA Omega  (Keating & Shadwick 2002 + RiskMetrics convention)
    -----------------------------------------------------------------------
    Replace equal-weighted gains/losses averages with exponentially weighted
    means (halflife = 42 days ≈ 2 months).  A bear-market print 8 months ago
    should not drag λ down the same as last week's data.

        w_t = exp(−(T−1−t) × ln2 / halflife),  normalised to sum = 1
        Ω_ewma = Σ w_t · max(r_t − τ, 0) / Σ w_t · max(τ − r_t, 0)

    2. ERL-based credibility discount  (replaces the broken cp×20 formula)
    -----------------------------------------------------------------------
    cp is mathematically constant at the hazard rate H (after normalization,
    R[0] = H always — see Adams & MacKay 2007 derivation).  The real BOCPD
    changepoint signal lives in the ERL: it drops sharply when a regime changes.

    We use an exponential decay over ERL:
        transition = exp(−erl / erl_halflife)
        λ = (1 − transition) × λ_omega + transition × 0.5

    At erl = 1   (just transitioned) : transition ≈ 1  → λ ≈ 0.5  (max uncertainty)
    At erl = 63  (3 months stable)   : transition ≈ 0.37 → partial trust in Ω
    At erl = 252 (1 year stable)     : transition ≈ 0.02 → nearly pure Ω signal

    When erl is not provided, falls back to the cp-based formula (backward compat).

    Parameters
    ----------
    window_arr    : (T, N) return window
    cp            : BOCPD changepoint probability (kept for backward compat)
    spy_idx       : column index of equity/market proxy (SPY)
    erl           : BOCPD expected run length (preferred credibility signal)
    rf_daily      : daily risk-free rate (Omega threshold τ)
    lam_min/max   : hard bounds on λ
    amplify       : sigmoid steepness (1 = flat Ω/(1+Ω), 2.5 = responsive)
    ewma_halflife : exponential decay halflife in trading days for Omega
    erl_halflife  : ERL halflife for credibility decay (63 days ≈ 3 months)
    """
    r = window_arr[:, spy_idx]
    T = len(r)

    # EWMA weights — recent returns count more (Becker et al. 2006)
    ew = np.exp(-np.log(2.0) * np.arange(T)[::-1] / ewma_halflife)
    ew /= ew.sum()

    gains  = float(np.sum(ew * np.maximum(r - rf_daily, 0.0)))
    losses = float(np.sum(ew * np.maximum(rf_daily - r, 0.0)))

    if losses < _EPS:
        log_omega = 3.0
    else:
        log_omega = float(np.log(np.clip(gains / losses, 1e-6, 1e6)))

    lam_omega = float(1.0 / (1.0 + np.exp(-amplify * log_omega)))
    lam_omega = float(np.clip(lam_omega, lam_min, lam_max))

    # ERL-based credibility discount (preferred) or cp fallback
    if erl is not None and erl > 0:
        transition = float(np.exp(-erl / erl_halflife))
    else:
        transition = float(1.0 - np.clip(1.0 - cp * 20.0, 0.0, 1.0))

    lam = (1.0 - transition) * lam_omega + transition * 0.5
    return float(np.clip(lam, lam_min, lam_max))


# ---------------------------------------------------------------------------
# Core optimisers
# ---------------------------------------------------------------------------

def amr_weights(
    returns: np.ndarray,
    lam: float = 0.50,
    max_weight: float = 0.35,
    l2_reg: float = 0.001,
) -> np.ndarray:
    """
    Asymmetric Minimum-Risk weights (long-only, SLSQP).

    Minimises:
        sqrt( mean(min(wᵀr, 0)²) ) − λ · mean(max(wᵀr, 0)) + l2_reg · ‖w − 1/N‖²

    The L2 regularisation term stabilises the solution by gently pulling toward
    equal weight — without it the optimiser can produce degenerate solutions on
    small universes.
    """
    N   = returns.shape[1]
    w0  = np.ones(N) / N
    w_eq = w0.copy()

    def objective(w: np.ndarray) -> float:
        r_p      = returns @ w
        downside = float(np.sqrt(np.mean(np.minimum(r_p, 0.0) ** 2) + _EPS))
        upside   = float(np.mean(np.maximum(r_p, 0.0)))
        reg      = float(l2_reg * np.sum((w - w_eq) ** 2))
        return downside - lam * upside + reg

    res = minimize(
        objective, w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * N,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 3000},
    )

    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, max_weight)
        return w / w.sum()

    logger.debug("AMR optimisation failed — equal weights fallback")
    return w0


def amr_cvar_weights(
    returns: np.ndarray,
    lam: float = 0.50,
    max_weight: float = 0.35,
    l2_reg: float = 0.001,
    alpha: float = 0.05,
    low_vol_penalty: float = 0.0,
) -> np.ndarray:
    """
    AMR with CVaR (Expected Shortfall) instead of semi-deviation, plus an
    optional cross-sectional low-volatility tilt (min-variance anomaly).

    Objective
    ---------
        CVaR_α(rₚ) − λ · mean(max(rₚ, 0))
        + low_vol_penalty · wᵀ · vol_rank
        + l2_reg · ‖w − 1/N‖²

    CVaR_α  = mean loss in the worst α fraction of days (α=5% by default).
              Captures fat tails better than semi-deviation; a $-20 day counts
              far more than ten $-2 days.

    vol_rank  = per-asset percentile rank of 1-year realized vol (0=lowest vol,
                1=highest vol). Penalising high-rank assets tilts toward lower-
                vol holdings — exploiting the low-volatility anomaly.

    low_vol_penalty  = 0.0  → pure CVaR-AMR (no tilt)
                     = 0.30 → strong low-vol bias (use during regime stress)
    """
    N    = returns.shape[1]
    w0   = np.ones(N) / N

    # Annualised individual vols → percentile ranks (0=lowest, 1=highest)
    indiv_vols = returns.std(axis=0) * np.sqrt(_ANN)
    if N > 1:
        vol_ranks = np.argsort(np.argsort(indiv_vols)) / (N - 1)
    else:
        vol_ranks = np.zeros(N)

    n_tail = max(1, int(np.ceil(alpha * len(returns))))

    def objective(w: np.ndarray) -> float:
        r_p   = returns @ w
        # CVaR: mean of the worst n_tail daily returns
        cvar  = -float(np.mean(np.sort(r_p)[:n_tail]))          # positive = bad
        up    = float(np.mean(np.maximum(r_p, 0.0)))
        vtilt = float(low_vol_penalty * np.dot(w, vol_ranks))    # penalise high-vol assets
        reg   = float(l2_reg * np.sum((w - w0) ** 2))
        return cvar - lam * up + vtilt + reg

    res = minimize(
        objective, w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * N,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-10, "maxiter": 3000},
    )

    if res.success and np.all(np.isfinite(res.x)):
        w = np.clip(res.x, 0.0, max_weight)
        return w / w.sum()

    logger.debug("AMR-CVaR optimisation failed — equal weights fallback")
    return w0


def hrp_weights(
    returns: np.ndarray,
    max_weight: float = 0.40,
) -> np.ndarray:
    """
    Hierarchical Risk Parity  (Lopez de Prado, J. Portfolio Management 2016).

    Three steps:
      1. Build a correlation-distance matrix and hierarchically cluster assets
         (single-linkage).  Similar assets end up in the same subtree.
      2. Quasi-diagonalise: reorder columns/rows so correlated assets are adjacent.
      3. Recursive bisection: each binary split allocates capital inversely
         proportional to each cluster's within-cluster variance.

    Benefits over mean-variance optimisation
    -----------------------------------------
    • No matrix inversion — immune to near-singular covariance.
    • Out-of-sample variance beats CLA by ~72% in Lopez de Prado's Monte Carlo.
    • Naturally diversifies across clusters rather than concentrating on the
      estimated minimum-variance portfolio which exploits noise.
    • On small universes (5 assets) the main gain is stability at regime transitions
      where covariance estimates are most unreliable.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    N   = returns.shape[1]
    cov = np.cov(returns.T)

    # Correlation → angular distance  d_ij = sqrt((1 - ρ_ij) / 2) ∈ [0, 1]
    std  = np.sqrt(np.diag(cov))
    std  = np.clip(std, _EPS, None)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -0.9999, 0.9999)
    dist = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))

    dist_condensed = squareform(dist, checks=False)
    link  = linkage(dist_condensed, method="single")
    order = list(leaves_list(link))   # quasi-diagonalised asset order

    def _cluster_var(idxs: list, w_local: dict) -> float:
        sub_w   = np.array([w_local[i] for i in idxs])
        sub_cov = cov[np.ix_(idxs, idxs)]
        return float(sub_w @ sub_cov @ sub_w + _EPS)

    def _bisect(items: list) -> dict:
        if len(items) == 1:
            return {items[0]: 1.0}
        mid   = len(items) // 2
        left, right   = items[:mid], items[mid:]
        w_left, w_right = _bisect(left), _bisect(right)
        v_l = _cluster_var(left,  w_left)
        v_r = _cluster_var(right, w_right)
        alpha = 1.0 - v_l / (v_l + v_r)
        result = {}
        for i, w in w_left.items():
            result[i] = w * alpha
        for i, w in w_right.items():
            result[i] = w * (1.0 - alpha)
        return result

    w_dict = _bisect(order)
    w = np.array([w_dict.get(i, 0.0) for i in range(N)])
    w = np.clip(w, 0.0, max_weight)
    s = w.sum()
    return w / s if s > _EPS else np.ones(N) / N


def inverse_vol_weights(
    returns: np.ndarray,
    max_weight: float = 0.35,
) -> np.ndarray:
    """Inverse-volatility portfolio (benchmark from AMR spec)."""
    vols    = returns.std(axis=0) * np.sqrt(_ANN)
    vols    = np.clip(vols, _EPS, None)
    inv_vol = 1.0 / vols
    w       = np.clip(inv_vol / inv_vol.sum(), 0.0, max_weight)
    return w / w.sum()


# ---------------------------------------------------------------------------
# Volatility targeting overlay
# ---------------------------------------------------------------------------

def vol_target_scale(
    weights: np.ndarray,
    recent_returns: np.ndarray,   # (vol_window, N) trailing returns
    target_vol: float = 0.10,
    leverage_cap: float = 1.50,
) -> float:
    """
    Scale factor s = min(σ_target / σ_est, leverage_cap).

    σ_est is computed from the trailing portfolio returns using the *current*
    weights, so there is no lookahead bias.
    """
    port_rets = recent_returns @ weights
    sigma_est = port_rets.std() * np.sqrt(_ANN)
    if sigma_est < _EPS:
        return 1.0
    return float(np.clip(target_vol / sigma_est, 0.0, leverage_cap))


def vol_target_scale_adaptive(
    weights: np.ndarray,
    recent_returns: np.ndarray,
    target_vol: float = 0.10,
    base_cap: float = 1.50,
    cp_signal: float = 0.0,
    erl_signal: float = 126.0,
) -> float:
    """
    Regime-adaptive vol-targeting scale factor.

    The leverage cap adjusts based on BOCPD signals:
      • Stable regime (high erl, low cp) → slightly higher cap (more confidence)
      • Regime transition (high cp)       → reduced cap (protect against unknown)

    Adaptive cap formula
    --------------------
        stability  = clip(erl / 126, 0, 1)   # 0=recent change, 1=6+ months stable
        cp_penalty = clip(cp × 20,   0, 1)   # 0=stable,        1=strong changepoint
        cap = base_cap + 0.25 × stability − 0.50 × cp_penalty
        cap = clip(cap, 0.75, base_cap + 0.25)

    Net effect:
      • Fully stable (erl≥126, cp≈0) → cap rises to base_cap + 0.25
      • Strong changepoint (cp≥0.05)  → cap drops to base_cap − 0.25 (min 0.75)
    """
    port_rets = recent_returns @ weights
    sigma_est = port_rets.std() * np.sqrt(_ANN)
    if sigma_est < _EPS:
        return 1.0

    stability  = float(np.clip(erl_signal / 126.0, 0.0, 1.0))
    cp_penalty = float(np.clip(cp_signal  * 20.0,  0.0, 1.0))
    adaptive_cap = base_cap + 0.25 * stability - 0.50 * cp_penalty
    adaptive_cap = float(np.clip(adaptive_cap, 0.75, base_cap + 0.25))

    return float(np.clip(target_vol / sigma_est, 0.0, adaptive_cap))
