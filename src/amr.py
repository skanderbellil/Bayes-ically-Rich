"""
Asymmetric Minimum-Risk (AMR) strategy with volatility targeting.

Design principles (as specified):
  • No expected-return estimation — avoids μ estimation error entirely
  • Objective: minimize downside_risk − λ · upside_potential
      downside_risk    = sqrt( mean( min(r_port, 0)² ) )
      upside_potential = mean( max(r_port, 0) )
  • Volatility targeting overlay: scale weights so portfolio targets 10% ann. vol
  • Weekly rebalancing, 252-day lookback window
  • Transaction costs: 5 bps per unit of turnover (on effective/scaled weights)

Strategies in this module
--------------------------
  amr          — core AMR with vol targeting
  inv_vol      — inverse-volatility baseline (also with vol targeting)
  amr_no_vt    — AMR without vol targeting (isolates overlay contribution)
  bocpd_amr    — BOCPD drives adaptive lookback + λ; AMR optimises
  hmm3_amr     — 3-state HMM blends regime-specific AMR portfolios + vol targeting
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

logger = logging.getLogger(__name__)

_EPS = 1e-8
_ANN = 252

AMR_STRATEGIES = ("amr", "inv_vol", "amr_no_vt", "bocpd_amr",
                  "bocpd_amr_v2", "bocpd_amr_v3", "bocpd_amr_v4", "hmm3_amr")


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


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AMRResult:
    strategy:   str
    returns:    pd.Series      # daily arithmetic returns (post-TC, post-scaling)
    weights:    pd.DataFrame   # unscaled optimal weights at each rebalance
    leverage:   pd.Series      # vol-targeting scale factor at each rebalance

    @property
    def cumulative(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def run_amr_backtest(
    returns: pd.DataFrame,
    strategy: str = "amr",
    lookback: int = 252,
    vol_window: int = 21,
    target_vol: float = 0.10,
    leverage_cap: float = 1.50,
    lam: float = 0.50,
    max_weight: float = 0.35,
    l2_reg: float = 0.001,
    tc: float = 0.0005,          # 5 bps per unit of turnover
    use_vol_target: bool = True,
    bocpd_hazard: float = 1 / 252,
) -> AMRResult:
    """
    Weekly-rebalanced backtest with optional volatility targeting overlay.

    Rebalance grid  : last trading day of each calendar week (Friday or prior).
    Lookback window : 252 trading days for optimisation.
    Vol window      : 21 trading days for volatility estimation.
    Transaction cost: applied to *effective* (scaled) weight turnover.

    Extra strategies
    ----------------
    bocpd_amr  : BOCPD pre-computed on SPY. At each rebalance the expected
                 run length sets the AMR lookback (recent regime data only)
                 and the changepoint probability reduces λ (be defensive
                 during transitions).
    hmm3_amr   : 3-state HMM (bull/sideways/bear). Each regime gets its own
                 AMR portfolio with a regime-appropriate λ. Final weights are
                 blended by forward-filtered regime probabilities + vol target.
    """
    from src.bayesian import min_variance_weights  # avoid circular import

    tickers  = returns.columns.tolist()
    n_assets = len(tickers)

    # ── Pre-compute BOCPD signals (O(T), done once before loop) ───────────
    bocpd_cp  = None
    bocpd_erl = None
    if strategy == "bocpd_amr":
        from src.regime_models import precompute_bocpd
        spy_col = "SPY" if "SPY" in tickers else tickers[0]
        spy_idx = tickers.index(spy_col)
        logger.info(f"  Pre-computing BOCPD on {spy_col} …")
        _cp, _erl = precompute_bocpd(returns.values[:, spy_idx], hazard=bocpd_hazard)
        bocpd_cp  = pd.Series(_cp,  index=returns.index)
        bocpd_erl = pd.Series(_erl, index=returns.index)

    if strategy in ("bocpd_amr_v2", "bocpd_amr_v3", "bocpd_amr_v4"):
        # Multi-asset BOCPD: SPY captures equity regime, TLT captures rate regime,
        # GLD captures flight-to-safety / inflation regime.
        # Weights: SPY 50%, TLT 30%, GLD 20% — equity regime is most predictive
        # for portfolio performance, bonds/gold add early warning of stress.
        from src.regime_models import precompute_bocpd_multi
        _bocpd_assets  = ["SPY", "TLT", "GLD"]
        _bocpd_weights = np.array([0.50, 0.30, 0.20])
        logger.info(f"  Pre-computing multi-asset BOCPD on {_bocpd_assets} …")
        _cp, _erl = precompute_bocpd_multi(
            returns, assets=_bocpd_assets,
            hazard=bocpd_hazard, weights=_bocpd_weights,
        )
        bocpd_cp  = pd.Series(_cp,  index=returns.index)
        bocpd_erl = pd.Series(_erl, index=returns.index)

    # ── HMM3 state (refitted periodically) ────────────────────────────────
    hmm3_model      = None
    hmm3_refit_ctr  = 0
    HMM3_REFIT_EVERY = 8   # every 8 weekly rebalances ≈ 2 months

    # Weekly rebalance dates
    try:
        rebal_dates = returns.resample("W-FRI").last().index
    except Exception:
        rebal_dates = returns.resample("W").last().index
    rebal_dates = rebal_dates[rebal_dates > returns.index[lookback + vol_window]]

    port_rets:    List[float] = []
    ret_dates:    List        = []
    weight_rows:  List        = []
    weight_dates: List        = []
    lev_vals:     List[float] = []

    prev_eff_w: Optional[np.ndarray] = None

    for i, rebal_t in enumerate(tqdm(rebal_dates, desc=f"  {strategy:<22}", leave=False)):
        hist = returns.loc[:rebal_t]
        if len(hist) < lookback + vol_window:
            continue

        window_arr = hist.values[-lookback:]   # standard optimisation window
        recent_arr = hist.values[-vol_window:] # vol estimation window

        # ── Core weights ──────────────────────────────────────────────────
        if strategy in ("amr", "amr_no_vt"):
            weights = amr_weights(window_arr, lam=lam,
                                  max_weight=max_weight, l2_reg=l2_reg)

        elif strategy == "inv_vol":
            weights = inverse_vol_weights(window_arr, max_weight=max_weight)

        elif strategy == "bocpd_amr":
            # ── BOCPD-AMR ────────────────────────────────────────────────
            # Signals at current rebalance date
            cp  = float(bocpd_cp.loc[:rebal_t].iloc[-1])
            erl = float(bocpd_erl.loc[:rebal_t].iloc[-1])

            # Adaptive lookback: use data since last likely change point
            # Short ERL → recent regime shift → use less history
            # Long  ERL → stable regime       → use full lookback
            adaptive_lb = int(np.clip(erl * 1.5, 42, lookback))
            opt_arr     = hist.values[-adaptive_lb:]

            # Adaptive λ: reduce upside participation during transitions
            # cp near 0 → stable → λ = 0.55  |  cp near 1 → transition → λ = 0.25
            lam_adaptive = 0.55 - 0.30 * float(np.clip(cp * 30, 0, 1))

            weights = amr_weights(opt_arr, lam=lam_adaptive,
                                  max_weight=max_weight, l2_reg=l2_reg)

        elif strategy == "bocpd_amr_v2":
            # ── BOCPD-AMR v2 — four statistical improvements ─────────────
            # Signals from multi-asset BOCPD (SPY+TLT+GLD aggregate)
            cp  = float(bocpd_cp.loc[:rebal_t].iloc[-1])
            erl = float(bocpd_erl.loc[:rebal_t].iloc[-1])

            # 1. Adaptive lookback (same logic as v1)
            adaptive_lb = int(np.clip(erl * 1.5, 42, lookback))
            opt_arr     = hist.values[-adaptive_lb:]

            # 2. Adaptive λ (same decay as v1)
            lam_adaptive = 0.55 - 0.30 * float(np.clip(cp * 30, 0, 1))

            # 3. Dynamic low-vol tilt (min-var anomaly):
            #    In calm regimes: small tilt (0.05) — let CVaR-AMR run freely
            #    During transitions: stronger tilt (up to 0.35) — flee to lower-vol assets
            #    This exploits the cross-sectional low-vol anomaly where it's strongest
            low_vol_penalty = 0.05 + 0.30 * float(np.clip(cp * 30, 0, 1))

            # 4. CVaR objective (5% Expected Shortfall) instead of semi-deviation
            weights = amr_cvar_weights(
                opt_arr, lam=lam_adaptive,
                max_weight=max_weight, l2_reg=l2_reg,
                alpha=0.05, low_vol_penalty=low_vol_penalty,
            )

        elif strategy == "bocpd_amr_v3":
            # ── BOCPD-AMR v3 — continuously-calibrated λ ─────────────────
            # Builds on all v2 improvements but replaces the hard-coded
            # λ formula with a data-derived continuous calibration.
            #
            # v2 used:  lam = 0.55 − 0.30 × clip(cp × 30, 0, 1)
            #           → three arbitrary constants, clips hard at cp > 0.033
            #
            # v3 uses:  lam = f(Omega ratio, cp)
            #           → Omega measures empirical upside/downside from data
            #           → cp applies a credibility discount (shrink toward 0.5)
            #           → no preset constants except optional floor/ceiling
            cp  = float(bocpd_cp.loc[:rebal_t].iloc[-1])
            erl = float(bocpd_erl.loc[:rebal_t].iloc[-1])

            # Adaptive lookback (same as v1/v2)
            adaptive_lb = int(np.clip(erl * 1.5, 42, lookback))
            opt_arr     = hist.values[-adaptive_lb:]

            # Continuous λ: SPY Omega ratio → amplified sigmoid → BOCPD discount
            _spy_idx = tickers.index("SPY") if "SPY" in tickers else 0
            lam_continuous = compute_continuous_lam(
                window_arr,           # full 252-day window for stable Omega
                cp=cp,
                spy_idx=_spy_idx,
                rf_daily=0.04 / _ANN,
            )

            # Use semi-deviation objective (not CVaR) so λ has real leverage.
            # CVaR at 5% = only 12 worst days dominate → same corner solution
            # regardless of λ.  Semi-deviation uses all negative days so the
            # upside term lam × E[max(r,0)] meaningfully shifts the portfolio.
            weights = amr_weights(
                opt_arr, lam=lam_continuous,
                max_weight=max_weight, l2_reg=l2_reg,
            )

        elif strategy == "bocpd_amr_v4":
            # ── BOCPD-AMR v4 — ERL-adaptive EWMA halflife ─────────────────
            #
            # Lessons from v3 diagnostic:
            #   • cp is always H (mathematical constant after normalization) → useless
            #   • ERL credibility discount fights Omega in BOTH directions → removed
            #   • Kelly guard on 21-day Sharpe fires too many false positives → removed
            #
            # Clean design:
            #   1. ERL-adaptive EWMA halflife  — regime-responsive Omega
            #      ERL tells us how long the current regime has lasted.  Instead of a
            #      fixed 42-day halflife, adapt it:
            #        halflife = clip(erl / 3,  14, 84)
            #      • Fresh regime (erl=10): halflife=14 → very responsive to recent data
            #      • Stable regime (erl=126): halflife=42 → standard (same as v3)
            #      • Long regime (erl=252): halflife=84 → stable estimate uses more history
            #      This is principled: the EWMA integrates ERL without fighting Omega.
            #      During crashes, Omega drops faster (shorter halflife → less history drag).
            #      During stable bulls, Omega is more reliable (longer halflife → less noise).
            #
            #   2. Adaptive lookback from ERL — unchanged from v1/v2/v3
            #
            # No credibility discount needed: Omega is self-sufficient.
            cp  = float(bocpd_cp.loc[:rebal_t].iloc[-1])
            erl = float(bocpd_erl.loc[:rebal_t].iloc[-1])

            # Adaptive lookback
            adaptive_lb = int(np.clip(erl * 1.5, 42, lookback))
            opt_arr     = hist.values[-adaptive_lb:]

            # ERL-adaptive EWMA halflife: short regime → responsive, long → stable
            _spy_idx = tickers.index("SPY") if "SPY" in tickers else 0
            _adaptive_hl = float(np.clip(erl / 3.0, 14.0, 84.0))
            lam_continuous = compute_continuous_lam(
                window_arr, cp=0.0,       # cp=0 disables the cp discount path
                erl=None,                 # erl=None disables ERL discount path
                spy_idx=_spy_idx, rf_daily=0.04 / _ANN,
                ewma_halflife=_adaptive_hl,
            )

            weights = amr_weights(opt_arr, lam=lam_continuous,
                                  max_weight=max_weight, l2_reg=l2_reg)

        elif strategy == "hmm3_amr":
            # ── 3-state HMM + AMR ────────────────────────────────────────
            from src.regime_models import HMM3

            if hmm3_model is None or hmm3_refit_ctr >= HMM3_REFIT_EVERY:
                hmm3_model    = HMM3().fit(window_arr)
                hmm3_refit_ctr = 0
            else:
                hmm3_refit_ctr += 1

            p_bull, p_side, p_bear = hmm3_model.regime_probs(window_arr)
            bull_m, side_m, bear_m = hmm3_model.regime_masks(window_arr)

            min_samples = max(n_assets + 5, 40)

            # Per-regime AMR with regime-appropriate λ
            # Bull → λ=0.65 (lean into upside); Bear → λ=0.25 (protect downside)
            def _amr_regime(mask, lam_r):
                arr = window_arr[mask]
                if len(arr) >= min_samples:
                    return amr_weights(arr, lam=lam_r,
                                       max_weight=max_weight, l2_reg=l2_reg)
                return amr_weights(window_arr, lam=lam_r,
                                   max_weight=max_weight, l2_reg=l2_reg)

            w_bull = _amr_regime(bull_m, 0.65)
            w_side = _amr_regime(side_m, 0.50)
            w_bear = _amr_regime(bear_m, 0.25)

            # Blend by regime probabilities
            w_blend = p_bull * w_bull + p_side * w_side + p_bear * w_bear
            w_blend = np.clip(w_blend, 0.0, max_weight)
            weights = w_blend / w_blend.sum()

            # Uncertainty penalisation: hedge toward equal when ambiguous
            # Ambiguity = 1 − max(p_bull, p_side, p_bear)  (0=certain, ~0.67=uniform)
            ambiguity   = 1.0 - max(p_bull, p_side, p_bear)
            safety      = np.clip(ambiguity * 0.4, 0.0, 0.25)
            weights     = (1 - safety) * weights + safety * np.ones(n_assets) / n_assets
            weights     = np.clip(weights, 0.0, max_weight)
            weights    /= weights.sum()

        else:
            weights = np.ones(n_assets) / n_assets

        # ── Volatility targeting ──────────────────────────────────────────
        if use_vol_target and strategy != "amr_no_vt":
            if strategy in ("bocpd_amr_v2", "bocpd_amr_v3"):
                # Use regime-adaptive cap: tighten during transitions, loosen when stable
                _cp_sig  = float(bocpd_cp.loc[:rebal_t].iloc[-1])
                _erl_sig = float(bocpd_erl.loc[:rebal_t].iloc[-1])
                scale = vol_target_scale_adaptive(
                    weights, recent_arr, target_vol, leverage_cap,
                    cp_signal=_cp_sig, erl_signal=_erl_sig,
                )
            elif strategy == "bocpd_amr_v4":
                _erl_sig = float(bocpd_erl.loc[:rebal_t].iloc[-1])
                _cp_sig  = float(bocpd_cp.loc[:rebal_t].iloc[-1])
                scale = vol_target_scale_adaptive(
                    weights, recent_arr, target_vol, leverage_cap,
                    cp_signal=_cp_sig, erl_signal=_erl_sig,
                )
            else:
                scale = vol_target_scale(weights, recent_arr, target_vol, leverage_cap)
        else:
            scale = 1.0

        eff_weights = weights * scale   # effective weights (may sum > 1 = leverage)

        # ── Transaction cost (on effective weight turnover) ───────────────
        tc_cost = 0.0
        if prev_eff_w is not None:
            turnover = float(np.sum(np.abs(eff_weights - prev_eff_w)))
            tc_cost  = tc * turnover

        # ── Holding period returns ────────────────────────────────────────
        next_t = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        period = returns.loc[rebal_t:next_t].iloc[1:]
        if len(period) == 0:
            continue

        pf_rets = period.values @ eff_weights  # leveraged daily returns
        if len(pf_rets) > 0:
            pf_rets = pf_rets.copy()
            pf_rets[0] -= tc_cost              # deduct TC on first day of period

        port_rets.extend(pf_rets.tolist())
        ret_dates.extend(period.index.tolist())
        weight_rows.append(weights)
        weight_dates.append(rebal_t)
        lev_vals.append(scale)

        prev_eff_w = eff_weights.copy()

    ret_s      = pd.Series(port_rets, index=ret_dates, name=strategy)
    weights_df = pd.DataFrame(weight_rows, index=weight_dates, columns=tickers)
    lev_s      = pd.Series(lev_vals, index=weight_dates, name="leverage")

    return AMRResult(strategy=strategy, returns=ret_s,
                     weights=weights_df, leverage=lev_s)


def run_all_amr(returns: pd.DataFrame, **kwargs) -> Dict[str, AMRResult]:
    """Run all AMR-family strategies. Extra kwargs forwarded to run_amr_backtest."""
    results = {}
    for strat in AMR_STRATEGIES:
        try:
            vt = strat != "amr_no_vt"
            results[strat] = run_amr_backtest(
                returns, strategy=strat, use_vol_target=vt, **kwargs)
        except Exception as exc:
            logger.error(f"AMR strategy '{strat}' failed: {exc}")
    return results


# ---------------------------------------------------------------------------
# Sensitivity analysis helpers
# ---------------------------------------------------------------------------

def sensitivity_sweep(
    returns: pd.DataFrame,
    param: str = "lam",
    values: list = None,
    **fixed_kwargs,
) -> Dict[str, AMRResult]:
    """
    Run AMR with varying values of one parameter.

    Parameters
    ----------
    param  : 'lam' | 'target_vol' | 'leverage_cap'
    values : list of values to sweep
    """
    if values is None:
        defaults = {"lam": [0.3, 0.4, 0.5, 0.6, 0.7],
                    "target_vol": [0.08, 0.09, 0.10, 0.11, 0.12],
                    "leverage_cap": [1.25, 1.40, 1.50, 1.60, 1.75]}
        values = defaults.get(param, [])

    results = {}
    for v in values:
        key = f"amr_{param}={v}"
        kwargs = {**fixed_kwargs, param: v}
        try:
            results[key] = run_amr_backtest(returns, strategy="amr", **kwargs)
        except Exception as exc:
            logger.error(f"Sweep {key} failed: {exc}")
    return results
