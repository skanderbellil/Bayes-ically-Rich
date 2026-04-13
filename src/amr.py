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

AMR_STRATEGIES = ("amr", "inv_vol", "amr_no_vt", "bocpd_amr", "hmm3_amr")


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
