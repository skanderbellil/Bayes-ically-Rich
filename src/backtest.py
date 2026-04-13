"""
Backtesting engine for PosteriorAlpha.

Strategies
----------
Original
  bayesian        — Bayesian shrinkage (hard window) + Mahalanobis λ
  equal_weight    — 1/N monthly rebalance
  min_variance    — long-only global minimum variance
  hist_max_sharpe — max Sharpe on full prior only (no adaptation)

Refined  (implementing all extensions from the original idea)
  bayesian_ewma   — EWMA recent estimation + transaction-cost penalty
  bayesian_hmm    — HMM regime filter: per-regime portfolios blended by
                    posterior regime probability + uncertainty penalisation
  bayesian_full   — everything: EWMA prior + HMM regime blend +
                    uncertainty penalisation + TC penalty
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.bayesian import (
    bayesian_posterior,
    compute_lambda,
    estimate_moments,
    estimate_moments_ewma,
    mahalanobis_distance,
    max_sharpe_weights,
    min_variance_weights,
)
from src.hmm_filter import RegimeHMM

logger = logging.getLogger(__name__)

# All strategy identifiers (order determines display order)
STRATEGIES = (
    "bayesian",
    "equal_weight",
    "min_variance",
    "hist_max_sharpe",
    "bayesian_ewma",
    "bayesian_hmm",
    "bayesian_full",
)

# How often to refit the HMM (every N rebalances); fitting every month is slow
_HMM_REFIT_EVERY = 6   # ≈ semi-annual


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    strategy: str
    returns:  pd.Series          # daily arithmetic portfolio returns
    weights:  pd.DataFrame       # weights at each rebalance date
    lambdas:  Optional[pd.Series] = None   # λ values (Bayesian strategies)
    bull_probs: Optional[pd.Series] = None # P(bull) at each rebalance (HMM strategies)

    @property
    def cumulative(self) -> pd.Series:
        return (1.0 + self.returns).cumprod()

    @property
    def n_days(self) -> int:
        return len(self.returns)


# ---------------------------------------------------------------------------
# Per-rebalance helper: build regime-specific portfolios (for HMM strategies)
# ---------------------------------------------------------------------------

def _regime_portfolios(
    returns: np.ndarray,
    hmm: RegimeHMM,
    rf: float,
    max_weight: float,
    w_prev: Optional[np.ndarray],
    tc: float,
    use_ewma: bool = False,
    ewma_halflife: int = 30,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute bull and bear regime-specific portfolios using HMM state assignments.

    Returns
    -------
    w_bull   : (N,) optimal weights for bull regime
    w_bear   : (N,) min-variance weights for bear regime (risk-off)
    p_bull   : P(bull | data[-1])
    """
    n = returns.shape[1]
    bull_mask, bear_mask = hmm.regime_mask(returns)

    bull_rets = returns[bull_mask]
    bear_rets = returns[bear_mask]

    min_samples = max(n + 5, 40)

    # ── Bull-regime portfolio ─────────────────────────────────────────────
    if len(bull_rets) >= min_samples:
        if use_ewma:
            mu_b, cov_b = estimate_moments_ewma(bull_rets, halflife=ewma_halflife)
        else:
            mu_b, cov_b = estimate_moments(bull_rets)
        w_bull = max_sharpe_weights(mu_b, cov_b, rf=rf, max_weight=max_weight,
                                    w_prev=w_prev, tc=tc)
    else:
        # Fall back to full-history max-Sharpe
        mu_all, cov_all = (estimate_moments_ewma(returns, ewma_halflife)
                           if use_ewma else estimate_moments(returns))
        w_bull = max_sharpe_weights(mu_all, cov_all, rf=rf, max_weight=max_weight,
                                    w_prev=w_prev, tc=tc)

    # ── Bear-regime portfolio  (min-variance = risk-off) ──────────────────
    if len(bear_rets) >= min_samples:
        _, cov_bear = estimate_moments(bear_rets)
        w_bear = min_variance_weights(cov_bear, max_weight=max_weight)
    else:
        w_bear = np.ones(n) / n   # equal weight as safe harbour

    p_bull = hmm.bull_prob(returns)
    return w_bull, w_bear, p_bull


def _uncertainty_blend(
    w_opt: np.ndarray,
    p_bull: float,
    max_safety: float = 0.30,
) -> np.ndarray:
    """
    Extension: uncertainty penalisation.

    "When divergence is high (you're unsure which regime you're in),
    shrink toward equal weights or minimum variance as a risk-off stance."

    Here we measure uncertainty by how ambiguous the HMM posterior is:
      uncertainty = 1 − 2·|p_bull − 0.5|
      (= 0 when we're certain of either regime, = 1 when completely ambiguous)

    We blend up to `max_safety` of the portfolio toward equal weight.
    """
    n           = len(w_opt)
    uncertainty = 1.0 - 2.0 * abs(p_bull - 0.5)   # ∈ [0, 1]
    safety      = max_safety * uncertainty           # ∈ [0, max_safety]
    w_eq        = np.ones(n) / n
    w_blended   = (1.0 - safety) * w_opt + safety * w_eq
    w_blended   = np.clip(w_blended, 0.0, 1.0)
    return w_blended / w_blended.sum()


# ---------------------------------------------------------------------------
# Core backtest function
# ---------------------------------------------------------------------------

def run_backtest(
    returns: pd.DataFrame,
    strategy: str = "bayesian",
    rebalance_freq: str = "ME",
    min_history: int = 252,
    recent_window: int = 60,
    ewma_halflife: int = 30,
    rf: float = 0.04,
    sensitivity: float = 1.5,
    max_weight: float = 0.20,
    tc: float = 0.001,
) -> BacktestResult:
    """
    Monthly-rebalanced backtest for any supported strategy.

    Parameters
    ----------
    recent_window : look-back for the hard-window Bayesian strategies
    ewma_halflife : half-life (days) for EWMA-based strategies
    tc            : one-way transaction cost rate (e.g. 0.001 = 10 bps)
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {STRATEGIES}.")

    tickers  = returns.columns.tolist()
    n_assets = len(tickers)

    try:
        rebal_grid = returns.resample(rebalance_freq).last().index
    except ValueError:
        rebal_grid = returns.resample("M").last().index

    rebal_dates = rebal_grid[rebal_grid > returns.index[min_history]]

    # State for HMM strategies
    hmm_model: Optional[RegimeHMM] = None
    hmm_refit_counter = 0

    # Transaction-cost state: previous weights
    prev_weights: Optional[np.ndarray] = None

    port_rets:    List[float] = []
    ret_dates:    List        = []
    weight_rows:  List        = []
    weight_dates: List        = []
    lambda_vals:  List[float] = []
    lambda_dates: List        = []
    bull_prob_vals:  List[float] = []
    bull_prob_dates: List        = []

    use_hmm  = strategy in ("bayesian_hmm", "bayesian_full")
    use_ewma = strategy in ("bayesian_ewma", "bayesian_full")
    use_tc   = strategy in ("bayesian_ewma", "bayesian_hmm", "bayesian_full")

    for i, rebal_t in enumerate(tqdm(rebal_dates, desc=f"  {strategy:<22}", leave=False)):
        hist = returns.loc[:rebal_t]
        if len(hist) < min_history + recent_window:
            continue

        prior_arr  = hist.values
        recent_arr = hist.values[-recent_window:]

        lam:    Optional[float] = None
        p_bull: Optional[float] = None
        w_tc = prev_weights if use_tc else None

        # ── HMM: refit every _HMM_REFIT_EVERY rebalances ─────────────────
        if use_hmm:
            if hmm_model is None or hmm_refit_counter >= _HMM_REFIT_EVERY:
                hmm_model = RegimeHMM().fit(prior_arr)
                hmm_refit_counter = 0
            else:
                hmm_refit_counter += 1

        # ── Weight computation ────────────────────────────────────────────
        if strategy == "equal_weight":
            weights = np.ones(n_assets) / n_assets

        elif strategy == "min_variance":
            _, cov = estimate_moments(prior_arr)
            weights = min_variance_weights(cov, max_weight)

        elif strategy == "hist_max_sharpe":
            mu, cov = estimate_moments(prior_arr)
            weights = max_sharpe_weights(mu, cov, rf=rf, max_weight=max_weight)

        elif strategy == "bayesian":
            mu_prior,  cov_prior  = estimate_moments(prior_arr)
            mu_recent, cov_recent = estimate_moments(recent_arr)
            divergence = mahalanobis_distance(mu_recent, mu_prior, cov_prior)
            lam        = compute_lambda(divergence, n_assets, sensitivity)
            mu_post, cov_post = bayesian_posterior(
                mu_prior, cov_prior, mu_recent, cov_recent, lam)
            weights = max_sharpe_weights(mu_post, cov_post, rf=rf, max_weight=max_weight)

        elif strategy == "bayesian_ewma":
            # EWMA replaces hard window for the 'recent' likelihood
            mu_prior, cov_prior = estimate_moments(prior_arr)
            mu_ewma,  cov_ewma  = estimate_moments_ewma(prior_arr, halflife=ewma_halflife)
            divergence = mahalanobis_distance(mu_ewma, mu_prior, cov_prior)
            lam        = compute_lambda(divergence, n_assets, sensitivity)
            mu_post, cov_post = bayesian_posterior(
                mu_prior, cov_prior, mu_ewma, cov_ewma, lam)
            weights = max_sharpe_weights(
                mu_post, cov_post, rf=rf, max_weight=max_weight,
                w_prev=w_tc, tc=tc)

        elif strategy == "bayesian_hmm":
            # Pure HMM: blend regime-specific portfolios by posterior regime prob
            w_bull, w_bear, p_bull = _regime_portfolios(
                prior_arr, hmm_model, rf=rf, max_weight=max_weight,
                w_prev=w_tc, tc=tc, use_ewma=False)
            w_blend = p_bull * w_bull + (1.0 - p_bull) * w_bear
            w_blend = np.clip(w_blend, 0.0, max_weight)
            weights = w_blend / w_blend.sum()
            # Uncertainty penalisation
            weights = _uncertainty_blend(weights, p_bull, max_safety=0.25)

        elif strategy == "bayesian_full":
            # Full combination: EWMA Bayesian shrinkage + HMM + uncertainty + TC
            # Step 1: EWMA Bayesian posterior for the current-regime estimate
            mu_prior, cov_prior = estimate_moments(prior_arr)
            mu_ewma,  cov_ewma  = estimate_moments_ewma(prior_arr, halflife=ewma_halflife)
            divergence = mahalanobis_distance(mu_ewma, mu_prior, cov_prior)
            lam        = compute_lambda(divergence, n_assets, sensitivity)
            mu_post, cov_post = bayesian_posterior(
                mu_prior, cov_prior, mu_ewma, cov_ewma, lam)

            # Step 2: HMM regime-specific portfolios (bull uses EWMA estimates)
            w_bull, w_bear, p_bull = _regime_portfolios(
                prior_arr, hmm_model, rf=rf, max_weight=max_weight,
                w_prev=w_tc, tc=tc, use_ewma=True, ewma_halflife=ewma_halflife)

            # Step 3: blend by HMM probability
            w_blend = p_bull * w_bull + (1.0 - p_bull) * w_bear
            w_blend = np.clip(w_blend, 0.0, max_weight)
            weights = w_blend / w_blend.sum()

            # Step 4: uncertainty penalisation
            weights = _uncertainty_blend(weights, p_bull, max_safety=0.25)

        # ── Apply weights for the holding period ──────────────────────────
        next_t = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else returns.index[-1]
        period = returns.loc[rebal_t:next_t].iloc[1:]
        if len(period) == 0:
            continue

        pf_rets = period.values @ weights

        port_rets.extend(pf_rets.tolist())
        ret_dates.extend(period.index.tolist())
        weight_rows.append(weights)
        weight_dates.append(rebal_t)

        if lam is not None:
            lambda_vals.append(lam)
            lambda_dates.append(rebal_t)
        if p_bull is not None:
            bull_prob_vals.append(p_bull)
            bull_prob_dates.append(rebal_t)

        prev_weights = weights.copy()

    returns_s  = pd.Series(port_rets, index=ret_dates, name=strategy)
    weights_df = pd.DataFrame(weight_rows, index=weight_dates, columns=tickers)
    lambdas_s  = (pd.Series(lambda_vals, index=lambda_dates, name="lambda")
                  if lambda_vals else None)
    bull_s     = (pd.Series(bull_prob_vals, index=bull_prob_dates, name="bull_prob")
                  if bull_prob_vals else None)

    return BacktestResult(
        strategy=strategy,
        returns=returns_s,
        weights=weights_df,
        lambdas=lambdas_s,
        bull_probs=bull_s,
    )


def run_all_strategies(
    returns: pd.DataFrame,
    **kwargs,
) -> Dict[str, BacktestResult]:
    """Run every strategy. Extra kwargs forwarded to run_backtest."""
    results = {}
    for strat in STRATEGIES:
        try:
            results[strat] = run_backtest(returns, strategy=strat, **kwargs)
        except Exception as exc:
            logger.error(f"Strategy '{strat}' failed: {exc}")
    return results
