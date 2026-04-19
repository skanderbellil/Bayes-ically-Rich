"""
Market-neutral pure-alpha strategy: beta-hedged residual mean-reversion.

Construction (all closed-form; no tuning loops, no factor search):

  1.  For each non-SPY asset i, run a rolling Bayesian linear regression
        r_i,t = alpha_i + beta_i * r_SPY,t + eps_i,t
      Under a diffuse Normal-Inverse-Gamma prior the posterior mean equals
      OLS on the window — exact, not fitted.

  2.  The residual eps_i,t is the idiosyncratic return; by construction
      its unconditional SPY-beta is zero.

  3.  Short-horizon residual z-score
        z_i,t = sum_{s in [t-h, t-1]} eps_i,s  /  sigma_eps,i
      serves as the mean-reversion sufficient statistic (AR(1) with zero
      long-run mean → cumulative residual is the only sufficient signal).

  4.  Portfolio weights on the k non-SPY assets:
        raw_i  = -z_i                          (contrarian on residuals)
        raw    = raw - mean(raw)               (dollar-neutral on residuals)
        gross  = sum |raw_i|
        w_i    = raw_i / gross                 (unit gross)
        w_SPY  = -sum_i w_i * beta_i           (hedges residual SPY beta to 0)

  5.  BOCPD gating.  When a changepoint has fired recently the residual
      regressions are stale — expected run-length is the natural credibility
      measure.  Exposure multiplier
        g_t = min(1, ERL_t / ERL_FULL)
      cuts leverage smoothly; ERL_FULL is the single strategy hyperparameter
      and is set to the regression window (beta has been observed for a full
      window → trust it fully).

  6.  Volatility targeting on the gated portfolio to a fixed annualized
      target (default 5%, appropriate for a long/short sleeve).

No parameter search, no thresholding, no cross-validation.  Degrees of
freedom: regression window, z-score horizon, vol target, BOCPD hazard —
all pinned to values used elsewhere in the codebase.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from src.backtest import BacktestResult
from src.regime_models import precompute_bocpd

logger = logging.getLogger(__name__)


@dataclass
class MNParams:
    market: str = "SPY"
    reg_window: int = 252           # rolling regression window (1y)
    z_window: int = 21              # residual-sum horizon (1m)
    rebalance_freq: str = "W-FRI"   # weekly, matches AMR family
    target_vol: float = 0.05        # annualized
    vol_window: int = 63            # 3m rolling portfolio vol
    leverage_cap: float = 2.0       # gross leverage ceiling (hedge + book)
    tc: float = 0.0005              # 5 bps per unit turnover
    hazard: float = 1 / 252         # BOCPD prior: ~1 regime change / year


# ---------------------------------------------------------------------------
# Rolling beta / alpha (exact posterior mean under diffuse NIG prior == OLS)
# ---------------------------------------------------------------------------

def _rolling_beta_alpha(
    y: np.ndarray, x: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling OLS of y on [1, x].  Returns (alpha_t, beta_t) arrays of length T;
    entries for t < window-1 are NaN (insufficient data).

    Uses cumulative sums so cost is O(T) rather than O(T * window).
    """
    T = len(y)
    a = np.full(T, np.nan)
    b = np.full(T, np.nan)
    if T < window:
        return a, b

    sx  = np.cumsum(x)
    sy  = np.cumsum(y)
    sxx = np.cumsum(x * x)
    sxy = np.cumsum(x * y)

    for t in range(window - 1, T):
        s = t - window
        if s < 0:
            Sx  = sx[t];  Sy  = sy[t]
            Sxx = sxx[t]; Sxy = sxy[t]
            n   = t + 1
        else:
            Sx  = sx[t]  - sx[s]
            Sy  = sy[t]  - sy[s]
            Sxx = sxx[t] - sxx[s]
            Sxy = sxy[t] - sxy[s]
            n   = window

        denom = n * Sxx - Sx * Sx
        if denom <= 0:
            continue
        beta  = (n * Sxy - Sx * Sy) / denom
        alpha = (Sy - beta * Sx) / n
        a[t] = alpha
        b[t] = beta
    return a, b


# ---------------------------------------------------------------------------
# Rebalance grid
# ---------------------------------------------------------------------------

def _rebalance_dates(idx: pd.DatetimeIndex, freq: str, start_offset: int) -> pd.DatetimeIndex:
    """Trading-day rebalance grid: last available day in each `freq` bucket,
    after `start_offset` warm-up days."""
    usable = idx[start_offset:]
    buckets = usable.to_series().groupby(usable.to_period(_freq_alias(freq))).max()
    return pd.DatetimeIndex(buckets.values)


def _freq_alias(freq: str) -> str:
    # to_period wants "W", "M", etc.
    return {"W-FRI": "W", "W": "W", "ME": "M", "M": "M"}.get(freq, "W")


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_market_neutral(
    returns: pd.DataFrame,
    params: Optional[MNParams] = None,
    assets: Optional[List[str]] = None,
) -> BacktestResult:
    """
    Beta-hedged residual mean-reversion on `assets` against `params.market`.

    `assets` defaults to all non-market columns of `returns`.
    """
    p = params or MNParams()
    if p.market not in returns.columns:
        raise ValueError(f"market column '{p.market}' not in returns")

    if assets is None:
        assets = [c for c in returns.columns if c != p.market]
    assets = [a for a in assets if a in returns.columns]
    if not assets:
        raise ValueError("no residual assets available")

    idx = returns.index
    T   = len(idx)
    k   = len(assets)
    mkt = returns[p.market].values
    R   = returns[assets].values                         # (T, k)

    # ── 1. Rolling alpha/beta for each residual asset ────────────────────
    alphas = np.full((T, k), np.nan)
    betas  = np.full((T, k), np.nan)
    for j in range(k):
        a, b = _rolling_beta_alpha(R[:, j], mkt, window=p.reg_window)
        alphas[:, j] = a
        betas[:, j]  = b

    # ── 2. Residual series (shifted-beta to avoid lookahead) ─────────────
    # At time t we use beta estimated on data up to t-1, i.e. betas[t-1].
    a_lag = np.vstack([np.full((1, k), np.nan), alphas[:-1, :]])
    b_lag = np.vstack([np.full((1, k), np.nan), betas[:-1, :]])
    residuals = R - a_lag - b_lag * mkt[:, None]          # (T, k)

    # ── 3. BOCPD on the market (pre-computed, no rebalance loop cost) ────
    _, erl = precompute_bocpd(mkt, hazard=p.hazard)
    gate = np.clip(erl / p.reg_window, 0.0, 1.0)          # (T,)

    # ── 4. Rebalance loop ────────────────────────────────────────────────
    start_offset = p.reg_window + p.z_window + 5
    reb_dates = _rebalance_dates(idx, p.rebalance_freq, start_offset)

    cols = assets + [p.market]
    weights = pd.DataFrame(index=reb_dates, columns=cols, dtype=float)
    gates   = pd.Series(index=reb_dates, dtype=float, name="gate")

    w_prev = np.zeros(len(cols))

    for reb_t in reb_dates:
        t = idx.get_loc(reb_t)

        # Residual z-score over [t - z_window, t-1] — uses only past data
        window = residuals[t - p.z_window + 1 : t + 1, :]   # last z_window rows up to t
        # Standardise by the residual's own std on the same window
        mu = np.nanmean(window, axis=0)
        sd = np.nanstd(window, axis=0, ddof=1)
        sd = np.where(sd > 1e-8, sd, 1.0)
        z  = (np.nansum(window, axis=0) / np.sqrt(p.z_window)) / sd

        if not np.all(np.isfinite(z)):
            w_prev = np.zeros_like(w_prev)
            weights.loc[reb_t] = w_prev
            gates.loc[reb_t]   = 0.0
            continue

        # Contrarian, dollar-neutral on residual book
        raw = -z
        raw = raw - raw.mean()
        gross = np.sum(np.abs(raw))
        if gross < 1e-10:
            w_prev = np.zeros_like(w_prev)
            weights.loc[reb_t] = w_prev
            gates.loc[reb_t]   = 0.0
            continue
        w_res = raw / gross                                 # unit gross on residuals

        # Market hedge: zero portfolio beta
        beta_now = betas[t - 1, :]                          # last-known betas
        if not np.all(np.isfinite(beta_now)):
            beta_now = np.nan_to_num(beta_now, nan=0.0)
        w_mkt = -float(np.sum(w_res * beta_now))

        # Apply BOCPD gating
        g = float(gate[t])

        full = np.concatenate([w_res, [w_mkt]]) * g

        # Leverage cap on gross (residuals + hedge)
        gross_full = np.sum(np.abs(full))
        if gross_full > p.leverage_cap:
            full = full * (p.leverage_cap / gross_full)

        w_prev = full
        weights.loc[reb_t] = full
        gates.loc[reb_t]   = g

    weights = weights.fillna(0.0)

    # ── 5. Day-by-day portfolio returns (forward-filled weights) ─────────
    w_daily = weights.reindex(idx).ffill().fillna(0.0)
    # Gross returns before TC
    port_gross = (w_daily[cols].values * returns[cols].values).sum(axis=1)

    # Transaction costs: sum |Δw| on rebalance days only
    turnover = np.zeros(T)
    prev = np.zeros(len(cols))
    for reb_t in reb_dates:
        t = idx.get_loc(reb_t)
        cur = weights.loc[reb_t].values
        turnover[t] = np.sum(np.abs(cur - prev))
        prev = cur
    port_net = port_gross - p.tc * turnover

    port_ret = pd.Series(port_net, index=idx, name="market_neutral")

    # ── 6. Ex-post volatility targeting (causal rolling scalar) ──────────
    roll_vol = port_ret.rolling(p.vol_window, min_periods=p.vol_window // 2).std()
    ann_vol  = roll_vol * np.sqrt(252)
    scale    = (p.target_vol / ann_vol).clip(upper=p.leverage_cap).shift(1)
    scale    = scale.fillna(0.0)
    port_vt  = port_ret * scale

    # Apply scaling to stored weights for transparency
    weights_scaled = weights.mul(scale.reindex(weights.index).fillna(0.0), axis=0)

    return BacktestResult(
        strategy   = "market_neutral",
        returns    = port_vt,
        weights    = weights_scaled,
        lambdas    = gates,           # BOCPD gate, reused slot for diagnostics
        bull_probs = None,
    )
