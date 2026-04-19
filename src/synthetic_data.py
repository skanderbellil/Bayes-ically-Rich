"""
Synthetic asset generation.

Two generators are provided:

  1. expand_universe()            — legacy linear-factor + Gaussian idio model.
                                    Fast, reproducible, but by construction
                                    contains no mean-reversion alpha.

  2. expand_universe_realistic()  — factor model with persistent (AR(1))
                                    idiosyncratic returns, log-AR(1) stochastic
                                    volatility, and Student-t innovations.
                                    These are the three features that
                                    distinguish real asset returns from iid
                                    Gaussian: autocorrelation of residuals,
                                    volatility clustering, and fat tails.
                                    This is the generator to use for
                                    validating residual stat-arb.

Generalisation hook
-------------------
`fit_params_from_real()` estimates all per-asset parameters directly from a
real return series + factor panel (OLS beta, residual AR(1) coefficient,
residual MoM Student-t df, log-vol AR(1) via variance-of-variance).  The
returned params can then be fed back into `simulate_asset()` to produce a
drop-in synthetic replacement with matching statistical signature.  When
more real data becomes available this is the single call that updates the
synthetic universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Legacy generator (kept for backwards compatibility with existing scripts)
# ---------------------------------------------------------------------------

# (name, description, [β_SPY, β_TLT, β_GLD, β_EEM, β_VNQ], daily_idio_vol)
SYNTHETIC_ASSETS = [
    ("QQQ_s",  "US Tech",             [ 1.20, -0.05,  0.00,  0.10, -0.10], 0.0050),
    ("IWM_s",  "US Small Cap",        [ 1.10, -0.02,  0.00,  0.05,  0.10], 0.0055),
    ("AGG_s",  "Agg Bonds",           [-0.05,  0.55,  0.05, -0.05,  0.00], 0.0020),
    ("TIP_s",  "TIPS",                [ 0.00,  0.45,  0.15,  0.00,  0.00], 0.0025),
    ("HYG_s",  "High Yield Bonds",    [ 0.35,  0.30,  0.00,  0.10,  0.10], 0.0040),
    ("DBC_s",  "Broad Commodities",   [ 0.10, -0.10,  0.65,  0.10,  0.00], 0.0060),
    ("VEA_s",  "Intl Developed",      [ 0.85,  0.05,  0.05,  0.15,  0.00], 0.0045),
    ("FXI_s",  "China Equities",      [ 0.30,  0.00,  0.00,  0.75,  0.00], 0.0080),
    ("IYR_s",  "US REITs (alt)",      [ 0.40,  0.05,  0.05,  0.05,  0.80], 0.0040),
]

BASE_FACTORS = ["SPY", "TLT", "GLD", "EEM", "VNQ"]


def expand_universe(returns: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Legacy linear-factor generator with Gaussian idio noise."""
    rng       = np.random.default_rng(seed)
    available = [c for c in BASE_FACTORS if c in returns.columns]
    factor    = returns[available].values

    synth: dict = {}
    for name, _desc, betas, sig_idio in SYNTHETIC_ASSETS:
        b    = np.array(betas[: len(available)])
        idio = rng.normal(0.0, sig_idio, size=len(returns))
        synth[name] = factor @ b + idio
    return pd.concat([returns, pd.DataFrame(synth, index=returns.index)], axis=1)


# ---------------------------------------------------------------------------
# Realistic generator
# ---------------------------------------------------------------------------


@dataclass
class AssetParams:
    """Per-asset statistical signature.  All fields are estimable from data."""
    name:       str
    betas:      np.ndarray          # (K,) factor loadings
    alpha_ann:  float = 0.0         # annualised expected excess (daily α = ann/252)
    idio_ar:    float = 0.0         # residual-return AR(1) coefficient; 0 ⇒ iid
    idio_sd:    float = 0.0         # long-run residual-return std (daily)
    sv_ar:      float = 0.95        # persistence of log-vol AR(1)
    sv_innov_sd: float = 0.15       # std of log-vol innovation
    t_df:       float = 8.0         # Student-t degrees of freedom; ∞ ⇒ Gaussian
    # Optional Ornstein-Uhlenbeck mean-reversion on the residual LEVEL
    # (cumulative residual return).  This is the Avellaneda-Lee
    # specification for statistical arbitrage: residual prices revert to a
    # long-run level with the given half-life in trading days.  When set,
    # `idio_ar` is ignored; `idio_sd` is interpreted as the innovation
    # std of the OU level process.
    ou_half_life: Optional[float] = None


@dataclass
class RealisticUniverseSpec:
    """Preset bundle of `AssetParams` sharing the same factor panel."""
    factor_names: Sequence[str]
    assets:       List[AssetParams] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Fit parameters from a real asset + factor panel
# ─────────────────────────────────────────────────────────────────────────────

def fit_params_from_real(
    asset: pd.Series,
    factors: pd.DataFrame,
    name: Optional[str] = None,
) -> AssetParams:
    """
    Estimate `AssetParams` from a real return series.

    Procedure (all closed-form MoM / OLS):
      • β, α via OLS regression on the factor panel.
      • ε_t = asset − α − β·factors; AR(1) ρ via sample autocorrelation.
      • idio_sd = std(ε).
      • Log-vol AR(1) via AR(1) fit on log |ε| residuals.
      • Student-t df via method-of-moments on excess kurtosis of ε:
          kurt_excess ≈ 6 / (df − 4)   ⇒   df ≈ 4 + 6/κ     (κ > 0)
        clipped to [4.5, 50] for numerical stability.
    """
    aligned = pd.concat([asset, factors], axis=1).dropna()
    y = aligned.iloc[:, 0].values
    X = aligned.iloc[:, 1:].values
    K = X.shape[1]

    # OLS (diffuse NIG posterior mean)
    Xd = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    alpha_d  = float(coef[0])
    betas    = coef[1:]

    eps = y - Xd @ coef
    sd  = float(np.std(eps, ddof=1)) + 1e-12

    # Residual AR(1)
    if len(eps) > 2:
        rho = float(np.corrcoef(eps[:-1], eps[1:])[0, 1])
    else:
        rho = 0.0
    rho = float(np.clip(rho, -0.4, 0.6))     # keep stationary & realistic

    # Log-vol AR(1) from |eps|
    abs_eps = np.log(np.abs(eps) + 1e-6)
    if len(abs_eps) > 2:
        sv_ar = float(np.corrcoef(abs_eps[:-1], abs_eps[1:])[0, 1])
    else:
        sv_ar = 0.95
    sv_ar  = float(np.clip(sv_ar, 0.5, 0.995))
    sv_sd  = float(np.std(abs_eps[1:] - sv_ar * abs_eps[:-1], ddof=1))
    sv_sd  = float(np.clip(sv_sd, 0.05, 0.8))

    # Student-t df from excess kurtosis
    z   = (eps - eps.mean()) / sd
    kurt_excess = float(((z**4).mean() - 3.0))
    t_df = 4.0 + 6.0 / kurt_excess if kurt_excess > 0.3 else 50.0
    t_df = float(np.clip(t_df, 4.5, 50.0))

    return AssetParams(
        name        = name or (asset.name if hasattr(asset, "name") else "asset"),
        betas       = betas.astype(float),
        alpha_ann   = alpha_d * 252.0,
        idio_ar     = rho,
        idio_sd     = sd,
        sv_ar       = sv_ar,
        sv_innov_sd = sv_sd,
        t_df        = t_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Simulate a single asset given parameters and a factor path
# ─────────────────────────────────────────────────────────────────────────────

def simulate_asset(
    params: AssetParams,
    factor_returns: np.ndarray,             # (T, K)
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a synthetic return series whose statistical signature matches
    `params`, driven by the supplied factor path.

    Model
    -----
      s_t   = exp(h_t - h_bar)              (relative vol innovation)
      h_t   = sv_ar · h_{t-1} + sv_innov_sd · η_t   (log-vol AR(1))
      u_t   = Student-t(df)  ·  idio_sd  ·  s_t
      ε_t   = idio_ar · ε_{t-1}  +  u_t             (AR(1) residual)
      r_t   = α/252  +  β·factor_t  +  ε_t

    Unconditional Var(ε) is normalised to `idio_sd²` by rescaling `u_t` after
    solving the AR(1) stationary variance so the reported idio_sd is honoured.
    """
    T, K = factor_returns.shape
    if len(params.betas) != K:
        raise ValueError(
            f"beta length {len(params.betas)} != number of factors {K}"
        )

    # Log-vol AR(1) around zero mean (h_bar implicitly 0)
    h = np.zeros(T)
    for t in range(1, T):
        h[t] = params.sv_ar * h[t - 1] + params.sv_innov_sd * rng.standard_normal()
    # Centre exp(h) to mean 1 so unconditional scale is preserved
    sv_mult = np.exp(h - h.mean() - 0.5 * h.var())

    # Student-t innovations scaled to unit variance
    df = params.t_df
    t_scale = np.sqrt((df - 2.0) / df) if df > 2 else 1.0
    u_raw = rng.standard_t(df, size=T) * t_scale    # Var ≈ 1

    if params.ou_half_life is not None:
        # OU on the residual LEVEL (Avellaneda-Lee): X_t = φ X_{t-1} + σ u_t
        #   half_life = ln 2 / ln(1/φ)  ⇒  φ = 0.5 ** (1 / half_life)
        # Residual return ε_t = X_t - X_{t-1} = (φ-1) X_{t-1} + σ u_t.
        phi = 0.5 ** (1.0 / max(params.ou_half_life, 1e-6))
        sigma = params.idio_sd * sv_mult         # per-step level innovation
        u = u_raw * sigma
        X = np.zeros(T)
        for t in range(1, T):
            X[t] = phi * X[t - 1] + u[t]
        eps = np.diff(X, prepend=0.0)
    else:
        # AR(1) on residual RETURNS.  Stationary variance ⇒
        # innovation scale = idio_sd · √(1 - ρ²).
        rho = params.idio_ar
        innov_scale = params.idio_sd * np.sqrt(max(1.0 - rho * rho, 1e-8))
        u = u_raw * sv_mult * innov_scale
        eps = np.zeros(T)
        for t in range(1, T):
            eps[t] = rho * eps[t - 1] + u[t]

    r = (params.alpha_ann / 252.0) + factor_returns @ params.betas + eps
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Default realistic preset (betas mirror the legacy table)
# ─────────────────────────────────────────────────────────────────────────────

# Idio AR coefficients chosen in the plausible range for each asset class.
# Bonds and commodities show the most persistent residuals empirically;
# broad equity indices show mild negative autocorrelation at daily horizon.
REALISTIC_ASSETS = [
    # name,    betas (SPY,TLT,GLD,EEM,VNQ),      α_ann, ρ,    σ_idio,  sv_ar, sv_sd, t_df
    ("QQQ_s",  [ 1.20, -0.05,  0.00,  0.10, -0.10],  0.01, -0.04, 0.0050, 0.92, 0.20, 7.0),
    ("IWM_s",  [ 1.10, -0.02,  0.00,  0.05,  0.10],  0.00, -0.03, 0.0055, 0.90, 0.22, 6.5),
    ("AGG_s",  [-0.05,  0.55,  0.05, -0.05,  0.00],  0.00,  0.08, 0.0020, 0.93, 0.18, 9.0),
    ("TIP_s",  [ 0.00,  0.45,  0.15,  0.00,  0.00],  0.00,  0.10, 0.0025, 0.92, 0.18, 9.0),
    ("HYG_s",  [ 0.35,  0.30,  0.00,  0.10,  0.10],  0.00,  0.06, 0.0040, 0.93, 0.22, 5.5),
    ("DBC_s",  [ 0.10, -0.10,  0.65,  0.10,  0.00],  0.00,  0.05, 0.0060, 0.91, 0.22, 7.0),
    ("VEA_s",  [ 0.85,  0.05,  0.05,  0.15,  0.00],  0.00, -0.02, 0.0045, 0.91, 0.20, 7.0),
    ("FXI_s",  [ 0.30,  0.00,  0.00,  0.75,  0.00],  0.00,  0.04, 0.0080, 0.90, 0.25, 5.0),
    ("IYR_s",  [ 0.40,  0.05,  0.05,  0.05,  0.80],  0.00,  0.05, 0.0040, 0.92, 0.22, 6.5),
]


def default_realistic_spec(factor_names: Sequence[str] = tuple(BASE_FACTORS)) -> RealisticUniverseSpec:
    assets = []
    for name, betas, a_ann, rho, sig, sv_ar, sv_sd, t_df in REALISTIC_ASSETS:
        assets.append(AssetParams(
            name=name,
            betas=np.array(betas[: len(factor_names)], dtype=float),
            alpha_ann=a_ann, idio_ar=rho, idio_sd=sig,
            sv_ar=sv_ar, sv_innov_sd=sv_sd, t_df=t_df,
        ))
    return RealisticUniverseSpec(factor_names=list(factor_names), assets=assets)


def expand_universe_realistic(
    returns: pd.DataFrame,
    spec: Optional[RealisticUniverseSpec] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Append realistic synthetic assets to `returns`.

    Uses the real factor paths present in `returns` (must include
    `spec.factor_names`) and overlays AR(1) residuals with log-vol AR(1)
    stochastic volatility and Student-t innovations.
    """
    spec       = spec or default_realistic_spec()
    available  = [c for c in spec.factor_names if c in returns.columns]
    if len(available) != len(spec.factor_names):
        missing = set(spec.factor_names) - set(available)
        raise ValueError(f"missing factor columns: {missing}")

    factor_ret = returns[available].values
    rng        = np.random.default_rng(seed)

    synth: dict = {}
    for p in spec.assets:
        # Each asset gets its own sub-RNG for reproducible per-asset draws
        sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        synth[p.name] = simulate_asset(p, factor_ret, sub)
    return pd.concat([returns, pd.DataFrame(synth, index=returns.index)], axis=1)


def fit_realistic_spec(
    returns: pd.DataFrame,
    factor_names: Sequence[str],
    asset_names: Sequence[str],
) -> RealisticUniverseSpec:
    """
    Fit a `RealisticUniverseSpec` to real data.

    For each `asset_names` column, estimate its params against `factor_names`.
    This is the function to call when plugging in a new real universe —
    the returned spec can then be fed back into `expand_universe_realistic`
    to produce matched synthetic siblings.
    """
    factors = returns[list(factor_names)]
    params  = [
        fit_params_from_real(returns[a], factors, name=f"{a}_s")
        for a in asset_names
    ]
    return RealisticUniverseSpec(factor_names=list(factor_names), assets=params)
