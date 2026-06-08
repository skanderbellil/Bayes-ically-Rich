"""Stage 2 — strategy research: models, optimisers, and regime signals."""
from posterioralpha.research.amr import (
    amr_cvar_weights,
    amr_weights,
    compute_continuous_lam,
    hrp_weights,
    inverse_vol_weights,
    vol_target_scale,
    vol_target_scale_adaptive,
)
from posterioralpha.research.bayesian import (
    bayesian_posterior,
    compute_lambda,
    estimate_moments,
    estimate_moments_ewma,
    mahalanobis_distance,
    max_sharpe_weights,
    min_variance_weights,
)
from posterioralpha.research.regimes import (
    BOCPD,
    HMM3,
    RegimeHMM,
    precompute_bocpd,
    precompute_bocpd_multi,
)

__all__ = [
    # Bayesian machinery
    "estimate_moments",
    "estimate_moments_ewma",
    "mahalanobis_distance",
    "compute_lambda",
    "bayesian_posterior",
    "max_sharpe_weights",
    "min_variance_weights",
    # AMR primitives
    "amr_weights",
    "amr_cvar_weights",
    "hrp_weights",
    "inverse_vol_weights",
    "compute_continuous_lam",
    "vol_target_scale",
    "vol_target_scale_adaptive",
    # Regime models
    "RegimeHMM",
    "BOCPD",
    "HMM3",
    "precompute_bocpd",
    "precompute_bocpd_multi",
]
