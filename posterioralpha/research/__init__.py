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
from posterioralpha.research.hurst import rolling_hurst, rs_hurst
from posterioralpha.research.intramonth import (
    intramonth_window_mask,
    wml_formation,
)
from posterioralpha.research.tsfm import (
    TimesFMVolForecaster,
    baseline_paths,
    log_rv_series,
    qlike,
    realized_target,
    realized_vol,
)
from posterioralpha.research.overlay import liquidity_vote, soft_sign
from posterioralpha.research.breadth import (
    average_pairwise_correlation,
    cross_sectional_ic,
    effective_breadth,
    empirical_transfer_coefficient,
    fundamental_law_sharpe,
    participation_ratio,
    transfer_coefficient,
)
from posterioralpha.research.gamma import (
    GammaSnapshot,
    bs_gamma,
    chain_gex,
    fetch_cboe_chain,
    fetch_gamma_snapshot,
    gex_time_series,
    zero_gamma_level,
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
    # Trend / Hurst primitives
    "rs_hurst",
    "rolling_hurst",
    # Intramonth momentum primitives
    "intramonth_window_mask",
    "wml_formation",
    # Macro risk overlay (liquidity × VIX-TS × credit vote)
    "liquidity_vote",
    "soft_sign",
    # Fundamental-Law diagnostics (effective breadth, transfer coefficient)
    "average_pairwise_correlation",
    "effective_breadth",
    "participation_ratio",
    "transfer_coefficient",
    "empirical_transfer_coefficient",
    "fundamental_law_sharpe",
    "cross_sectional_ic",
    # Dealer gamma exposure (GEX)
    "bs_gamma",
    "chain_gex",
    "zero_gamma_level",
    "fetch_cboe_chain",
    "fetch_gamma_snapshot",
    "gex_time_series",
    "GammaSnapshot",
    # Realized-vol forecasting primitives + the TimesFM adapter
    # (the foundation model itself is an optional extra: pip install -e .[tsfm])
    "realized_vol",
    "log_rv_series",
    "realized_target",
    "baseline_paths",
    "qlike",
    "TimesFMVolForecaster",
]
