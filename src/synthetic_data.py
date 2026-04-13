"""
Synthetic asset universe expansion via factor model.

Generates additional asset return series:
  r_synth = Σ β_j · r_factor_j + σ_idio · ε

Factors are the 5 real ETFs: SPY, TLT, GLD, EEM, VNQ.
Seed is fixed so results are reproducible across runs.

These synthetic assets are clearly labelled (_s suffix) so results can be
interpreted knowing they are derived, not independently observed.
"""
import numpy as np
import pandas as pd

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
    """
    Append synthetic assets to `returns` using a factor model.

    Parameters
    ----------
    returns : (T, N) daily arithmetic returns — must contain BASE_FACTORS columns
    seed    : RNG seed for reproducibility

    Returns
    -------
    DataFrame with real + synthetic columns, same index as `returns`.
    """
    rng            = np.random.default_rng(seed)
    available      = [c for c in BASE_FACTORS if c in returns.columns]
    factor_ret     = returns[available].values        # (T, K)

    synth: dict = {}
    for name, _desc, betas, sig_idio in SYNTHETIC_ASSETS:
        beta_vec     = np.array(betas[: len(available)])
        idio         = rng.normal(0.0, sig_idio, size=len(returns))
        synth[name]  = factor_ret @ beta_vec + idio

    return pd.concat([returns, pd.DataFrame(synth, index=returns.index)], axis=1)
