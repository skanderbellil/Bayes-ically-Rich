"""
Macro risk overlay — the liquidity × VIX-TS × credit vote (stage 2: research).

A fit-free, three-signal exposure dial validated in the experiments thread
(run_liquidity_predictive / run_liquidity_continuous):

  liq_raw    : 6-month change in Fed net liquidity (publication-lagged)
  vix_raw    : VIX3M/VIX − 1   (implied-vol term structure, contango degree)
  credit_raw : HYG − IEF trailing relative return  (credit risk appetite)

Each raw signal is mapped to a strength in [0, 1] by a *soft sign*:

  strength = clip(0.5 + raw / (4·MAD_w), 0, 1)

which preserves the natural zero threshold (raw 0 → 0.5), saturates at ±2
trailing MADs, and is never optimised on returns.  The overlay exposure is the
mean of the three strengths.  All inputs must be lagged by the caller (or use
``liquidity_vote`` which applies the standard causal shifts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

H_LIQ = 126          # 6-month net-liquidity change horizon (peak lead)
PUB_LAG = 5          # FRED publication delay, business days
CREDIT_WIN = 21      # credit relative-strength window
MAD_WIN = 756        # trailing window for the MAD scale (≈3y)


def soft_sign(raw: pd.Series, mad_win: int = MAD_WIN) -> pd.Series:
    """clip(0.5 + raw/(4·MAD), 0, 1): continuous strength, no return fitting."""
    mad = raw.rolling(mad_win, min_periods=252).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    return (0.5 + raw / (4.0 * mad.replace(0, np.nan))).clip(0.0, 1.0)


def liquidity_vote(
    net_liquidity: pd.Series,
    vix: pd.Series,
    vix3m: pd.Series,
    hyg: pd.Series,
    ief: pd.Series,
    binary: bool = False,
) -> pd.DataFrame:
    """
    Build the three causal signal strengths and the vote exposure.

    Inputs are *levels* (net liquidity in any unit, VIX/VIX3M closes, HYG/IEF
    prices) on a shared daily index.  Returns a DataFrame with columns
    ``liq, vix_ts, credit, exposure`` — all shifted so that the value at date t
    only uses information available before t (publication lag on liquidity,
    one-day trade lag on everything).
    """
    idx = net_liquidity.index
    liq_raw = net_liquidity.diff(H_LIQ).shift(PUB_LAG)
    vix_raw = (vix3m / vix - 1.0).reindex(idx)
    credit_raw = (hyg.pct_change(CREDIT_WIN) - ief.pct_change(CREDIT_WIN)).reindex(idx)

    if binary:
        s = pd.DataFrame({
            "liq": (liq_raw > 0).astype(float),
            "vix_ts": (vix_raw > 0).astype(float),
            "credit": (credit_raw > 0).astype(float),
        })
    else:
        s = pd.DataFrame({
            "liq": soft_sign(liq_raw),
            "vix_ts": soft_sign(vix_raw),
            "credit": soft_sign(credit_raw),
        })
    s = s.shift(1)                      # trade on yesterday's reading
    s["exposure"] = s[["liq", "vix_ts", "credit"]].mean(axis=1)
    return s
