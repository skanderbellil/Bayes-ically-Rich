"""
Hurst-exponent trend primitives (stage 2: research).

The Hurst exponent H estimated by rescaled-range (R/S) analysis quantifies the
persistence of a return series:

    H > 0.5  → trending / persistent   (momentum regime)
    H = 0.5  → random walk
    H < 0.5  → mean-reverting / anti-persistent

These are pure, causal functions: ``rolling_hurst`` stores H at the *last* day
of each trailing window, so a value at index ``t`` uses only data up to and
including ``t`` — no lookahead. They back the Hurst-bull timing study and the
multi-asset cross-sectional ranking study in ``experiments/``.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-8


def rs_hurst(x: np.ndarray) -> float:
    """
    Estimate the Hurst exponent of ``x`` via rescaled-range (R/S) analysis.

    Splits the series into segments at a handful of log-spaced scales, computes
    the mean rescaled range R/S at each scale, and returns the slope of
    ``log(R/S)`` vs ``log(scale)`` — the Hurst exponent. Returns 0.5 (random
    walk) when the series is too short or degenerate to estimate.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20 or not np.all(np.isfinite(x)):
        return 0.5
    scales = np.unique(np.logspace(np.log10(10), np.log10(N // 2), 6).astype(int))
    rs_list, good = [], []
    for n in scales:
        if n < 4 or n > N // 2:
            continue
        K = N // n
        vals = []
        for k in range(K):
            seg = x[k * n:(k + 1) * n]
            z = np.cumsum(seg - seg.mean())
            R = float(z.max() - z.min())
            S = float(seg.std(ddof=1))
            if S > _EPS:
                vals.append(R / S)
        if vals:
            rs_list.append(float(np.mean(vals)))
            good.append(int(n))
    if len(good) < 3:
        return 0.5
    slope, _ = np.polyfit(np.log(good), np.log(rs_list), 1)
    return float(slope)


def rolling_hurst(r: np.ndarray, win: int) -> np.ndarray:
    """
    Rolling Hurst exponent over a trailing window of ``win`` observations.

    Returns an array the same length as ``r``; entry ``t`` is the Hurst
    exponent of ``r[t-win:t]`` (stored at the window's last index, NaN until
    the first full window is available).
    """
    H = np.full(len(r), np.nan)
    for t in range(win, len(r) + 1):
        H[t - 1] = rs_hurst(r[t - win:t])
    return H
