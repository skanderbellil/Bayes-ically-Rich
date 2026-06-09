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

import warnings

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

    Every trailing window has the same length, hence the same R/S scales, so
    the per-scale segment statistics are computed for all windows at once. Any
    window that is non-finite or contains a (near-)constant segment falls back
    to the scalar :func:`rs_hurst`, keeping results identical to the naive loop.
    """
    r = np.asarray(r, dtype=float)
    n_total = len(r)
    H = np.full(n_total, np.nan)
    if n_total < win:
        return H

    # Short windows (and degenerate scale sets) are exactly the cases where
    # rs_hurst returns the 0.5 fallback; defer to it rather than special-case.
    scales = np.unique(np.logspace(np.log10(10), np.log10(win // 2), 6).astype(int))
    scales = [int(n) for n in scales if 4 <= n <= win // 2]
    if win < 20 or len(scales) < 3:
        for t in range(win, n_total + 1):
            H[t - 1] = rs_hurst(r[t - win:t])
        return H

    from numpy.lib.stride_tricks import sliding_window_view

    windows = sliding_window_view(r, win)            # (n_total-win+1, win)
    n_win   = windows.shape[0]
    log_n   = np.log(np.asarray(scales, dtype=float))

    # log(mean R/S) at each scale for every window; NaN marks a scale whose
    # segments were all (near-)constant — those windows fall back to scalar.
    logrs = np.full((n_win, len(scales)), np.nan)
    for si, n in enumerate(scales):
        K = win // n
        segs = windows[:, : K * n].reshape(n_win, K, n)       # (n_win, K, n)
        demeaned = segs - segs.mean(axis=2, keepdims=True)
        z = np.cumsum(demeaned, axis=2)
        R = z.max(axis=2) - z.min(axis=2)                     # (n_win, K)
        S = segs.std(axis=2, ddof=1)                          # (n_win, K)
        valid = S > _EPS
        rs = np.where(valid, R / np.where(valid, S, 1.0), np.nan)
        with warnings.catch_warnings(), np.errstate(invalid="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows → NaN
            mean_rs = np.nanmean(np.where(valid, rs, np.nan), axis=1)
        logrs[:, si] = np.log(mean_rs)

    finite_win = np.all(np.isfinite(windows), axis=1)
    clean      = finite_win & np.all(np.isfinite(logrs), axis=1)

    # Vectorised least-squares slope of log(R/S) vs log(scale) over clean rows.
    x = log_n - log_n.mean()
    denom = np.dot(x, x)
    y = logrs - logrs.mean(axis=1, keepdims=True)
    slopes = (y @ x) / denom                                  # (n_win,)

    out = np.where(clean, slopes, np.nan)
    H[win - 1:] = out

    # Fall back to the scalar estimator for any non-clean window (non-finite
    # input, or a scale with all-constant segments) to match rs_hurst exactly.
    for wi in np.nonzero(~clean)[0]:
        H[wi + win - 1] = rs_hurst(r[wi : wi + win])
    return H
