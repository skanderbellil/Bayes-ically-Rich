"""
Cost-aware backtest + integral baselines for the kinematic signals.

Conventions (shared with the rest of the repo, kept deliberately strict):
  • A signal dated t is known at the close of t and earns the asset return over
    t → t+1. We therefore align position_t with return_{t+1} (one bar lag); no
    same-bar fills.
  • Position = the signal clipped to ±`max_pos` (a unit-scaled tilt, so a
    z-score of +2 caps at the limit). Turnover = Σ|Δposition|.
  • Costs are charged on turnover in basis points of notional. We always report
    BOTH gross and net, plus annualised turnover.

Nothing here consumes a smoother. Signals arrive pre-built from `signals.py`,
which only exposes causal (filtered / one-step-innovation) quantities.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

ANN = 252


# ─────────────────────────────────────────────────────────────────────────────
# Core position → PnL engine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    gross: pd.Series                  # daily gross return of the strategy
    net: pd.Series                    # daily net (after costs) return
    position: pd.Series               # position held INTO each day
    turnover: pd.Series               # |Δposition| per day
    stats: Dict[str, float] = field(default_factory=dict)


def run_signal_backtest(
    signal: pd.Series,
    asset_ret: pd.Series,
    cost_bps: float = 1.0,
    max_pos: float = 1.0,
    scale: float = 1.0,
) -> BacktestResult:
    """
    Backtest a single-asset tilt. `signal` is the causal tilt dated t; it is
    lagged one bar so position held into t+1 was decided at close t, then
    multiplied by the t+1 return.

    cost_bps : round-trip cost per unit turnover, in bps of notional.
    max_pos  : position cap (|pos| ≤ max_pos).
    """
    sig, ret = signal.align(asset_ret, join="inner")
    pos_target = (scale * sig).clip(-max_pos, max_pos).fillna(0.0)
    position = pos_target.shift(1).fillna(0.0)          # held INTO the day → no lookahead
    turnover = position.diff().abs().fillna(position.abs())

    gross = position * ret
    cost = turnover * (cost_bps * 1e-4)
    net = gross - cost

    res = BacktestResult(gross=gross, net=net, position=position, turnover=turnover)
    res.stats = _bt_stats(gross, net, turnover)
    return res


def _ann_sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(ANN))


def _bt_stats(gross: pd.Series, net: pd.Series, turnover: pd.Series) -> Dict[str, float]:
    n = max(len(net.dropna()), 1)
    return {
        "sharpe_gross": _ann_sharpe(gross),
        "sharpe_net": _ann_sharpe(net),
        "ann_ret_net": float(net.mean() * ANN),
        "ann_vol": float(net.std() * np.sqrt(ANN)),
        "ann_turnover": float(turnover.mean() * ANN),
        "hit_rate": float((net.dropna() > 0).mean()),
        "n_days": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward driver
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    log_price: pd.Series,
    fit_signal_fn: Callable[[pd.Series, pd.Series], pd.Series],
    n_folds: int = 5,
    min_train: int = 500,
) -> pd.Series:
    """
    Expanding-window walk-forward. The series is cut into `n_folds` contiguous
    OOS blocks; for block k, `fit_signal_fn(train_log_price, full_log_price)` is
    called with everything strictly before the block as TRAIN and must return a
    causal signal aligned to `full_log_price.index`. Only the block's slice of
    that signal is kept, so all parameters were fit out-of-sample.

    Returns the stitched OOS signal over the whole evaluable span.
    """
    idx = log_price.index
    n = len(idx)
    start = max(min_train, n // (n_folds + 1))
    bounds = np.linspace(start, n, n_folds + 1).astype(int)

    pieces = []
    for k in range(n_folds):
        lo, hi = bounds[k], bounds[k + 1]
        if hi <= lo:
            continue
        train = log_price.iloc[:lo]
        sig_full = fit_signal_fn(train, log_price)        # causal, full index
        pieces.append(sig_full.iloc[lo:hi])
    return pd.concat(pieces).reindex(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Integral baselines (section 6): EMA / MACD are the first integral of price
# ─────────────────────────────────────────────────────────────────────────────

def ema_signal(price: pd.Series, span: int = 50, zwindow: int = 252) -> pd.Series:
    """Trend tilt from price vs its EMA — the canonical layer-1 smoother.
    (price − EMA)/EMA, causally z-scored."""
    ema = price.ewm(span=span, adjust=False).mean()
    raw = (price - ema) / ema
    mu = raw.rolling(zwindow, min_periods=60).mean()
    sd = raw.rolling(zwindow, min_periods=60).std()
    return (raw - mu) / sd.replace(0.0, np.nan)


def macd_signal(
    price: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9,
    zwindow: int = 252,
) -> pd.Series:
    """
    MACD histogram (fast EMA − slow EMA, minus its signal line) — a finite-
    difference velocity proxy on the smoothed price, causally z-scored. The
    direct integral-layer competitor to the Kalman filtered-velocity signal.
    """
    macd = price.ewm(span=fast, adjust=False).mean() - price.ewm(span=slow, adjust=False).mean()
    hist = macd - macd.ewm(span=sig, adjust=False).mean()
    mu = hist.rolling(zwindow, min_periods=60).mean()
    sd = hist.rolling(zwindow, min_periods=60).std()
    return (hist - mu) / sd.replace(0.0, np.nan)


# ─────────────────────────────────────────────────────────────────────────────
# Path-signature corner (section 6, OVERFIT-PRONE — flagged loudly)
# ─────────────────────────────────────────────────────────────────────────────

def truncated_signature_depth2(path_2d: np.ndarray) -> np.ndarray:
    """
    Depth-2 truncated path signature of a 2-D path (time, value), computed by
    hand so no `esig`/`signatory` dependency is needed.

    Returns [S1_t, S1_v, S2_tt, S2_tv, S2_vt, S2_vv] — the level-1 increments and
    level-2 iterated integrals. Level-2 cross terms encode lead/lag & area
    (Lévy area), which is exactly the kind of rich feature that overfits on a
    short single-asset sample. USE ONLY inside walk-forward and read the result
    with heavy skepticism — this is the flagged overfit corner.
    """
    p = np.asarray(path_2d, float)
    d = np.diff(p, axis=0)                       # increments
    s1 = d.sum(axis=0)                           # level 1: total displacement
    # level 2: ∫∫ dX_i dX_j  via cumulative left-point rule
    cum = np.vstack([np.zeros(2), np.cumsum(d, axis=0)[:-1]])
    s2 = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            s2[i, j] = np.sum(cum[:, i] * d[:, j])
    return np.array([s1[0], s1[1], s2[0, 0], s2[0, 1], s2[1, 0], s2[1, 1]])


def signature_trend_signal(
    log_price: pd.Series, window: int = 40, train_frac: float = 0.5,
    zwindow: int = 252,
) -> pd.Series:
    """
    Roll a depth-2 signature over the last `window` bars, fit a linear map from
    signature → next-day return on the TRAIN slice only, and apply it OOS. The
    fitted weights never see the evaluation slice, but the feature space is rich
    and the sample is short — treat any edge here as the overfit baseline, not a
    result. Returned as a causally z-scored tilt for comparability.
    """
    lp = log_price.dropna()
    vals = lp.values
    t = np.arange(len(vals), dtype=float)
    feats, idxs = [], []
    for end in range(window, len(vals)):
        seg = np.column_stack([t[end - window:end], vals[end - window:end]])
        seg = seg - seg[0]                       # translate to origin
        feats.append(truncated_signature_depth2(seg))
        idxs.append(lp.index[end])
    X = np.asarray(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)     # standardise features

    # target = next-day LOG return aligned to each feature row
    logret_next = np.diff(vals)
    y = np.array([logret_next[lp.index.get_loc(ix)] if lp.index.get_loc(ix) < len(logret_next)
                  else np.nan for ix in idxs])
    ntr = int(train_frac * len(X))
    Xtr, ytr = X[:ntr], y[:ntr]
    ok = ~np.isnan(ytr)
    # ridge for stability (the depth-2 features are collinear)
    lam = 1.0
    A = Xtr[ok].T @ Xtr[ok] + lam * np.eye(X.shape[1])
    w = np.linalg.solve(A, Xtr[ok].T @ ytr[ok])
    pred = X @ w
    raw = pd.Series(pred, index=idxs)
    mu = raw.rolling(zwindow, min_periods=60).mean()
    sd = raw.rolling(zwindow, min_periods=60).std()
    return ((raw - mu) / sd.replace(0.0, np.nan)).reindex(log_price.index)
