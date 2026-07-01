"""
The integral ladder + adaptive rung selection.

Going UP the kinematic ladder (integrate) instead of DOWN (differentiate) flips
the noise story: differencing amplifies high-frequency noise, integration cancels
it (see `experiments/run_ladder_regime.py` and the `integral_ladder` figure). The
rungs, all CAUSAL, on log price `p`:

    deviation  (P) :  p − slow_baseline                 — the order-0 "position"
    absement   (I) :  leaky integral of the deviation   — first integral, "absement"
    velocity   (D) :  Δp                                 — first derivative

These are exactly the P, I, D terms of a controller steering price toward fair
value. The empirical finding is that *which rung carries the signal changes over
time* (regime-dependent), so `RegimeRungSelector` learns, on TRAIN only, which
rung to trust in each trend/chop regime and switches between them out-of-sample.

The leaky (not pure) integral is deliberate: a pure cumsum is non-stationary
(lag-1 autocorr ≈ 1.0) — control theory calls this "integral windup". A decay
keeps the integral bounded and stationary, the integral-ladder analogue of the
filter's process-noise regularisation.
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from posterioralpha.kinematic.signals import causal_zscore


# ─────────────────────────────────────────────────────────────────────────────
# Ladder primitives (all causal)
# ─────────────────────────────────────────────────────────────────────────────

def leaky_integral(x: pd.Series, halflife: float = 60.0) -> pd.Series:
    """
    Exponentially-decaying running integral:  I_t = λ·I_{t-1} + x_t,
    λ = ½^(1/halflife). Bounded and stationary (avoids integral windup), unlike a
    raw cumulative sum. Causal — I_t uses only x_{≤t}.
    """
    lam = 0.5 ** (1.0 / halflife)
    out = np.empty(len(x))
    acc = 0.0
    vals = x.values
    for t in range(len(vals)):
        v = vals[t]
        acc = lam * acc + (0.0 if np.isnan(v) else v)
        out[t] = acc
    return pd.Series(out, index=x.index)


def efficiency_ratio(price: pd.Series, window: int = 30) -> pd.Series:
    """
    Kaufman efficiency ratio: |net move| / total path length over `window`.
    ≈1 → clean trend (the path went somewhere); ≈0 → chop (lots of motion, no
    progress). Causal. This is the trend-vs-chop axis the rung selector gates on.
    """
    net_move = (price - price.shift(window)).abs()
    path = price.diff().abs().rolling(window).sum()
    return (net_move / path.replace(0.0, np.nan)).rename("ER")


def ladder_rungs(
    log_price: pd.Series, baseline_span: int = 100,
    integral_halflife: float = 60.0, z_window: int = 252,
) -> pd.DataFrame:
    """
    The three causal rung signals, each z-scored to a comparable scale and
    oriented so that 'go long' is positive:

      deviation-reversion :  −z(p − EMA_baseline)        fade the stretch
      absement-trend      :  +z(leaky∫ deviation)        ride sustained displacement
      velocity-trend      :  +z(Δp)                      ride recent drift
    """
    base = log_price.ewm(span=baseline_span, adjust=False).mean()
    dev = log_price - base
    absn = leaky_integral(dev, integral_halflife)
    vel = log_price.diff()
    return pd.DataFrame({
        "deviation-reversion": -causal_zscore(dev, z_window),
        "absement-trend":      +causal_zscore(absn, z_window),
        "velocity-trend":      +causal_zscore(vel, z_window),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive rung selection (learns rung→regime on TRAIN, switches OOS)
# ─────────────────────────────────────────────────────────────────────────────

def _ic(sig: pd.Series, fwd: pd.Series) -> float:
    df = pd.concat([sig, fwd], axis=1).dropna()
    return float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman")) if len(df) >= 20 else np.nan


class RegimeRungSelector:
    """
    Picks which ladder rung to trade as a function of the trend/chop regime.

    fit() splits TRAIN into `n_regimes` efficiency-ratio buckets, and for each
    bucket picks the rung with the highest train IC against next-day returns.
    transform() classifies each OOS day into a bucket (TRAIN thresholds) and
    emits that bucket's chosen rung — a single causal signal that switches rungs
    as the regime changes. Low-parameter (one rung choice per regime) by design,
    because the conditional edge is small and easy to overfit.
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.edges_: Optional[np.ndarray] = None
        self.choice_: Dict[int, str] = {}
        self.train_ic_: Dict[int, Dict[str, float]] = {}

    def fit(self, rungs: pd.DataFrame, ER: pd.Series, fwd_ret: pd.Series) -> "RegimeRungSelector":
        qs = np.linspace(0, 1, self.n_regimes + 1)[1:-1]
        self.edges_ = ER.quantile(qs).values
        reg = np.digitize(ER.values, self.edges_)
        reg = pd.Series(reg, index=ER.index)
        for k in range(self.n_regimes):
            idx = reg.index[reg.values == k]
            ics = {c: _ic(rungs[c].reindex(idx), fwd_ret.reindex(idx)) for c in rungs.columns}
            self.train_ic_[k] = ics
            # best by train IC; NaN-safe (default to reversion if a bucket is empty)
            best = max(ics, key=lambda c: (ics[c] if np.isfinite(ics[c]) else -1e9))
            self.choice_[k] = best
        return self

    def regimes(self, ER: pd.Series) -> pd.Series:
        return pd.Series(np.digitize(ER.values, self.edges_), index=ER.index)

    def transform(self, rungs: pd.DataFrame, ER: pd.Series) -> pd.Series:
        reg = self.regimes(ER)
        out = pd.Series(np.nan, index=rungs.index)
        for k, col in self.choice_.items():
            mask = (reg == k).reindex(rungs.index, fill_value=False)
            out[mask.values] = rungs[col].reindex(rungs.index)[mask.values]
        return out

    def summary(self) -> pd.DataFrame:
        rows = []
        for k in range(self.n_regimes):
            row = {"regime": k, "chosen_rung": self.choice_.get(k, "—")}
            row.update({f"trainIC[{c}]": self.train_ic_.get(k, {}).get(c, np.nan)
                        for c in ("deviation-reversion", "absement-trend", "velocity-trend")})
            rows.append(row)
        return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-sectional plumbing (a single asset lacks the breadth to learn the flip;
# the cross-section supplies it — see experiments/run_ladder_cross_section.py)
# ─────────────────────────────────────────────────────────────────────────────

def per_name_rungs(log_prices: pd.DataFrame, **kw) -> Dict[str, pd.DataFrame]:
    """
    Apply `ladder_rungs` to every column of a (dates × names) log-price frame.
    Returns {rung_name: wide frame of that rung across names}, all causal.
    """
    cols = {"deviation-reversion": {}, "absement-trend": {}, "velocity-trend": {}}
    for name in log_prices.columns:
        r = ladder_rungs(log_prices[name].dropna(), **kw)
        for k in cols:
            cols[k][name] = r[k]
    return {k: pd.DataFrame(v).reindex(log_prices.index) for k, v in cols.items()}


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last available trading day of each calendar month in `index`."""
    marks = pd.Series(index=index, dtype=float).resample("ME").last().index
    days = [index[index <= d][-1] for d in marks if (index <= d).any()]
    return pd.DatetimeIndex(sorted(set(days)))


def cross_sectional_ls(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, rebal: pd.DatetimeIndex,
    cost_bps: float = 5.0, quantile: float = 1 / 3, min_names: int = 30,
) -> pd.Series:
    """
    Monthly cross-sectional tercile long-short. At each rebalance, long the top
    `quantile` of `signal`, short the bottom, equal-weight, hold to the next
    rebalance, charge `cost_bps` on the traded notional Σ|Δw|. Returns the net
    per-period return series. Both `signal` and `fwd_ret` are (dates × names),
    aligned to `rebal`; `fwd_ret` is next-period return per name.
    """
    sig = signal.reindex(rebal)
    rets, prev_w = [], pd.Series(0.0, index=signal.columns)
    for d in rebal[:-1]:
        s = sig.loc[d].dropna()
        f = fwd_ret.loc[d].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_names:
            rets.append(np.nan)
            continue
        s = s[common]
        lo, hi = s.quantile(quantile), s.quantile(1 - quantile)
        w = pd.Series(0.0, index=signal.columns)
        longs, shorts = s[s >= hi].index, s[s <= lo].index
        w[longs] = 1.0 / len(longs)
        w[shorts] = -1.0 / len(shorts)
        turn = (w - prev_w).abs().sum()
        prev_w = w
        rets.append(float((w[common] * f[common]).sum()) - turn * cost_bps * 1e-4)
    return pd.Series(rets, index=rebal[:-1]).dropna()


def cross_sectional_ic(
    signal: pd.DataFrame, fwd_ret: pd.DataFrame, rebal: pd.DatetimeIndex,
    min_names: int = 30,
) -> pd.Series:
    """Per-rebalance cross-sectional Spearman IC of `signal` vs next-period return."""
    sig = signal.reindex(rebal)
    out = {}
    for d in rebal[:-1]:
        s = sig.loc[d].dropna()
        f = fwd_ret.loc[d].dropna()
        common = s.index.intersection(f.index)
        if len(common) >= min_names:
            out[d] = s[common].corr(f[common], method="spearman")
    return pd.Series(out)


def regime_gated_blend(
    rungs: Dict[str, pd.DataFrame], efficiency: pd.Series,
    trend_rung: str = "deviation-reversion", chop_rung: str = "absement-trend",
) -> pd.DataFrame:
    """
    Parameter-free structural blend: weight the chop rung by (1 − efficiency) and
    the trend rung by efficiency, per date. Low efficiency ratio → choppy market
    → lean on the integral rung; high → trending → lean on reversion. This is the
    combination that WORKS (conditioning on a stable market state), as opposed to
    weighting by recent performance, which chases mean-reverting edges and loses.
    """
    er = efficiency.clip(0, 1)
    w_chop = (1 - er)
    return rungs[chop_rung].mul(w_chop, axis=0) + rungs[trend_rung].mul(er, axis=0)
