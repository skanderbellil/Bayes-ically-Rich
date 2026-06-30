"""
Kinematic state-space filters — the "generative layer" (price = noisy
observation of a latent state).

THE IDEA
--------
Treat the printed (log) price as a noisy measurement of a hidden kinematic
state and let a Kalman filter infer the state. Three models live here, all
sharing the same causal/diagnostic contract:

  KinematicKalman   latent [level, velocity, acceleration] with a
                    constant-acceleration (white-noise-jerk) transition. The
                    headline "generative layer" — gives causal velocity and
                    acceleration estimates from price alone.

  LocalLevelKalman  latent [level] only, random-walk transition. price = level
                    + noise. Its one-step innovation (print minus predicted
                    level) is the mean-reversion signal of section 3.

  PairsKalman       latent [alpha, beta] random walk, observation
                    y_A = alpha + beta * y_B + noise. Time-varying hedge ratio
                    for a pair (section 4).

═══════════════════════════════════════════════════════════════════════════════
CAUSALITY CONTRACT — read this before trading anything out of this module.
═══════════════════════════════════════════════════════════════════════════════
The #1 lookahead trap with a Kalman filter is feeding the RTS *smoother* into a
signal. The smoother at time t conditions on the WHOLE sample (data after t),
so it is hindsight. This module makes the split physical:

  * `FilterResult.state_filt`  (a_{t|t})     — uses data ≤ t  → CAUSAL, tradeable
  * `FilterResult.state_pred`  (a_{t|t-1})   — uses data ≤ t-1 → CAUSAL
  * `FilterResult.innovation`  (y_t - ŷ_{t|t-1}) — CAUSAL (the new info at t)
  * `smooth()` output (a_{t|T})              — uses data ≤ T  → DIAGNOSTIC ONLY

Every field on `FilterResult` is causal by construction. The smoother is a
SEPARATE method returning a SEPARATE array, never stored on `FilterResult`, so
it cannot leak into a signal by accident. Parameters (q, r) are fit by MLE on a
TRAIN slice only — see `fit_mle`.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

_LOG2PI = float(np.log(2.0 * np.pi))

# MLE floors. Daily closes carry very little measurement noise, so the
# innovation-variance/process-noise optimisers happily walk r (or q) toward 0
# and underflow. These floors are tiny relative to log-price scales (~1e-4) yet
# keep the recursion finite. A floored r ≈ 0 is still the honest "the print is
# essentially the fair level" solution — it just no longer NaNs.
_Q_FLOOR = 1e-12
_R_FLOOR = 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterResult:
    """
    Output of a forward Kalman pass. EVERY field is causal: the value at row t
    uses observations up to (state_filt/innovation) or before (state_pred) time
    t, never after. Safe to feed any of these into a trading signal.
    """
    state_filt: np.ndarray      # (T, k)  a_{t|t}    filtered state  [CAUSAL]
    cov_filt: np.ndarray        # (T, k, k) P_{t|t}
    state_pred: np.ndarray      # (T, k)  a_{t|t-1}  one-step predicted [CAUSAL]
    cov_pred: np.ndarray        # (T, k, k) P_{t|t-1}
    obs_pred: np.ndarray        # (T,)    ŷ_{t|t-1}  predicted observation [CAUSAL]
    innovation: np.ndarray      # (T,)    y_t - ŷ_{t|t-1}              [CAUSAL]
    innovation_var: np.ndarray  # (T,)    S_t = Var(innovation)        [CAUSAL]
    loglik: float               # Σ log N(y_t | ŷ_{t|t-1}, S_t)

    @property
    def level(self) -> np.ndarray:
        return self.state_filt[:, 0]

    @property
    def velocity(self) -> np.ndarray:
        return self.state_filt[:, 1]

    @property
    def acceleration(self) -> np.ndarray:
        if self.state_filt.shape[1] < 3:
            raise AttributeError("model has no acceleration state")
        return self.state_filt[:, 2]

    @property
    def std_innovation(self) -> np.ndarray:
        """Standardised innovation v_t / sqrt(S_t) — unit variance if model is
        well specified. The raw material for the mean-reversion signal."""
        return self.innovation / np.sqrt(np.clip(self.innovation_var, 1e-300, None))


# ─────────────────────────────────────────────────────────────────────────────
# Generic linear-Gaussian forward pass + RTS smoother
# ─────────────────────────────────────────────────────────────────────────────

def _kalman_forward(
    y: np.ndarray,
    F: np.ndarray,
    Q: np.ndarray,
    R: float,
    a0: np.ndarray,
    P0: np.ndarray,
    H: Optional[np.ndarray] = None,
    H_t: Optional[np.ndarray] = None,
) -> FilterResult:
    """
    Forward Kalman pass for a scalar-observation linear-Gaussian model.

    Either a constant `H` (k,) or a time-varying `H_t` (T, k) must be given.
    Missing observations (NaN in `y`) are handled as pure prediction steps
    (no update), which keeps the recursion causal across gaps.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    k = F.shape[0]

    state_filt = np.empty((T, k))
    cov_filt = np.empty((T, k, k))
    state_pred = np.empty((T, k))
    cov_pred = np.empty((T, k, k))
    obs_pred = np.empty(T)
    innov = np.empty(T)
    innov_var = np.empty(T)
    loglik = 0.0

    a, P = a0.copy(), P0.copy()
    for t in range(T):
        # ── predict ──────────────────────────────────────────────────────
        if t == 0:
            a_pred, P_pred = a0.copy(), P0.copy()
        else:
            a_pred = F @ a
            P_pred = F @ P @ F.T + Q
        state_pred[t] = a_pred
        cov_pred[t] = P_pred

        h = H if H is not None else H_t[t]
        yhat = float(h @ a_pred)
        S = float(h @ P_pred @ h + R)
        Sd = max(S, 1e-300)                           # guard division/log
        obs_pred[t] = yhat
        innov_var[t] = S

        if np.isnan(y[t]):
            # no measurement — carry the prediction forward
            innov[t] = np.nan
            a, P = a_pred, P_pred
        else:
            v = y[t] - yhat
            innov[t] = v
            K = (P_pred @ h) / Sd                     # (k,)
            a = a_pred + K * v
            P = P_pred - np.outer(K, h @ P_pred)
            loglik += -0.5 * (_LOG2PI + np.log(Sd) + v * v / Sd)

        state_filt[t] = a
        cov_filt[t] = P

    return FilterResult(
        state_filt=state_filt, cov_filt=cov_filt,
        state_pred=state_pred, cov_pred=cov_pred,
        obs_pred=obs_pred, innovation=innov, innovation_var=innov_var,
        loglik=float(loglik),
    )


def _rts_smoother(res: FilterResult, F: np.ndarray) -> np.ndarray:
    """
    Rauch–Tung–Striebel backward smoother → a_{t|T}.

    DIAGNOSTIC ONLY. Conditions on the full sample, so it is hindsight; it is
    used for plots and for the noise-amplification reference, NEVER for a trade.
    Time-invariant F only (so not for PairsKalman, whose H — not F — varies).
    """
    T, k = res.state_filt.shape
    xs = res.state_filt.copy()
    Ps = res.cov_filt.copy()
    for t in range(T - 2, -1, -1):
        Ppred_next = res.cov_pred[t + 1]
        try:
            C = res.cov_filt[t] @ F.T @ np.linalg.inv(Ppred_next)
        except np.linalg.LinAlgError:
            C = res.cov_filt[t] @ F.T @ np.linalg.pinv(Ppred_next)
        xs[t] = res.state_filt[t] + C @ (xs[t + 1] - res.state_pred[t + 1])
        Ps[t] = res.cov_filt[t] + C @ (Ps[t + 1] - Ppred_next) @ C.T
    return xs


# ─────────────────────────────────────────────────────────────────────────────
# Constant-acceleration kinematic filter
# ─────────────────────────────────────────────────────────────────────────────

class KinematicKalman:
    """
    Latent state x = [level, velocity, acceleration]ᵀ with constant-acceleration
    dynamics driven by white-noise jerk, observed as a noisy level.

    Transition (unit time step):
        x_t = F x_{t-1} + w,   w ~ N(0, q·Qj)
        F = [[1, 1, ½],
             [0, 1, 1 ],
             [0, 0, 1 ]]
    Observation:
        y_t = level_t + v,     v ~ N(0, r)
        H = [1, 0, 0]

    Process noise is the standard discretised continuous-white-noise-jerk model
    (Bar-Shalom), so the entire transition noise is ONE scalar q (the jerk
    spectral density). With one scalar r, the model has just two parameters and
    the Q/R sensitivity sweep collapses to a clean 1-D sweep of log10(q/r).

    Apply to LOG price: velocity ≈ drift (per-bar log-return), acceleration ≈
    its curvature, and the noise scale r is roughly stationary across decades.
    """

    F = np.array([[1.0, 1.0, 0.5],
                  [0.0, 1.0, 1.0],
                  [0.0, 0.0, 1.0]])
    H = np.array([1.0, 0.0, 0.0])

    # discretised white-noise-jerk process-noise shape (multiplied by scalar q)
    Qj = np.array([[1 / 20, 1 / 8, 1 / 6],
                   [1 / 8, 1 / 3, 1 / 2],
                   [1 / 6, 1 / 2, 1.0]])

    def __init__(self, q: float = 1e-6, r: float = 1e-4):
        self.q = float(q)
        self.r = float(r)

    # ── priors ────────────────────────────────────────────────────────────
    def _init_state(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y0 = float(y[~np.isnan(y)][0])
        a0 = np.array([y0, 0.0, 0.0])
        P0 = np.diag([self.r, 1e-2, 1e-4])   # diffuse-ish on velocity/accel
        return a0, P0

    # ── inference ───────────────────────────────────────────────────────────
    def filter(self, y: np.ndarray) -> FilterResult:
        """Causal forward pass. Use these outputs for signals."""
        a0, P0 = self._init_state(np.asarray(y, float))
        return _kalman_forward(np.asarray(y, float), self.F, self.q * self.Qj,
                               self.r, a0, P0, H=self.H)

    def smooth(self, y: np.ndarray) -> np.ndarray:
        """RTS-smoothed states (T, 3). DIAGNOSTIC ONLY — never trade these."""
        return _rts_smoother(self.filter(y), self.F)

    def loglik(self, y: np.ndarray) -> float:
        return self.filter(y).loglik

    # ── calibration (TRAIN ONLY) ─────────────────────────────────────────────
    def fit_mle(self, y_train: np.ndarray, verbose: bool = False) -> "KinematicKalman":
        """
        Maximum-likelihood (q, r) on a TRAIN slice via the prediction-error
        decomposition (the filter's own log-likelihood). Optimised in log-space
        for positivity and scale. Mutates and returns self.

        This is offline calibration on training data — it never touches the
        evaluation slice, so it introduces no lookahead into the signal.
        """
        y = np.asarray(y_train, float)
        r0 = max(0.5 * np.nanvar(np.diff(y[~np.isnan(y)])), 1e-10)
        x0 = np.array([np.log(r0 * 1e-3), np.log(r0)])   # [log q, log r]

        def nll(theta):
            self.q = max(np.exp(theta[0]), _Q_FLOOR)
            self.r = max(np.exp(theta[1]), _R_FLOOR)
            ll = self.filter(y).loglik
            return -ll if np.isfinite(ll) else 1e12

        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 400})
        self.q = float(max(np.exp(res.x[0]), _Q_FLOOR))
        self.r = float(max(np.exp(res.x[1]), _R_FLOOR))
        if verbose:
            print(f"  KinematicKalman MLE: q={self.q:.3e} r={self.r:.3e} "
                  f"q/r={self.q / self.r:.3e} loglik={-res.fun:.1f}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Local-level filter (random-walk level + noise)
# ─────────────────────────────────────────────────────────────────────────────

class LocalLevelKalman:
    """
    Scalar local-level model:  level_t = level_{t-1} + w,  y_t = level_t + v.

    The one-step innovation v_t = y_t − level_{t|t-1} is exactly "print minus
    the filter's prediction of fair level". Its standardised form is tested as a
    mean-reversion signal (section 3). The single ratio q/r is the model's
    signal-to-noise: large q/r → trusts the print → short memory (fast EWMA);
    small q/r → heavy smoothing (slow EWMA). This is why the local level is the
    Bayesian sibling of the EMA integral baseline.
    """

    F = np.array([[1.0]])
    H = np.array([1.0])

    def __init__(self, q: float = 1e-5, r: float = 1e-4):
        self.q = float(q)
        self.r = float(r)

    def filter(self, y: np.ndarray) -> FilterResult:
        y = np.asarray(y, float)
        y0 = float(y[~np.isnan(y)][0])
        a0 = np.array([y0])
        P0 = np.array([[self.r]])
        return _kalman_forward(y, self.F, np.array([[self.q]]), self.r,
                               a0, P0, H=self.H)

    def smooth(self, y: np.ndarray) -> np.ndarray:
        return _rts_smoother(self.filter(y), self.F)

    def loglik(self, y: np.ndarray) -> float:
        return self.filter(y).loglik

    def fit_mle(self, y_train: np.ndarray, verbose: bool = False) -> "LocalLevelKalman":
        y = np.asarray(y_train, float)
        r0 = max(0.5 * np.nanvar(np.diff(y[~np.isnan(y)])), 1e-10)
        x0 = np.array([np.log(r0 * 1e-2), np.log(r0)])

        def nll(theta):
            self.q = max(np.exp(theta[0]), _Q_FLOOR)
            self.r = max(np.exp(theta[1]), _R_FLOOR)
            ll = self.filter(y).loglik
            return -ll if np.isfinite(ll) else 1e12

        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 400})
        self.q = float(max(np.exp(res.x[0]), _Q_FLOOR))
        self.r = float(max(np.exp(res.x[1]), _R_FLOOR))
        if verbose:
            print(f"  LocalLevelKalman MLE: q={self.q:.3e} r={self.r:.3e} "
                  f"q/r={self.q / self.r:.3e}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic hedge-ratio filter (time-varying beta for a pair)
# ─────────────────────────────────────────────────────────────────────────────

class PairsKalman:
    """
    Time-varying linear hedge for a pair, state x = [alpha, beta]ᵀ random walk:

        x_t = x_{t-1} + w,   w ~ N(0, diag(q, q))
        y_A,t = alpha_t + beta_t · y_B,t + v,   v ~ N(0, r)

    The observation matrix H_t = [1, y_B,t] is time-varying (that is what makes
    beta adaptive), so the RTS smoother above (time-invariant F-only) is not
    wired for this class — and we never want a smoothed beta in a trade anyway.

    `filter` returns the causal alpha_t, beta_t (a_{t|t}) and the residual
    spread y_A,t − ŷ_{t|t-1}, whose innovation is the tradeable spread.
    Operate on LOG prices.
    """

    F = np.array([[1.0, 0.0], [0.0, 1.0]])

    def __init__(self, q: float = 1e-5, r: float = 1e-3):
        self.q = float(q)
        self.r = float(r)

    def filter(self, y_a: np.ndarray, y_b: np.ndarray) -> FilterResult:
        y_a = np.asarray(y_a, float)
        y_b = np.asarray(y_b, float)
        H_t = np.column_stack([np.ones_like(y_b), y_b])     # (T, 2)
        # OLS-ish prior for [alpha, beta] from the first chunk to start sane
        n0 = min(60, len(y_a))
        b0 = np.polyfit(y_b[:n0], y_a[:n0], 1)              # [slope, intercept]
        a0 = np.array([b0[1], b0[0]])
        P0 = np.diag([self.r, 1.0])
        Q = np.diag([self.q, self.q])
        return _kalman_forward(y_a, self.F, Q, self.r, a0, P0, H_t=H_t)

    def loglik(self, y_a: np.ndarray, y_b: np.ndarray) -> float:
        return self.filter(y_a, y_b).loglik

    def fit_mle(self, y_a_train: np.ndarray, y_b_train: np.ndarray,
                verbose: bool = False) -> "PairsKalman":
        ya, yb = np.asarray(y_a_train, float), np.asarray(y_b_train, float)
        resid0 = ya - np.polyval(np.polyfit(yb, ya, 1), yb)
        r0 = max(np.var(resid0), 1e-8)
        x0 = np.array([np.log(r0 * 1e-3), np.log(r0)])

        def nll(theta):
            self.q = max(np.exp(theta[0]), _Q_FLOOR)
            self.r = max(np.exp(theta[1]), _R_FLOOR)
            ll = self.filter(ya, yb).loglik
            return -ll if np.isfinite(ll) else 1e12

        res = minimize(nll, x0, method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-2, "maxiter": 400})
        self.q = float(max(np.exp(res.x[0]), _Q_FLOOR))
        self.r = float(max(np.exp(res.x[1]), _R_FLOOR))
        if verbose:
            print(f"  PairsKalman MLE: q={self.q:.3e} r={self.r:.3e}")
        return self
