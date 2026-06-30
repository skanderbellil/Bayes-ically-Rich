"""
Regime layer (section 5) — a 2-state HMM on the filtered velocity state that
toggles trend vs mean-revert dynamics, used to GATE the momentum and reversion
signals.

CAUSALITY — the subtle trap here
--------------------------------
hmmlearn's `score_samples` / `predict_proba` return the forward–BACKWARD
posterior γ_t = P(state_t | x_0 … x_{T-1}): it conditions on the WHOLE window,
i.e. the future. Using it to gate a trade at t is lookahead. (The repo's older
`RegimeHMM` calls `score_samples` and labels it "forward-filtered" — it is not.)

This module therefore implements the forward recursion BY HAND from the fitted
Gaussian parameters, so the regime belief at t, α̂_t = P(state_t | x_0 … x_t),
uses observations only up to and including t. Parameters are EM-fit on a TRAIN
slice; the forward pass that produces tradeable probabilities runs over the full
series but, being forward-only, never peeks past t.
"""
from typing import Optional

import numpy as np


class VelocityRegimeHMM:
    """
    2-state Gaussian HMM on the (filtered) velocity series.

    The two latent states are read as:
      • TREND  — emission mean velocity sits away from zero (directional drift
                 persists) → momentum signals should work.
      • REVERT — emission mean velocity sits near zero (choppy/oscillating)
                 → mean-reversion signals should work.

    `trend_prob` returns the causal P(trend | velocity ≤ t), which the backtester
    uses to blend the momentum and reversion tilts.
    """

    def __init__(self, n_iter: int = 100, random_state: int = 42):
        self.n_iter = n_iter
        self.random_state = random_state
        self._model = None
        self.trend_state: int = 0
        self.revert_state: int = 1

    # ── fitting (TRAIN ONLY) ────────────────────────────────────────────────
    def fit(self, velocity_train: np.ndarray) -> "VelocityRegimeHMM":
        v = np.asarray(velocity_train, float).reshape(-1, 1)
        v = v[~np.isnan(v).ravel()]
        try:
            from hmmlearn import hmm as _hmm
            m = _hmm.GaussianHMM(
                n_components=2, covariance_type="full",
                n_iter=self.n_iter, tol=1e-3, random_state=self.random_state,
            )
            m.fit(v)
            self._model = m
            # TREND = state whose emission mean velocity is furthest from zero.
            absmean = np.abs(m.means_[:, 0])
            self.trend_state = int(np.argmax(absmean))
            self.revert_state = 1 - self.trend_state
        except Exception:
            self._model = None
        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ── causal forward filter (hand-rolled, no backward pass) ───────────────
    def _forward_posteriors(self, velocity: np.ndarray) -> np.ndarray:
        """
        α̂_t = P(state_t | x_0 … x_t), normalised forward variable. (T, 2).
        Pure forward recursion → causal. NaN velocities are skipped (the belief
        is carried forward through the transition only).
        """
        from scipy.stats import norm

        m = self._model
        log_pi = np.log(np.clip(m.startprob_, 1e-12, None))
        log_A = np.log(np.clip(m.transmat_, 1e-12, None))
        mu = m.means_[:, 0]
        sd = np.sqrt(np.clip(m.covars_[:, 0, 0], 1e-18, None))

        x = np.asarray(velocity, float)
        T = len(x)
        post = np.full((T, 2), 0.5)
        log_alpha = None
        for t in range(T):
            if np.isnan(x[t]):
                if log_alpha is not None:
                    # transition-only carry: α_t ∝ Σ_i α_{t-1}(i) A(i,·)
                    a = log_alpha[:, None] + log_A
                    log_alpha = _logsumexp_axis0(a)
                    log_alpha -= _logsumexp(log_alpha)
                    post[t] = np.exp(log_alpha)
                continue
            log_e = norm.logpdf(x[t], loc=mu, scale=sd)
            if log_alpha is None:
                log_alpha = log_pi + log_e
            else:
                a = log_alpha[:, None] + log_A          # (prev, next)
                log_alpha = _logsumexp_axis0(a) + log_e
            log_alpha -= _logsumexp(log_alpha)          # normalise (forward var)
            post[t] = np.exp(log_alpha)
        return post

    def trend_prob(self, velocity: np.ndarray) -> np.ndarray:
        """Causal P(trend regime | velocity ≤ t), length-T array in [0, 1]."""
        if not self.is_fitted:
            return np.full(len(velocity), 0.5)
        return self._forward_posteriors(velocity)[:, self.trend_state]

    # ── reporting ───────────────────────────────────────────────────────────
    def summary(self) -> dict:
        if not self.is_fitted:
            return {}
        m = self._model
        return {
            "trend_mean_vel": float(m.means_[self.trend_state, 0]),
            "revert_mean_vel": float(m.means_[self.revert_state, 0]),
            "trend_persist": float(m.transmat_[self.trend_state, self.trend_state]),
            "revert_persist": float(m.transmat_[self.revert_state, self.revert_state]),
        }


# ── log-sum-exp helpers (kept local; avoid scipy import churn) ──────────────
def _logsumexp(v: np.ndarray) -> float:
    mx = np.max(v)
    return float(mx + np.log(np.sum(np.exp(v - mx))))


def _logsumexp_axis0(a: np.ndarray) -> np.ndarray:
    mx = np.max(a, axis=0)
    return mx + np.log(np.sum(np.exp(a - mx), axis=0))
