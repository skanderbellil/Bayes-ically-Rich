"""
Advanced regime detection models.

BOCPD  — Bayesian Online Change Point Detection (Adams & MacKay 2007)
         1D Normal-Gamma conjugate prior: exact, fast, no fixed state count.
         "Soul of the idea": pure online Bayesian updating — at every step
         it asks 'has the distribution changed?' and updates its belief.
         Output: changepoint_prob, expected_run_length.

HMM3   — 3-state Gaussian HMM (bull / sideways / bear).
         Captures normal bull, grinding/uncertain, and crisis regimes
         that a 2-state model merges. Forward-filtered posteriors only
         (no Viterbi backward pass → no lookahead bias).
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_MAX_RL = 504   # cap run-length distribution at 2 years to keep O(1) memory


# ─────────────────────────────────────────────────────────────────────────────
# BOCPD
# ─────────────────────────────────────────────────────────────────────────────

class BOCPD:
    """
    Bayesian Online Change Point Detection on a scalar signal.

    Model: data within each regime ~ Normal(μ, 1/τ), with a
    Normal-Gamma conjugate prior.  At each step the hazard H
    is the prior probability of a change point occurring.

    Use on a market summary signal (e.g. SPY daily return, first PC).
    Pre-compute the full series before the backtest loop for speed.

    Key outputs
    -----------
    changepoint_prob     : P(regime just changed)
    expected_run_length  : E[days since last change point]
                           Short → recent regime shift; Long → stable regime
    """

    def __init__(
        self,
        hazard: float = 1 / 252,    # ~1 regime change per year
        mu0: float = 0.0,
        kappa0: float = 1.0,
        alpha0: float = 2.0,
        beta0: float = 1e-4,        # non-informative variance prior
    ):
        self.H      = hazard
        self.mu0    = mu0
        self.kappa0 = kappa0
        self.alpha0 = alpha0
        self.beta0  = beta0
        self._reset()

    # ── Internal state ────────────────────────────────────────────────────

    def _reset(self):
        self.R     = np.array([1.0])          # P(run_length = l)
        self.mu    = np.array([self.mu0])
        self.kappa = np.array([self.kappa0])
        self.alpha = np.array([self.alpha0])
        self.beta  = np.array([self.beta0])

    # ── Update ────────────────────────────────────────────────────────────

    def update(self, x: float) -> "BOCPD":
        """Assimilate one new scalar observation."""
        pred = self._predictive_pdf(float(x))

        R_cp   = np.array([float(np.sum(self.R * pred)) * self.H])
        R_grow = self.R * pred * (1.0 - self.H)

        new_R = np.concatenate([R_cp, R_grow])
        s = new_R.sum()
        self.R = new_R / s if s > 0 else new_R

        self._update_params(float(x))
        self._truncate()
        return self

    def _predictive_pdf(self, x: float) -> np.ndarray:
        """Student-t predictive density p(x | run_length = l) for each l."""
        from scipy.stats import t as t_dist
        df    = 2.0 * self.alpha
        scale = np.sqrt(
            np.clip(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa), 1e-12, None)
        )
        return t_dist.pdf(x, df=df, loc=self.mu, scale=scale)

    def _update_params(self, x: float):
        k, m, a, b = self.kappa, self.mu, self.alpha, self.beta
        self.kappa = np.concatenate([[self.kappa0], k + 1.0])
        self.mu    = np.concatenate([[self.mu0],    (k * m + x) / (k + 1.0)])
        self.alpha = np.concatenate([[self.alpha0], a + 0.5])
        self.beta  = np.concatenate([[self.beta0],  b + k * (x - m) ** 2 / (2.0 * (k + 1.0))])

    def _truncate(self):
        """Cap run-length distribution at _MAX_RL to keep memory bounded."""
        if len(self.R) > _MAX_RL:
            tail = self.R[_MAX_RL:].sum()
            self.R     = self.R[:_MAX_RL];     self.R[-1]     += tail
            self.kappa = self.kappa[:_MAX_RL]
            self.mu    = self.mu[:_MAX_RL]
            self.alpha = self.alpha[:_MAX_RL]
            self.beta  = self.beta[:_MAX_RL]

    # ── Signals ───────────────────────────────────────────────────────────

    @property
    def changepoint_prob(self) -> float:
        """P(change point at the most recent observation)."""
        return float(self.R[0]) if len(self.R) else 0.0

    @property
    def expected_run_length(self) -> float:
        """E[days since last change point]."""
        return float(np.dot(np.arange(len(self.R)), self.R))

    # ── Batch pre-computation ─────────────────────────────────────────────

    def fit_history(self, series: np.ndarray) -> "BOCPD":
        """Warm up the model by processing a 1D array of past observations."""
        self._reset()
        for x in series:
            self.update(float(x))
        return self


def precompute_bocpd(
    returns_series: np.ndarray,
    hazard: float = 1 / 252,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BOCPD once over a full 1D series and return arrays of signals.

    Returns
    -------
    cp_probs  : (T,) changepoint probability at each day
    erl       : (T,) expected run length at each day
    """
    model  = BOCPD(hazard=hazard)
    cp, rl = [], []
    for x in returns_series:
        model.update(float(x))
        cp.append(model.changepoint_prob)
        rl.append(model.expected_run_length)
    return np.array(cp), np.array(rl)


# ─────────────────────────────────────────────────────────────────────────────
# 3-state HMM
# ─────────────────────────────────────────────────────────────────────────────

class HMM3:
    """
    3-state Gaussian HMM: bull (0), sideways (1), bear (2).

    States are ranked by the mean return of the first asset so labels
    are consistent across refits.  Posteriors are forward-filtered
    (score_samples argmax) — no lookahead bias.
    """

    BULL     = 0
    SIDEWAYS = 1
    BEAR     = 2

    def __init__(self, n_iter: int = 80, random_state: int = 42):
        self.n_iter       = n_iter
        self.random_state = random_state
        self._model       = None
        self._state_map   = {0: 0, 1: 1, 2: 2}   # internal HMM state → label

    def fit(self, returns: np.ndarray) -> "HMM3":
        try:
            from hmmlearn import hmm as _hmm
            m = _hmm.GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=self.n_iter,
                tol=1e-3,
                random_state=self.random_state,
            )
            m.fit(returns)
            self._model = m

            # Label by mean of first asset: highest→bull, middle→sideways, lowest→bear
            ranked = np.argsort(m.means_[:, 0])[::-1]   # descending
            self._state_map = {
                int(ranked[0]): self.BULL,
                int(ranked[1]): self.SIDEWAYS,
                int(ranked[2]): self.BEAR,
            }
        except Exception as exc:
            logger.debug(f"HMM3 fit failed: {exc}")
            self._model = None
        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def regime_probs(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        (p_bull, p_sideways, p_bear) at the last timestep.
        Forward-filtered — uses only data up to that point.
        """
        if not self.is_fitted:
            return (1/3, 1/3, 1/3)
        try:
            _, post = self._model.score_samples(returns)
            raw = post[-1]                     # (3,) forward posteriors at last step
            p   = [0.0, 0.0, 0.0]
            for internal, label in self._state_map.items():
                p[label] = float(np.clip(raw[internal], 0, 1))
            return tuple(p)
        except Exception:
            return (1/3, 1/3, 1/3)

    def state_sequence(self, returns: np.ndarray) -> np.ndarray:
        """Forward-filtered label at each timestep (0=bull,1=side,2=bear)."""
        if not self.is_fitted:
            return np.ones(len(returns), dtype=int)   # default: sideways
        try:
            _, post = self._model.score_samples(returns)
            raw = np.argmax(post, axis=1)
            return np.array([self._state_map[int(s)] for s in raw], dtype=int)
        except Exception:
            return np.ones(len(returns), dtype=int)

    def regime_masks(
        self, returns: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Boolean masks (bull, sideways, bear) for each day."""
        s = self.state_sequence(returns)
        return s == self.BULL, s == self.SIDEWAYS, s == self.BEAR
