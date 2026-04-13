"""
Hidden Markov Model regime filter for PosteriorAlpha.

Extension from the original idea:
  "HMM Regime Filter — instead of a single prior, have per-regime priors and
   use the HMM to assign posterior regime probabilities, then mix regime-specific
   Sharpe-optimal weights."

Design
------
  • 2-state Gaussian HMM fitted on the full available return history.
  • States are labelled bull (higher mean return) and bear (lower / negative).
  • At each rebalance we compute:
      p_bull = P(regime = bull | all returns up to t)
      p_bear = 1 − p_bull
  • We then build regime-specific portfolios and blend them:
      w_final = p_bull · w_bull + p_bear · w_bear
  • When the regime is ambiguous (p_bull ≈ 0.5) we also apply an
    uncertainty penalisation that shrinks toward equal weight.
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm as _hmm
    _HMM_OK = True
except ImportError:
    _HMM_OK = False
    logger.warning("hmmlearn not available — HMM strategies will degrade gracefully.")


class RegimeHMM:
    """
    2-state Gaussian HMM for bull / bear regime detection.

    Usage
    -----
    model = RegimeHMM().fit(returns_array)
    p_bull = model.bull_prob(returns_array)   # scalar in [0, 1]
    states = model.state_sequence(returns_array)  # (T,) int array
    """

    def __init__(self, n_iter: int = 60, random_state: int = 42):
        self.n_iter       = n_iter
        self.random_state = random_state
        self._model: Optional[object] = None
        self.bull_state: int = 0
        self.bear_state: int = 1

    # ── Fitting ───────────────────────────────────────────────────────────

    def fit(self, returns: np.ndarray) -> "RegimeHMM":
        """
        Fit a 2-state Gaussian HMM on (T, N) returns.
        Identifies bull vs bear by comparing the mean return of the first asset.
        """
        if not _HMM_OK:
            return self

        model = _hmm.GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=self.n_iter,
            tol=1e-3,
            random_state=self.random_state,
        )
        try:
            model.fit(returns)
            self._model = model
            # Label states: higher mean on asset 0 → bull
            mean0 = model.means_[:, 0]
            self.bull_state = int(np.argmax(mean0))
            self.bear_state = 1 - self.bull_state
        except Exception as exc:
            logger.debug(f"HMM fit failed: {exc}")
            self._model = None

        return self

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # ── Inference ─────────────────────────────────────────────────────────

    def bull_prob(self, returns: np.ndarray) -> float:
        """
        P(bull regime | observed returns) at the final timestep.
        Returns 0.5 (maximum uncertainty) if model is not fitted.
        """
        if not self.is_fitted:
            return 0.5
        try:
            _, posteriors = self._model.score_samples(returns)
            return float(np.clip(posteriors[-1, self.bull_state], 0.0, 1.0))
        except Exception:
            return 0.5

    def state_sequence(self, returns: np.ndarray) -> np.ndarray:
        """Most-likely state at each timestep via Viterbi decoding."""
        if not self.is_fitted:
            return np.zeros(len(returns), dtype=int)
        try:
            return self._model.predict(returns)
        except Exception:
            return np.zeros(len(returns), dtype=int)

    def regime_mask(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (bull_mask, bear_mask) boolean arrays of length T.
        """
        states = self.state_sequence(returns)
        return states == self.bull_state, states == self.bear_state
