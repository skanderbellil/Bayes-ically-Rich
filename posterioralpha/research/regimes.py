"""
Regime-detection models (strategy-research layer).

Every change-point / regime model the backtest engines consume lives here.

RegimeHMM — 2-state Gaussian HMM (bull / bear). Per-regime priors blended by
            posterior regime probability. Forward-filtered (no Viterbi
            lookahead). Consumed by the Bayesian backtest engine.

BOCPD     — Bayesian Online Change Point Detection (Adams & MacKay 2007)
            1D Normal-Gamma conjugate prior: exact, fast, no fixed state count.
            "Soul of the idea": pure online Bayesian updating — at every step
            it asks 'has the distribution changed?' and updates its belief.
            Output: changepoint_prob, expected_run_length.

HMM3      — 3-state Gaussian HMM (bull / sideways / bear).
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


def precompute_bocpd_multi(
    returns_df,
    assets: list,
    hazard: float = 1 / 252,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run BOCPD independently on multiple assets and return aggregated signals.

    cp aggregation  →  MAX across assets
    erl aggregation →  weighted average

    Why max for cp?
    Weighted averaging washes out individual asset spikes — the result ends up
    pinned to the hazard rate (≈0.004) at all times, making the signal useless.
    Taking the MAX preserves any asset's changepoint signal: if TLT fires during
    a rate shock but SPY hasn't reacted yet, max-cp catches it early.

    Why weighted avg for erl?
    ERL measures "how long has this regime been running" — a weighted average
    gives a stable estimate of cross-asset regime age.  We give SPY the most
    weight since its ERL most directly reflects the equity regime we're
    allocating into.
    """
    available = [a for a in assets if a in returns_df.columns]
    if not available:
        raise ValueError(f"None of {assets} found in returns_df columns")

    if weights is None:
        w = np.ones(len(available)) / len(available)
    else:
        w = np.array(weights[: len(available)], dtype=float)
        w /= w.sum()

    all_cp, all_erl = [], []
    for asset in available:
        cp_i, erl_i = precompute_bocpd(returns_df[asset].values, hazard=hazard)
        all_cp.append(cp_i)
        all_erl.append(erl_i)

    # MAX preserves spike sensitivity; weighted avg provides stable regime-age estimate
    cp_agg  = np.max(np.stack(all_cp,  axis=1), axis=1)
    erl_agg = sum(w[i] * all_erl[i] for i in range(len(all_erl)))
    return np.array(cp_agg), np.array(erl_agg)


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


# ─────────────────────────────────────────────────────────────────────────────
# 2-state HMM regime filter
# ─────────────────────────────────────────────────────────────────────────────

class RegimeHMM:
    """
    2-state Gaussian HMM for bull / bear regime detection.

    Extension from the original idea:
      "HMM Regime Filter — instead of a single prior, have per-regime priors and
       use the HMM to assign posterior regime probabilities, then mix regime-
       specific Sharpe-optimal weights."

    Design
    ------
      • 2-state Gaussian HMM fitted on the full available return history.
      • States are labelled bull (higher mean return) and bear (lower / negative).
      • At each rebalance we compute p_bull = P(regime = bull | returns up to t).
      • The Bayesian backtest engine builds regime-specific portfolios and blends
        them by p_bull, with an uncertainty penalty when p_bull ≈ 0.5.

    Usage
    -----
    model  = RegimeHMM().fit(returns_array)
    p_bull = model.bull_prob(returns_array)        # scalar in [0, 1]
    states = model.state_sequence(returns_array)   # (T,) int array
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
        try:
            from hmmlearn import hmm as _hmm
        except ImportError:
            logger.warning("hmmlearn not available — RegimeHMM degrades gracefully.")
            self._model = None
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
            logger.debug(f"RegimeHMM fit failed: {exc}")
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
        """
        Most-likely state at each timestep via forward-filtered argmax.

        Uses the forward pass only (no backward smoothing), so the state
        assigned to day t depends only on data up to day t — zero lookahead bias.
        Previously used Viterbi (model.predict) which runs a backward pass and
        assigns states with full hindsight over the window.
        """
        if not self.is_fitted:
            return np.zeros(len(returns), dtype=int)
        try:
            _, posteriors = self._model.score_samples(returns)
            return np.argmax(posteriors, axis=1).astype(int)
        except Exception:
            return np.zeros(len(returns), dtype=int)

    def regime_mask(self, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (bull_mask, bear_mask) boolean arrays of length T."""
        states = self.state_sequence(returns)
        return states == self.bull_state, states == self.bear_state
