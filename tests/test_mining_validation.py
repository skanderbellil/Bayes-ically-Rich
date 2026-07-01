"""Statistical sanity checks for posterioralpha.mining.validation (all seeded)."""
import numpy as np
import pytest

from posterioralpha.mining.validation import (
    block_bootstrap_pvalue,
    deflated_sharpe_ratio,
    random_windows,
)


# ── block_bootstrap_pvalue ───────────────────────────────────────────────────

def test_bootstrap_pvalue_significant_on_profitable_series():
    """mean 0.001 / sd 0.005 daily is a ~3+ annualised Sharpe: p must be tiny."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.005, 1000)
    p, lo, hi = block_bootstrap_pvalue(r, rng=np.random.default_rng(1), n_boot=200)
    assert p < 0.05
    assert lo < hi
    assert lo > 0.0, "95% CI of a ~3-Sharpe strategy should exclude zero"


def test_bootstrap_pvalue_insignificant_on_noise():
    """Zero-mean noise: about half the bootstrap Sharpes fall below zero."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0, 0.005, 1000)
    p, lo, hi = block_bootstrap_pvalue(r, rng=np.random.default_rng(3), n_boot=200)
    assert p > 0.2
    assert lo < 0.0 < hi


def test_bootstrap_pvalue_short_series_degenerate():
    rng = np.random.default_rng(4)
    p, lo, hi = block_bootstrap_pvalue(np.ones(30) * 0.001, rng=rng)
    assert (p, lo, hi) == (1.0, 0.0, 0.0)


# ── random_windows ───────────────────────────────────────────────────────────

def test_random_windows_within_bounds():
    rng = np.random.default_rng(5)
    n_obs, window_len, min_start = 1000, 100, 50
    wins = random_windows(n_obs, n_windows=200, window_len=window_len,
                          rng=rng, min_start=min_start)
    assert len(wins) == 200
    for start, end in wins:
        assert min_start <= start
        assert end == start + window_len
        assert end <= n_obs


def test_random_windows_degenerate_returns_full_span():
    rng = np.random.default_rng(6)
    wins = random_windows(n_obs=80, n_windows=5, window_len=100, rng=rng,
                          min_start=0)
    assert wins == [(0, 80)]


# ── deflated_sharpe_ratio ────────────────────────────────────────────────────

def _positive_returns(seed=7, T=1000):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0005, 0.01, T)


def test_dsr_decreases_with_more_trials():
    r = _positive_returns()
    var = 0.002  # variance of the searched trial Sharpes (daily units)
    dsrs = [deflated_sharpe_ratio(r, n_trials=n, trial_sharpes_var=var)
            for n in (2, 10, 100, 1000, 10000)]
    for a, b in zip(dsrs, dsrs[1:]):
        assert b < a, f"DSR must strictly decrease with n_trials: {dsrs}"


def test_dsr_bounded_in_unit_interval():
    r = _positive_returns(seed=8)
    for n in (2, 50, 5000):
        dsr = deflated_sharpe_ratio(r, n_trials=n, trial_sharpes_var=0.001)
        assert 0.0 <= dsr <= 1.0


def test_dsr_high_for_strong_sharpe_few_trials():
    rng = np.random.default_rng(9)
    r = rng.normal(0.001, 0.005, 1000)  # ~3 annualised Sharpe
    dsr = deflated_sharpe_ratio(r, n_trials=2, trial_sharpes_var=1e-6)
    assert dsr > 0.95


def test_dsr_degenerate_inputs():
    assert deflated_sharpe_ratio(np.ones(30), n_trials=10,
                                 trial_sharpes_var=0.01) == 0.0  # too short
    assert deflated_sharpe_ratio(np.zeros(200), n_trials=10,
                                 trial_sharpes_var=0.01) == 0.0  # zero std
