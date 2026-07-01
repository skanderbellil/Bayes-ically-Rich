"""Closed-form invariants for posterioralpha.validation.metrics.compute_metrics."""
import numpy as np
import pandas as pd
import pytest

from posterioralpha.validation.metrics import compute_metrics

_ANN = 252


def _series(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx)


class TestConstantReturn:
    """A constant daily return has a closed-form CAGR and zero volatility."""

    R = 0.0004        # daily return
    N = 504           # exactly 2 years of trading days

    @pytest.fixture(scope="class")
    def metrics(self):
        return compute_metrics(_series([self.R] * self.N), rf=0.04)

    def test_cagr_closed_form(self, metrics):
        expected = (1.0 + self.R) ** _ANN - 1.0
        assert metrics["CAGR"] == pytest.approx(expected, rel=1e-9)

    def test_total_return_closed_form(self, metrics):
        expected = (1.0 + self.R) ** self.N - 1.0
        assert metrics["Total Return"] == pytest.approx(expected, rel=1e-9)

    def test_zero_volatility(self, metrics):
        assert metrics["Volatility"] == pytest.approx(0.0, abs=1e-12)

    def test_win_rate_is_one(self, metrics):
        assert metrics["Win Rate"] == 1.0

    def test_no_drawdown(self, metrics):
        assert metrics["Max DD"] == pytest.approx(0.0, abs=1e-9)


def test_sharpe_closed_form_alternating_series():
    """Alternating a±b returns: mean and std are known exactly, so the Sharpe
    (with the module's rf convention and 1e-9 guard) is hand-computable."""
    a, b, n, rf = 0.001, 0.01, 504, 0.04
    r = _series([a + b, a - b] * (n // 2))
    m = compute_metrics(r, rf=rf)

    mean_excess = a - rf / _ANN
    # sample std (ddof=1) of an alternating ±b series of even length n
    std_excess = b * np.sqrt(n / (n - 1))
    expected_sharpe = (mean_excess * _ANN) / (std_excess * np.sqrt(_ANN) + 1e-9)
    assert m["Sharpe"] == pytest.approx(expected_sharpe, rel=1e-9)

    expected_vol = std_excess * np.sqrt(_ANN)
    assert m["Volatility"] == pytest.approx(expected_vol, rel=1e-9)


def test_max_drawdown_known_value():
    """+10% then -50%: peak 1.1 -> trough 0.55, so max drawdown is -50%."""
    m = compute_metrics(_series([0.10, -0.50]))
    assert m["Max DD"] == pytest.approx(-0.5, abs=1e-6)


def test_max_drawdown_multi_leg():
    """Drawdown must be measured from the running peak, not the start."""
    # cum: 1.2 -> 0.96 -> 1.248 -> 0.7488 ; peak 1.248 -> trough 0.7488 = -40%
    m = compute_metrics(_series([0.20, -0.20, 0.30, -0.40]))
    assert m["Max DD"] == pytest.approx(-0.4, abs=1e-6)


def test_zero_return_series_does_not_blow_up():
    m = compute_metrics(_series([0.0] * 300), rf=0.04)
    assert m, "expected non-empty metrics dict"
    for key, val in m.items():
        assert np.isfinite(val), f"{key} is not finite: {val}"
    assert m["Total Return"] == pytest.approx(0.0, abs=1e-12)
    assert m["CAGR"] == pytest.approx(0.0, abs=1e-12)
    assert m["Max DD"] == pytest.approx(0.0, abs=1e-12)
    assert m["Win Rate"] == 0.0


def test_empty_series_returns_empty_dict():
    assert compute_metrics(pd.Series([], dtype=float)) == {}


def test_all_nan_series_returns_empty_dict():
    assert compute_metrics(_series([np.nan] * 10)) == {}


def test_benchmark_beta_of_self_is_one():
    """Regressing a series on itself must give beta ~ 1 and alpha ~ 0."""
    rng = np.random.default_rng(7)
    r = _series(rng.normal(0.0005, 0.01, 500))
    m = compute_metrics(r, benchmark=r, rf=0.04)
    # 1e-4 tolerance: compute_metrics adds a 1e-9 guard to the covariance
    # denominator, which perturbs beta by ~1e-5 at daily-return magnitudes.
    assert m["Beta"] == pytest.approx(1.0, abs=1e-4)
    assert m["Alpha"] == pytest.approx(0.0, abs=1e-4)
