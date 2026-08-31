"""Causality tripwires for the TimesFM volatility bake-off.

The bake-off computes each baseline as a single pass over the whole return
history and then reads off the evaluation dates — a big speedup, but only valid
if every path is genuinely causal. These tests pin that: flipping the sign of
every return strictly after a cutoff must leave all predictions at or before
the cutoff bit-for-bit unchanged, exactly the tripwire `test_no_lookahead.py`
runs against the AMR engine.

None of this touches TimesFM itself (a 200M-parameter download); the model
adapter is exercised only for its import-error contract.
"""
import numpy as np
import pandas as pd
import pytest

from posterioralpha.research.tsfm import (
    TimesFMVolForecaster,
    bayes_path,
    build_contexts,
    causal_bias_correction,
    ewma_path,
    har_path,
    log_rv_series,
    newey_west_t,
    qlike,
    realized_target,
    rw_path,
)

N_DAYS = 1500
HORIZON = 21
CUTOFF = 1000


def _returns(seed: int = 7) -> np.ndarray:
    """Vol-clustering returns: a GARCH-ish path, so the vol models have signal."""
    rng = np.random.default_rng(seed)
    r = np.zeros(N_DAYS)
    v = 1e-4
    for i in range(N_DAYS):
        v = 2e-6 + 0.9 * v + 0.08 * (r[i - 1] ** 2 if i else 0.0)
        r[i] = rng.standard_normal() * np.sqrt(v)
    return r


@pytest.fixture(scope="module")
def rets():
    return _returns()


@pytest.fixture(scope="module")
def tampered(rets):
    """Same history, but every return after CUTOFF is sign-flipped and doubled."""
    r = rets.copy()
    r[CUTOFF + 1:] = -2.0 * r[CUTOFF + 1:]
    return r


def _eval_idxs():
    return np.arange(400, N_DAYS - HORIZON, 21)


# ------------------------------------------------------------- causality ---
@pytest.mark.parametrize("name", ["rw", "ewma94", "bayes"])
def test_filter_paths_are_causal(rets, tampered, name):
    fn = {"rw": lambda r: rw_path(r, HORIZON),
          "ewma94": ewma_path,
          "bayes": bayes_path}[name]
    a, b = fn(rets), fn(tampered)
    ok = np.isfinite(a[: CUTOFF + 1])
    assert ok.sum() > 100, "expected a warmed-up path before the cutoff"
    np.testing.assert_allclose(a[: CUTOFF + 1][ok], b[: CUTOFF + 1][ok], rtol=0, atol=0)


def test_har_refit_is_causal(rets, tampered):
    idxs = _eval_idxs()
    idxs = idxs[idxs <= CUTOFF]
    a = har_path(rets, HORIZON, idxs)
    b = har_path(tampered, HORIZON, idxs)
    np.testing.assert_allclose(a[idxs], b[idxs], rtol=0, atol=0)


def test_har_uses_only_closed_target_windows(rets):
    """A pair whose target window is still open must not enter the fit.

    Perturbing returns inside (t-horizon, t] changes the *features* at t, so we
    check the weaker, sharper property directly: the fit at t is unchanged when
    only returns strictly after t move.
    """
    idxs = np.array([800])
    r2 = rets.copy()
    r2[801:] *= -3.0
    assert har_path(rets, HORIZON, idxs)[800] == har_path(r2, HORIZON, idxs)[800]


def test_timesfm_contexts_are_causal(rets, tampered):
    idxs = _eval_idxs()
    idxs = idxs[idxs <= CUTOFF]
    for a, b in zip(build_contexts(rets, idxs, HORIZON, 256),
                    build_contexts(tampered, idxs, HORIZON, 256)):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_bayes_sentinel_cannot_leak(rets):
    """The appended sentinel return must never influence a forecast we read."""
    base = bayes_path(rets, sentinel=0.0)
    for s in (0.5, -0.25, 1e3):
        np.testing.assert_allclose(base, bayes_path(rets, sentinel=s), rtol=0, atol=0)


def test_bayes_path_matches_truncated_reruns(rets):
    """One pass over the full history == re-running at each evaluation date."""
    full = bayes_path(rets)
    for t in (500, 900, 1300):
        np.testing.assert_allclose(full[t], bayes_path(rets[: t + 1])[-1],
                                   rtol=1e-12, atol=0)


# ---------------------------------------------------------- construction ---
def test_target_is_the_next_horizon_window(rets):
    """target[t] must be the realized vol of days t+1..t+horizon and nothing else."""
    t = 700
    got = realized_target(rets, HORIZON)[t]
    want = np.log(np.sqrt(252 * np.mean(rets[t + 1: t + 1 + HORIZON] ** 2)))
    assert got == pytest.approx(want, rel=1e-12)


def test_rw_path_is_the_trailing_window(rets):
    t = 700
    want = np.log(np.sqrt(252 * np.mean(rets[t - HORIZON + 1: t + 1] ** 2)))
    assert rw_path(rets, HORIZON)[t] == pytest.approx(want, rel=1e-12)


def test_log_rv_warms_up_as_nan(rets):
    x = log_rv_series(rets, HORIZON)
    assert np.isnan(x[: HORIZON - 1]).all()
    assert np.isfinite(x[HORIZON - 1:]).all()


# --------------------------------------------------------------- scoring ---
def test_qlike_is_minimised_at_the_truth():
    truth = 0.20
    losses = {s: qlike(np.array([truth]), np.array([s])).item()
              for s in (0.10, 0.15, 0.20, 0.25, 0.40)}
    assert min(losses, key=losses.get) == truth
    assert losses[truth] == pytest.approx(0.0, abs=1e-12)
    # QLIKE punishes under-prediction harder than over-prediction
    assert losses[0.10] > losses[0.40]


def test_bias_correction_is_causal():
    """The correction at date i may only use residuals from dates before i-1."""
    dates = pd.to_datetime([f"2020-{m:02d}-01" for m in range(1, 13)] * 2).sort_values()
    panel = pd.DataFrame({
        "date": dates,
        "model": ["m"] * len(dates),
        "resid": np.linspace(0.0, 1.0, len(dates)),
    })
    corr = causal_bias_correction(panel, min_obs=2)
    assert corr.iloc[:4].eq(0.0).all(), "no correction before min_obs is reached"

    tampered = panel.copy()
    tampered.loc[tampered.index[-6:], "resid"] += 10.0
    corr2 = causal_bias_correction(tampered, min_obs=2)
    n_early = len(dates) - 8
    np.testing.assert_allclose(corr.iloc[:n_early], corr2.iloc[:n_early])


def test_newey_west_t_on_pure_noise_is_small():
    rng = np.random.default_rng(0)
    t = newey_west_t(rng.standard_normal(400))
    assert abs(t) < 3.0


def test_newey_west_t_detects_a_real_mean():
    rng = np.random.default_rng(0)
    t = newey_west_t(rng.standard_normal(400) + 0.5)
    assert t > 5.0


# --------------------------------------------------------------- adapter ---
def test_forecaster_reports_missing_dependency_clearly():
    """Without TimesFM installed the adapter must say how to install it."""
    f = TimesFMVolForecaster()
    assert f.max_context > 0 and f.batch_size > 0
    try:
        import timesfm  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match=r"\[tsfm\]"):
            f.load()
