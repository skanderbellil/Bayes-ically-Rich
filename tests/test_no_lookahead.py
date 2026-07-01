"""Lookahead tripwire for the AMR backtest engine.

Decisions made at rebalance date t must depend only on data at or before t.
We run the cheap "inv_vol" strategy on synthetic data, then flip the sign of
every return strictly AFTER a cutoff date and rerun: all weight/leverage rows
at or before the cutoff must be bit-for-bit reproducible.
"""
import numpy as np
import pandas as pd
import pytest

from posterioralpha.backtest import run_amr_backtest

N_DAYS = 1200
N_ASSETS = 4


def _synthetic_returns(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", periods=N_DAYS)
    cols = ["SPY", "TLT", "GLD", "EEM"][:N_ASSETS]
    # mildly correlated returns with distinct vols so inv_vol is non-trivial
    vols = np.array([0.010, 0.006, 0.008, 0.013])[:N_ASSETS]
    raw = rng.standard_normal((N_DAYS, N_ASSETS))
    common = rng.standard_normal((N_DAYS, 1))
    data = 0.0003 + (0.7 * raw + 0.3 * common) * vols
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture(scope="module")
def base_result():
    returns = _synthetic_returns()
    res = run_amr_backtest(returns, strategy="inv_vol")
    return returns, res


def test_backtest_produces_output(base_result):
    _, res = base_result
    assert len(res.weights) > 50, "expected many weekly rebalances"
    assert len(res.returns) > 500
    assert res.returns.notna().all()
    # unscaled inv_vol weights: long-only, fully invested
    w = res.weights.values
    assert (w >= -1e-12).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-8)


def test_weights_before_cutoff_unaffected_by_future_data(base_result):
    returns, res = base_result

    # cutoff: a rebalance date roughly 2/3 through the weight history
    cutoff = res.weights.index[int(len(res.weights) * 2 / 3)]

    # corrupt ONLY the future: flip the sign of every return after the cutoff
    perturbed = returns.copy()
    future = perturbed.index > cutoff
    assert future.sum() > 200, "cutoff too late — perturbation would be trivial"
    perturbed.loc[future] *= -1.0

    res2 = run_amr_backtest(perturbed, strategy="inv_vol")

    w1 = res.weights.loc[:cutoff]
    w2 = res2.weights.loc[:cutoff]
    assert len(w1) > 20
    assert w1.index.equals(w2.index), "rebalance grid changed before cutoff"
    np.testing.assert_allclose(
        w1.values, w2.values, rtol=0, atol=1e-12,
        err_msg="weights at/before cutoff changed when only FUTURE returns "
                "were perturbed — lookahead bias in the AMR engine",
    )

    # the vol-target leverage decision at t must also be future-independent
    lev1 = res.leverage.loc[:cutoff]
    lev2 = res2.leverage.loc[:cutoff]
    np.testing.assert_allclose(lev1.values, lev2.values, rtol=0, atol=1e-12)

    # and the future must actually have mattered: weights after the cutoff differ
    tail1 = res.weights.loc[cutoff:].iloc[1:]
    tail2 = res2.weights.loc[cutoff:].iloc[1:]
    assert not np.allclose(tail1.values, tail2.values), (
        "perturbing future returns changed nothing — test has no power"
    )


def test_returns_before_cutoff_unaffected_by_future_data(base_result):
    """Realised portfolio returns up to the cutoff are also reproducible."""
    returns, res = base_result
    cutoff = res.weights.index[int(len(res.weights) * 2 / 3)]

    perturbed = returns.copy()
    perturbed.loc[perturbed.index > cutoff] *= -1.0
    res2 = run_amr_backtest(perturbed, strategy="inv_vol")

    r1 = res.returns.loc[:cutoff]
    r2 = res2.returns.loc[:cutoff]
    assert len(r1) > 200
    assert r1.index.equals(r2.index)
    np.testing.assert_allclose(r1.values, r2.values, rtol=0, atol=1e-12)
