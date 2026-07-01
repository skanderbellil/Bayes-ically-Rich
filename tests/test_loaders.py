"""Sanity checks for the small bundled-dataset loaders (fully offline).

Only datasets under ~5 MB are exercised so the suite stays fast; loaders whose
backing file is missing or too large are skipped, not failed.
"""
from pathlib import Path

import pandas as pd
import pytest

from posterioralpha.data import loaders

MAX_BYTES = 5 * 1024 * 1024  # 5 MB


def _skip_unless_small(path: Path):
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    if path.stat().st_size > MAX_BYTES:
        pytest.skip(f"{path.name} larger than 5 MB — skipped for speed")


SMALL_LOADERS = [
    pytest.param(loaders.load_portfolio_prices, loaders.PORTFOLIO_CSV,
                 id="portfolio_prices"),
    pytest.param(loaders.load_portfolio_returns, loaders.PORTFOLIO_CSV,
                 id="portfolio_returns"),
    pytest.param(loaders.load_sp500_prices, loaders.SP500_CSV,
                 id="sp500_prices"),
    pytest.param(loaders.load_net_liquidity, loaders.NET_LIQUIDITY_CSV,
                 id="net_liquidity"),
    pytest.param(loaders.load_fred_macro, loaders.FRED_MACRO_CSV,
                 id="fred_macro"),
]


@pytest.mark.parametrize("loader,path", SMALL_LOADERS)
def test_loader_basic_invariants(loader, path):
    _skip_unless_small(path)
    df = loader()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty, "loaded DataFrame is empty"
    assert df.shape[1] >= 1

    # Date index: datetime, strictly increasing (sorted + no duplicates)
    assert isinstance(df.index, pd.DatetimeIndex), f"index is {type(df.index)}"
    assert df.index.is_monotonic_increasing, "index not sorted ascending"
    assert df.index.is_unique, "index has duplicate dates"

    # No column may be entirely NaN
    all_nan = [c for c in df.columns if df[c].isna().all()]
    assert not all_nan, f"all-NaN columns: {all_nan}"


def test_portfolio_returns_are_plausible_daily_returns():
    _skip_unless_small(loaders.PORTFOLIO_CSV)
    rets = loaders.load_portfolio_returns()
    assert not rets.isna().any().any(), "returns contain NaNs after dropna"
    # Daily ETF arithmetic returns: > -100% always, and |r| < 60% for SPY/TLT/
    # GLD/EEM/VNQ over any realistic sample.
    assert (rets > -1.0).all().all()
    assert rets.abs().max().max() < 0.6


def test_portfolio_prices_are_positive():
    _skip_unless_small(loaders.PORTFOLIO_CSV)
    prices = loaders.load_portfolio_prices()
    assert (prices.dropna() > 0).all().all(), "adjusted closes must be positive"
