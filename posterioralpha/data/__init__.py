"""Stage 1 — data access: live download, synthetic universe, bundled loaders."""
from posterioralpha.data.loaders import (
    load_portfolio_prices,
    load_portfolio_returns,
    load_sp500_prices,
)
from posterioralpha.data.market import (
    download_returns,
    get_sp500_tickers,
    sample_tickers,
)
from posterioralpha.data.synthetic import expand_universe

__all__ = [
    "download_returns",
    "get_sp500_tickers",
    "sample_tickers",
    "expand_universe",
    "load_portfolio_prices",
    "load_portfolio_returns",
    "load_sp500_prices",
]
