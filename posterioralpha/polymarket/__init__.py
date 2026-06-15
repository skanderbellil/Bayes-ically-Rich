"""Polymarket cross-market momentum — a prediction-market strategy module.

A self-contained domain subpackage (like ``pead`` and ``council``) that spans all
four pipeline stages on a fundamentally different asset class: **binary prediction
markets** rather than the time-series equity/ETF portfolios elsewhere in the repo.
Outcome-token prices are implied probabilities, edge comes from probability
*mispricing/drift* rather than mean-variance, and the natural coordinate is
log-odds.

  data      → ``fetch``    : live Gamma (metadata) + CLOB (price history) panel
  research  → ``signals``  : log-odds momentum, cross-sectional z-score, Bayes shrink
  backtest  → ``backtest`` : no-lookahead, cost-aware long/short engine
  validation→ reuse ``posterioralpha.validation.compute_metrics``
"""
from .backtest import PMParams, PMResult, run_polymarket_momentum
from .fetch import build_price_panel, fetch_markets, fetch_token_history
from .signals import (
    bayesian_shrink,
    cross_sectional_score,
    from_logodds,
    logodds_momentum,
    to_logodds,
)

__all__ = [
    "build_price_panel",
    "fetch_markets",
    "fetch_token_history",
    "to_logodds",
    "from_logodds",
    "logodds_momentum",
    "bayesian_shrink",
    "cross_sectional_score",
    "PMParams",
    "PMResult",
    "run_polymarket_momentum",
]
