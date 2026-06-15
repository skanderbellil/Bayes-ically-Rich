"""Polymarket cross-market momentum — a prediction-market strategy module.

A self-contained domain subpackage (like ``pead`` and ``council``) that spans all
four pipeline stages on a fundamentally different asset class: **binary prediction
markets** rather than the time-series equity/ETF portfolios elsewhere in the repo.
Outcome-token prices are implied probabilities, edge comes from probability
*mispricing/drift* rather than mean-variance, and the natural coordinate is
log-odds.

  data      → ``fetch``      : live Gamma (metadata) + CLOB (price history + book) panel
  research  → ``signals``    : log-odds momentum, cross-sectional z-score, Bayes shrink
              ``volatility`` : upside/downside semivol, vol-skew, sharp-move detection
  events    → ``events``     : event-study table (sharp move → actual outcome), BOCPD-tagged
  backtest  → ``backtest``   : no-lookahead, cost-aware long/short engine
  validation→ reuse ``posterioralpha.validation.compute_metrics``
"""
from .backtest import PMParams, PMResult, run_polymarket_momentum
from .events import band_matched_calibration, bracket_outcomes, build_event_table
from .fetch import (
    build_price_panel,
    fetch_markets,
    fetch_order_book,
    fetch_token_history,
    order_book_features,
    resolution_outcomes,
)
from .signals import (
    bayesian_shrink,
    cross_sectional_score,
    from_logodds,
    logodds_momentum,
    to_logodds,
)
from .trades import TradeParams, TradeResult, simulate_event_trades, summarize_trades
from .volatility import (
    downside_vol,
    logodds_returns,
    realized_vol,
    sharp_up,
    upside_vol,
    vol_skew,
)

__all__ = [
    # data
    "build_price_panel",
    "fetch_markets",
    "fetch_token_history",
    "fetch_order_book",
    "order_book_features",
    "resolution_outcomes",
    # research: momentum signals
    "to_logodds",
    "from_logodds",
    "logodds_momentum",
    "bayesian_shrink",
    "cross_sectional_score",
    # research: volatility
    "logodds_returns",
    "realized_vol",
    "upside_vol",
    "downside_vol",
    "vol_skew",
    "sharp_up",
    # events
    "build_event_table",
    "bracket_outcomes",
    "band_matched_calibration",
    # backtest
    "PMParams",
    "PMResult",
    "run_polymarket_momentum",
    # event-trade harvest
    "TradeParams",
    "TradeResult",
    "simulate_event_trades",
    "summarize_trades",
]
