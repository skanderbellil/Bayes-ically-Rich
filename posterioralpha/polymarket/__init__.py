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
from .categorize import classify_markets, market_category
from .crossoutcome import (
    coupling_summary,
    event_panels,
    rank_calibration,
    redistribution_drift,
    relative_value_reversion,
)
from .efficiency import brier_skill, build_brier_frame
from .events import band_matched_calibration, bracket_outcomes, build_event_table
from .fieldshape import bucket_by_shape, constituent_rows, field_shape_events
from .papertrade import load_ledger, update_ledger
from . import smartflow_papertrade
from .fetch import (
    build_price_panel,
    fetch_market_resolution,
    fetch_markets,
    fetch_order_book,
    fetch_token_history,
    order_book_features,
    resolution_outcomes,
    token_resolution,
)
from .signals import (
    bayesian_shrink,
    cross_sectional_score,
    from_logodds,
    logodds_momentum,
    to_logodds,
)
from .smartmoney import (
    FollowParams,
    FollowResult,
    MMScreen,
    cohort_token_notional,
    mm_fingerprint,
    run_smart_money_follow,
    screen_market_makers,
    select_universe,
    trader_equity_curve,
)
from .behavior import (
    SpecialistParams,
    SpecialistResult,
    run_specialist_follow,
    stack_trades,
    token_domains,
    trader_token_pnl,
)
from .flow import (
    FlowParams,
    FlowResult,
    flow_signal_panels,
    liquidity_spread,
    run_flow_book,
)
from .tradequality import (
    build_trade_features,
    summarize_quality,
    timing_persistence,
    wallet_lead_scores,
)
from .traders import (
    build_token_panel,
    candidate_universe,
    fetch_leaderboard,
    fetch_many_traders,
    fetch_trader_trades,
    fetch_volume_wallets,
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
from .weather import (
    Bucket,
    ForecastConsensus,
    bucket_prob,
    compute_edges,
    fetch_token_window,
    fetch_weather_event,
    parse_bucket,
    point_in_time_forecast,
    realized_tmax,
)

__all__ = [
    # data
    "build_price_panel",
    "fetch_markets",
    "fetch_token_history",
    "fetch_order_book",
    "order_book_features",
    "resolution_outcomes",
    "fetch_market_resolution",
    "token_resolution",
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
    # categorisation
    "market_category",
    "classify_markets",
    # efficiency
    "build_brier_frame",
    "brier_skill",
    # field shape (peaked vs flat) calibration
    "field_shape_events",
    "constituent_rows",
    "bucket_by_shape",
    # paper-trade tracker
    "load_ledger",
    "update_ledger",
    # weather edge (London max-temperature, forecast vs market)
    "Bucket",
    "ForecastConsensus",
    "parse_bucket",
    "bucket_prob",
    "point_in_time_forecast",
    "realized_tmax",
    "fetch_weather_event",
    "fetch_token_window",
    "compute_edges",
    # cross-outcome structure
    "event_panels",
    "coupling_summary",
    "rank_calibration",
    "redistribution_drift",
    "relative_value_reversion",
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
    # smart-money: trader-level data
    "fetch_leaderboard",
    "candidate_universe",
    "fetch_volume_wallets",
    "fetch_trader_trades",
    "fetch_many_traders",
    "build_token_panel",
    # smart-money: follow engine
    "MMScreen",
    "mm_fingerprint",
    "cohort_token_notional",
    "select_universe",
    "trader_equity_curve",
    "FollowParams",
    "FollowResult",
    "run_smart_money_follow",
    "screen_market_makers",
    # behaviour: shared primitives + domain-specialist follow
    "stack_trades",
    "token_domains",
    "trader_token_pnl",
    "SpecialistParams",
    "SpecialistResult",
    "run_specialist_follow",
    # behaviour: order-flow signals (breadth / imbalance), beta-neutral
    "flow_signal_panels",
    "run_flow_book",
    "liquidity_spread",
    "FlowParams",
    "FlowResult",
    # behaviour: trade-level quality tells + timing persistence
    "build_trade_features",
    "summarize_quality",
    "wallet_lead_scores",
    "timing_persistence",
]
