"""Cost models for PEAD strategy: bid-ask, short borrow, market impact.

Real trading costs by market-cap bucket (empirical estimates from broker surveys):

Market Cap       | Bid-Ask Spread | Short Borrow | Market Impact @ $1B |
Mega Cap         | 0.05%          | 0.5–1.0%     | 0.05–0.10%
Large Cap        | 0.10%          | 1.0–2.0%     | 0.10–0.15%
Mid Cap          | 0.20%          | 2.0–3.0%     | 0.15–0.30%
Small Cap        | 0.30–0.50%     | 3.0–5.0%     | 0.30–0.50%
Micro Cap        | 0.50–1.00%     | 5.0–8.0%     | 0.50–1.00%
Nano Cap         | 1.00–2.00%     | 8.0–12.0%    | 1.00–2.00%

These are conservative estimates; actual costs depend on:
  - Execution quality (VWAP, TWAP, POI, etc.)
  - Prime broker relationships (rebates for large firms)
  - Market conditions (liquidity, volatility, time of day)
  - Portfolio construction (simultaneous rebalancing, single vs multi-leg)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostModel:
    """Cost parameters by market cap bucket."""

    bucket: str
    bid_ask_pct: float  # One-way, so round-trip = 2x
    short_borrow_annual_pct: float
    market_impact_annual_pct: float = 0.0  # Scales with AUM


def get_cost_models() -> dict[str, CostModel]:
    """Return cost models for each cap bucket (conservative estimates)."""
    return {
        "Mega Cap": CostModel(
            bucket="Mega Cap",
            bid_ask_pct=0.05,
            short_borrow_annual_pct=0.75,
            market_impact_annual_pct=0.0,
        ),
        "Large Cap": CostModel(
            bucket="Large Cap",
            bid_ask_pct=0.10,
            short_borrow_annual_pct=1.50,
            market_impact_annual_pct=0.0,
        ),
        "Mid Cap": CostModel(
            bucket="Mid Cap",
            bid_ask_pct=0.20,
            short_borrow_annual_pct=2.50,
            market_impact_annual_pct=0.0,
        ),
        "Small Cap": CostModel(
            bucket="Small Cap",
            bid_ask_pct=0.40,
            short_borrow_annual_pct=4.00,
            market_impact_annual_pct=0.0,
        ),
        "Micro Cap": CostModel(
            bucket="Micro Cap",
            bid_ask_pct=0.75,
            short_borrow_annual_pct=6.50,
            market_impact_annual_pct=0.0,
        ),
        "Nano Cap": CostModel(
            bucket="Nano Cap",
            bid_ask_pct=1.50,
            short_borrow_annual_pct=10.00,
            market_impact_annual_pct=0.0,
        ),
    }


def compute_quarterly_cost(cost_model: CostModel, annual_ls_return: float) -> float:
    """Compute total quarterly trading cost (in basis points).

    Costs:
      1. Bid-ask: round-trip open + close = 4 × one-way spread (2 legs × 2 ways)
      2. Short borrow: paid annually, so 1/4 per quarter
      3. Commissions: assumed 1–2 bps per side (negligible)

    Args:
      cost_model: Cost parameters for the bucket
      annual_ls_return: Annual LS return (for slippage scaling)

    Returns:
      Total quarterly cost in basis points (to subtract from quarterly return)
    """
    # Bid-ask: round-trip = 2 sides × 2 ways (entry + exit) × one-way spread
    bid_ask_bps = 2 * 2 * cost_model.bid_ask_pct * 100  # Convert to bps

    # Short borrow: quarterly = annual / 4
    borrow_bps = cost_model.short_borrow_annual_pct * 100 / 4

    # Market impact (scales with AUM, not relevant for this backtest)
    impact_bps = 0

    total_bps = bid_ask_bps + borrow_bps + impact_bps
    return total_bps


def apply_costs_to_returns(
    quarterly_returns: list[float],
    cap_bucket: str,
) -> tuple[list[float], list[float]]:
    """Apply costs to quarterly LS returns.

    Args:
      quarterly_returns: List of quarterly LS returns (in decimal form, e.g., 0.02 = 2%)
      cap_bucket: Market cap bucket name

    Returns:
      Tuple of (gross_returns, net_returns)
    """
    cost_models = get_cost_models()
    cost_model = cost_models.get(cap_bucket, cost_models["Mid Cap"])

    quarterly_cost_bps = compute_quarterly_cost(
        cost_model, np.mean(quarterly_returns) * 100 if quarterly_returns else 0
    )
    quarterly_cost_decimal = quarterly_cost_bps / 10_000

    net_returns = [r - quarterly_cost_decimal for r in quarterly_returns]

    return quarterly_returns, net_returns


def capacity_constraint(
    aum: float,
    cap_bucket: str,
    annual_trading_volume_pct: float = 2.0,  # % of market cap traded per year
) -> dict:
    """Estimate market capacity and scalability limits.

    Args:
      aum: Assets under management (dollars)
      cap_bucket: Market cap bucket name
      annual_trading_volume_pct: Annual trading volume as % of bucket market cap

    Returns:
      Dict with capacity metrics
    """
    # Approximate market caps (2026 estimates)
    bucket_market_caps = {
        "Mega Cap": 50e12,  # $50T (top 10 companies)
        "Large Cap": 15e12,  # $15T
        "Mid Cap": 5e12,  # $5T
        "Small Cap": 2e12,  # $2T
        "Micro Cap": 0.5e12,  # $500B
        "Nano Cap": 0.1e12,  # $100B
    }

    market_cap = bucket_market_caps.get(cap_bucket, 2e12)
    annual_bucket_volume = market_cap * (annual_trading_volume_pct / 100)
    annual_capacity = annual_bucket_volume  # Max annual trading without >2% market share

    years_to_capacity = aum / (annual_capacity / 4) if annual_capacity > 0 else float("inf")

    return {
        "bucket": cap_bucket,
        "aum": aum,
        "market_cap": market_cap,
        "annual_bucket_volume": annual_bucket_volume,
        "annual_capacity": annual_capacity,
        "quarterly_capacity": annual_capacity / 4,
        "max_aum_no_impact": annual_capacity / 4,
        "utilization_at_aum": (aum / (annual_capacity / 4)) if annual_capacity > 0 else float("inf"),
    }
