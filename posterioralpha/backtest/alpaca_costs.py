"""Alpaca retail-broker cost model (2026) for deployability stress-tests.

Grounded in Alpaca's published 2026 terms:
  - US equities/ETFs trade COMMISSION-FREE ($0 per share, $0 per trade).
  - Regulatory pass-through on SELLS only: SEC Section 31 fee + FINRA TAF.
    Tiny — modelled as ~0.3 bps of sell notional.
  - SHORTING is ETB-only: easy-to-borrow names borrow at $0/yr; hard-to-borrow
    (HTB) names are simply *not shortable* on the Trading API. Small/micro/nano
    caps are predominantly HTB → treat as not shortable.
  - Margin (leverage > 1x) accrues ~6.25%/yr (non-elite) on the borrowed notional.
  - PDT rule retired 2026-06-04: no $25k minimum, no day-trade count limit
    (Reg-T 2x intraday still applies). Holding-period strategies are unaffected.

The binding *real* cost for a commission-free retail trader is therefore the
bid-ask half-spread paid on every entry/exit (scaled by turnover), plus margin
interest on any leverage, plus the hard constraint that you cannot short HTB.

Spread estimates below are conservative current half-spreads (one-way, bps);
round-trip = 2x. They are tighter than the legacy ``pead.costs`` desk estimates
because retail marketable orders on liquid names cross a single NBBO half-spread.
"""
from __future__ import annotations

from dataclasses import dataclass

# One-way (half-spread) execution cost in basis points, by liquidity tier.
# Round-trip entry+exit = 2x these.
HALF_SPREAD_BPS = {
    "ETF":       1.0,   # SPY/TLT/GLD/EEM/VNQ — penny-wide, deep books
    "Mega Cap":  2.0,
    "Large Cap": 4.0,
    "Mid Cap":   10.0,
    "Small Cap": 25.0,
    "Micro Cap": 60.0,
    "Nano Cap":  120.0,
}

# Easy-to-borrow (shortable at $0) vs hard-to-borrow (not shortable on Alpaca).
SHORTABLE_ETB = {"ETF", "Mega Cap", "Large Cap"}      # reliably ETB
NOT_SHORTABLE = {"Mid Cap", "Small Cap", "Micro Cap", "Nano Cap"}  # mostly HTB

COMMISSION_BPS = 0.0          # Alpaca: commission-free US equities/ETFs
REG_SELL_FEE_BPS = 0.3        # SEC + FINRA TAF pass-through, sells only
MARGIN_ANNUAL = 0.0625        # non-elite USD margin rate (Elite: 0.0475)


@dataclass(frozen=True)
class AlpacaCosts:
    """Retail Alpaca cost parameters (all rates in decimal, bps where noted)."""

    commission_bps: float = COMMISSION_BPS
    reg_sell_fee_bps: float = REG_SELL_FEE_BPS
    margin_annual: float = MARGIN_ANNUAL

    def roundtrip_cost(self, tier: str) -> float:
        """Round-trip execution cost (decimal) for one entry+exit in `tier`.

        = 2 x half-spread + 2 x commission + 1 x reg sell fee.
        """
        half = HALF_SPREAD_BPS.get(tier, HALF_SPREAD_BPS["Mid Cap"])
        bps = 2 * half + 2 * self.commission_bps + self.reg_sell_fee_bps
        return bps / 1e4

    def oneway_cost(self, tier: str, is_sell: bool = False) -> float:
        """One-way execution cost (decimal) for a single fill in `tier`."""
        half = HALF_SPREAD_BPS.get(tier, HALF_SPREAD_BPS["Mid Cap"])
        bps = half + self.commission_bps + (self.reg_sell_fee_bps if is_sell else 0.0)
        return bps / 1e4

    def turnover_cost(self, turnover: float, tier: str = "ETF") -> float:
        """Execution cost (decimal) for a given fractional turnover (sum |Δw|).

        `turnover` is total one-way notional traded as a fraction of equity, so
        cost = turnover x (half-spread + commission). Reg sell fee applied to
        the selling half (turnover/2).
        """
        half = HALF_SPREAD_BPS.get(tier, HALF_SPREAD_BPS["ETF"])
        exec_bps = turnover * (half + self.commission_bps)
        sell_bps = (turnover / 2.0) * self.reg_sell_fee_bps
        return (exec_bps + sell_bps) / 1e4

    def margin_drag(self, gross_exposure: float, days: int = 1) -> float:
        """Daily-or-period margin interest (decimal) on leverage above 1.0x.

        gross_exposure = sum of |position weights|. Only the part above 1.0 is
        financed on margin.
        """
        borrowed = max(gross_exposure - 1.0, 0.0)
        return borrowed * self.margin_annual * (days / 252.0)

    @staticmethod
    def is_shortable(tier: str) -> bool:
        """Whether a name in this liquidity tier can be shorted on Alpaca (ETB)."""
        return tier in SHORTABLE_ETB


def oneway_bps(tier: str) -> float:
    """Convenience: one-way half-spread in bps for a tier (for tc= wiring)."""
    return HALF_SPREAD_BPS.get(tier, HALF_SPREAD_BPS["Mid Cap"])
