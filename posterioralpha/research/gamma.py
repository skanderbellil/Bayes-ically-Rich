"""
Dealer gamma exposure (GEX) — options-flow research primitive (stage 2).

Market-maker gamma positioning shapes intraday equity dynamics:

  - **Long gamma** (positive GEX): dealers hedge by selling strength / buying
    weakness → volatility *suppression*, mean-reverting tape.
  - **Short gamma** (negative GEX): dealers hedge by buying strength / selling
    weakness → volatility *amplification*, trending / fragile tape.

The **zero-gamma flip** is the spot level where net dealer gamma crosses zero —
above it dealers are typically long gamma (stabilising), below it short gamma
(destabilising).

This module computes GEX from an options chain (Black-Scholes gamma × open
interest, calls long / puts short by the standard dealer convention).  A
*snapshot* is available from yfinance's current chain; a GEX **backtest**
needs historical open-interest-by-strike, which yfinance does not provide —
feed a historical chain into :func:`chain_gex` to extend this to a time series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_SQRT_2PI = np.sqrt(2.0 * np.pi)
_CONTRACT_MULTIPLIER = 100          # shares per option contract
_PCT_MOVE = 0.01                    # express GEX as $ gamma per 1% spot move


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def bs_gamma(S, K, T, sigma, r: float = 0.0) -> np.ndarray:
    """
    Black-Scholes gamma ∂²V/∂S² (same for calls and puts).

    Vectorised over array inputs; returns 0 where inputs are degenerate
    (non-positive T or sigma).
    """
    S = np.asarray(S, float); K = np.asarray(K, float)
    T = np.asarray(T, float); sigma = np.asarray(sigma, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = _norm_pdf(d1) / (S * sigma * np.sqrt(T))
    return np.where((T > 0) & (sigma > 0) & (S > 0) & (K > 0), gamma, 0.0)


def chain_gex(
    strikes: np.ndarray,
    open_interest: np.ndarray,
    ivs: np.ndarray,
    is_call: np.ndarray,
    spot: float,
    t_years: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """
    Signed dollar gamma exposure per contract row, in $ per 1% spot move.

    Dealer convention: long call gamma (+), short put gamma (−).  The total GEX
    is ``chain_gex(...).sum()``; per-strike aggregation is a groupby on strike.
    """
    gamma = bs_gamma(spot, strikes, t_years, ivs, r)
    sign = np.where(is_call, 1.0, -1.0)
    oi = np.nan_to_num(np.asarray(open_interest, float), nan=0.0)
    return sign * gamma * oi * _CONTRACT_MULTIPLIER * spot * spot * _PCT_MOVE


def zero_gamma_level(
    strikes: np.ndarray,
    open_interest: np.ndarray,
    ivs: np.ndarray,
    is_call: np.ndarray,
    spot: float,
    t_years: np.ndarray,
    r: float = 0.0,
    span: float = 0.20,
    n: int = 201,
) -> Optional[float]:
    """
    Spot level where net GEX crosses zero, searched over ±``span`` around spot.

    Returns the crossing nearest the current spot, or ``None`` if net gamma
    does not change sign within the band.
    """
    grid = np.linspace(spot * (1 - span), spot * (1 + span), n)
    net = np.array([
        chain_gex(strikes, open_interest, ivs, is_call, s, t_years, r).sum()
        for s in grid
    ])
    sign_change = np.where(np.diff(np.sign(net)) != 0)[0]
    if len(sign_change) == 0:
        return None
    # pick the crossing closest to spot; linear-interpolate within the bracket
    i = sign_change[np.argmin(np.abs(grid[sign_change] - spot))]
    x0, x1, y0, y1 = grid[i], grid[i + 1], net[i], net[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x0)


def gex_time_series(
    chains: pd.DataFrame,
    r: float = 0.0,
    date_col: str = "date",
    strike_col: str = "strike",
    oi_col: str = "open_interest",
    iv_col: str = "iv",
    type_col: str = "type",
    expiry_col: str = "expiry",
    spot_col: str = "spot",
) -> pd.Series:
    """
    Build a daily total-GEX series from a history of option chains.

    ``chains`` is long-format: one row per (date, expiry, strike, option type)
    with open interest, implied vol, and the underlying spot for that date.
    This is the bridge for free historical-options datasets (e.g. OptionsDX,
    DoltHub, CBOE EOD) — load them into this shape and get a backtestable GEX
    series that conditions equity returns/vol on the gamma regime.

    ``type_col`` values starting with 'c'/'C' are treated as calls.  Returns a
    Series indexed by date (total dealer GEX, $ per 1% move).
    """
    df = chains.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_t"] = (pd.to_datetime(df[expiry_col]) - df[date_col]).dt.days.clip(lower=1) / 365.0
    df["_call"] = df[type_col].astype(str).str.lower().str.startswith("c")

    out = {}
    for dt, g in df.groupby(date_col):
        spot = float(g[spot_col].iloc[0])
        gex = chain_gex(
            g[strike_col].values, g[oi_col].values, g[iv_col].values,
            g["_call"].values, spot, g["_t"].values, r,
        )
        out[dt] = float(gex.sum())
    s = pd.Series(out).sort_index()
    s.name = "gex"
    return s


@dataclass
class GammaSnapshot:
    spot: float
    total_gex: float           # $ gamma per 1% move (billions if /1e9)
    flip_level: Optional[float]
    by_strike: pd.DataFrame    # index=strike, columns=[gex, call_oi, put_oi]
    n_contracts: int

    @property
    def regime(self) -> str:
        return "long-gamma (vol-suppressing)" if self.total_gex >= 0 \
            else "short-gamma (vol-amplifying)"


def fetch_gamma_snapshot(
    ticker: str = "SPY",
    max_expiries: int = 12,
    r: float = 0.0,
    moneyness: float = 0.25,
) -> GammaSnapshot:
    """
    Compute a current dealer-GEX snapshot from yfinance's live options chain.

    Aggregates up to ``max_expiries`` expirations, keeps strikes within
    ``moneyness`` of spot, and returns total GEX, the zero-gamma flip level,
    and the per-strike profile.  Requires network access.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    spot = float(tk.fast_info["lastPrice"])
    exps = list(tk.options or [])[:max_expiries]
    if not exps:
        raise RuntimeError(f"no options chain available for {ticker}")

    now = pd.Timestamp.utcnow().normalize().tz_localize(None)
    rows = []
    for e in exps:
        t_years = max((pd.Timestamp(e) - now).days, 0) / 365.0
        if t_years <= 0:
            t_years = 1 / 365.0
        oc = tk.option_chain(e)
        for df, call in ((oc.calls, True), (oc.puts, False)):
            d = df[["strike", "openInterest", "impliedVolatility"]].copy()
            d["is_call"] = call
            d["t_years"] = t_years
            rows.append(d)

    chain = pd.concat(rows, ignore_index=True)
    chain = chain[(chain["strike"] >= spot * (1 - moneyness)) &
                  (chain["strike"] <= spot * (1 + moneyness))]
    chain["impliedVolatility"] = chain["impliedVolatility"].replace(0, np.nan)
    chain = chain.dropna(subset=["impliedVolatility"])

    gex = chain_gex(
        chain["strike"].values, chain["openInterest"].values,
        chain["impliedVolatility"].values, chain["is_call"].values,
        spot, chain["t_years"].values, r,
    )
    chain = chain.assign(gex=gex)

    by_strike = chain.groupby("strike").agg(
        gex=("gex", "sum"),
        call_oi=("openInterest", lambda s: s[chain.loc[s.index, "is_call"]].sum()),
        put_oi=("openInterest", lambda s: s[~chain.loc[s.index, "is_call"]].sum()),
    )
    flip = zero_gamma_level(
        chain["strike"].values, chain["openInterest"].values,
        chain["impliedVolatility"].values, chain["is_call"].values,
        spot, chain["t_years"].values, r,
    )
    return GammaSnapshot(
        spot=spot, total_gex=float(gex.sum()), flip_level=flip,
        by_strike=by_strike, n_contracts=len(chain),
    )
