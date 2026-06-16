"""Order-flow signals from trader behaviour — breadth & imbalance, beta-neutral.

The smart-money studies followed *who* was trading; this asks whether the
*aggregate flow itself* — independent of any one wallet's identity — predicts the
cross-section of next-day moves. Two signals, both formed from the stacked fills
of the (non-MM) candidate pool over a trailing window ending at *t*:

  * **breadth** — the number of *distinct* wallets net-buying a token minus those
    net-selling it. A token many independent traders are piling into is a
    consensus signal that a single whale's size can't fake. (The first probe:
    ≥3 distinct buyers → roughly double the next-day drift of thin-breadth names.)
  * **imbalance** (OFI) — net signed dollars (buys − sells). The classic
    order-flow-imbalance pressure signal, in dollar space.

The catch on this universe is a pervasive **favorite-drift beta**: priced tokens
skew toward favorites that resolved Yes, so *any* long book looks good (a naive
long-all is already ~0.5 Sharpe). To measure signal rather than beta, the engine
trades the signals **cross-sectionally and dollar-neutral** — long the top
fraction, short the bottom fraction by the day's cross-sectional z-score — so the
common drift cancels and only the *ranking* power of the flow is paid out.

All signals are causal (trailing window up to *t*, book held into *t+1*), marked
to the CLOB mid on $1 gross, with a turnover cost — the same convention as the
rest of the subpackage.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .behavior import stack_trades
from .signals import cross_sectional_score

logger = logging.getLogger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Signal panels
# ---------------------------------------------------------------------------

def flow_signal_panels(
    traders: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    is_mm: dict[str, bool] | None = None,
    lookback: int = 7,
) -> dict[str, pd.DataFrame]:
    """Trailing-window breadth & imbalance panels aligned to the price grid.

    Returns ``{"breadth": df, "imbalance": df}`` (dates × tokens). Each value at
    date *t* aggregates the pool's fills over the trailing ``lookback`` days
    ending at *t* — strictly causal, so it can trade *t → t+1*. Market makers are
    excluded if an ``is_mm`` map is given (their churn is not "flow").
    """
    is_mm = is_mm or {}
    tokens = list(price_panel.columns)
    dates = price_panel.index
    keep = {w: t for w, t in traders.items() if not is_mm.get(w, False)}
    T = stack_trades(keep, tokens)
    if T.empty:
        z = pd.DataFrame(0.0, index=dates, columns=tokens)
        return {"breadth": z, "imbalance": z.copy()}

    T["buy"] = T["side"].values == "BUY"
    # daily per-token: distinct buyers, distinct sellers, net signed $
    g = T.groupby(["day", "asset"])
    buyers  = g.apply(lambda d: d.loc[d["buy"], "wallet"].nunique()).unstack(fill_value=0.0)
    sellers = g.apply(lambda d: d.loc[~d["buy"], "wallet"].nunique()).unstack(fill_value=0.0)
    net_usd = g["signed_usd"].sum().unstack(fill_value=0.0)

    def _roll(daily: pd.DataFrame) -> pd.DataFrame:
        d = daily.reindex(columns=tokens, fill_value=0.0).reindex(dates, fill_value=0.0)
        return d.rolling(lookback, min_periods=1).sum()

    # informed flow: dollars weighted by *pay-up* urgency (the dominant tell in
    # TRADE_QUALITY) — only fills that crossed the mid in their favour contribute,
    # signed by side. A consensus of aggressive, urgent fills, not raw volume.
    mid = price_panel.ffill()
    midser = mid.stack(); midser.index = midser.index.set_names(["day", "asset"])
    T = T.merge(midser.rename("mid"), on=["day", "asset"], how="left").dropna(subset=["mid"])
    prem = np.where(T["buy"].values, T["price"].values - T["mid"].values,
                    T["mid"].values - T["price"].values)
    T["informed"] = np.sign(T["signed_usd"].values) * T["usdcSize"].values * np.maximum(prem, 0.0)
    inf_usd = T.groupby(["day", "asset"])["informed"].sum().unstack(fill_value=0.0)

    breadth   = _roll(buyers) - _roll(sellers)     # net distinct-trader consensus
    imbalance = _roll(net_usd)                      # net signed dollars
    informed  = _roll(inf_usd)                      # pay-up-weighted directional flow
    # only meaningful where the token is live (priced) at t
    live = price_panel.notna()
    return {"breadth": breadth.where(live), "imbalance": imbalance.where(live),
            "informed": informed.where(live)}


# ---------------------------------------------------------------------------
# Cross-sectional, dollar-neutral backtest
# ---------------------------------------------------------------------------

@dataclass
class FlowParams:
    signal: str = "breadth"     # breadth | imbalance
    lookback: int = 7
    holding: int = 1            # rebalance every N days
    top_frac: float = 0.2       # fraction per long / short leg
    gross: float = 1.0
    min_names: int = 8          # need this many live tokens to trade a date
    cost: float = 0.005
    contrarian: bool = False    # fade the flow instead of following it


@dataclass
class FlowResult:
    returns: pd.Series
    gross_returns: pd.Series
    turnover: pd.Series
    quantile_drift: pd.Series   # mean next-day Δp by signal quintile (calibration)
    params: FlowParams = field(default_factory=FlowParams)


def run_flow_book(
    price_panel: pd.DataFrame,
    signal_panel: pd.DataFrame,
    params: FlowParams = FlowParams(),
) -> FlowResult:
    """Dollar-neutral cross-sectional book on a flow signal.

    Each rebalance: cross-sectionally z-score the signal across live tokens, go
    long the top ``top_frac`` and short the bottom ``top_frac`` (equal dollars),
    hold into *t+1* for w·Δp, charge turnover cost. Dollar-neutrality strips the
    favorite-drift beta so the Sharpe reflects the signal's *ranking* power.

    Also returns ``quantile_drift`` — mean next-day Δp by signal quintile, pooled
    over all token-days — the descriptive calibration behind the book.
    """
    p = price_panel.sort_index()
    dates = p.index
    score = cross_sectional_score(signal_panel.reindex_like(p), min_names=params.min_names)
    dp_fwd = p.shift(-1) - p

    sign = -1.0 if params.contrarian else 1.0
    rets, gross_rets, turn = [], [], []
    prev_w = pd.Series(0.0, index=p.columns)

    for i, t in enumerate(dates):
        if i % params.holding == 0:
            row = score.loc[t].dropna()
            w = pd.Series(0.0, index=p.columns)
            n = len(row)
            k = max(1, int(round(n * params.top_frac)))
            if n >= 2 * k:
                order = row.sort_values()
                leg = params.gross / 2.0 / k
                w[order.index[-k:]] = sign * leg
                w[order.index[:k]] = -sign * leg
        else:
            w = prev_w.copy()
            w[~p.columns.isin(score.loc[t].dropna().index)] = 0.0

        to = float((w - prev_w).abs().sum())
        pnl = float((w * dp_fwd.loc[t].reindex(p.columns).fillna(0.0)).sum()) if i + 1 < len(dates) else 0.0
        rets.append(pnl - params.cost * to)
        gross_rets.append(pnl)
        turn.append(to)
        prev_w = w

    # calibration: mean forward Δp by signal quintile (pooled)
    s = score.values.ravel()
    d = dp_fwd.values.ravel()
    m = ~(np.isnan(s) | np.isnan(d))
    qd = pd.Series(dtype=float)
    if m.sum() > 50:
        q = pd.qcut(pd.Series(s[m]), 5, labels=False, duplicates="drop")
        qd = pd.Series(d[m]).groupby(q.values).mean()
        qd.index = [f"Q{i+1}" for i in qd.index]

    return FlowResult(
        returns=pd.Series(rets, index=dates, name=params.signal),
        gross_returns=pd.Series(gross_rets, index=dates, name=params.signal + "_gross"),
        turnover=pd.Series(turn, index=dates, name="turnover"),
        quantile_drift=qd,
        params=params,
    )
