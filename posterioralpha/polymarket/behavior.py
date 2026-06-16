"""Trader-behaviour primitives — the shared layer under the smart-money studies.

``smartmoney.py`` answered "follow the recent winners?" (no — flow reverts). This
module factors out the reusable machinery for the *next* questions about trader
behaviour, so each new signal is a thin consumer rather than a re-plumbing:

  * ``stack_trades``       — every candidate's fills on the priced universe in one
    long frame, with signed dollars/shares and a calendar day. The substrate for
    order-flow, breadth, aggressiveness and entry-timing signals.
  * ``token_domains``      — a topic label per outcome token (reusing
    ``categorize.market_category`` on the market title), so skill and flow can be
    conditioned on domain (politics / sports / macro / …). The efficiency and
    field-shape studies showed forecasting skill is *domain-specific*; following a
    trader globally dilutes their politics edge with their sports noise.
  * ``trader_token_pnl``   — each wallet's marked PnL **attributed per token**
    (cash flow of that token's fills + its marked inventory). Summing the columns
    of a domain gives a wallet's point-in-time *in-domain* skill — the input to
    domain-specialist selection.

Everything is causal: positions/cash are cumulative sums of fills, marked against
prices up to *t*, so any value at *t* uses only information available by *t*.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .categorize import market_category
from .smartmoney import trader_position_panel

logger = logging.getLogger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Stacked trade frame
# ---------------------------------------------------------------------------

def stack_trades(traders: dict[str, pd.DataFrame], tokens: list[str]) -> pd.DataFrame:
    """All wallets' fills on the priced universe, in one long frame.

    Columns added: ``wallet``, ``day`` (normalised date), ``signed_usd`` /
    ``signed_sh`` (+ on BUY, − on SELL). Trades on tokens outside ``tokens`` are
    dropped so every row joins cleanly to the price panel.
    """
    tok = set(tokens)
    frames = []
    for w, t in traders.items():
        tt = t[t["asset"].isin(tok)]
        if tt.empty:
            continue
        tt = tt.copy()
        tt["wallet"] = w
        frames.append(tt)
    if not frames:
        return pd.DataFrame(columns=list(next(iter(traders.values())).columns)
                            + ["wallet", "day", "signed_usd", "signed_sh"])
    T = pd.concat(frames, ignore_index=True)
    T["day"] = T["timestamp"].dt.normalize()
    buy = T["side"].values == "BUY"
    T["signed_usd"] = np.where(buy, T["usdcSize"].values, -T["usdcSize"].values)
    T["signed_sh"] = np.where(buy, T["size"].values, -T["size"].values)
    return T


# ---------------------------------------------------------------------------
# Token → domain
# ---------------------------------------------------------------------------

def token_titles(traders: dict[str, pd.DataFrame]) -> pd.Series:
    """Most-frequent market title per outcome token, across all wallets."""
    frames = [t[["asset", "title"]] for t in traders.values() if not t.empty]
    if not frames:
        return pd.Series(dtype=object)
    allt = pd.concat(frames, ignore_index=True)
    return (allt.groupby("asset")["title"]
            .agg(lambda s: s.value_counts().index[0] if len(s) else ""))


def token_domains(traders: dict[str, pd.DataFrame], tokens: list[str]) -> pd.Series:
    """Coarse topic label per token (politics/geopolitics/crypto/macro/sports/...).

    Reuses the keyword classifier on each token's market title, so domain-aware
    studies share one consistent taxonomy with the price-based ones.
    """
    titles = token_titles(traders)
    dom = {tok: market_category(titles.get(tok, "")) for tok in tokens}
    s = pd.Series(dom, name="domain")
    return s.reindex(tokens)


# ---------------------------------------------------------------------------
# Per-token marked PnL (→ per-domain skill)
# ---------------------------------------------------------------------------

def trader_token_pnl(
    trades: pd.DataFrame,
    tokens: list[str],
    dates: pd.DatetimeIndex,
    price_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Marked PnL of one wallet attributed per token (dates × tokens).

    For each token: cumulative cash flow of its fills (−$ buy, +$ sell) plus its
    marked inventory (net shares × price). Summed over a domain's tokens this is
    the wallet's point-in-time PnL *in that domain*; summed over all tokens it
    matches the global equity curve in ``smartmoney``.
    """
    t = trades[trades["asset"].isin(tokens)].copy()
    if t.empty:
        return pd.DataFrame(0.0, index=dates, columns=tokens)
    t["day"] = t["timestamp"].dt.normalize()
    buy = t["side"].values == "BUY"
    t["cash"] = np.where(buy, -t["usdcSize"].values, t["usdcSize"].values)
    t["signed"] = np.where(buy, t["size"].values, -t["size"].values)

    cash = (t.groupby(["day", "asset"])["cash"].sum().unstack(fill_value=0.0)
            .reindex(columns=tokens, fill_value=0.0)
            .reindex(dates, fill_value=0.0).cumsum())
    shares = (t.groupby(["day", "asset"])["signed"].sum().unstack(fill_value=0.0)
              .reindex(columns=tokens, fill_value=0.0)
              .reindex(dates, fill_value=0.0).cumsum().clip(lower=0.0))
    px = price_panel.reindex(index=dates, columns=tokens).ffill().fillna(0.0)
    return cash + shares * px


# ---------------------------------------------------------------------------
# Domain-specialist follow engine
# ---------------------------------------------------------------------------

@dataclass
class SpecialistParams:
    """Configuration for the domain-specialist follow backtest."""
    book: str = "specialist"   # specialist | global | anti_special | all_leaders
    n_per_domain: int = 5      # leaders followed per domain (specialist/anti)
    rebalance: int = 7         # re-select every N days
    burn_in: int = 45          # history before first selection
    gross: float = 1.0
    cost: float = 0.005
    min_active: int = 2        # min rankable wallets in a domain to trade it


@dataclass
class SpecialistResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    n_cohort: pd.Series
    params: SpecialistParams = field(default_factory=SpecialistParams)


def run_specialist_follow(
    traders: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    token_domain: pd.Series,
    is_mm: dict[str, bool],
    params: SpecialistParams = SpecialistParams(),
) -> SpecialistResult:
    """Follow domain specialists — point-in-time, mirror-and-exit, by topic.

    Skill is ranked *within domain*: a wallet's PnL on politics tokens chooses
    who we copy on politics, independently of its sports book. The ``global``
    book ranks by total PnL (the original smart-money baseline) to isolate
    whether domain-conditioning is what helps; ``anti_special`` (bottom per
    domain) and ``all_leaders`` (every non-MM wallet) bracket the claim.
    """
    panel = price_panel.sort_index()
    dates = panel.index
    tokens = list(panel.columns)
    pool = [w for w in traders if not is_mm.get(w, False)]   # MMs never followed here

    positions = {w: trader_position_panel(traders[w], tokens, dates) for w in pool}
    tok_pnl   = {w: trader_token_pnl(traders[w], tokens, dates, panel) for w in pool}

    domains = sorted({d for d in token_domain.reindex(tokens).dropna().unique()})
    tokens_by_dom = {d: [t for t in tokens if token_domain.get(t) == d] for d in domains}
    tokens_by_dom = {d: tl for d, tl in tokens_by_dom.items() if tl}
    domains = list(tokens_by_dom)

    dom_pnl = {d: pd.DataFrame({w: tok_pnl[w][tokens_by_dom[d]].sum(axis=1) for w in pool})
               for d in domains}
    global_pnl = pd.DataFrame({w: tok_pnl[w].sum(axis=1) for w in pool})
    k_global = params.n_per_domain * max(1, len(domains))

    first_sel = dates[min(params.burn_in, len(dates) - 1)]
    weights = pd.DataFrame(0.0, index=dates, columns=tokens)
    prev_w = pd.Series(0.0, index=tokens)
    cohort_by_dom: dict[str, list[str]] = {}
    cohort_global: list[str] = []
    rets, gross_rets, turn, n_cohort = [], [], [], []

    for i, t in enumerate(dates):
        if t >= first_sel and (i - params.burn_in) % params.rebalance == 0:
            if params.book == "global":
                cohort_global = _rank_pick(global_pnl.loc[t], k_global, "top", params.min_active)
            else:
                side = "bottom" if params.book == "anti_special" else "top"
                n = 10_000 if params.book == "all_leaders" else params.n_per_domain
                cohort_by_dom = {d: _rank_pick(dom_pnl[d].loc[t], n, side, params.min_active)
                                 for d in domains}

        agg = pd.Series(0.0, index=tokens)
        if params.book == "global":
            for w in cohort_global:
                agg = agg.add(positions[w].loc[t], fill_value=0.0)
            followed = set(cohort_global)
        else:
            followed = set()
            for d in domains:
                cohort_d = cohort_by_dom.get(d, [])
                if not cohort_d:
                    continue
                td = tokens_by_dom[d]
                sub = pd.Series(0.0, index=td)
                for w in cohort_d:
                    sub = sub.add(positions[w].loc[t, td], fill_value=0.0)
                agg.loc[td] = agg.loc[td].add(sub, fill_value=0.0)
                followed |= set(cohort_d)

        dollar = agg * panel.loc[t].reindex(tokens).fillna(0.0)
        gsum = float(dollar.abs().sum())
        w_t = dollar / gsum * params.gross if gsum > _EPS else pd.Series(0.0, index=tokens)

        weights.loc[t] = w_t
        to = float((w_t - prev_w).abs().sum())
        cost = params.cost * to
        if i + 1 < len(dates):
            dp = (panel.iloc[i + 1] - panel.iloc[i]).reindex(tokens).fillna(0.0)
            pnl = float((w_t * dp).sum())
        else:
            pnl = 0.0
        rets.append(pnl - cost)
        gross_rets.append(pnl)
        turn.append(to)
        n_cohort.append(len(followed))
        prev_w = w_t

    return SpecialistResult(
        returns=pd.Series(rets, index=dates, name=params.book),
        gross_returns=pd.Series(gross_rets, index=dates, name=params.book + "_gross"),
        weights=weights,
        turnover=pd.Series(turn, index=dates, name="turnover"),
        n_cohort=pd.Series(n_cohort, index=dates, name="n_cohort"),
        params=params,
    )


def _rank_pick(pnl_row: pd.Series, n: int, side: str, min_active: int) -> list[str]:
    """Top/bottom-``n`` rankable wallets from one PnL row (drops flat/dormant)."""
    s = pnl_row.dropna()
    s = s[s.abs() > _EPS]
    if len(s) < min_active:
        return []
    ranked = s.sort_values(ascending=False)
    return list(ranked.index[-n:] if side == "bottom" else ranked.index[:n])
