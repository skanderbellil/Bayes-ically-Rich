"""Follow the smart money — point-in-time copy-trading on Polymarket.

The hypothesis is **trader momentum**: wallets that have been *winning* keep
making good trades, so a book that mirrors the recent winners' net positions —
after screening out market makers — should earn their edge minus frictions.

Three pieces, all designed to keep the test honest:

1. **Market-maker screen** (``mm_fingerprint``). A leaderboard winner can be a
   genuine forecaster *or* a market maker harvesting spread. We never want to
   "follow" an MM — their inventory is noise, flipped thousands of times. We
   fingerprint each wallet from its fills (two-sidedness, churn/turnover,
   edge-per-dollar, trade count) and flag the MM-like ones.

2. **Point-in-time skill** (``trader_equity_curves``). The whole study lives or
   dies on avoiding lookahead. At every rebalance date *t* we rank candidates by
   the PnL they had realised **strictly before *t***, reconstructed from each
   wallet's own fills marked against the CLOB price panel — never by the
   current, end-of-sample leaderboard. The leaderboard only seeds *which wallets
   exist* in the pool (a documented survivorship caveat, not a timing leak).

3. **Mirror-and-exit follow** (``run_smart_money_follow``). The cohort's target
   in each outcome token is the selected leaders' *aggregate net dollar
   inventory*: we hold proportional to their conviction while they hold and
   decay to zero as they exit. Marked to the CLOB mid daily on $1 gross, with a
   turnover cost — the same additive, $1-of-max-loss convention as the rest of
   the subpackage.

Everything is expressed **per outcome token** (the specific asset a trader
bought). A token's price ∈ (0, 1) rises when its outcome becomes more likely and
settles to 0/1 at resolution, so "short the Yes" needs no sign gymnastics — it
simply shows up as the cohort holding the *No* token, a different asset already
in the panel. Net shares per token are therefore non-negative (you cannot sell
what you never bought; capped trade history is clipped at zero).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Universe: which tokens does the cohort actually trade?
# ---------------------------------------------------------------------------

def cohort_token_notional(traders: dict[str, pd.DataFrame]) -> pd.Series:
    """Total USD traded per outcome token across all candidate wallets.

    Used to pick the copyable universe: the tokens the pool collectively pushed
    the most money through (and which we can therefore price and mark cleanly).
    """
    frames = [t[["asset", "usdcSize"]] for t in traders.values() if not t.empty]
    if not frames:
        return pd.Series(dtype=float)
    allt = pd.concat(frames, ignore_index=True)
    return allt.groupby("asset")["usdcSize"].sum().sort_values(ascending=False)


def select_universe(traders: dict[str, pd.DataFrame], top_k: int = 150) -> list[str]:
    """Top-``top_k`` outcome tokens by cohort trading notional."""
    notional = cohort_token_notional(traders)
    return [str(t) for t in notional.head(top_k).index]


# ---------------------------------------------------------------------------
# Market-maker fingerprint
# ---------------------------------------------------------------------------

@dataclass
class MMScreen:
    """Thresholds for flagging a wallet as market-maker-like."""
    min_trades: int = 200          # below this, churn metrics are too noisy to call MM
    two_sided: float = 0.35        # min(buy,sell)$/total per token, volume-weighted
    turnover: float = 6.0          # gross $ traded / peak net $ inventory
    edge_per_dollar: float = 0.004 # realised PnL / gross $ traded (MMs scalp thin)


def mm_fingerprint(
    trades: pd.DataFrame,
    price_panel: pd.DataFrame,
    asof: Optional[pd.Timestamp] = None,
    screen: MMScreen = MMScreen(),
) -> dict:
    """Behavioural fingerprint of one wallet, on fills strictly before ``asof``.

    Returns the raw metrics plus an ``is_mm`` flag. A market maker's signature is
    *all three* of: heavy two-sided round-tripping, high turnover relative to net
    inventory, and a thin edge per traded dollar — combined with a high trade
    count (so we never tar an occasional trader with the MM brush).
    """
    t = trades if asof is None else trades[trades["timestamp"] < asof]
    n = len(t)
    out = {"n_trades": n, "two_sided": np.nan, "turnover": np.nan,
           "edge_per_dollar": np.nan, "gross_usd": 0.0, "is_mm": False}
    if n == 0:
        return out

    signed_sh = np.where(t["side"].values == "BUY", t["size"].values, -t["size"].values)
    signed_usd = np.where(t["side"].values == "BUY", t["usdcSize"].values, -t["usdcSize"].values)
    gross_usd = float(t["usdcSize"].sum())
    out["gross_usd"] = gross_usd

    tmp = t.assign(_susd=signed_usd, _ssh=signed_sh)
    # two-sidedness: per token, how balanced are buys vs sells (1 = pure round-trip)
    def _two_sided(g: pd.DataFrame) -> float:
        buy = g.loc[g["side"] == "BUY", "usdcSize"].sum()
        sell = g.loc[g["side"] == "SELL", "usdcSize"].sum()
        tot = buy + sell
        return 0.0 if tot <= 0 else 2.0 * min(buy, sell) / tot
    by_tok = tmp.groupby("asset")
    ts_w = by_tok.apply(_two_sided)
    vol_w = by_tok["usdcSize"].sum()
    out["two_sided"] = float((ts_w * vol_w).sum() / (vol_w.sum() + _EPS))

    # turnover: gross $ traded vs the largest net $ inventory ever carried
    net_sh_by_tok = tmp.groupby("asset")["_ssh"].cumsum()
    peak_net_usd = 0.0
    for tok, g in tmp.assign(_cum=net_sh_by_tok).groupby("asset"):
        peak_net_usd = max(peak_net_usd, float((g["_cum"].abs() * g["price"]).max()))
    out["turnover"] = gross_usd / (peak_net_usd + _EPS)

    # edge per traded dollar: realised+marked PnL / gross $ (MMs scalp thin)
    pnl = _wallet_pnl_asof(t, price_panel, asof)
    out["edge_per_dollar"] = pnl / (gross_usd + _EPS)

    out["is_mm"] = bool(
        n >= screen.min_trades
        and out["two_sided"] >= screen.two_sided
        and out["turnover"] >= screen.turnover
        and out["edge_per_dollar"] <= screen.edge_per_dollar
    )
    return out


def screen_market_makers(
    traders: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    volume_watch: Optional[set] = None,
    screen: "MMScreen" = None,
) -> dict[str, bool]:
    """Flag MM-like wallets across a pool → {wallet: is_mm}.

    A wallet is flagged if its behavioural fingerprint trips the MM thresholds,
    *or* it sits on the high-volume watchlist with many trades and a thin edge —
    the two complementary signatures of spread-harvesting rather than forecasting.
    """
    screen = screen or MMScreen()
    volume_watch = volume_watch or set()
    out: dict[str, bool] = {}
    for w, t in traders.items():
        fp = mm_fingerprint(t, price_panel, asof=None, screen=screen)
        out[w] = bool(fp["is_mm"] or (w in volume_watch and fp["n_trades"] >= 200
                                      and fp["edge_per_dollar"] <= 0.01))
    return out


def _wallet_pnl_asof(
    trades: pd.DataFrame,
    price_panel: pd.DataFrame,
    asof: Optional[pd.Timestamp],
) -> float:
    """Marked PnL of a wallet at ``asof``: cash flow + inventory × current price.

    Restricted to tokens in ``price_panel`` (the copyable universe), so it is the
    wallet's PnL *on the markets we would actually mirror* — the same footing the
    follow-book is judged on.
    """
    t = trades if asof is None else trades[trades["timestamp"] < asof]
    if t.empty:
        return 0.0
    cols = set(price_panel.columns)
    t = t[t["asset"].isin(cols)]
    if t.empty:
        return 0.0
    cash = float(np.where(t["side"].values == "BUY",
                          -t["usdcSize"].values, t["usdcSize"].values).sum())
    signed_sh = np.where(t["side"].values == "BUY", t["size"].values, -t["size"].values)
    net = pd.Series(signed_sh, index=t["asset"].values).groupby(level=0).sum()
    # price at asof (last available on/before asof), else final price (resolution)
    px_row = (price_panel.loc[:asof].ffill().iloc[-1]
              if asof is not None and (price_panel.index <= asof).any()
              else price_panel.ffill().iloc[-1])
    marked = float((net.clip(lower=0.0) * px_row.reindex(net.index).fillna(0.0)).sum())
    return cash + marked


# ---------------------------------------------------------------------------
# Per-trader daily position & equity panels
# ---------------------------------------------------------------------------

def trader_position_panel(
    trades: pd.DataFrame,
    tokens: list[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Daily net shares held per token for one wallet (dates × tokens).

    Cumulative signed fills (BUY +size, SELL −size) on the daily calendar,
    forward-filled, clipped at 0 (a token can't be held short; capped history can
    otherwise produce spurious negatives). Tokens outside ``tokens`` are dropped.
    """
    t = trades[trades["asset"].isin(tokens)].copy()
    if t.empty:
        return pd.DataFrame(0.0, index=dates, columns=tokens)
    t["day"] = t["timestamp"].dt.normalize()
    t["signed"] = np.where(t["side"].values == "BUY", t["size"].values, -t["size"].values)
    daily = t.groupby(["day", "asset"])["signed"].sum().unstack(fill_value=0.0)
    daily = daily.reindex(columns=tokens, fill_value=0.0)
    pos = daily.reindex(dates, fill_value=0.0).cumsum().clip(lower=0.0)
    return pos


def trader_cash_panel(
    trades: pd.DataFrame,
    tokens: list[str],
    dates: pd.DatetimeIndex,
) -> pd.Series:
    """Daily cumulative cash flow for one wallet (−$ on BUY, +$ on SELL).

    Restricted to ``tokens`` so cash and inventory marking share one footing.
    """
    t = trades[trades["asset"].isin(tokens)].copy()
    if t.empty:
        return pd.Series(0.0, index=dates)
    t["day"] = t["timestamp"].dt.normalize()
    t["cash"] = np.where(t["side"].values == "BUY",
                         -t["usdcSize"].values, t["usdcSize"].values)
    daily = t.groupby("day")["cash"].sum()
    return daily.reindex(dates, fill_value=0.0).cumsum()


def trader_equity_curve(
    pos: pd.DataFrame,
    cash: pd.Series,
    price_panel: pd.DataFrame,
) -> pd.Series:
    """Marked equity over time: cash(t) + Σ shares(t) × price(t).

    The point-in-time skill series — its value at *t* uses only fills before *t*
    (``pos``/``cash`` are causal cumulants) and prices up to *t*.
    """
    px = price_panel.reindex(index=pos.index, columns=pos.columns).ffill().fillna(0.0)
    marked = (pos * px).sum(axis=1)
    return cash.reindex(pos.index).fillna(0.0) + marked


# ---------------------------------------------------------------------------
# Follow engine
# ---------------------------------------------------------------------------

@dataclass
class FollowParams:
    """Configuration for the mirror-and-exit follow backtest."""
    book: str = "smart_money"      # smart_money | all_leaders | anti_smart | with_mm
    n_follow: int = 20             # cohort size (top-N by point-in-time PnL)
    rebalance: int = 7             # re-select the cohort every N days
    burn_in: int = 45              # days of history before the first selection (MM screen + skill)
    gross: float = 1.0             # $ notional
    cost: float = 0.005            # per-unit turnover cost
    min_active_traders: int = 3    # need at least this many rankable traders to trade a date


@dataclass
class FollowResult:
    returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    n_cohort: pd.Series            # traders actually followed per day
    cohort_log: dict               # rebalance date → list[wallet] followed
    params: FollowParams = field(default_factory=FollowParams)


def run_smart_money_follow(
    traders: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    is_mm: dict[str, bool],
    params: FollowParams = FollowParams(),
) -> FollowResult:
    """Walk-forward copy-trading book on a token price panel.

    At each rebalance date the cohort is chosen **point-in-time** by marked PnL
    realised before that date; the target token weights are the cohort's
    aggregate net *dollar* inventory, normalised to ``gross`` and held into the
    next day for w·Δp, net of turnover cost.

    Four books (``params.book``) form a falsifiable panel:
      smart_money — top-N non-MM wallets by point-in-time PnL (the hypothesis)
      all_leaders — every non-MM wallet, inventory-weighted (does *selection* help?)
      anti_smart  — bottom-N non-MM wallets (negative control: follow the losers)
      with_mm     — top-N by PnL *including* MMs (does the MM screen matter?)
    """
    panel = price_panel.sort_index()
    dates = panel.index
    tokens = list(panel.columns)

    # ── Pre-compute per-trader causal position / cash / equity panels ─────────
    pool = list(traders.keys())
    if params.book in ("smart_money", "all_leaders", "anti_smart"):
        pool = [w for w in pool if not is_mm.get(w, False)]
    positions = {w: trader_position_panel(traders[w], tokens, dates) for w in pool}
    cashflows = {w: trader_cash_panel(traders[w], tokens, dates) for w in pool}
    equity    = {w: trader_equity_curve(positions[w], cashflows[w], panel) for w in pool}
    equity_df = pd.DataFrame(equity)                      # dates × wallet, point-in-time PnL

    first_sel = dates[min(params.burn_in, len(dates) - 1)]

    weights = pd.DataFrame(0.0, index=dates, columns=tokens)
    prev_w = pd.Series(0.0, index=tokens)
    cohort: list[str] = []
    cohort_log: dict = {}
    rets, gross_rets, turn, n_cohort = [], [], [], []

    for i, t in enumerate(dates):
        # ── re-select the cohort every `rebalance` days, after burn-in ───────
        if t >= first_sel and (i - params.burn_in) % params.rebalance == 0:
            cohort = _select_cohort(equity_df.loc[t], params)
            cohort_log[t] = cohort

        # ── target weights = cohort aggregate net dollar inventory ───────────
        if cohort:
            agg = pd.Series(0.0, index=tokens)
            for w in cohort:
                agg = agg.add(positions[w].loc[t], fill_value=0.0)
            dollar = agg * panel.loc[t].reindex(tokens).fillna(0.0)  # shares → $ inventory
            gsum = float(dollar.abs().sum())
            w_t = dollar / gsum * params.gross if gsum > _EPS else pd.Series(0.0, index=tokens)
        else:
            w_t = pd.Series(0.0, index=tokens)

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
        n_cohort.append(len(cohort))
        prev_w = w_t

    return FollowResult(
        returns=pd.Series(rets, index=dates, name=params.book),
        gross_returns=pd.Series(gross_rets, index=dates, name=params.book + "_gross"),
        weights=weights,
        turnover=pd.Series(turn, index=dates, name="turnover"),
        n_cohort=pd.Series(n_cohort, index=dates, name="n_cohort"),
        cohort_log=cohort_log,
        params=params,
    )


def _select_cohort(equity_row: pd.Series, params: FollowParams) -> list[str]:
    """Point-in-time cohort from one date's marked-PnL row across the pool.

    Only wallets that are *rankable* (have a non-NaN, non-flat PnL — i.e. they
    have actually traded universe tokens by now) are eligible, so we never pad a
    cohort with dormant accounts.
    """
    s = equity_row.dropna()
    s = s[s.abs() > _EPS]                       # has taken real positions by now
    if len(s) < params.min_active_traders:
        return []
    ranked = s.sort_values(ascending=False)
    if params.book == "all_leaders":
        return list(ranked.index)                 # follow the whole pool (no selection)
    if params.book == "anti_smart":
        return list(ranked.index[-params.n_follow:])
    return list(ranked.index[:params.n_follow])   # smart_money / with_mm: top-N by PnL
