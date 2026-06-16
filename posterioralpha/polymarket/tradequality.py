"""Trade-level behavioural tells — what marks an *informed* Polymarket fill?

The follow/flow studies aggregate positions and flow. This one zooms to the
individual fill and asks which observable properties of a trade predict that it
was *right* — i.e. that the token drifted in the trade's direction afterwards.
Four behavioural tells, each a hypothesis about informed trading:

  * **aggressiveness** — did the trader cross the spread / pay up? We don't have a
    taker flag, but we have each fill's price and the day's CLOB mid, so a BUY
    executed *above* the mid (or a SELL *below*) reveals urgency: paying up to get
    filled now is a conviction/information tell.
  * **entry vs add** — a wallet's *first* fill in a market (initiating a position)
    may carry more information than averaging into an existing one.
  * **conviction size** — a fill that is large *relative to that wallet's own
    typical size* (a within-wallet z-score), i.e. "betting big".
  * **patience** — a wallet-level tell: do traders who *hold* (long average
    holding time) forecast better than flippers?

Outcome per fill = the **forward directional drift**: the token's price change
over the next ``fwd`` days, signed by the trade side (so a profitable BUY and a
profitable SELL both score positive). Causal — it uses only prices strictly
after the trade. A second, coarser outcome marks each fill against the token's
eventual **resolution** (final ≈0/1 price).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .behavior import stack_trades

logger = logging.getLogger(__name__)

_EPS = 1e-9


def build_trade_features(
    traders: dict[str, pd.DataFrame],
    price_panel: pd.DataFrame,
    is_mm: dict[str, bool] | None = None,
    fwd: int = 3,
    min_price: float = 0.05,
    max_price: float = 0.95,
) -> pd.DataFrame:
    """One row per fill on the priced universe with behavioural tells + outcome.

    Columns: wallet, day, asset, side, usdcSize, mid, premium (signed, >0 = paid
    up / urgent), is_entry, size_z (within-wallet log-size z-score), hold_days
    (wallet mean holding time), dir_fwd (forward ``fwd``-day directional drift),
    resolved_correct (1/0/NaN vs the token's eventual 0/1 resolution).

    Fills on tokens priced outside ``[min_price, max_price]`` that day are dropped
    (near-resolved books are illiquid and their "drift" is mechanical).
    """
    is_mm = is_mm or {}
    tokens = list(price_panel.columns)
    keep = {w: t for w, t in traders.items() if not is_mm.get(w, False)}
    T = stack_trades(keep, tokens)
    if T.empty:
        return pd.DataFrame()

    p = price_panel.sort_index()
    mid = p.ffill()
    fwd_drift = (p.shift(-fwd) - p)               # token forward Δp
    final = p.ffill().iloc[-1]                    # eventual (resolution) price per token

    # map each fill's (day, asset) to mid + forward drift
    midser = mid.stack(); midser.index = midser.index.set_names(["day", "asset"])
    fwdser = fwd_drift.stack(); fwdser.index = fwdser.index.set_names(["day", "asset"])
    T = T.merge(midser.rename("mid"), on=["day", "asset"], how="left")
    T = T.merge(fwdser.rename("fwd"), on=["day", "asset"], how="left")
    T = T.dropna(subset=["mid", "fwd"])
    T = T[(T["mid"] >= min_price) & (T["mid"] <= max_price)]
    if T.empty:
        return pd.DataFrame()

    buy = T["side"].values == "BUY"
    # aggressiveness: signed premium to mid in the trade's favour-direction
    T["premium"] = np.where(buy, T["price"] - T["mid"], T["mid"] - T["price"])
    # forward directional drift (BUY: +fwd, SELL: -fwd)
    T["dir_fwd"] = np.where(buy, T["fwd"], -T["fwd"])
    # resolution correctness
    fin = T["asset"].map(final.to_dict())
    res = np.where(fin >= 0.9, 1.0, np.where(fin <= 0.1, 0.0, np.nan))
    T["resolved_correct"] = np.where(buy, res, 1.0 - res)

    # entry vs add: first fill per (wallet, token) in time
    T = T.sort_values("timestamp")
    T["is_entry"] = ~T.duplicated(["wallet", "asset"], keep="first")

    # conviction: within-wallet log-size z-score
    T["lsize"] = np.log(T["usdcSize"].clip(lower=1.0))
    g = T.groupby("wallet")["lsize"]
    T["size_z"] = (T["lsize"] - g.transform("mean")) / (g.transform("std") + _EPS)

    # patience: wallet mean holding days (entry→exit per token, simple proxy)
    T["hold_days"] = T["wallet"].map(_wallet_hold_days(keep, tokens))
    return T


def _wallet_hold_days(traders: dict[str, pd.DataFrame], tokens: list[str]) -> dict[str, float]:
    """Mean span (days) between first and last fill per (wallet, token)."""
    out: dict[str, float] = {}
    tok = set(tokens)
    for w, t in traders.items():
        tt = t[t["asset"].isin(tok)]
        if tt.empty:
            continue
        span = tt.groupby("asset")["timestamp"].agg(lambda s: (s.max() - s.min()).total_seconds() / 86400.0)
        out[w] = float(span.mean()) if len(span) else 0.0
    return out


def _stat(x: pd.Series) -> tuple[float, float, int]:
    """Mean, t-stat, n of a series (drop NaN)."""
    x = x.dropna()
    n = len(x)
    if n < 2:
        return float("nan"), float("nan"), n
    return float(x.mean()), float(x.mean() / (x.std() / np.sqrt(n) + _EPS)), n


def summarize_quality(feat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Bucketed forward-drift / hit-rate tables for each behavioural tell."""
    out: dict[str, pd.DataFrame] = {}

    def _table(groups: list[tuple[str, pd.Series]]) -> pd.DataFrame:
        rows = []
        for label, mask in groups:
            m, tt, n = _stat(feat.loc[mask, "dir_fwd"])
            hit = feat.loc[mask, "resolved_correct"].dropna()
            rows.append([label, n, m, tt, float(hit.mean()) if len(hit) else float("nan")])
        return pd.DataFrame(rows, columns=["bucket", "n", "dir_fwd", "t", "hit_rate"])

    out["aggressiveness"] = _table([
        ("paid up >1c", feat["premium"] > 0.01),
        ("at/inside mid", feat["premium"].between(-0.01, 0.01)),
        ("below mid <-1c", feat["premium"] < -0.01),
    ])
    out["entry_vs_add"] = _table([
        ("entry (first fill)", feat["is_entry"]),
        ("add-on", ~feat["is_entry"]),
    ])
    out["conviction"] = _table([
        ("big bet (z>1)", feat["size_z"] > 1.0),
        ("normal (|z|<=1)", feat["size_z"].abs() <= 1.0),
        ("small (z<-1)", feat["size_z"] < -1.0),
    ])
    med = feat["hold_days"].median()
    out["patience"] = _table([
        (f"patient (hold>{med:.0f}d)", feat["hold_days"] > med),
        (f"flipper (hold<={med:.0f}d)", feat["hold_days"] <= med),
    ])
    return out
