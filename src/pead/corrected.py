"""Corrected PEAD analysis primitives.

Fixes five flaws found in the original pipeline:
  1. Tradeability: forward returns in signal_panel are anchored to p_t0 (the close
     on/before the announcement). Earnings drop after the close, so p_t0 is not a
     valid entry. The earliest tradeable entry is the t+1 close. Because the stored
     returns are LOG returns, the tradeable drift is additive:
         drift_t1_to_t21 = ret_t21 - ret_t1      (log, %)
  2. Log->simple: never average log returns cross-sectionally and treat the mean as a
     simple return. Convert each leg with expm1(logret/100) BEFORE averaging.
  3. Real benchmark: compare against cached real SPY quarterly returns, not a mock.
  4. Trailing percentile: select on an expanding/trailing SUE percentile, not the
     in-quarter quantile (which peeks at names that report later in the quarter).
  5. Sharpe: annualize quarterly returns with sqrt(4); flag overlap caveat to caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_panel(path: str = "results/pead/signal_panel.csv") -> pd.DataFrame:
    """Load the signal panel and add the tradeable drift column."""
    p = pd.read_csv(path)
    p["announcement_date"] = pd.to_datetime(p["announcement_date"], utc=True, errors="coerce")
    p = p.dropna(subset=["sue", "ret_t1", "ret_t21"])
    # Tradeable drift: enter at t+1 close, exit at t+21 close (log %, additive).
    p["drift_t1_to_t21"] = p["ret_t21"] - p["ret_t1"]
    p["q"] = p["announcement_date"].dt.to_period("Q")
    return p


def log_pct_to_simple(logret_pct: pd.Series | np.ndarray) -> np.ndarray:
    """Convert a log return expressed in percent to a simple return (decimal)."""
    return np.expm1(np.asarray(logret_pct, dtype=float) / 100.0)


def portfolio_quarter_return(group_logret_pct: pd.Series) -> float:
    """Equal-weight simple return of a basket, from per-name LOG returns (%).

    Converts each name to a simple return first, then averages. This is the
    correct order; averaging logs then exponentiating understates dispersion
    and can drive (1+r) negative for bad quarters.
    """
    return float(log_pct_to_simple(group_logret_pct).mean())


def annual_and_sharpe(quarterly_simple: list[float]) -> tuple[float, float, float]:
    """Return (annual_return, sharpe_sqrt4, max_drawdown) from quarterly simple returns."""
    r = np.asarray(quarterly_simple, dtype=float)
    if len(r) == 0 or r.std() == 0:
        return 0.0, 0.0, 0.0
    mean = r.mean()
    annual = (1 + mean) ** 4 - 1
    sharpe = np.sqrt(4) * mean / r.std()
    eq = np.cumprod(1 + r)
    dd = (eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)
    return annual, sharpe, float(dd.min())


def trailing_percentile_mask(df: pd.DataFrame, sue_col: str = "sue", q: float = 0.75) -> pd.Series:
    """Boolean mask: keep names whose SUE exceeds the EXPANDING percentile of all
    SUE observed strictly before that name's announcement date.

    Avoids in-quarter look-ahead. Names before enough history are dropped.
    """
    d = df.sort_values("announcement_date")
    sue = d[sue_col].to_numpy()
    keep = np.zeros(len(d), dtype=bool)
    # Expanding percentile; require a minimum warmup so the threshold is meaningful.
    warmup = 200
    for i in range(len(d)):
        if i < warmup:
            continue
        thr = np.quantile(sue[:i], q)
        keep[i] = sue[i] > thr
    return pd.Series(keep, index=d.index).reindex(df.index).fillna(False)


def backtest(
    panel: pd.DataFrame,
    bucket: str | None,
    ret_col: str,
    use_trailing: bool,
    q: float = 0.75,
    cost_annual: float = 0.0,
) -> dict:
    """Quarterly long-only PEAD backtest with correct return math.

    Args:
      panel: from load_panel().
      bucket: market_cap bucket to restrict to, or None for all caps.
      ret_col: 'ret_t21' (as-backtested, incl. jump) or 'drift_t1_to_t21' (tradeable).
      use_trailing: trailing-percentile selection vs in-quarter quantile.
      q: SUE percentile threshold.
      cost_annual: annual cost (decimal) subtracted from the annualized return.
    """
    df = panel if bucket is None else panel[panel["market_cap"] == bucket]
    if use_trailing:
        df = df[trailing_percentile_mask(df, q=q)]

    quarterly = []
    npos = []
    for _, g in df.groupby("q"):
        sel = g if use_trailing else g[g["sue"] > g["sue"].quantile(q)]
        if len(sel) == 0:
            continue
        quarterly.append(portfolio_quarter_return(sel[ret_col]))
        npos.append(len(sel))

    annual, sharpe, mdd = annual_and_sharpe(quarterly)
    return {
        "annual_gross": annual,
        "annual_net": annual - cost_annual,
        "sharpe": sharpe,
        "max_dd": mdd_clip(mdd),
        "n_quarters": len(quarterly),
        "avg_positions": float(np.mean(npos)) if npos else 0.0,
        "quarterly": quarterly,
    }


def mdd_clip(mdd: float) -> float:
    """Long-only drawdown cannot exceed -100%; clip for sanity reporting."""
    return max(mdd, -1.0)


def load_spy_quarterly(path: str = "results/pead/spy_quarterly.csv") -> pd.DataFrame:
    """Load cached real SPY quarterly returns (decimal)."""
    s = pd.read_csv(path)
    return s
