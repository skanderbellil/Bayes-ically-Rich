"""
Analytics: metrics, per-name contribution, fragility index, dispersion, pairs,
tier attribution.

All returns are daily SIMPLE returns in USD. Cumulative/total figures compound;
"contribution" figures use an additive daily decomposition so per-name pieces
sum to the book's arithmetic PnL (the standard Brinson-style approximation --
stated here as an assumption).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    UNIVERSE, RISK_FREE_ANNUAL, TRADING_DAYS, HARD_TO_BORROW,
    BORROW_APR_DEFAULT, BORROW_APR_HTB,
)
from portfolio import (
    equal_weight_book, per_name_weight_paths, borrow_daily_drag, cum_nav,
)


# ---------------------------------------------------------------------------
# 3. Metrics table
# ---------------------------------------------------------------------------
def max_drawdown(nav):
    """Max drawdown (negative number) of a NAV path."""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def drawdown_series(nav):
    return nav / nav.cummax() - 1.0


def metrics(ret, name="", rf=RISK_FREE_ANNUAL):
    """Return a dict of the standard metrics for a daily-return series."""
    r = ret.dropna()
    nav = cum_nav(r)
    ytd = float(nav.iloc[-1] - 1.0) if len(nav) else np.nan
    ann_vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann_ret = float((1.0 + r.mean()) ** TRADING_DAYS - 1.0)  # geometric-ish
    excess = r.mean() * TRADING_DAYS - rf
    sharpe = float(excess / ann_vol) if ann_vol > 0 else np.nan
    return {
        "name": name,
        "YTD_return": ytd,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "Sharpe_rf4": sharpe,
        "max_drawdown": max_drawdown(nav),
        "worst_day": float(r.min()),
        "best_day": float(r.max()),
        "pct_positive_days": float((r > 0).mean()),
        "n_days": int(len(r)),
    }


def metrics_table(series_dict, rf=RISK_FREE_ANNUAL):
    rows = [metrics(s, name=k, rf=rf) for k, s in series_dict.items()]
    return pd.DataFrame(rows).set_index("name")


# ---------------------------------------------------------------------------
# 4. Per-name contribution to L/S PnL (+ waterfall data)
# ---------------------------------------------------------------------------
def name_contributions(returns, long_members, short_members, apply_borrow=True):
    """Additive per-name contribution to the DOLLAR-NEUTRAL L/S total return.

    Long name i contributes  sum_t  w_i(t) * r_i(t).
    Short name j contributes  sum_t -w_j(t) * r_j(t)  (minus its borrow accrual).
    The sum of all contributions equals the arithmetic (sum-of-daily) L/S return,
    which we also return for reconciliation.
    """
    long_members = [m for m in long_members if m in returns.columns]
    short_members = [m for m in short_members if m in returns.columns]

    WL = per_name_weight_paths(returns, long_members)
    WS = per_name_weight_paths(returns, short_members)

    contrib = {}
    # Longs: +w*r
    for m in long_members:
        contrib[m] = float((WL[m] * returns[m].reindex(WL.index).fillna(0.0)).sum())
    # Shorts: -w*r  minus borrow accrued on that name's weight
    for m in short_members:
        gross = -float((WS[m] * returns[m].reindex(WS.index).fillna(0.0)).sum())
        if apply_borrow:
            apr = BORROW_APR_HTB if m in HARD_TO_BORROW else BORROW_APR_DEFAULT
            borrow = float((WS[m] * (apr / TRADING_DAYS)).sum())
            contrib[m] = gross - borrow
        else:
            contrib[m] = gross

    c = pd.Series(contrib).sort_values(ascending=False)
    total = float(c.sum())
    return c, total


# ---------------------------------------------------------------------------
# 5. One-name fragility index (the "Nebius test")
# ---------------------------------------------------------------------------
def fragility_index(returns, long_members, short_members, apply_borrow=True):
    """Rerun the dollar-neutral L/S three ways: full, excluding the single best
    contributor, and excluding the single worst contributor. Report each run's
    total return and the spread between them.

    Interpretation: a SMALL spread => alpha is broad/thesis-driven; a LARGE
    spread => the book is hostage to one or two names (fragile, name-lucky)."""
    contrib, _ = name_contributions(returns, long_members, short_members, apply_borrow)
    best = contrib.idxmax()
    worst = contrib.idxmin()

    def run(excl):
        lm = [m for m in long_members if m != excl]
        sm = [m for m in short_members if m != excl]
        lr = equal_weight_book(returns, lm)
        sr = equal_weight_book(returns, sm)
        borrow = pd.Series(0.0, index=returns.index)
        if apply_borrow:
            ws = per_name_weight_paths(returns, [m for m in sm if m in returns.columns])
            for d in returns.index:
                borrow.loc[d] = borrow_daily_drag(
                    [m for m in sm if m in returns.columns],
                    ws.loc[d] if d in ws.index else None)
        ls = lr - sr - borrow
        return float(cum_nav(ls).iloc[-1] - 1.0)

    full = run(None)
    ex_best = run(best)
    ex_worst = run(worst)
    runs = {
        "full": full,
        f"ex_best({best})": ex_best,
        f"ex_worst({worst})": ex_worst,
    }
    spread = max(runs.values()) - min(runs.values())
    return dict(runs=runs, best=best, worst=worst,
                best_contrib=float(contrib[best]),
                worst_contrib=float(contrib[worst]),
                spread=spread,
                drop_best_impact=full - ex_best,
                drop_worst_impact=full - ex_worst)


# ---------------------------------------------------------------------------
# 6. Within-category dispersion (fragile_middle)
# ---------------------------------------------------------------------------
def category_dispersion(returns, members):
    """Return (cum_df, monthly_xsection_std) for a category.

    cum_df: cumulative return path per name.
    monthly_xsection_std: for each month, the cross-sectional std of that month's
    per-name returns -- i.e. how much the names DISAGREE. Thesis says this should
    be HIGH inside the short basket (idiosyncratic blow-ups, not a clean factor).
    """
    members = [m for m in members if m in returns.columns]
    R = returns[members]
    cum = (1.0 + R.fillna(0.0)).cumprod() - 1.0

    # Per-name monthly returns, then cross-sectional std across names each month.
    monthly = (1.0 + R.fillna(0.0)).groupby([R.index.year, R.index.month]).prod() - 1.0
    monthly.index = [f"{y:04d}-{m:02d}" for (y, m) in monthly.index]
    xstd = monthly.std(axis=1, ddof=1)
    return cum, monthly, xstd


# ---------------------------------------------------------------------------
# 7. Pair spread: long-CRWV/short-NBIS and the inverse
# ---------------------------------------------------------------------------
def pair_spread(returns, a="CRWV", b="NBIS"):
    """Cumulative return of a $1-long-a / $1-short-b pair and its inverse."""
    out = {}
    if a in returns.columns and b in returns.columns:
        long_a_short_b = returns[a] - returns[b]
        long_b_short_a = returns[b] - returns[a]
        out[f"long_{a}_short_{b}"] = cum_nav(long_a_short_b) - 1.0
        out[f"long_{b}_short_{a}"] = cum_nav(long_b_short_a) - 1.0
    return out


# ---------------------------------------------------------------------------
# 8. Tier attribution on the long book
# ---------------------------------------------------------------------------
def tier_attribution(returns, long_members):
    """Total (compounded) return of each long tier's equal-weight book, plus its
    additive contribution to the equal-weight-of-tiers long book.

    NOTE: the main long book equal-weights NAMES, not tiers. Here we also report
    the per-tier standalone return so we can see which tier carried H1."""
    from config import LONG_TIERS
    rows = {}
    for tier in LONG_TIERS:
        members = [m for m in long_members
                   if UNIVERSE[m]["tier"] == tier and m in returns.columns]
        if not members:
            continue
        tr = equal_weight_book(returns, members)
        rows[tier] = dict(
            tier_return=float(cum_nav(tr).iloc[-1] - 1.0),
            ann_vol=float(tr.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            n_names=len(members),
        )
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Rolling beta of the L/S book (for chart iv)
# ---------------------------------------------------------------------------
def rolling_beta_series(ls_ret, spy_ret, window):
    cov = ls_ret.rolling(window).cov(spy_ret)
    var = spy_ret.rolling(window).var()
    return cov / var
