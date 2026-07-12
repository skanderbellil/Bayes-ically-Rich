"""
Portfolio construction: equal-weight books, three L/S variants, borrow haircut.

NO-LOOKAHEAD RULE (guardrail from the brief)
--------------------------------------------
Every weight or scaling factor used to earn a return on day t is fixed using
information available at the PRIOR month-end (t-1 month or earlier). Concretely:
  * Equal weights are reset at each month-end and held (drifting) through the
    next month -- the reset uses only names that already had data at that
    month-end.
  * The beta-neutral short-scaling factor for month M is the 60-day rolling
    beta computed THROUGH the last trading day of month M-1.

We never optimize weights (guardrail: this is diagnosis, not curve-fitting).
Equal weight, full stop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    UNIVERSE, RISK_FREE_ANNUAL, TRADING_DAYS, BETA_WINDOW,
    BORROW_APR_DEFAULT, BORROW_APR_HTB, HARD_TO_BORROW,
)


# ---------------------------------------------------------------------------
# Equal-weight book returns with monthly rebalance + intra-month drift
# ---------------------------------------------------------------------------
def _month_end_index(idx):
    """Trading-day month-ends present in `idx` (last obs of each calendar month)."""
    s = pd.Series(idx, index=idx)
    return s.groupby([idx.year, idx.month]).last().values


def _target_weights(valid_mask, target):
    """Target weights among currently-valid members. `target` is None (equal) or
    a Series of desired relative weights; it is restricted to valid names and
    renormalized to sum to 1."""
    valid = valid_mask[valid_mask].index
    if target is None:
        n = len(valid)
        w = pd.Series(0.0, index=valid_mask.index)
        if n:
            w.loc[valid] = 1.0 / n
        return w
    t = target.reindex(valid_mask.index).fillna(0.0).clip(lower=0.0)
    t.loc[~valid_mask.astype(bool)] = 0.0
    s = t.sum()
    return t / s if s > 0 else t


def _resolve_target(target, d, members):
    """Normalize `target` into a Series-or-None for reset date `d`.

    * None                       -> equal weight (returns None)
    * Series (static tilt)       -> itself, restricted to members
    * DataFrame (time-varying)   -> the most recent row with index <= d,
                                    i.e. weights decided at the prior rebalance
                                    (no lookahead).
    """
    if target is None:
        return None
    if isinstance(target, pd.DataFrame):
        prior = target.index[target.index <= d]
        if len(prior) == 0:
            return None                      # not enough history yet -> equal
        return target.loc[prior[-1]].reindex(members)
    return target.reindex(members)


def inverse_vol_targets(returns, members, window, reset_dates):
    """Time-varying inverse-volatility target weights.

    For each reset date r, weight_i proportional to 1 / (trailing `window`-day
    vol of name i, measured on returns UP TO AND INCLUDING r). Those weights are
    then applied to the NEXT month, so the decision uses only past data -> no
    lookahead. Names with insufficient history fall back to equal weight.
    Returns a DataFrame indexed by reset date.
    """
    members = [m for m in members if m in returns.columns]
    R = returns[members]
    rows = {}
    for r in sorted(reset_dates):
        hist = R.loc[:r].tail(window)
        vol = hist.std(ddof=1)
        inv = 1.0 / vol.replace(0.0, np.nan)
        if inv.notna().sum() == 0:
            inv = pd.Series(1.0, index=members)     # equal fallback
        inv = inv.fillna(inv.mean())
        rows[r] = inv / inv.sum()
    return pd.DataFrame(rows).T


def equal_weight_book(returns, members, rebalance="M", target=None):
    """Daily return of a monthly-rebalanced book.

    Weights are reset to `target` (default: equal) at each month-end among names
    with valid data, then DRIFT with realized returns until the next reset. Honest
    "set weights, let it run, rebalance monthly" -- no lookahead, since the reset
    only uses names already trading at the reset date. Passing a non-equal
    `target` (e.g. balance-sheet fragility weights) gives a tilted book.
    """
    members = [m for m in members if m in returns.columns]
    R = returns[members].copy()
    dates = R.index
    month_ends = set(pd.Timestamp(d) for d in _month_end_index(dates))
    reset_dates = {dates[0]} | month_ends

    book_ret = pd.Series(0.0, index=dates)
    w = None
    for i, d in enumerate(dates):
        if w is None or d in reset_dates:
            w = _target_weights(R.loc[d].notna(), _resolve_target(target, d, members))
        r_t = R.loc[d].fillna(0.0)
        book_ret.iloc[i] = float((w * r_t).sum())
        grown = w * (1.0 + r_t)
        tot = grown.sum()
        w = grown / tot if tot > 0 else w
    return book_ret


def per_name_weight_paths(returns, members, target=None):
    """DataFrame of each member's effective weight each day (drifting, monthly
    reset to `target`; default equal). Used for per-name contribution & borrow."""
    members = [m for m in members if m in returns.columns]
    R = returns[members].copy()
    dates = R.index
    month_ends = set(pd.Timestamp(d) for d in _month_end_index(dates))
    reset_dates = {dates[0]} | month_ends

    W = pd.DataFrame(0.0, index=dates, columns=members)
    w = None
    for d in dates:
        if w is None or d in reset_dates:
            w = _target_weights(R.loc[d].notna(), _resolve_target(target, d, members))
        W.loc[d] = w
        r_t = R.loc[d].fillna(0.0)
        grown = w * (1.0 + r_t)
        tot = grown.sum()
        w = grown / tot if tot > 0 else w
    return W


# ---------------------------------------------------------------------------
# Borrow-cost haircut on the short book
# ---------------------------------------------------------------------------
def borrow_daily_drag(short_members, short_weights_row=None):
    """Daily borrow cost (as a positive number to SUBTRACT from short PnL) for a
    $1 short book. Names in HARD_TO_BORROW accrue 15% APR, the rest 4% APR.

    If `short_weights_row` (a Series of name->weight summing to ~1) is given, the
    drag is weight-weighted; otherwise equal weight is assumed."""
    members = list(short_members)
    if short_weights_row is None:
        w = pd.Series(1.0 / len(members), index=members)
    else:
        w = short_weights_row.reindex(members).fillna(0.0)
    apr = pd.Series([BORROW_APR_HTB if m in HARD_TO_BORROW else BORROW_APR_DEFAULT
                     for m in members], index=members)
    daily = (w * apr).sum() / TRADING_DAYS
    return float(daily)


# ---------------------------------------------------------------------------
# Rolling beta helper (vs SPY)
# ---------------------------------------------------------------------------
def rolling_beta(book_ret, mkt_ret, window):
    """Rolling beta of `book_ret` on `mkt_ret` over `window` days."""
    cov = book_ret.rolling(window).cov(mkt_ret)
    var = mkt_ret.rolling(window).var()
    return cov / var


# ---------------------------------------------------------------------------
# The three portfolio variants
# ---------------------------------------------------------------------------
def build_variants(returns, long_members, short_members, spy_ret,
                   apply_borrow=True, short_frag_weights=None):
    """Construct daily-return series for the three portfolio variants.

    (a) long_only         : the long book alone.
    (b) dollar_neutral     : +100% long book, -100% short book, rebalanced
                             monthly (equal weight each side). 100/100 gross.
    (c) beta_neutral       : long book minus a beta-scaled short book, where the
                             short notional is multiplied by k = beta_long/beta_short
                             (both trailing-60d betas vs SPY, fixed at prior
                             month-end -> no lookahead), so the combined book is
                             ~beta neutral.

    Borrow drag (if apply_borrow) is charged daily against whatever short
    notional the variant carries.
    Returns dict(name -> Series) plus a dict of diagnostics.
    """
    long_ret = equal_weight_book(returns, long_members)
    short_ret = equal_weight_book(returns, short_members)
    short_w = per_name_weight_paths(returns, short_members)

    # Daily borrow drag for a $1 short book, using the drifting short weights.
    borrow = pd.Series(0.0, index=returns.index)
    if apply_borrow:
        for d in returns.index:
            borrow.loc[d] = borrow_daily_drag(
                [m for m in short_members if m in returns.columns],
                short_w.loc[d] if d in short_w.index else None)

    # ---- (a) long only ----
    v_long = long_ret.copy()

    # ---- (b) dollar-neutral 100/100 ----
    # short leg PnL to the L/S book is -(short_ret) minus borrow on $1 short.
    v_dn = long_ret - short_ret - borrow

    # ---- (c) beta-neutral ----
    # Rolling 60d betas vs SPY; the scale k for month M uses betas as of the last
    # day of month M-1 (shift onto month-ends, forward-fill within the month).
    beta_long = rolling_beta(long_ret, spy_ret, BETA_WINDOW)
    beta_short = rolling_beta(short_ret, spy_ret, BETA_WINDOW)

    dates = returns.index
    month_ends = set(pd.Timestamp(d) for d in _month_end_index(dates))
    k = pd.Series(np.nan, index=dates)
    last_k = 1.0
    for d in dates:
        # At a month boundary, refresh k from the beta ratio observed *up to* the
        # prior day (betas are trailing, so beta.loc[d] already excludes future).
        if d in month_ends:
            bl, bs = beta_long.loc[d], beta_short.loc[d]
            if np.isfinite(bl) and np.isfinite(bs) and abs(bs) > 1e-6:
                last_k = float(np.clip(bl / bs, 0.0, 3.0))  # cap leverage sanity
        k.loc[d] = last_k
    # Apply k with a one-day lag so the scale is strictly known before the day.
    k_applied = k.shift(1).bfill()
    v_bn = long_ret - k_applied * short_ret - k_applied * borrow

    variants = {
        "long_only": v_long,
        "dollar_neutral": v_dn,
        "beta_neutral": v_bn,
    }
    diag = dict(
        long_ret=long_ret, short_ret=short_ret, borrow=borrow,
        beta_long=beta_long, beta_short=beta_short, k=k_applied,
        short_weights=short_w,
    )

    # ---- (d) dollar-neutral, BALANCE-SHEET-AWARE short ----
    # Same 100/100 structure, but the short leg is weighted by ex-ante financing
    # fragility (see fundamentals.py) instead of equal weight. Held as a static
    # structural tilt (monthly reset to the same fragility target). This is the
    # "fit the idea" test of iteration-1 conclusion #5.
    if short_frag_weights is not None:
        short_ret_f = equal_weight_book(returns, short_members,
                                        target=short_frag_weights)
        short_w_f = per_name_weight_paths(returns, short_members,
                                          target=short_frag_weights)
        borrow_f = pd.Series(0.0, index=returns.index)
        if apply_borrow:
            for d in returns.index:
                borrow_f.loc[d] = borrow_daily_drag(
                    [m for m in short_members if m in returns.columns],
                    short_w_f.loc[d] if d in short_w_f.index else None)
        variants["dollar_neutral_bsaware"] = long_ret - short_ret_f - borrow_f
        diag["short_ret_frag"] = short_ret_f
        diag["borrow_frag"] = borrow_f
        diag["short_weights_frag"] = short_w_f

    # ---- (e) dollar-neutral, INVERSE-VOLATILITY (risk-parity) both legs ----
    # The lesson from (d): the book's damage came from a few HYPER-VOLATILE names
    # sized equally by DOLLARS, not from balance sheet. Equalizing RISK instead of
    # dollars is the simple, general, no-lookahead, no-tuning fix. Weights are
    # proportional to 1/trailing-60d-vol, decided at each month-end and applied to
    # the next month. Directly caps the single-name-hostage problem from iter 1.
    dates = returns.index
    reset_dates = {dates[0]} | set(pd.Timestamp(d) for d in _month_end_index(dates))
    lv = inverse_vol_targets(returns, long_members, BETA_WINDOW, reset_dates)
    sv = inverse_vol_targets(returns, short_members, BETA_WINDOW, reset_dates)
    long_ret_rp = equal_weight_book(returns, long_members, target=lv)
    short_ret_rp = equal_weight_book(returns, short_members, target=sv)
    short_w_rp = per_name_weight_paths(returns, short_members, target=sv)
    borrow_rp = pd.Series(0.0, index=returns.index)
    if apply_borrow:
        for d in returns.index:
            borrow_rp.loc[d] = borrow_daily_drag(
                [m for m in short_members if m in returns.columns],
                short_w_rp.loc[d] if d in short_w_rp.index else None)
    variants["dollar_neutral_riskparity"] = long_ret_rp - short_ret_rp - borrow_rp
    diag["long_ret_rp"] = long_ret_rp
    diag["short_ret_rp"] = short_ret_rp
    diag["invvol_long"] = lv
    diag["invvol_short"] = sv

    return variants, diag


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def benchmark_returns(returns):
    """SPY and the QQQ/XLI equal-weight blend as daily-return Series."""
    out = {}
    if "SPY" in returns:
        out["SPY"] = returns["SPY"]
    if "QQQ" in returns and "XLI" in returns:
        out["BLEND(QQQ+XLI)"] = 0.5 * returns["QQQ"] + 0.5 * returns["XLI"]
    return out


def cum_nav(ret, start=1.0):
    """Cumulative NAV path from a daily-return series (compounded)."""
    return start * (1.0 + ret.fillna(0.0)).cumprod()
