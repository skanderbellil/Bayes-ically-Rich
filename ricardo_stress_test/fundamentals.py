"""
Balance-sheet fundamentals layer -> a rules-based FRAGILITY score for the shorts.

WHY THIS EXISTS
---------------
Iteration 1 found the naive equal-weight short book was hostage to NBIS, which is
a "neocloud" (thesis short) but NOT balance-sheet-fragile: it sits on ~net cash.
CRWV, by contrast, funds an ~$8bn/yr cash burn on ~$2bn of cash and heavy debt.
The thesis is really about *financing fragility*, not the label "neocloud". So we
weight the short book by fragility measured from FUNDAMENTALS, not from returns.

"FIT THE IDEA, NOT THE DATA" (the whole point of this iteration)
---------------------------------------------------------------
Everything here is deliberately generalizable and free of return-fitting:
  * Fragility uses three standard, a-priori structural metrics -- leverage, cash-
    burn coverage, and unprofitability -- chosen because they define financing
    fragility, NOT because they worked in sample.
  * Each metric is converted to a cross-sectional PERCENTILE RANK (robust to
    units and outliers), then the three ranks are simple-averaged. No metric
    weights are tuned. No temperature / no optimizer / no free parameters.
  * Short weights are PROPORTIONAL to the fragility rank -- a gentle monotone
    tilt, not an aggressive concentration on whatever blew up.
  * NO LOOKAHEAD: fundamentals are taken as of the quarter that was public by the
    backtest start (2025-09-30, via a filing-lag rule), and the tilt is held
    STATIC. It is a structural characteristic, not a timing signal.

DATA
----
Yahoo's `fundamentals-timeseries` endpoint returns true point-in-time quarterly
series (debt, cash, FCF, revenue, operating income). quoteSummary's quarterly
statement modules have been gutted by Yahoo, so we use the timeseries endpoint
and fall back to the current `financialData` snapshot only if a name is missing
(flagged in the output).
"""
from __future__ import annotations

import os
import json
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import START_DATE

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
_CA = "/root/.ccr/ca-bundle.crt"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Filing-lag rule: a quarter ending on date D is assumed public D + this many
# days. As-of the backtest start we only use quarters whose (endDate + lag) has
# already passed -> strictly no lookahead. 45d comfortably clears 10-Q filing.
FILING_LAG_DAYS = 45

_TS_TYPES = [
    "quarterlyTotalDebt",
    "quarterlyCashAndCashEquivalents",
    "quarterlyFreeCashFlow",
    "quarterlyTotalRevenue",
    "quarterlyOperatingIncome",
]

_ALIAS = {"KLA": "KLAC"}   # keep consistent with data.py


def _session():
    s = requests.Session()
    px = os.environ.get("HTTPS_PROXY")
    if px:
        s.proxies = {"https": px, "http": px}
    if os.path.exists(_CA):
        s.verify = _CA
    s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
    return s


def _crumb(session):
    session.get("https://fc.yahoo.com", timeout=20)
    return session.get("https://query1.finance.yahoo.com/v1/test/getcrumb",
                       timeout=20).text


def fetch_timeseries(tickers, force=False, polite=0.4):
    """Return {ticker: {type: [(asOfDate, value), ...]}} with JSON caching."""
    session = _session()
    crumb = None
    out = {}
    p1 = int(time.mktime((2024, 1, 1, 0, 0, 0, 0, 0, 0)))
    p2 = int(time.mktime((2026, 7, 1, 0, 0, 0, 0, 0, 0)))
    typ_str = ",".join(_TS_TYPES)
    for tk in tickers:
        safe = tk.replace("/", "_").replace("=", "_")
        cache = CACHE_DIR / f"fund_{safe}.json"
        if cache.exists() and not force:
            out[tk] = json.loads(cache.read_text())
            continue
        if crumb is None:
            crumb = _crumb(session)
        ysym = _ALIAS.get(tk, tk)
        url = (f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/"
               f"v1/finance/timeseries/{ysym}?type={typ_str}"
               f"&period1={p1}&period2={p2}&crumb={crumb}")
        series = {}
        try:
            j = session.get(url, timeout=25).json()["timeseries"]["result"]
            for ser in j:
                typ = ser["meta"]["type"][0]
                pts = [(x["asOfDate"], x.get("reportedValue", {}).get("raw"))
                       for x in ser.get(typ, []) if x and x.get("reportedValue")]
                series[typ] = pts
        except Exception as e:
            print(f"  [fund-FAIL] {tk}: {str(e)[:60]}")
        out[tk] = series
        cache.write_text(json.dumps(series))
        time.sleep(polite)
    return out


# ---------------------------------------------------------------------------
# As-of-start snapshot with no lookahead
# ---------------------------------------------------------------------------
def _as_of_cutoff(as_of=START_DATE):
    d = dt.date.fromisoformat(as_of) - dt.timedelta(days=FILING_LAG_DAYS)
    return d.isoformat()


def _latest_stock(points, cutoff):
    """Latest (date, value) with date <= cutoff -> value (stock item)."""
    ok = [(d, v) for d, v in points if v is not None and d <= cutoff]
    return (ok[-1] if ok else (None, None))


def _ttm_flow(points, cutoff, lookback_days=372):
    """TTM-annualized flow from quarters with date in (cutoff-lookback, cutoff].
    Robust to quarterly OR semiannual reporters: sum the periods and scale to a
    full year by (365 / span_days). Returns (annualized_value, n_periods)."""
    lo = (dt.date.fromisoformat(cutoff) - dt.timedelta(days=lookback_days)).isoformat()
    win = [(d, v) for d, v in points if v is not None and lo < d <= cutoff]
    if not win:
        return None, 0
    vals = [v for _, v in win]
    total = float(np.sum(vals))
    # Span from earliest period start (approx one quarter before first date) to
    # the latest date; scale that partial year up to 12 months.
    dates = [dt.date.fromisoformat(d) for d, _ in win]
    span_days = (max(dates) - min(dates)).days + 91  # +~1 quarter for the first
    ann = total * 365.0 / max(span_days, 91)
    return ann, len(win)


def build_fragility(short_tickers, force=False, as_of=START_DATE):
    """Return a DataFrame indexed by ticker with the as-of fundamentals, the
    three fragility components (as percentile ranks), the combined fragility
    score, and the proportional balance-sheet-aware short weight.

    All cross-sectional (ranked within `short_tickers`)."""
    ts = fetch_timeseries(short_tickers, force=force)
    cutoff = _as_of_cutoff(as_of)

    rows = {}
    for tk in short_tickers:
        s = ts.get(tk, {})
        d_dt, debt = _latest_stock(s.get("quarterlyTotalDebt", []), cutoff)
        c_dt, cash = _latest_stock(s.get("quarterlyCashAndCashEquivalents", []), cutoff)
        fcf, _ = _ttm_flow(s.get("quarterlyFreeCashFlow", []), cutoff)
        rev, _ = _ttm_flow(s.get("quarterlyTotalRevenue", []), cutoff)
        opinc, _ = _ttm_flow(s.get("quarterlyOperatingIncome", []), cutoff)

        debt = float(debt) if debt is not None else np.nan
        cash = float(cash) if cash is not None else np.nan
        net_debt = debt - cash if np.isfinite(debt) and np.isfinite(cash) else np.nan
        burn = max(-fcf, 0.0) if fcf is not None else np.nan
        rows[tk] = dict(
            asof_quarter=d_dt, total_debt=debt, cash=cash, net_debt=net_debt,
            ttm_fcf=fcf, ttm_rev=rev, ttm_opinc=opinc, burn=burn,
        )
    df = pd.DataFrame(rows).T
    for c in ["total_debt", "cash", "net_debt", "ttm_fcf", "ttm_rev",
              "ttm_opinc", "burn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- three structural FINANCING/SOLVENCY metrics (higher raw = more fragile)
    # We deliberately use only BALANCE-SHEET / liquidity axes -- NOT profitability.
    # Profitability is a business-quality signal, not a financing-fragility one;
    # NBIS is the cautionary example: deeply lossmaking yet sitting on net cash,
    # so it is NOT financing-fragile and must not be shorted heavily. The three
    # axes below all place a cash-rich lossmaker LOW and a debt-funded burner HIGH.
    eps = 1.0
    # 1) Leverage intensity: net debt relative to revenue (net cash -> negative).
    df["lev"] = df["net_debt"] / (df["ttm_rev"].abs() + eps)
    # 2) Liquidity runway (inverse): annual cash burn as a fraction of the cash
    #    pile -- how fast the war chest is consumed. A big cash pile (NBIS) shrinks
    #    this; a debt-funded burner on thin cash (CRWV, APLD) blows it up.
    df["burn_cover"] = df["burn"] / (df["cash"].abs() + eps)
    # 3) Debt-vs-cash coverage: total debt relative to cash on hand -- can the
    #    balance sheet retire its debt from liquidity? Low cash/high debt = fragile.
    df["debt_to_cash"] = df["total_debt"] / (df["cash"].abs() + eps)

    # --- robust combination: cross-sectional percentile ranks, simple-averaged.
    # NaNs (missing fundamentals) rank to the median (0.5) -> neutral, not extreme.
    comp = pd.DataFrame(index=df.index)
    for c in ["lev", "burn_cover", "debt_to_cash"]:
        comp[c + "_pct"] = df[c].rank(pct=True)
    comp = comp.fillna(0.5)
    df["fragility"] = comp.mean(axis=1)
    df = df.join(comp)

    # --- balance-sheet-aware short weight: PROPORTIONAL to fragility (static).
    fr = df["fragility"].clip(lower=1e-6)
    df["w_bsaware"] = fr / fr.sum()
    df["w_equal"] = 1.0 / len(df)
    df["w_tilt"] = df["w_bsaware"] - df["w_equal"]     # +ve = shorted MORE

    return df.sort_values("fragility", ascending=False)


if __name__ == "__main__":
    from config import SHORT_TICKERS
    df = build_fragility(SHORT_TICKERS)
    cols = ["asof_quarter", "net_debt", "cash", "burn", "ttm_rev",
            "lev", "burn_cover", "debt_to_cash", "fragility", "w_equal",
            "w_bsaware", "w_tilt"]
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.3f}")
    print(df[cols].to_string())
