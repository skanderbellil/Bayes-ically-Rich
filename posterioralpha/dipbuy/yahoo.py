"""Minimal Yahoo chart-API client (no yfinance dependency), cached on disk.

Used for (i) SPY / European index ETFs daily OHLC + adjusted close (core portfolio,
index gap), and (ii) recent intraday bars with pre/post-market (1m ~ last 7-30 days,
5m/15m ~ last 60 days, 60m ~ last 730 days) for execution checks the CTA EOD file
cannot do. Yahoo carries only currently-listed tickers -> intraday checks are
survivorship-biased and only cover the recent window; they are robustness checks.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .paths import YAHOO_DIR

URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}?{qs}"


def _get(url: str, timeout: int = 30) -> dict:
    for attempt in range(4):
        r = subprocess.run(["curl", "-s", "-m", str(timeout), "-A", "Mozilla/5.0", url], capture_output=True, text=True)
        try:
            j = json.loads(r.stdout)
            if j.get("chart", {}).get("result"):
                return j
            if j.get("chart", {}).get("error"):
                return j
        except json.JSONDecodeError:
            pass
        time.sleep(2 ** attempt)
    return {}


def chart(sym: str, interval: str = "1d", start: str | None = None, end: str | None = None,
          rng: str | None = None, prepost: bool = False, cache: bool = True, max_age_h: float = 1e9) -> pd.DataFrame:
    """OHLCV(+adjclose) indexed by exchange-local timestamp. Empty frame if unavailable."""
    key = f"{sym}_{interval}_{start}_{end}_{rng}_{int(prepost)}".replace("/", "-")
    path = Path(YAHOO_DIR) / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if cache and path.exists() and (time.time() - path.stat().st_mtime) / 3600 < max_age_h:
        j = json.loads(path.read_text())
    else:
        qs = f"interval={interval}&includePrePost={'true' if prepost else 'false'}&events=div%2Csplit"
        if rng:
            qs += f"&range={rng}"
        else:
            p1 = int(pd.Timestamp(start or "1990-01-01").timestamp()); p2 = int(pd.Timestamp(end or pd.Timestamp.now()).timestamp())
            qs += f"&period1={p1}&period2={p2}"
        j = _get(URL.format(sym=sym, qs=qs))
        if j:
            path.write_text(json.dumps(j))
    try:
        res = j["chart"]["result"][0]
    except (KeyError, TypeError, IndexError):
        return pd.DataFrame()
    ts = res.get("timestamp")
    if not ts:
        return pd.DataFrame()
    tz = res["meta"].get("exchangeTimezoneName", "America/New_York")
    q = res["indicators"]["quote"][0]
    df = pd.DataFrame({k: q.get(k) for k in ("open", "high", "low", "close", "volume")},
                      index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz))
    if "adjclose" in res["indicators"]:
        df["adjclose"] = res["indicators"]["adjclose"][0]["adjclose"]
    df = df.astype(float)
    df.attrs["meta"] = res["meta"]
    df.attrs["events"] = res.get("events", {})
    return df


def daily(sym: str, start: str = "2012-01-01", end: str | None = None) -> pd.DataFrame:
    df = chart(sym, "1d", start=start, end=end or "2026-11-01")
    if df.empty:
        return df
    df.index = pd.DatetimeIndex(df.index.tz_localize(None).normalize())
    return df[~df.index.duplicated(keep="last")].dropna(subset=["open", "close"])


def spy_daily() -> pd.DataFrame:
    """SPY with gap (open/prev close), intraday (close/open), total return (adjclose)."""
    s = daily("SPY", "2012-01-01")
    s["ret"] = s.adjclose.pct_change()
    s["gap"] = s.open / s.close.shift(1) - 1
    s["intra"] = s.close / s.open - 1
    return s
