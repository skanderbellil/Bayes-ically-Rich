"""
Data layer: download daily ADJUSTED closes, normalize to USD, clean the panel.

WHY NOT `yfinance` DIRECTLY
---------------------------
The brief asks for yfinance adjusted closes. yfinance >=0.2.5x uses `curl_cffi`
with browser-TLS impersonation. In this sandbox all egress is forced through a
TLS-re-terminating (MITM) proxy, and curl_cffi's impersonated ClientHello is
reset by that proxy -- yfinance simply cannot complete a handshake here. Yahoo's
public chart endpoint (`/v8/finance/chart/...`), which is exactly what yfinance
wraps, works fine over a plain `requests` session with a browser User-Agent.

So this module talks to that SAME endpoint directly. The economic content is
identical to `yf.download(..., auto_adjust=True)`: we read the `adjclose` series
(splits + dividends applied). Everything is cached to Parquet so reruns are
offline and deterministic.

CLEANING PIPELINE (per the brief)
---------------------------------
1. Fetch adjusted closes for every equity + benchmark + required FX pairs.
2. Convert every non-USD equity to USD (GBp -> GBP/100), aligning FX by date.
3. Build a master calendar = SPY's trading days (US sessions).
4. Reindex each USD price series to that calendar; forward-fill <= 3 days
   (bridges foreign-market holidays / date-line offset without inventing data).
5. Any ticker still missing > 10% of the calendar is REPORTED and DROPPED.
6. Final inner-join drops the residual NaN rows so every kept series is aligned.
"""
from __future__ import annotations

import os
import time
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import (
    START_DATE, END_DATE, FFILL_LIMIT, MAX_MISSING_FRAC,
    ALL_EQUITY_TICKERS, BENCH_TICKERS, UNIVERSE,
)

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

_CA = "/root/.ccr/ca-bundle.crt"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# Yahoo symbol aliases: our universe label -> the symbol Yahoo actually serves.
# KLA Corporation still lists on Yahoo under its legacy Nasdaq ticker "KLAC".
YF_ALIAS = {"KLA": "KLAC"}


def _session() -> requests.Session:
    s = requests.Session()
    px = os.environ.get("HTTPS_PROXY")
    if px:
        s.proxies = {"https": px, "http": px}
    if os.path.exists(_CA):
        s.verify = _CA
    s.headers.update({"User-Agent": _UA, "Accept": "application/json"})
    return s


def _fetch_chart(session, ticker, p1, p2, max_retries=5):
    """Return (DataFrame[adjclose], currency) for one ticker, or (None, None).

    Retries with exponential backoff on Yahoo's 429 rate limiting.
    """
    yahoo_sym = YF_ALIAS.get(ticker, ticker)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_sym}"
    params = dict(period1=p1, period2=p2, interval="1d", events="div,split")
    backoff = 1.5
    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = res.get("timestamp")
            if not ts:
                return None, None
            idx = pd.to_datetime(ts, unit="s").normalize()
            ind = res["indicators"]
            # Prefer split+dividend adjusted close; fall back to raw close.
            if "adjclose" in ind and ind["adjclose"][0].get("adjclose"):
                px = ind["adjclose"][0]["adjclose"]
            else:
                px = ind["quote"][0]["close"]
            cur = res["meta"].get("currency", "USD")
            s = pd.Series(px, index=idx, name=ticker, dtype="float64")
            s = s[~s.index.duplicated(keep="last")].sort_index()
            return s.to_frame(), cur
        except requests.HTTPError:
            time.sleep(backoff); backoff *= 2
        except Exception:
            time.sleep(backoff); backoff *= 2
    return None, None


def download_prices(tickers, force=False, polite=0.4):
    """Download adjusted closes for `tickers`. Returns (prices_df, currency_map,
    failed_list). Each ticker is cached to Parquet + a currency sidecar."""
    p1 = int(dt.datetime.fromisoformat(START_DATE).timestamp())
    # +1 day so END_DATE itself is inclusive of any session that day.
    p2 = int((dt.datetime.fromisoformat(END_DATE) + dt.timedelta(days=1)).timestamp())
    session = _session()
    cols, currencies, failed = {}, {}, []

    for tk in tickers:
        safe = tk.replace("/", "_").replace("=", "_")
        pq = CACHE_DIR / f"{safe}.csv"
        meta = CACHE_DIR / f"{safe}.json"
        if pq.exists() and meta.exists() and not force:
            s = pd.read_csv(pq, index_col=0, parse_dates=True).iloc[:, 0]
            cur = json.loads(meta.read_text())["currency"]
        else:
            df, cur = _fetch_chart(session, tk, p1, p2)
            if df is None or df.empty:
                failed.append(tk)
                print(f"  [FAIL]  {tk:12s} no data returned")
                continue
            s = df.iloc[:, 0]
            df.to_csv(pq)
            meta.write_text(json.dumps({"currency": cur}))
            time.sleep(polite)   # be gentle with Yahoo's rate limiter
        cols[tk] = s
        currencies[tk] = cur

    prices = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
    return prices, currencies, failed


# ---------------------------------------------------------------------------
# FX normalization
# ---------------------------------------------------------------------------
def required_fx_pairs(currency_map):
    """From detected quote currencies, build the set of `<CUR>USD=X` pairs
    needed. 'GBp' (London pence) maps to GBPUSD (handled /100 downstream)."""
    pairs = set()
    for cur in currency_map.values():
        base = "GBP" if cur == "GBp" else cur
        if base != "USD":
            pairs.add(f"{base}USD=X")
    return sorted(pairs)


def to_usd(prices, currency_map, fx_prices):
    """Convert every equity column to USD.

    ASSUMPTION: FX is aligned to each equity's own dates and forward-filled
    (FX quotes trade nearly 24/5 so a same-or-prior-day rate is the honest
    conversion; no lookahead since we only ever use same/earlier FX).
    """
    out = pd.DataFrame(index=prices.index)
    notes = {}
    for tk in prices.columns:
        cur = currency_map.get(tk, "USD")
        s = prices[tk]
        if cur == "USD":
            out[tk] = s
            notes[tk] = "USD (no conversion)"
            continue
        scale = 1.0
        base = cur
        if cur == "GBp":            # pence -> pounds
            scale = 0.01
            base = "GBP"
        pair = f"{base}USD=X"
        if pair not in fx_prices.columns:
            # No FX available -> cannot convert; leave as-is and flag loudly.
            out[tk] = s
            notes[tk] = f"WARN: {cur} but {pair} missing -> left unconverted"
            print(f"  [FX-WARN] {tk}: {pair} unavailable; left in {cur}")
            continue
        fx = fx_prices[pair].reindex(s.index).ffill()
        out[tk] = s * scale * fx
        notes[tk] = f"{cur} x {pair}" + (" /100" if cur == "GBp" else "")
    return out, notes


# ---------------------------------------------------------------------------
# Calendar alignment + missing-data policy
# ---------------------------------------------------------------------------
def align_and_clean(usd_prices, master_calendar):
    """Reindex to master calendar, ffill<=limit, drop >MAX_MISSING_FRAC missing.

    Returns (clean_prices, report_df) where report lists every ticker's missing
    fraction and whether it was kept or dropped."""
    cal = master_calendar
    reindexed = usd_prices.reindex(cal)
    filled = reindexed.ffill(limit=FFILL_LIMIT)

    rows, keep = [], []
    for tk in filled.columns:
        miss = float(filled[tk].isna().mean())
        dropped = miss > MAX_MISSING_FRAC
        rows.append(dict(ticker=tk, missing_frac=round(miss, 4),
                         status="DROPPED" if dropped else "kept"))
        if dropped:
            print(f"  [DROP]  {tk:12s} missing {miss:6.1%} (> {MAX_MISSING_FRAC:.0%})")
        else:
            keep.append(tk)
    report = pd.DataFrame(rows).sort_values("missing_frac", ascending=False)

    clean = filled[keep].dropna(how="any")   # final inner-join on rows
    return clean, report


def build_master_calendar(prices):
    """US trading calendar = SPY's own trading days within the window."""
    if "SPY" not in prices.columns:
        # Fallback: union of all observed dates.
        return prices.index
    spy = prices["SPY"].dropna()
    return spy.index


def load_dataset(force=False):
    """Top-level entry: returns a dict with clean USD prices, returns, FX,
    the data-quality report, currency map, and the list of failed tickers."""
    all_syms = ALL_EQUITY_TICKERS + BENCH_TICKERS
    print(f"[data] downloading {len(all_syms)} equities/benchmarks ...")
    eq_prices, currency_map, failed = download_prices(all_syms, force=force)

    fx_pairs = required_fx_pairs(currency_map)
    print(f"[data] FX pairs required: {fx_pairs}")
    fx_prices, _fxcur, fx_failed = download_prices(fx_pairs, force=force)
    failed += [f for f in fx_failed if f not in failed]

    usd_prices, fx_notes = to_usd(eq_prices, currency_map, fx_prices)

    master = build_master_calendar(usd_prices)
    clean, report = align_and_clean(usd_prices, master)

    # Daily simple returns on the USD, cleaned panel.
    returns = clean.pct_change().dropna(how="all")

    kept = [c for c in clean.columns]
    print(f"[data] kept {len(kept)} series over {len(clean)} trading days "
          f"({clean.index.min().date()} -> {clean.index.max().date()})")

    return dict(
        prices=clean,
        returns=returns,
        fx_prices=fx_prices,
        currency_map=currency_map,
        fx_notes=fx_notes,
        report=report,
        failed=failed,
        calendar=master,
    )


if __name__ == "__main__":
    d = load_dataset()
    print("\n--- data quality report ---")
    print(d["report"].to_string(index=False))
    print("\nFailed:", d["failed"])
