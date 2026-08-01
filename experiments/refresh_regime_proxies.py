#!/usr/bin/env python3
"""
Refresh the exogenous regime-proxy datasets the live regime gate reads
======================================================================

The validated-regime strategy (run_validated_regime_paper_update.py) computes
"the CURRENT regime" from three committed daily series. Those files were
snapshotted once (2026-06-22/24) and nothing ever refreshed them — so the live
gate silently froze on a June view of the world (proxy_value stuck at 199.48
for five straight weeks of entries). This script keeps them current:

  datasets/gpr_daily.csv.gz    Caldara-Iacoviello daily GPR      matteoiacoviello.com
  datasets/epu_daily.csv       daily US EPU                      policyuncertainty.com
  datasets/btc_usd_daily.csv   BTC-USD daily close               Coinbase Exchange API

Merge policy: append-only union by date (fresh download wins on overlapping
dates, since sources revise recent days). A refreshed file is only written if
it does not shrink and does not move its last date backwards — a bad download
can never eat committed history.

Sources publish daily with a few days' lag, so the hourly cron doesn't need to
hit them every run: a source is skipped while its file's last date is within
--if-stale-days (default 2) of today. --force refreshes everything.

Exit status: non-zero if any *attempted* source failed (the workflow step
surfaces that loudly); sources that succeed are still written.

Usage:
  python experiments/refresh_regime_proxies.py [--if-stale-days 2] [--force]
"""
from __future__ import annotations
import argparse
import io
import logging
import sys
import time
from datetime import date, timedelta

import pandas as pd
import _bootstrap  # noqa: F401

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

GPR_FILE = _bootstrap.ROOT / "datasets" / "gpr_daily.csv.gz"
EPU_FILE = _bootstrap.ROOT / "datasets" / "epu_daily.csv"
BTC_FILE = _bootstrap.ROOT / "datasets" / "btc_usd_daily.csv"

# Some of these hosts 403 the default python-requests UA.
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) posterioralpha-proxy-refresh/1.0"}


def _download(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    import requests
    delay = 2.0
    last = None
    for _ in range(retries):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
            if resp.status_code == 200:
                return resp.content
            last = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except Exception as e:  # noqa: BLE001 — network errors retried, re-raised below
            last = e
        time.sleep(delay)
        delay *= 2
    raise last


def _merge_write(path, old: pd.DataFrame, new: pd.DataFrame, label: str) -> None:
    """Union by date, fresh rows winning on overlap; refuse to shrink history."""
    merged = (pd.concat([old, new], ignore_index=True)
                .drop_duplicates("date", keep="last")
                .sort_values("date").reset_index(drop=True))
    if len(merged) < len(old) or merged["date"].max() < old["date"].max():
        raise RuntimeError(f"{label}: merged result would lose history "
                           f"({len(old)}→{len(merged)} rows, last {old['date'].max()}→{merged['date'].max()})")
    merged.to_csv(path, index=False)
    logger.info("%s: %d rows (+%d), last date %s", label, len(merged), len(merged) - len(old),
                merged["date"].max())


def _is_fresh(path, if_stale_days: int) -> bool:
    if not path.exists():
        return False
    last = pd.to_datetime(pd.read_csv(path)["date"]).max().date()
    return last >= date.today() - timedelta(days=if_stale_days)


def refresh_gpr() -> None:
    """Daily geopolitical-risk index. Distributed as an Excel sheet; the column
    layout occasionally shifts, so locate the date column by name."""
    last_err = None
    for url in ("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
                "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xlsx"):
        try:
            raw = pd.read_excel(io.BytesIO(_download(url)))
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
    else:
        raise last_err
    date_col = next((c for c in raw.columns if str(c).strip().lower() in ("date", "day")), None)
    if date_col is None:
        raise RuntimeError(f"GPR: no date column in {list(raw.columns)[:8]}")
    cols = {}
    for want in ("GPRD", "GPRD_ACT", "GPRD_THREAT"):
        got = next((c for c in raw.columns if str(c).strip().upper() == want), None)
        if got is None:
            raise RuntimeError(f"GPR: missing column {want}")
        cols[want] = got
    d = pd.to_numeric(raw[date_col], errors="coerce")
    if d.notna().mean() > 0.9:                      # integer YYYYMMDD form
        dates = pd.to_datetime(d.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
    else:
        dates = pd.to_datetime(raw[date_col], errors="coerce")
    new = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"),
                        **{k: pd.to_numeric(raw[v], errors="coerce") for k, v in cols.items()}})
    new = new.dropna(subset=["date", "GPRD"]).reset_index(drop=True)
    if len(new) < 100:
        raise RuntimeError(f"GPR: implausibly small download ({len(new)} rows)")
    _merge_write(GPR_FILE, pd.read_csv(GPR_FILE), new, "gpr_daily.csv.gz")


def refresh_epu() -> None:
    """Daily US Economic Policy Uncertainty index."""
    raw = pd.read_csv(io.BytesIO(_download(
        "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv")))
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    val_col = next((c for c in raw.columns if "policy" in c or c == "epu"), None)
    if not {"year", "month", "day"} <= set(raw.columns) or val_col is None:
        raise RuntimeError(f"EPU: unexpected columns {list(raw.columns)}")
    dates = pd.to_datetime(dict(year=pd.to_numeric(raw["year"], errors="coerce"),
                                month=pd.to_numeric(raw["month"], errors="coerce"),
                                day=pd.to_numeric(raw["day"], errors="coerce")), errors="coerce")
    new = pd.DataFrame({"date": dates.dt.strftime("%Y-%m-%d"),
                        "EPU": pd.to_numeric(raw[val_col], errors="coerce").round(2)})
    new = new.dropna().reset_index(drop=True)
    if len(new) < 100:
        raise RuntimeError(f"EPU: implausibly small download ({len(new)} rows)")
    _merge_write(EPU_FILE, pd.read_csv(EPU_FILE), new, "epu_daily.csv")


def refresh_btc() -> None:
    """BTC-USD daily closes from the keyless Coinbase Exchange candles API
    (max 300 candles per request — page from the last cached date to today)."""
    import json
    old = pd.read_csv(BTC_FILE)
    start = pd.to_datetime(old["date"]).max() - pd.Timedelta(days=3)   # re-fetch a few days for revisions
    rows = []
    while start.date() <= date.today():
        end = min(start + pd.Timedelta(days=290), pd.Timestamp.now())
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity=86400&start={start.strftime('%Y-%m-%dT00:00:00Z')}"
               f"&end={end.strftime('%Y-%m-%dT23:59:59Z')}")
        candles = json.loads(_download(url))
        # each candle: [epoch, low, high, open, close, volume]
        rows += [(pd.Timestamp(c[0], unit="s").strftime("%Y-%m-%d"), float(c[4])) for c in candles]
        start = end + pd.Timedelta(days=1)
    new = pd.DataFrame(rows, columns=["date", "close"]).drop_duplicates("date").sort_values("date")
    if new.empty:
        raise RuntimeError("BTC: empty download")
    _merge_write(BTC_FILE, old, new, "btc_usd_daily.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--if-stale-days", type=int, default=2,
                    help="skip a source whose file already reaches within N days of today")
    ap.add_argument("--force", action="store_true", help="refresh every source regardless of freshness")
    args = ap.parse_args()

    failures = []
    for label, path, fn in (("GPR", GPR_FILE, refresh_gpr),
                            ("EPU", EPU_FILE, refresh_epu),
                            ("BTC", BTC_FILE, refresh_btc)):
        if not args.force and _is_fresh(path, args.if_stale_days):
            logger.info("%s: fresh enough (last date within %dd) — skipping", label, args.if_stale_days)
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — one bad source must not block the others
            logger.error("%s refresh FAILED: %s", label, e)
            failures.append(label)
    if failures:
        logger.error("proxy refresh failed for: %s (files left untouched)", ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
