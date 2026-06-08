"""Checkpoint/resume data fetcher for the PEAD pipeline (yfinance).

For each ticker we persist two artifacts under data/raw/pead/:
  - <TICKER>__earnings.csv : analyst earnings surprises (EPS Estimate, Reported EPS,
    Surprise(%)) dated by announcement -> the SUE signal source.
  - <TICKER>__prices.csv   : auto-adjusted daily closes -> forward returns.

Resume is automatic: a ticker with both files present (or a recorded failure) is
skipped. A status ledger tracks OK / NO_EARNINGS / NO_PRICES / ERROR so a long
pull can be interrupted and restarted without re-hitting Yahoo.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")


def _earn_path(d: Path, t: str) -> Path:
    return d / f"{t}__earnings.csv"


def _price_path(d: Path, t: str) -> Path:
    return d / f"{t}__prices.csv"


def fetch_one(ticker: str, out_dir: Path, earn_limit: int = 80) -> str:
    """Fetch earnings surprises + adjusted prices for one ticker.

    Returns a status string. Idempotent: skips work already on disk.
    """
    import yfinance as yf

    ep, pp = _earn_path(out_dir, ticker), _price_path(out_dir, ticker)
    if ep.exists() and pp.exists():
        return "CACHED"

    tk = yf.Ticker(ticker)

    # --- earnings surprises (the signal) ---
    try:
        ed = tk.get_earnings_dates(limit=earn_limit)
    except Exception as e:  # noqa: BLE001 - yfinance raises many ad-hoc types
        return f"ERROR:earnings:{type(e).__name__}"
    if ed is None or len(ed) == 0:
        return "NO_EARNINGS"
    ed = ed.dropna(subset=["Reported EPS", "EPS Estimate"], how="any")
    if len(ed) < 8:  # need history to standardize the surprise into a SUE
        return "NO_EARNINGS"
    ed.to_csv(ep)

    # --- adjusted prices (forward returns) ---
    try:
        px = tk.history(period="max", auto_adjust=True)
    except Exception as e:  # noqa: BLE001
        return f"ERROR:prices:{type(e).__name__}"
    if px is None or len(px) == 0:
        return "NO_PRICES"
    px[["Close"]].to_csv(pp)

    return "OK"


def fetch_universe(
    tickers: list[str],
    out_dir: str,
    ledger_path: str,
    sleep: float = 0.4,
    log_every: int = 25,
) -> pd.DataFrame:
    """Fetch all tickers with checkpoint/resume; returns the status ledger."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ledger = {}
    lp = Path(ledger_path)
    if lp.exists():
        prev = pd.read_csv(lp, index_col=0)
        ledger = prev["status"].to_dict()

    todo = [t for t in tickers if t not in ledger]
    print(f"fetch_universe: {len(tickers)} total, {len(todo)} to fetch, "
          f"{len(ledger)} already in ledger")

    for i, t in enumerate(todo, 1):
        status = fetch_one(t, out)
        ledger[t] = status
        if i % log_every == 0 or i == len(todo):
            s = pd.Series(ledger)
            print(f"  [{i}/{len(todo)}] last={t}:{status} | "
                  f"OK={(s=='OK').sum()} CACHED={(s=='CACHED').sum()} "
                  f"NO_EARN={(s=='NO_EARNINGS').sum()} "
                  f"ERR={s.str.startswith('ERROR').sum()}")
            pd.DataFrame({"status": ledger}).to_csv(lp)
        if status not in ("CACHED",):
            time.sleep(sleep)

    df = pd.DataFrame({"status": ledger})
    df.to_csv(lp)
    return df
