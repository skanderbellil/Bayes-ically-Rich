#!/usr/bin/env python3
"""Re-fetch full OHLC for the Mega+Large PEAD names (the original fetch saved Close only).

OHLC lets us model an intraday stop exactly: a long stops at the open if the day GAPPED
below the level, otherwise at the level if the day's LOW touched it. Saves to
data/raw/pead/<TICKER>__ohlc.csv. Checkpoint/resume. Requires signal_panel.csv.
"""
import warnings; warnings.filterwarnings("ignore")
import time
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths

import yfinance as yf

panel = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
bucket = panel.drop_duplicates("ticker").set_index("ticker")["market_cap"].to_dict()
names = sorted(t for t, b in bucket.items() if b in ("Mega Cap", "Large Cap"))

done = skip = err = 0
for i, t in enumerate(names, 1):
    fp = paths.RAW_DIR / f"{t}__ohlc.csv"
    if fp.exists():
        skip += 1
        continue
    try:
        h = yf.Ticker(t).history(period="max", auto_adjust=True)
    except Exception:
        err += 1
        continue
    if h is None or len(h) == 0:
        err += 1
        continue
    h[["Open", "High", "Low", "Close"]].to_csv(fp)
    done += 1
    if done % 25 == 0:
        print(f"  [{i}/{len(names)}] fetched {done}, cached {skip}, err {err}")
    time.sleep(0.2)

print(f"OHLC re-fetch done: {done} fetched, {skip} cached, {err} err, {len(names)} total")
