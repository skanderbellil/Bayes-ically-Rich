"""Enrich signal stock-days with Yahoo ex-dividend dates and earnings dates (live tickers only).

For every symbol with a gap signal (either side) from 2015 on: Yahoo daily bars with dividend/split events
and yfinance earnings dates. A Yahoo history is accepted for a given signal day only if Yahoo's
close-to-close return that day matches the CTA return within 1.5 points (guards against ticker reuse).
Writes data/raw/dipbuy/yahoo_events.parquet (one row per signal stock-day) and updates panel.exdiv_any.
"""
import sys
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import backtest as bt, yahoo
from posterioralpha.dipbuy.paths import RAW_DIR

P = pd.read_parquet(RAW_DIR / "panel.parquet")
thr = np.maximum(bt.THR_FLOOR, bt.THR_SD * P.sd60)
base = (P.year >= 2015) & P.Q.notna() & (P.prev_close >= 5) & P.sd60.notna() & (P.gap.abs() < 0.5) & ~P.split
cand = P[base & (P.agap.abs() >= thr)][["sym", "date", "ret"]].copy()
syms = sorted(cand.sym.unique())
print(len(cand), "signal stock-days,", len(syms), "symbols", flush=True)


def ysym(s):
    return s.replace("/", "-")


def one(s):
    out = {"sym": s}
    try:
        d = yahoo.daily(ysym(s), "2014-01-01")
        ev = d.attrs.get("events", {}) if not d.empty else {}
        out["divs"] = sorted({pd.Timestamp(int(k), unit="s").tz_localize("UTC").tz_convert("America/New_York").normalize().tz_localize(None)
                              for k in ev.get("dividends", {})})
        out["yret"] = d.close.pct_change() if not d.empty else pd.Series(dtype=float)
    except Exception:
        out["divs"], out["yret"] = [], pd.Series(dtype=float)
    try:
        import yfinance as yf
        e = yf.Ticker(ysym(s)).get_earnings_dates(limit=100)
        if e is not None and len(e):
            e = e.reset_index()
            e.columns = ["ts"] + list(e.columns[1:])
            out["earn"] = pd.DataFrame({"ts": e.ts.dt.tz_convert("America/New_York").dt.tz_localize(None),
                                        "est": e["EPS Estimate"].notna()})
    except Exception:
        pass
    return out


with ThreadPoolExecutor(4) as ex:
    res = {}
    for i, r in enumerate(ex.map(one, syms)):
        res[r["sym"]] = r
        if i % 200 == 0:
            print(i, flush=True)

days = pd.DatetimeIndex(sorted(P.date.unique()))
nxt = pd.Series(days[1:], index=days[:-1])
rows = []
for (s, dt, r) in cand.itertuples(index=False):
    o = res.get(s, {})
    yr = o.get("yret", pd.Series(dtype=float))
    ok = dt in yr.index and np.isfinite(yr.get(dt, np.nan)) and abs(yr[dt] - r) < 0.015
    exd = ok and dt in set(o.get("divs", []))
    e = o.get("earn")
    earn, covered = np.nan, np.nan
    if ok and e is not None:
        # gap-relevant announcement: after the previous close or before today's open
        ts = e.ts
        day = ts.dt.normalize()
        aft = ts.dt.hour >= 16
        unknown = (ts.dt.hour == 0) & (ts.dt.minute == 0)
        eff = day.where(~aft, day.map(lambda x: nxt.get(x, x + pd.Timedelta(days=1))))
        hit = (eff == dt) | (unknown & (day.map(lambda x: nxt.get(x, x)) == dt))
        earn = bool(hit.any())
        past = e[e.ts < dt]
        covered = bool(past.tail(4).est.any()) if len(past) else np.nan
    rows.append((s, dt, ok, exd, earn, covered))
Y = pd.DataFrame(rows, columns=["sym", "date", "yahoo_match", "y_exdiv", "earnings_day", "eps_covered"])
Y.to_parquet(RAW_DIR / "yahoo_events.parquet", index=False)
print(Y[["yahoo_match", "y_exdiv"]].mean().to_dict(), Y.earnings_day.value_counts(dropna=False).to_dict())
P = P.merge(Y[["sym", "date", "y_exdiv"]], on=["sym", "date"], how="left")
P["exdiv_any"] = P.exdiv | P.y_exdiv.fillna(False).astype(bool)
P.drop(columns="y_exdiv").to_parquet(RAW_DIR / "panel.parquet", index=False)
print("exdiv_any on signal days:", P.loc[P.set_index(["sym", "date"]).index.isin(Y.set_index(["sym", "date"]).index), "exdiv_any"].mean())
