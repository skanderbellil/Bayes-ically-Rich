"""Yahoo intraday features for NYSE / NYSE American names in the recent window (live tickers only).

60m bars + pre-market (2023-12 -> 2026-10): pre_last = last pre-market print before 09:30 (bar close),
    y_open = first regular-session trade (open of the 09:30 bar), h1_close = price at ~10:30,
    h1_typ = (H+L+C)/3 of the 09:30-10:30 bar (first-hour VWAP proxy).
5m bars + pre-market (last ~60 sessions): vwap5/15/30 = volume-weighted typical price of the first 1/3/6 bars,
    px0935 / px0945 = close of the 09:30 / 09:40 bar, pre5_last = last pre-market print.
Writes data/raw/dipbuy/intraday_60m.parquet and intraday_5m.parquet (one row per symbol-day).
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import time as T

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import yahoo
from posterioralpha.dipbuy.paths import RAW_DIR

P = pd.read_parquet(RAW_DIR / "panel.parquet", columns=["sym", "date", "Q", "prev_close"])
syms = sorted(P[(P.date >= "2023-12-01") & P.Q.notna() & (P.prev_close >= 5)].sym.unique())
syms = sorted(set(syms) | {"SPY"})
print(len(syms), "symbols", flush=True)


def feats60(s):
    d = yahoo.chart(s.replace("/", "-"), "60m", rng="730d", prepost=True)
    if d.empty:
        return None
    d = d.dropna(subset=["close"]); d["day"] = d.index.tz_localize(None).normalize(); t = d.index.time
    pre = d[t < T(9, 30)].groupby("day").close.last()
    h1 = d[(t >= T(9, 30)) & (t < T(10, 30))].groupby("day").first()
    out = pd.DataFrame({"pre_last": pre, "y_open": h1.open, "h1_close": h1.close, "h1_typ": (h1.high + h1.low + h1.close) / 3})
    out["sym"] = s
    return out.rename_axis("date").reset_index()


def feats5(s):
    d = yahoo.chart(s.replace("/", "-"), "5m", rng="60d", prepost=True)
    if d.empty:
        return None
    d = d.dropna(subset=["close"]); d["day"] = d.index.tz_localize(None).normalize(); t = d.index.time
    pre = d[t < T(9, 30)].groupby("day").close.last()
    reg = d[(t >= T(9, 30)) & (t < T(16, 0))].copy()
    reg["typ"] = (reg.high + reg.low + reg.close) / 3
    reg["k"] = reg.groupby("day").cumcount()
    out = {"pre5_last": pre, "y5_open": reg[reg.k == 0].set_index("day").open}
    for n, m in ((1, 5), (3, 15), (6, 30)):
        w = reg[reg.k < n]
        g = w.assign(pv=w.typ * w.volume).groupby("day")
        vw = g.pv.sum() / g.volume.sum().replace(0, np.nan)
        out[f"vwap{m}"] = vw.fillna(w.groupby("day").typ.mean())
    out["px0935"] = reg[reg.k == 0].set_index("day").close
    out["px0945"] = reg[reg.k == 2].set_index("day").close
    o = pd.DataFrame(out); o["sym"] = s
    return o.rename_axis("date").reset_index()


for name, fn in (("intraday_60m", feats60), ("intraday_5m", feats5)):
    with ThreadPoolExecutor(4) as ex:
        parts = [x for x in ex.map(fn, syms) if x is not None]
    D = pd.concat(parts, ignore_index=True)
    D.to_parquet(RAW_DIR / f"{name}.parquet", index=False)
    print(name, len(D), "rows", D.sym.nunique(), "symbols", D.date.min(), D.date.max(), flush=True)
