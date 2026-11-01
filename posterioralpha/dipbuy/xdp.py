"""NYSE Pillar XDP sample files (https://ftp.nyse.com/Historical Data Samples/) rendered as CSV.

Message types used:
  3   Symbol Index Mapping : sym, exchange code, security type (C common, P preferred, E ETF, U unit, W warrant, T test ...)
  220 Trade                : time, price, qty and four sale-condition bytes
        cond2 'O' = market-centre OPENING auction print, '6' = CLOSING auction print, '5' = re-opening
        cond3 'T' = extended-hours trade; cond4 'I' = odd lot, 'Q' = official open, 'M' = official close
  105 Imbalance            : pre-auction indicative data (published from 07:00/08:00 to the open)
        [5] reference price, [6] paired qty, [7] total imbalance qty, [9] auction time, [10] auction type
        ('M' core open, 'C' close, ...), [11] side, [12] continuous-book clearing price (the indicative open
        actually published pre-open on NYSE), [15] indicative match price (0 pre-open on NYSE)
Only a handful of sample days are public: trades 2023-10-02/03 (NYSE, NYSE American), imbalances
2020-09-09 (NYSE) and 2026-07-01/02 (NYSE, NYSE American).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


def _open(path: Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "rt")


def read_symbols_and(path: Path, msg: str):
    """Return (symbols DataFrame, list of split rows for message type ``msg``)."""
    syms, rows = [], []
    pref = msg + ","
    with _open(path) as f:
        for line in f:
            if line.startswith(pref):
                rows.append(line.rstrip("\n").split(","))
            elif line.startswith("3,"):
                p = line.rstrip("\n").split(",")
                syms.append((p[2], p[5], p[6]))
    S = pd.DataFrame(syms, columns=["sym", "exch", "stype"]).drop_duplicates("sym", keep="last")
    return S, rows


def read_trades(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    S, rows = read_symbols_and(path, "220")
    T = pd.DataFrame([(r[3], r[2], float(r[6]), int(r[7]), r[8], r[9], r[10], r[11] if len(r) > 11 else "") for r in rows],
                     columns=["sym", "time", "px", "qty", "c1", "c2", "c3", "c4"])
    T["ext"] = T.c3.isin(["T", "U"])
    return S, T


def auction_summary(T: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: opening/closing auction price & qty, first regular trade, early-session activity,
    first-N-minute VWAP/volume on this venue and the 09:35 / 09:45 last price."""
    reg = T[~T.ext & (T.time >= "09:30:00") & (T.time < "16:00:01")]
    o = T[T.c2 == "O"].groupby("sym").agg(auc_open=("px", "first"), auc_open_qty=("qty", "sum"), auc_open_t=("time", "first"))
    c = T[T.c2 == "6"].groupby("sym").agg(auc_close=("px", "last"), auc_close_qty=("qty", "sum"))
    pre = T[T.time < "09:30:00"].groupby("sym").agg(pre_last=("px", "last"), pre_qty=("qty", "sum"), pre_n=("px", "size"))
    f = reg.groupby("sym").agg(first_reg=("px", "first"), hi_reg=("px", "max"), lo_reg=("px", "min"),
                               last_reg=("px", "last"), vol_reg=("qty", "sum"))
    out = [o, c, pre, f]
    for m in (5, 15, 30):
        w = reg[reg.time < f"{9 + (30 + m) // 60:02d}:{(30 + m) % 60:02d}:00"]
        g = w.assign(pv=w.px * w.qty).groupby("sym")
        out.append(pd.DataFrame({f"vwap{m}": g.pv.sum() / g.qty.sum(), f"vol{m}": g.qty.sum()}))
    for hhmm in ("09:35", "09:45"):
        w = reg[reg.time < hhmm + ":00"]
        out.append(w.groupby("sym").px.last().rename(f"px{hhmm.replace(':', '')}"))
    return pd.concat(out, axis=1)


def read_imbalances(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    S, rows = read_symbols_and(path, "105")
    num = lambda x: float(x) if x.strip() not in ("", "0") else np.nan
    I = pd.DataFrame([(r[3], r[2], num(r[5]), float(r[6] or 0), float(r[7] or 0), r[9], r[10], r[11],
                       num(r[12]), num(r[15])) for r in rows if len(r) > 15],
                     columns=["sym", "time", "ref_px", "paired", "imb", "auc_time", "auc_type", "side", "book_px", "ind_px"])
    return S, I


def indicative_open(I: pd.DataFrame, cutoffs=("09:00", "09:15", "09:25", "09:28", "09:29:30")) -> pd.DataFrame:
    """Last indicative match price (and paired qty) published before each cutoff for the core open auction."""
    I = I[I.auc_type == "M"].sort_values("time").copy()
    # NYSE leaves the indicative-match field at 0 pre-open; the continuous-book clearing price is the
    # published indicative opening price. Use it when the match price is absent.
    I["ind_px"] = I.ind_px.fillna(I.book_px)
    out = []
    for c in cutoffs:
        w = I[(I.time < (c if len(c) > 5 else c + ":00")) & I.ind_px.notna()]
        g = w.groupby("sym").last()
        out.append(g[["ind_px", "paired", "ref_px"]].add_suffix("_" + c.replace(":", "")))
    return pd.concat(out, axis=1)
