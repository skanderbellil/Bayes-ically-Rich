"""CTA (Consolidated Tape) end-of-day summary files -> daily per-venue panel.

Source: https://ftp.nyse.com/cts_summary_files/CTA.Summary.EODSUM.YYYYMMDD.csv
(public, free, 2014-01-02 onward). One file per trading day lists, for *every*
security on the CTA tape that day (NYSE-, NYSE American-, Arca- and Cboe-listed;
NOT Nasdaq-listed), a consolidated summary row plus one row per trading venue
with that venue's open / high / low / last / volume.

Why this matters for the study:
* The primary listing venue's open (NYSE / NYSE American) is its first
  regular-session print, i.e. the **official opening auction** price, and its
  last trade is the **closing auction** price (verified against XDP trade
  conditions in ``experiments/run_dipbuy_verify_open.py``).
* Each file is a point-in-time snapshot, so delisted names are present up to
  their last trading day: **no survivorship filter**.
* Consolidated high/low/volume (all venues, regular session) let us measure
  how often the auction print is the day's extreme across the whole market.

Two schemas exist: "legacy" (2014 - early 2019; ``Con_EOD``/``Part_EOD``) and
"Pillar" (2019 -; ``ConsEOD``/``ConsolidatedEODSummary`` ...). Each file holds
several snapshots (16:15, 17:05, 18:40, 20:04 ET); the latest one wins.
"""
from __future__ import annotations

import csv
import io
import re
import subprocess
from statistics import median

import numpy as np
import pandas as pd

BASE_URL = "https://ftp.nyse.com/cts_summary_files/"

# ETFs on the CTA tape kept for market benchmarks / gap references.
ETF_KEEP = {"SPY", "IWM", "IWC", "IJR", "MDY", "IVV", "VTI", "DIA", "EFA", "IEUR", "VGK", "FEZ"}
KEEP_PRIMARY = {"NYSE", "AMEX"}

_VENUE_ALIASES = {
    "NYSE_MKT": "AMEX", "NYSE_AMERICAN": "AMEX", "NYSE_AMEX": "AMEX", "AMEX": "AMEX",
    "NYSE_ARCA": "ARCA", "ARCA": "ARCA",
    "BATS": "BZX", "BATS_BZX": "BZX", "CBOE_BZX": "BZX",
    "NSDQ": "NASDAQ", "NASDAQ": "NASDAQ",
}


def norm_venue(v: str) -> str:
    v = v.strip().upper().replace("-", "_")
    return _VENUE_ALIASES.get(v, v)


def _px(s: str) -> float:
    s = s.strip().lstrip("_")
    try:
        x = float(s)
    except ValueError:
        return np.nan
    return x if x > 0 else np.nan


def _vol(s: str) -> float:
    s = s.strip().lstrip("_")
    try:
        return float(s)
    except ValueError:
        return np.nan


def list_eod_files(timeout: int = 120) -> pd.DataFrame:
    """Return DataFrame(date, name, size) of the available EODSUM files."""
    html = subprocess.run(["curl", "-s", "-m", str(timeout), BASE_URL], capture_output=True, text=True).stdout
    rows = re.findall(r'(CTA\.Summary\.EODSUM\.(\d{8})\.csv)</a>\s+\S+\s+\S+\s+(\d+)', html)
    df = pd.DataFrame(rows, columns=["name", "date", "size"]).drop_duplicates("name")
    df["date"] = pd.to_datetime(df.date)
    df["size"] = df["size"].astype(int)
    return df.sort_values("date").reset_index(drop=True)


def fetch(name: str, timeout: int = 600) -> bytes:
    r = subprocess.run(["curl", "-s", "-f", "--retry", "4", "--retry-delay", "3", "-m", str(timeout), BASE_URL + name],
                       capture_output=True)
    if r.returncode != 0 or len(r.stdout) < 1000:
        raise IOError(f"download failed for {name}: rc={r.returncode} len={len(r.stdout)}")
    return r.stdout


def parse_eod(raw: bytes | str, keep_all: bool = False) -> pd.DataFrame:
    """Parse one EODSUM file into one row per symbol (last snapshot wins).

    Columns: sym, net, prim, finstat,
      cons_hi, cons_lo, cons_last, cons_vol            (consolidated, regular session)
      p_open, p_hi, p_lo, p_last, p_vol                (primary listing venue)
      fin_vol                                          (FINRA TRF = off-exchange volume)
      alt_open_med, alt_open_lo, alt_open_hi, nven     (other venues' first prints)
    """
    text = raw.decode("latin-1") if isinstance(raw, bytes) else raw
    rdr = csv.reader(io.StringIO(text))
    header = None
    for row in rdr:
        if len(row) > 8:
            header = [h.strip() for h in row]
            break
    if header is None:
        raise ValueError("no header")
    legacy = header[0] == "MsgCat"
    cons: dict[str, tuple] = {}
    part: dict[tuple, tuple] = {}
    if legacy:
        # MsgCat,MsgType,Network,SeqNum,OutputTime,Symbol,Suffix,PartId,PrClsDate,HighPrice,LowPrice,LastPrice,TotVolume,OpenPrice,Tick,FinStat,ShortSale
        for r in rdr:
            if len(r) < 14:
                continue
            mt = r[1]
            sym = (r[5].strip() + (" " + r[6].strip() if r[6].strip() else "")).strip()
            t = r[4]
            traded = r[8].strip() in ("", "_")
            rec = (t, norm_venue(r[7]), r[2].strip(), _px(r[9]), _px(r[10]), _px(r[11]), _vol(r[12]), _px(r[13]),
                   r[15].strip() if len(r) > 15 else "", traded)
            if mt == "Con_EOD":
                if sym not in cons or cons[sym][0] <= t:
                    cons[sym] = rec
            elif mt == "Part_EOD":
                k = (sym, rec[1])
                if k not in part or part[k][0] <= t:
                    part[k] = rec
    else:
        # MsgCat/Type,TransTime,SeqNum,Symbol,instrumentType,partiId,prevClosePriceDate,lastClosePrice,highPrice,lowPrice,openPrice,totalVolume,ssr,primaryListingMarketPartiId,financialStatusInd,numberOfParticipants
        for r in rdr:
            if len(r) < 12:
                continue
            mt = r[0].strip()
            sym = r[3].strip()
            t = r[1]
            traded = r[6].strip() in ("", "_")
            if mt.startswith("Cons"):
                prim = norm_venue(r[13]) if len(r) > 13 and r[13].strip() else norm_venue(r[5])
                rec = (t, prim, "", _px(r[8]), _px(r[9]), _px(r[7]), _vol(r[11]), np.nan,
                       r[14].strip() if len(r) > 14 else "", traded)
                if sym not in cons or cons[sym][0] <= t:
                    cons[sym] = rec
            elif mt.startswith("Part"):
                rec = (t, norm_venue(r[5]), "", _px(r[8]), _px(r[9]), _px(r[7]), _vol(r[11]), _px(r[10]), "", traded)
                k = (sym, rec[1])
                if k not in part or part[k][0] <= t:
                    part[k] = rec
    by_sym: dict[str, list] = {}
    for (sym, ven), rec in part.items():
        by_sym.setdefault(sym, []).append(rec)
    out = []
    for sym, c in cons.items():
        if legacy:
            # legacy Con_EOD.PartId is the venue of the LAST trade, not the listing market.
            # Network A = NYSE-listed; network B = NYSE American / Arca / Cboe-listed (resolved later).
            prim = "NYSE" if c[2] == "A" else ("ARCA" if sym in ETF_KEEP else "AMEX?")
        else:
            prim = c[1]
        if not keep_all and prim not in KEEP_PRIMARY and prim != "AMEX?" and sym not in ETF_KEEP:
            continue
        pv = prim.rstrip("?")
        p = None; fin_vol = amex_vol = arca_vol = 0.0; alt = []
        for rec in by_sym.get(sym, []):
            if not rec[9] or not (rec[6] > 0):
                continue  # venue did not trade today (row echoes a prior close)
            if rec[1] == "AMEX":
                amex_vol = rec[6]
            elif rec[1] == "ARCA":
                arca_vol = rec[6]
            if rec[1] == pv:
                p = rec
            elif rec[1] == "FINRA":
                fin_vol = rec[6]
            elif rec[7] == rec[7]:
                alt.append(rec[7])
        if p is None:
            p = (None, pv, "", np.nan, np.nan, np.nan, 0.0, np.nan, "", False)
        out.append((sym, c[2], prim, c[8], c[3], c[4], c[5], c[6] if c[9] else 0.0,
                    p[7], p[3], p[4], p[5], p[6], fin_vol, amex_vol, arca_vol,
                    median(alt) if alt else np.nan, min(alt) if alt else np.nan, max(alt) if alt else np.nan, len(alt)))
    cols = ["sym", "net", "prim", "finstat", "cons_hi", "cons_lo", "cons_last", "cons_vol",
            "p_open", "p_hi", "p_lo", "p_last", "p_vol", "fin_vol", "amex_vol", "arca_vol", "alt_open_med", "alt_open_lo", "alt_open_hi", "nven"]
    df = pd.DataFrame(out, columns=cols)
    for c in cols[4:]:
        df[c] = df[c].astype("float32") if c not in ("cons_vol", "p_vol", "fin_vol", "amex_vol", "arca_vol") else df[c].astype("float64")
    df["nven"] = df.nven.astype("int16")
    return df
