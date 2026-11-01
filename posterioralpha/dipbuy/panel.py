"""Build the analysis panel from the stacked CTA EOD data (pure transformation, no I/O beyond reading).

Conventions (kept from the first pass where possible):
  open   = primary-venue opening-auction print (NYSE / NYSE American official open)
  close  = primary-venue last print (closing auction when one prints)
  gap    = open / prev close - 1 ; ret = close / prev close - 1 ; intra = close / open - 1
  ar     = ret - SPY total return ; agap = gap - SPY gap
  sd60   = std of ar over the prior 60 sessions (min 50), excluding today
  Q      = quintile of PRIOR calendar-year median daily $ volume (consolidated), among stock-years with
           prior-year median price >= $5 and >= 200 sessions (1 = most neglected)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

COMMON_RE = re.compile(r"^[A-Z]{1,5}(/[A-F])?$")
SPLIT_RATIOS = np.array([1 / 2, 1 / 3, 2 / 3, 1 / 4, 1 / 5, 1 / 10, 2 / 5])   # 2:1 3:1 3:2 4:1 5:1 10:1 5:2


def resolve_listing(p: pd.DataFrame, pillar_start="2019-02-01", amex_share_thr=0.02) -> pd.Series:
    """Map legacy 'AMEX?' (CTA network B) symbols to AMEX or drop.

    Rule 1: the symbol's listing market as reported in the Pillar-era files (if it survived into 2019+
            under the same ticker and was kept, it is NYSE/AMEX listed; ETFs listed on Arca/Cboe are not kept).
    Rule 2 (names gone before 2019): NYSE American venue share of consolidated volume > threshold
            AND the NYSE American venue printed an open on most days (an opening auction runs there).
    """
    leg = p[p.prim == "AMEX?"]
    pil = p[(p.date >= pillar_start) & p.prim.isin(["AMEX", "NYSE"])].groupby("sym").prim.agg(lambda s: s.mode().iat[0])
    st = leg.groupby("sym").agg(amex=("amex_vol", "sum"), cons=("cons_vol", "sum"), has_open=("p_open", lambda s: s.notna().mean()),
                                last=("date", "max"))
    st["share"] = st.amex / st.cons.replace(0, np.nan)
    lab = pd.Series(index=st.index, dtype=object)
    in_pil = st.index.intersection(pil.index)
    lab.loc[in_pil] = pil.loc[in_pil]
    rest = st.index.difference(in_pil)
    ok = (st.loc[rest, "share"] > amex_share_thr) & (st.loc[rest, "has_open"] > 0.5)
    lab.loc[rest] = np.where(ok, "AMEX", "DROP")
    return lab


def build(raw: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    p = raw.copy()
    p["sym"] = p.sym.astype(str)
    p["prim"] = p.prim.astype(str)
    p["exdiv"] = p.sym.str.endswith(" XD")
    p["sym"] = p.sym.str.replace(" XD", "", regex=False)
    # early-2014 files report volume in round lots: rescale days where SPY volume is implausibly small
    sv = p[p.sym == "SPY"].set_index("date").cons_vol
    mult = pd.Series(np.where(sv < 1e7, 100.0, 1.0), index=sv.index)
    m = p.date.map(mult).fillna(1.0)
    for c in ("cons_vol", "p_vol", "fin_vol", "amex_vol", "arca_vol"):
        p[c] = p[c] * m
    p = p[p.sym.map(lambda s: bool(COMMON_RE.match(s)))]
    lab = resolve_listing(p)
    leg = p.prim == "AMEX?"
    p.loc[leg, "prim"] = p.loc[leg, "sym"].map(lab).fillna("DROP")
    p = p[p.prim.isin(["NYSE", "AMEX"])]
    p = p.sort_values(["sym", "date"]).reset_index(drop=True)

    # trading-day index so that "previous row" means "previous session"
    days = pd.DatetimeIndex(sorted(raw.date.unique()))
    p["di"] = p.date.map(pd.Series(np.arange(len(days)), index=days))
    g = p.groupby("sym", sort=False)
    consecutive = g.di.diff() == 1
    p["prev_close"] = g.p_last.shift(1).where(consecutive)
    p["prev_close_cons"] = g.cons_last.shift(1).where(consecutive)
    p["open"] = p.p_open; p["close"] = p.p_last
    p["gap"] = p.open / p.prev_close - 1
    p["ret"] = p.close / p.prev_close - 1
    p["intra"] = p.close / p.open - 1
    # splits: both the open and the close sit on a standard split ratio of the prior close and volume
    # in shares jumps by ~1/ratio (splits are pre-announced, so excluding them is not look-ahead in spirit)
    r_o = (1 + p.gap).values; r_c = (1 + p.ret).values
    dist = lambda r: np.abs(np.nan_to_num(r, nan=9)[:, None] / SPLIT_RATIOS[None, :] - 1)
    k = dist(r_o).argmin(1)
    ratio = SPLIT_RATIOS[k]
    tight = (dist(r_o).min(1) < 0.02) & (np.abs(r_c / ratio - 1) < 0.04)
    g = p.groupby("sym", sort=False)
    vpre = g.cons_vol.transform(lambda x: x.shift(1).rolling(60, min_periods=20).median())
    vpost = g.cons_vol.transform(lambda x: x[::-1].rolling(20, min_periods=5).median()[::-1])
    p["split"] = tight & (vpost / vpre >= 0.75 / ratio).fillna(False).values
    for c in ("gap", "ret", "intra"):
        p.loc[p.split, c] = np.nan
    p.loc[p.ret.abs() > 3, ["gap", "ret"]] = np.nan          # >300% moves: data errors / symbol reuse

    s = spy[["ret", "gap", "intra"]].rename(columns=lambda c: "spy_" + c)
    p = p.join(s, on="date")
    p["ar"] = p.ret - p.spy_ret
    p["agap"] = p.gap - p.spy_gap
    p["aintra"] = p.intra - p.spy_intra
    g = p.groupby("sym", sort=False)
    p["sd60"] = g.ar.transform(lambda x: x.shift(1).rolling(60, min_periods=50).std())
    p["vm60"] = g.cons_vol.transform(lambda x: x.shift(1).rolling(60, min_periods=50).median())
    p["dvol"] = p.cons_vol * p.close
    p["year"] = p.date.dt.year

    ys = p.groupby(["sym", "year"]).agg(med_dvol=("dvol", "median"), med_px=("close", "median"), n=("ret", "size")).reset_index()
    ys["year"] += 1
    ys = ys[(ys.med_px >= 5) & (ys.n >= 200) & (ys.med_dvol > 0)]
    ys["Q"] = ys.groupby("year").med_dvol.transform(lambda s: pd.qcut(s, 5, labels=False) + 1)
    p = p.merge(ys.rename(columns={"med_dvol": "prior_dvol"})[["sym", "year", "Q", "prior_dvol"]], on=["sym", "year"], how="left")
    return p
