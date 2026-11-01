"""European small/mid-cap daily panel from Yahoo (secondary test).

Universe: financedatabase equities whose *primary* listing is on a venue with an opening and closing
call auction: Xetra (.DE), Euronext Paris/Amsterdam/Brussels/Lisbon, Borsa Italiana, BME, SIX,
Wiener Börse, Nasdaq Stockholm/Helsinki/Copenhagen, Oslo Børs. LSE is excluded: its small caps trade on
SETSqx periodic auctions, and prices are quoted in pence.
Daily open/close on these venues are the auction uncrossing prices for names that print in the auction.
CAVEATS: (1) Yahoo carries only currently listed tickers, so the panel is survivorship-biased.
(2) The dataset cannot tell an auction print from a later first trade for names without an auction
execution. (3) Volume is the venue (Yahoo) volume, not consolidated volume across MTFs.
Index gap/return reference: iShares STOXX Europe 600 (EXSA.DE, Xetra).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import yahoo

EXCH = {"GER": "Xetra", "PAR": "Euronext Paris", "AMS": "Euronext Amsterdam", "BRU": "Euronext Brussels",
        "LIS": "Euronext Lisbon", "MIL": "Borsa Italiana", "MCE": "BME Madrid", "EBS": "SIX", "VIE": "Vienna",
        "STO": "Nasdaq Stockholm", "HEL": "Nasdaq Helsinki", "CPH": "Nasdaq Copenhagen", "OSL": "Oslo"}
INDEX = "EXSA.DE"


def universe() -> pd.DataFrame:
    import financedatabase as fd
    eq = fd.Equities().select().reset_index()
    eq = eq[eq.exchange.isin(EXCH) & eq.market_cap.isin(["Nano Cap", "Micro Cap", "Small Cap", "Mid Cap", "Large Cap", "Mega Cap"])]
    eq = eq[~eq.symbol.str.contains(r"[\^=]")]
    return eq[["symbol", "name", "exchange", "country", "currency", "market_cap", "sector"]].drop_duplicates("symbol")


def fetch_all(symbols, start="2013-01-01", workers=4) -> pd.DataFrame:
    def one(s):
        d = yahoo.daily(s, start)
        if d.empty:
            return None
        d = d[["open", "high", "low", "close", "volume"]].copy()
        d["sym"] = s
        return d.rename_axis("date").reset_index()
    with ThreadPoolExecutor(workers) as ex:
        parts = [x for x in ex.map(one, symbols) if x is not None]
    return pd.concat(parts, ignore_index=True)


def build(raw: pd.DataFrame, idx: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    p = raw.merge(meta[["symbol", "exchange", "market_cap", "currency"]].rename(columns={"symbol": "sym"}), on="sym")
    p = p[(p.volume > 0) & (p.open > 0) & (p.close > 0) & (p.high >= p.low)]
    p = p.sort_values(["sym", "date"]).reset_index(drop=True)
    days = pd.DatetimeIndex(sorted(idx.index))
    p = p[p.date.isin(days)]
    p["di"] = p.date.map(pd.Series(np.arange(len(days)), index=days))
    g = p.groupby("sym", sort=False)
    cons = g.di.diff() == 1
    p["prev_close"] = g.close.shift(1).where(cons)
    p["gap"] = p.open / p.prev_close - 1
    p["ret"] = p.close / p.prev_close - 1
    p["intra"] = p.close / p.open - 1
    p.loc[p.ret.abs() > 1, ["gap", "ret", "intra"]] = np.nan
    p["open_bad"] = (p.open > p.high * 1.005) | (p.open < p.low * 0.995)
    ix = pd.DataFrame({"idx_ret": idx.adjclose.pct_change(), "idx_gap": idx.open / idx.close.shift(1) - 1,
                       "idx_intra": idx.close / idx.open - 1})
    p = p.join(ix, on="date")
    p["ar"] = p.ret - p.idx_ret; p["agap"] = p.gap - p.idx_gap
    g = p.groupby("sym", sort=False)
    p["sd60"] = g.ar.transform(lambda x: x.shift(1).rolling(60, min_periods=50).std())
    p["dvol"] = p.volume * p.close
    p["year"] = p.date.dt.year
    ys = p.groupby(["sym", "year"]).agg(med_dvol=("dvol", "median"), med_px=("close", "median"), n=("ret", "size")).reset_index()
    ys["year"] += 1
    ys = ys[(ys.med_px >= 5) & (ys.n >= 200) & (ys.med_dvol > 0)]
    ys["Q"] = ys.groupby("year").med_dvol.transform(lambda s: pd.qcut(s, 5, labels=False) + 1)
    return p.merge(ys.rename(columns={"med_dvol": "prior_dvol"})[["sym", "year", "Q", "prior_dvol"]], on=["sym", "year"], how="left")
