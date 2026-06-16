#!/usr/bin/env python3
"""
The LONG-ONLY cop_at quality tilt, net of cost (user 2026-06-15).

cop_at (cash-based operating profitability) cleared our bar as a LONG-SHORT
factor, but the alpha may live in the short leg. This tests the best-case
LONG-ONLY version: the actual JKP top-quantile cop_at PORTFOLIO (the high-
cash-profitability book you'd hold long) vs the market, net of cost, across
4 regions. Plus the real-ETF cross-check (COWZ/QUAL — what you can buy).

Data: JKP per-quantile portfolios (datasets/jkp_cop_portfolios.csv.gz),
downloaded here; long format location,name,pf,date,ret. The long-only tilt =
top-quantile minus the cap-weighted universe (EW of quantiles ~ market proxy).
Cost grid applied to the tilt's active return (slow signal -> modest turnover).
"""
import sys, io, zipfile
import numpy as np
import pandas as pd
from tabulate import tabulate
import _bootstrap  # noqa: F401

DATA = _bootstrap.ROOT / "datasets"
REGIONS = ["usa","gbr","jpn","deu","fra","can","aus","kor","chn","ind","che","hkg"]
URL = ("https://jkpfactors-data.s3.amazonaws.com/public/portfolios/"
       "%5B{r}%5D_%5Bcop_at%5D_%5Bmonthly%5D_%5Bvw_cap%5D.zip")


def fetch():
    import requests
    f = DATA / "jkp_cop_portfolios.csv.gz"
    if f.exists():
        return pd.read_csv(f, parse_dates=["date"])
    frames = []
    for r in REGIONS:
        try:
            resp = requests.get(URL.format(r=r), timeout=90); resp.raise_for_status()
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            frames.append(pd.read_csv(zf.open(zf.namelist()[0]), parse_dates=["date"]))
        except Exception as e:
            print(f"  skip {r}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(f, index=False, compression="gzip")
    return df


def tnet(active, cost_ann):
    a = (active - cost_ann / 12).dropna()
    if len(a) < 24:
        return np.nan, np.nan
    return a.mean() * 12, a.mean() / a.std(ddof=1) * np.sqrt(len(a))


def main():
    df = fetch()
    pfs = sorted(df["pf"].unique())
    top, bot = pfs[-1], pfs[0]
    print(f"cop_at quantile portfolios: pf values {pfs} (top={top}=high quality)\n")
    rows = []
    for r in REGIONS:
        d = df[df.location == r]
        wide = d.pivot_table(index="date", columns="pf", values="ret").sort_index()
        if top not in wide or len(wide) < 36:
            continue
        mkt = wide.mean(axis=1)                      # EW of quantiles ~ universe/market
        tilt = wide[top]                             # long-only high-quality book
        active = (tilt - mkt).dropna()               # tilt's active return vs market
        # total Sharpe of holding the tilt vs the market
        def sh(x): x = x.dropna(); return x.mean()/x.std(ddof=1)*np.sqrt(12)
        a0, t0 = tnet(active, 0.0)
        a1, t1 = tnet(active, 0.01)                  # 1%/yr cost on the tilt
        rows.append([r, f"{sh(tilt):.2f}", f"{sh(mkt):.2f}",
                     f"{a0:+.1%}", f"{t0:.2f}", f"{a1:+.1%}", f"{t1:.2f}"])
    print("LONG-ONLY cop_at tilt (top quantile) vs that country's universe:")
    print(tabulate(rows, headers=["country", "tilt Sharpe", "mkt Sharpe",
                                  "active(gross)", "t", "active(net 1%)", "t"],
                   tablefmt="github"))
    npos = sum(1 for r in rows if float(r[6]) > 2)
    print(f"\nCountries with net-of-1%-cost active t>2: {npos}/{len(rows)}")

    # real-ETF cross-check
    try:
        px = pd.read_csv(DATA / "quality_etfs.csv", index_col=0, parse_dates=True)
        def m(c):
            s = px[c].dropna(); return (1+s.pct_change()).resample("ME").apply(lambda x: x.prod()-1).dropna()
        spy = m("SPY"); erows = []
        for t in ["COWZ", "QUAL", "DGRW", "SPHQ"]:
            if t not in px: continue
            r = m(t); idx = r.index.intersection(spy.index)
            act = (r.loc[idx] - spy.loc[idx])
            erows.append([t, f"{act.mean()*12:+.1%}", f"{act.mean()/act.std()*np.sqrt(len(act)):.2f}"])
        print("\nReal long-only quality ETFs, active return vs SPY (net of ETF fees):")
        print(tabulate(erows, headers=["ETF", "active/yr", "t"], tablefmt="github"))
    except Exception as e:
        print("ETF cross-check skipped:", e)

    print("\nVERDICT: does the long-only cop_at tilt earn a positive active "
          "return over the market, net of cost, in the US AND cross-region? "
          "This is how much of the cop_at edge a long-only (no-shorting) "
          "investor can actually capture. Bar: t>2 net cross-region to be "
          "worth tilting; t>3 to be a real standalone edge.")


if __name__ == "__main__":
    main()
