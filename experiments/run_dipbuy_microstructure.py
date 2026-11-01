"""Suspects B (capacity) and D (tradability of the signal) on the public NYSE XDP sample days.

B: for every NYSE / NYSE American common stock on 2023-10-02/03, the opening-auction $ volume and the
   first-5/15/30-minute $ volume *on the listing venue*, by liquidity quintile -> what share a $2k / $5k / $20k
   ticket would be. (Listing-venue volume understates the whole market's first-30-minute volume; consolidated
   first-30-minute volume is not in the samples, so a x(1/venue share) scaling is also shown.)
D: imbalance-feed indicative match prices published before the open (09:00 ... 09:29:30) on 2020-09-09 (NYSE),
   2026-07-01/02 (NYSE + NYSE American): how close to the official open, and does a gap signal computed from
   the indicative price select the same trades as the one computed from the official open?
Requires the panel (experiments/run_dipbuy_build_panel.py).
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import xdp
from posterioralpha.dipbuy.paths import XDP_DIR, DOCS_DIR, RAW_DIR

pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
P = pd.read_parquet(RAW_DIR / "panel.parquet")
P["thr"] = np.maximum(0.03, 3 * P.sd60)

# ---------------------------------------------------------------- B. capacity
cap_rows = []
for day in ("2023-10-02", "2023-10-03"):
    d = P[P.date == day].set_index("sym")
    for venue, fn in (("NYSE", "NYSE"), ("AMEX", "AMEX")):
        S, T = xdp.read_trades(XDP_DIR / f"EQY_US_TAQ_{fn}_TRADES_{day.replace('-', '')}.gz")
        A = xdp.auction_summary(T)
        X = d[d.prim == venue].join(A, how="inner")
        X = X[X.auc_open.notna() & X.Q.notna()]
        X["auc_usd"] = X.auc_open * X.auc_open_qty
        for m in (5, 15, 30):
            X[f"usd{m}"] = X[f"vwap{m}"] * X[f"vol{m}"]
        X["venue_share"] = X.p_vol / X.cons_vol
        X["usd30_cons_est"] = X.usd30 / X.venue_share.clip(lower=0.02)
        X["day"] = day
        cap_rows.append(X)
C = pd.concat(cap_rows)
out = {}
for q, g in C.groupby("Q"):
    r = {"N stock-days": len(g), "NYSE American share": (g.prim == "AMEX").mean(),
         "median prior-yr $vol/day": g.prior_dvol.median(), "median opening auction $": g.auc_usd.median(),
         "p25 opening auction $": g.auc_usd.quantile(.25), "median 1st-30min $ (listing venue)": g.usd30.median(),
         "median 1st-30min $ (consolidated est.)": g.usd30_cons_est.median()}
    for tk in (2000, 5000, 20000):
        r[f"${tk//1000}k / median auction"] = tk / g.auc_usd.median()
        r[f"share of names where ${tk//1000}k > 10% of auction"] = (tk > 0.10 * g.auc_usd).mean()
        r[f"${tk//1000}k / median 1st-30min cons. $"] = tk / g.usd30_cons_est.median()
    out[int(q)] = r
CAP = pd.DataFrame(out)
print("=== B. Capacity at the open (XDP sample days 2023-10-02/03, all common stocks) ===")
print(CAP.to_string(float_format=lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}"))
CAP.to_csv(DOCS_DIR / "capacity_auction.csv")
# same, by listing venue
share = (C.auc_open_qty / C.p_vol).groupby(C.prim).median().rename("median auction qty / listing-venue day volume")
print("\n", share); share.to_csv(DOCS_DIR / "auction_share_of_venue_volume.csv")
byv = C.groupby(["prim", "Q"]).auc_usd.median().unstack()
print("\nmedian opening-auction $ by listing venue x Q:\n", byv.round(0))
byv.to_csv(DOCS_DIR / "capacity_auction_by_venue.csv")
# auction print vs first-5-min VWAP / 09:35 / 09:45 on ordinary days: noise of the auction print by Q
C["auc_vs_vwap5"] = C.auc_open / C.vwap5 - 1
C["auc_vs_0945"] = C.auc_open / C.px0945 - 1
noise = C.groupby("Q")[["auc_vs_vwap5", "auc_vs_0945"]].agg(lambda s: s.abs().median())
print("\nmedian |auction print / later price - 1| on ordinary days:\n", noise.round(4))
noise.to_csv(DOCS_DIR / "auction_noise_sample_days.csv")

# ---------------------------------------------------------------- D. pre-open indicative signal
rows = []
for day, files in (("2020-09-09", [("NYSE", "NYSE_IMBALANCES_20200909")]),
                   ("2026-07-01", [("NYSE", "EQY_US_NYSE_REF_IMBALANCES_20260701.gz"), ("AMEX", "EQY_US_AMEX_REF_IMBALANCES_20260701.gz")]),
                   ("2026-07-02", [("NYSE", "EQY_US_NYSE_REF_IMBALANCES_20260702.gz"), ("AMEX", "EQY_US_AMEX_REF_IMBALANCES_20260702.gz")])):
    d = P[P.date == day].set_index("sym")
    for venue, fn in files:
        S, I = xdp.read_imbalances(XDP_DIR / fn)
        ind = xdp.indicative_open(I)
        X = d[d.prim == venue].join(ind, how="left")
        X["day"] = day
        rows.append(X)
D = pd.concat(rows)
D = D[D.Q.notna() & D.open.notna() & D.prev_close.notna() & (D.prev_close >= 5)]
res = {}
for c in ("0900", "0915", "0925", "0928", "092930"):
    ip = D[f"ind_px_{c}"]
    dev = (ip / D.open - 1).abs()
    agap_ind = ip / D.prev_close - 1 - D.spy_gap
    sig_real = D.agap <= -D.thr
    sig_ind = agap_ind <= -D.thr
    res[c] = {"has indicative": ip.notna().mean(), "median |ind/open-1|": dev.median(), "p90 |ind/open-1|": dev.quantile(.9),
              "real signals": int(sig_real.sum()), "pre-open signals": int(sig_ind.sum()),
              "recall (real caught pre-open)": (sig_ind & sig_real).sum() / max(sig_real.sum(), 1),
              "precision": (sig_ind & sig_real).sum() / max(sig_ind.sum(), 1),
              "mean open->close, pre-open signals": D.intra[sig_ind].mean(),
              "mean open->close, real signals": D.intra[sig_real].mean()}
R = pd.DataFrame(res)
print("\n=== D. Indicative (pre-open) vs official open, NYSE/NYSE American common stocks >= $5 ===")
print(R.to_string(float_format=lambda x: f"{x:,.4f}"))
R.to_csv(DOCS_DIR / "preopen_indicative.csv")
byq = D.assign(dev=(D["ind_px_0928"] / D.open - 1).abs(), has=D["ind_px_0928"].notna()).groupby("Q")[["has", "dev"]].median()
byq["has"] = D.assign(has=D["ind_px_0928"].notna()).groupby("Q").has.mean()
print("\nby Q at 09:28 (share with an indicative price, median |ind/open-1|):\n", byq.round(4))
byq.to_csv(DOCS_DIR / "preopen_indicative_by_q.csv")
