"""Suspect A, step 1: what IS the 'open' in the CTA EOD summary?

Reconciles the primary-venue open/last in the CTA EOD summary with the NYSE XDP trade
tape (sale conditions identify the auction prints) for the public sample days.
Also measures, from the tape itself, the opening-auction size distribution, the gap between
the auction print and the first continuous trade, and how often the auction print is the
venue's regular-session high or low.
"""
import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import cts, xdp
from posterioralpha.dipbuy.paths import XDP_DIR, DOCS_DIR, ensure_dirs

ensure_dirs()
eq = lambda a, b: np.isclose(a, b, rtol=1e-6, atol=1e-4)
lines = []
for day in ("20231002", "20231003"):
    E = cts.parse_eod(open(XDP_DIR / f"CTA.Summary.EODSUM.{day}.csv", "rb").read()).set_index("sym")
    for venue, fn in (("NYSE", f"EQY_US_TAQ_NYSE_TRADES_{day}.gz"), ("AMEX", f"EQY_US_TAQ_AMEX_TRADES_{day}.gz")):
        S, T = xdp.read_trades(XDP_DIR / fn)
        A = auction_summary = xdp.auction_summary(T)
        X = S.set_index("sym").join(A, how="inner").join(E, how="inner")
        L = X[(X.stype == "C") & (X.prim == venue) & X.auc_open.notna()]
        r = {"day": day, "venue": venue, "listed common stocks w/ opening auction": len(L),
             "EOD open == opening-auction print": eq(L.p_open, L.auc_open).mean(),
             "EOD open == first regular-session trade": eq(L.p_open, L.first_reg).mean(),
             "share with a closing-auction print": L.auc_close.notna().mean(),
             "EOD last == closing-auction print (when one printed)": eq(L.p_last, L.auc_close)[L.auc_close.notna()].mean(),
             "EOD last == last regular-session trade": eq(L.p_last, L.last_reg).mean(),
             "share with pre-open trades on this venue": (L.pre_n > 0).mean(),
             "auction print == venue regular-session low": eq(L.auc_open, L.lo_reg).mean(),
             "auction print == consolidated low": eq(L.auc_open, L.cons_lo).mean(),
             "median opening-auction $": (L.auc_open * L.auc_open_qty).median(),
             "share of opening auctions < $2k": ((L.auc_open * L.auc_open_qty) < 2000).mean(),
             "share of opening auctions < $20k": ((L.auc_open * L.auc_open_qty) < 20000).mean(),
             "median opening-auction qty share of consolidated vol": (L.auc_open_qty / L.cons_vol).median()}
        lines.append(r)
R = pd.DataFrame(lines).set_index(["day", "venue"]).T
pd.set_option("display.width", 200)
print(R.to_string(float_format=lambda x: f"{x:,.3f}"))
R.to_csv(DOCS_DIR / "verify_open_fields.csv")
