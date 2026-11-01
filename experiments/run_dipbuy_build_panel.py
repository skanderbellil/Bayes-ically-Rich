"""Stack the per-day CTA files and build the analysis panel -> data/raw/dipbuy/panel.parquet."""
import _bootstrap  # noqa: F401
import pandas as pd

from posterioralpha.dipbuy import panel, yahoo
from posterioralpha.dipbuy.paths import CTS_DAY_DIR, RAW_DIR, ensure_dirs

ensure_dirs()
files = sorted(CTS_DAY_DIR.glob("*.parquet"))
raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
print(f"{len(files)} days, {len(raw):,} raw rows")
spy = yahoo.spy_daily()
spy.to_parquet(RAW_DIR / "spy.parquet")
P = panel.build(raw, spy)
P["exdiv_any"] = P.exdiv          # Yahoo ex-dividend dates are merged in by build_dipbuy_yahoo_events.py
P.to_parquet(RAW_DIR / "panel.parquet", index=False)
print(f"panel: {len(P):,} rows, {P.sym.nunique():,} symbols; by listing: {P.prim.value_counts().to_dict()}")
print("splits flagged:", int(P.split.sum()), " legacy ex-div rows:", int(P.exdiv.sum()))
print("symbols per year:", P.groupby("year").sym.nunique().to_dict())
last = P.groupby("sym").date.max()
print("symbols whose last session is before 2026-10-01 (delisted/renamed):", int((last < "2026-10-01").sum()))
