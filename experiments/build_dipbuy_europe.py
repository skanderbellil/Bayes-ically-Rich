"""Fetch the European daily panel (Yahoo) -> data/raw/dipbuy/europe_raw.parquet."""
import _bootstrap  # noqa: F401
from posterioralpha.dipbuy import europe, yahoo
from posterioralpha.dipbuy.paths import RAW_DIR, ensure_dirs

ensure_dirs()
U = europe.universe()
U.to_parquet(RAW_DIR / "europe_universe.parquet")
print(len(U), "symbols\n", U.exchange.value_counts().to_dict(), flush=True)
raw = europe.fetch_all(list(U.symbol))
raw.to_parquet(RAW_DIR / "europe_raw.parquet", index=False)
idx = yahoo.daily(europe.INDEX, "2012-01-01"); idx.to_parquet(RAW_DIR / "europe_index.parquet")
print(raw.sym.nunique(), "symbols with data;", len(raw), "rows", raw.date.min(), raw.date.max())
