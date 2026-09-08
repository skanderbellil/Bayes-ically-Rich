"""Derive a committed venue table from the cached CoinGecko /tickers responses.

Writes data/crypto_panel/venues.csv: for each token, whether a European retail
account can buy it on a tier-1 exchange, on any retail CEX, and the median
bid-ask spread across tier-1 venues. The raw JSON cache is large and gitignored;
this table is small and committed, so the notebooks run offline.
"""
import glob, gzip, json, os
import numpy as np
import pandas as pd

PANEL = "/home/user/Bayes-ically-Rich/data/crypto_panel"
TIER1 = {"binance", "coinbase exchange", "kraken", "bitstamp", "bitvavo", "bitpanda"}
RETAIL = TIER1 | {"crypto.com exchange", "kucoin", "okx", "bybit", "gate.io"}

rows = []
for f in sorted(glob.glob(f"{PANEL}/raw/tickers_*.json.gz")):
    gid = os.path.basename(f)[len("tickers_"):-len(".json.gz")]
    d = json.load(gzip.open(f, "rt"))
    if "__error__" in d or not d.get("tickers"):
        rows.append({"gecko_id": gid, "tier1": False, "retail": False,
                     "n_tier1_venues": 0, "spread_pct": np.nan}); continue
    ok = [t for t in d["tickers"] if not t.get("is_anomaly") and not t.get("is_stale")]
    venues = {t["market"]["name"].lower() for t in ok}
    t1 = [t for t in ok if t["market"]["name"].lower() in TIER1
          and t.get("bid_ask_spread_percentage") is not None]
    rows.append({"gecko_id": gid, "tier1": bool(venues & TIER1),
                 "retail": bool(venues & RETAIL),
                 "n_tier1_venues": len(venues & TIER1),
                 "spread_pct": float(np.median([t["bid_ask_spread_percentage"] for t in t1]))
                               if t1 else np.nan})
V = pd.DataFrame(rows)
V.to_csv(f"{PANEL}/venues.csv", index=False)
print(f"{len(V)} tokens -> {PANEL}/venues.csv  "
      f"(tier1 {int(V.tier1.sum())}, any retail {int(V.retail.sum())})")
