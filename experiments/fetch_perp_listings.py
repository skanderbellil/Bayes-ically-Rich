"""Point-in-time perp shortability: OKX listTime + Hyperliquid first-funding date."""
import gzip, json, time, urllib.request
from pathlib import Path
import pandas as pd

RAW = Path("/home/user/Bayes-ically-Rich/data/crypto_panel/raw")
OUT = Path("/home/user/Bayes-ically-Rich/data/crypto_panel")
UA = {"User-Agent": "crypto-strategy-research/1.0"}


def post(url, body, key):
    p = RAW / f"{key}.json.gz"
    if p.exists():
        return json.load(gzip.open(p, "rt"))
    time.sleep(0.15)
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={**UA, "Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(req, timeout=60).read())
    except Exception as e:
        d = {"__error__": repr(e)}
    json.dump(d, gzip.open(p, "wt"))
    return d


def get(url, key):
    p = RAW / f"{key}.json.gz"
    if p.exists():
        return json.load(gzip.open(p, "rt"))
    time.sleep(0.15)
    try:
        d = json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read())
    except Exception as e:
        d = {"__error__": repr(e)}
    json.dump(d, gzip.open(p, "wt"))
    return d


# --- OKX: listTime is the instrument's first listing --------------------------
okx = get("https://www.okx.com/api/v5/public/instruments?instType=SWAP", "okx_swaps")
okx_list = {}
for x in okx.get("data", []):
    if x["instId"].endswith("-USDT-SWAP"):
        s = x["instId"].split("-")[0]
        t = pd.Timestamp(int(x["listTime"]), unit="ms").normalize()
        okx_list[s] = min(okx_list.get(s, pd.Timestamp("2100-01-01")), t)
print(f"OKX USDT perps: {len(okx_list)}")

# --- Hyperliquid: earliest funding record == listing date ---------------------
meta = post("https://api.hyperliquid.xyz/info", {"type": "meta"}, "hl_meta")
hl_coins = [x["name"] for x in meta["universe"]]
hl_list = {}
for i, c in enumerate(hl_coins):
    d = post("https://api.hyperliquid.xyz/info",
             {"type": "fundingHistory", "coin": c, "startTime": 1672531200000}, f"hlfund_{c}")
    if isinstance(d, list) and d:
        hl_list[c] = pd.Timestamp(d[0]["time"], unit="ms").normalize()
    if (i + 1) % 50 == 0:
        print(f"  hyperliquid {i+1}/{len(hl_coins)}  dated {len(hl_list)}")
print(f"Hyperliquid perps dated: {len(hl_list)}")

# --- CoinGecko id -> ticker ---------------------------------------------------
cg = get("https://api.coingecko.com/api/v3/coins/list", "cg_coins_list")
tick = {c["id"]: c["symbol"].upper() for c in cg}

un = pd.read_csv(OUT / "universe.csv")
un["ticker"] = un.gecko_id.map(tick)
un["okx_perp_listed"] = un.ticker.map(okx_list)
un["hl_perp_listed"] = un.ticker.map(hl_list)
un["perp_listed"] = un[["okx_perp_listed", "hl_perp_listed"]].min(axis=1)
un.to_csv(OUT / "perp_listings.csv", index=False)

n = un.perp_listed.notna().sum()
print(f"\nuniverse tokens with a perp anywhere: {n}/{len(un)}")
print(f"  OKX only {un.okx_perp_listed.notna().sum()}, HL only {un.hl_perp_listed.notna().sum()}")
print(f"  earliest listing {un.perp_listed.min()}, median {un.perp_listed.median()}")
print(f"wrote {OUT/'perp_listings.csv'}")
