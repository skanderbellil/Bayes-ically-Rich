"""Spot liquidity snapshot for the non-perp names: order-book depth, spread, volume.

CoinGecko /coins/{id}/tickers?depth=true returns, per venue: 2% market depth in
USD (cost_to_move_up/down_usd), bid-ask spread, 24h volume, and a trust_score
that flags wash-traded venues. Free tier, no key, heavily rate limited.
"""
import gzip, json, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd

RAW = Path("/home/user/Bayes-ically-Rich/data/crypto_panel/raw")
OUT = Path("/home/user/Bayes-ically-Rich/data/crypto_panel")
UA = {"User-Agent": "crypto-strategy-research/1.0"}


def get(url, key, gap=7.0, retries=5):
    p = RAW / f"{key}.json.gz"
    if p.exists():
        return json.load(gzip.open(p, "rt"))
    delay = 15.0
    for i in range(retries):
        time.sleep(gap)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                d = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 504) and i < retries - 1:
                time.sleep(delay); delay *= 2; continue
            d = {"__error__": f"HTTP {e.code}"}; break
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay); delay *= 2; continue
            d = {"__error__": repr(e)}; break
    json.dump(d, gzip.open(p, "wt"))
    return d


need = pd.read_csv("/tmp/claude-0/-home-user-Bayes-ically-Rich/426807f8-74e3-5df8-815c-2a2439ec415c/scratchpad/need_liquidity.csv")["gecko_id"].tolist()
print(f"{len(need)} tokens to fetch", flush=True)

rows = []
for i, gid in enumerate(need):
    d = get(f"https://api.coingecko.com/api/v3/coins/{gid}/tickers?depth=true&order=volume_desc",
            f"tickers_{gid}")
    if "__error__" in d or not d.get("tickers"):
        rows.append({"gecko_id": gid, "error": str(d.get("__error__", "no tickers"))})
    else:
        t = pd.DataFrame([{
            "venue": x["market"]["name"],
            "trust": x.get("trust_score"),
            "anomaly": bool(x.get("is_anomaly")),
            "stale": bool(x.get("is_stale")),
            "vol_usd": (x.get("converted_volume") or {}).get("usd") or 0.0,
            "spread_pct": x.get("bid_ask_spread_percentage"),
            "up": x.get("cost_to_move_up_usd") or 0.0,
            "down": x.get("cost_to_move_down_usd") or 0.0,
        } for x in d["tickers"]])
        # trust_score comes back null on the free tier, so venue quality is
        # filtered on the flags that ARE populated: is_anomaly / is_stale.
        clean = t[~t.anomaly & ~t.stale]
        quoting = clean[clean[["up", "down"]].notna().any(axis=1)]
        depth = (quoting[["up", "down"]].mean(axis=1)).sum()
        sp = quoting.dropna(subset=["spread_pct"])
        spread = float((sp.spread_pct * sp.vol_usd).sum() / sp.vol_usd.sum()) if sp.vol_usd.sum() else None
        rows.append({"gecko_id": gid, "n_venues": len(t),
                     "n_clean": int(len(clean)), "n_quoting": int(len(quoting)),
                     "depth_2pct_usd": depth,
                     "adv_usd_quoting": float(quoting.vol_usd.sum()),
                     "adv_usd_clean": float(clean.vol_usd.sum()),
                     "adv_usd_all": float(t.vol_usd.sum()),
                     "spread_pct": spread,
                     "flagged_vol_share": float(1 - clean.vol_usd.sum() / t.vol_usd.sum())
                                          if t.vol_usd.sum() else 0.0})
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(need)}", flush=True)

L = pd.DataFrame(rows)
L.to_csv(OUT / "liquidity.csv", index=False)
ok = L[L.get("depth_2pct_usd").notna()] if "depth_2pct_usd" in L else L
print(f"\nfetched {len(L)}; with depth data: {len(ok)}")
print(f"wrote {OUT/'liquidity.csv'}", flush=True)
