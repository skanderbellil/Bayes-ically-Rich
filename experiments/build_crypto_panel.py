"""Build a long-history, survivorship-free crypto panel from free endpoints.

  fees   : DefiLlama fee adapters (full history, free, no key)
  prices : coins.llama.fi paged 500 points/request (free, no key, back to ~2019)

Writes data/crypto_panel/{prices,fees}.csv.gz plus a universe manifest.
Raw JSON is cached gzipped under data/crypto_panel/raw/ (gitignored - large).
"""
from __future__ import annotations
import gzip, json, sys, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd

REPO = Path("/home/user/Bayes-ically-Rich")
OUT = REPO / "data" / "crypto_panel"
RAW = OUT / "raw"
RAW.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "crypto-strategy-research/1.0"}
START = pd.Timestamp("2019-01-01")
MIN_ALLTIME_FEES = 10e6
_last = {}


def get(url, key, min_gap=0.22, retries=4):
    path = RAW / f"{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    host = url.split("/")[2]
    delay = 4.0
    for i in range(retries):
        gap = time.time() - _last.get(host, 0)
        if gap < min_gap:
            time.sleep(min_gap - gap)
        _last[host] = time.time()
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                payload = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 502, 503, 504) and i < retries - 1:
                time.sleep(delay); delay *= 2; continue
            payload = {"__error__": f"HTTP {e.code}"}; break
        except Exception as e:
            if i < retries - 1:
                time.sleep(delay); delay *= 2; continue
            payload = {"__error__": repr(e)}; break
    with gzip.open(path, "wt") as fh:
        json.dump(payload, fh)
    return payload


# ---------------------------------------------------------------- 1. universe
ov = get("https://api.llama.fi/overview/fees?excludeTotalDataChart=true"
         "&excludeTotalDataChartBreakdown=true", "fees_overview")
cands = [p for p in ov["protocols"] if (p.get("totalAllTime") or 0) >= MIN_ALLTIME_FEES]
cands.sort(key=lambda p: -(p.get("totalAllTime") or 0))
print(f"[1] {len(cands)} fee adapters with >= ${MIN_ALLTIME_FEES/1e6:.0f}M all-time fees", flush=True)

fee_by_gecko: dict[str, list[pd.Series]] = {}
meta: dict[str, dict] = {}
skipped = {"no_gecko": 0, "doublecounted": 0, "error": 0, "empty": 0}
parents_seen: set[str] = set()
resolved_via_parent = 0
slug_of: dict[str, list[str]] = {}

for i, p in enumerate(cands):
    slug = p.get("slug")
    if not slug:
        continue
    d = get(f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyFees"
            f"&excludeTotalDataChartBreakdown=true", f"fees_{slug.replace('/', '_')}")
    if "__error__" in d:
        skipped["error"] += 1; continue
    if d.get("doublecounted"):
        skipped["doublecounted"] += 1; continue
    gid = d.get("gecko_id")
    if not gid:
        # Child adapters (uniswap-v3, aave-v3, hyperliquid-perps, ...) carry no
        # gecko_id of their own but point at "parent#<slug>", which does. Resolve
        # through the parent, but keep the CHILD's daily series: summing the
        # children reproduces the parent total without double-counting it.
        par = (d.get("parentProtocol") or "")
        if par.startswith("parent#"):
            pslug = par.split("#", 1)[1]
            pd_ = get(f"https://api.llama.fi/summary/fees/{pslug}?dataType=dailyFees"
                      f"&excludeTotalDataChartBreakdown=true", f"fees_{pslug}")
            gid = pd_.get("gecko_id") if "__error__" not in pd_ else None
            if gid:
                parents_seen.add(pslug)
                resolved_via_parent += 1
    if not gid:
        skipped["no_gecko"] += 1; continue
    chart = d.get("totalDataChart") or []
    if not chart:
        skipped["empty"] += 1; continue
    s = pd.Series({pd.Timestamp(t, unit="s").normalize(): float(v or 0.0) for t, v in chart})
    s = s[~s.index.duplicated()].sort_index()
    fee_by_gecko.setdefault(gid, []).append(s)
    slug_of.setdefault(gid, []).append(slug)
    m = meta.setdefault(gid, {"gecko_id": gid, "symbol": d.get("symbol"),
                              "adapters": [], "category": d.get("category"),
                              "alltime_fees": 0.0})
    m["adapters"].append(slug)
    m["alltime_fees"] += p.get("totalAllTime") or 0.0
    if (i + 1) % 50 == 0:
        print(f"    fees {i+1}/{len(cands)}  tokens so far {len(fee_by_gecko)}", flush=True)

# Safety net: if a parent aggregate ALSO appeared in the candidate list next to
# its own children, drop the parent so its fees are not counted twice.
for gid, slugs in list(slug_of.items()):
    dupes = [i for i, sl in enumerate(slugs) if sl in parents_seen and len(slugs) > 1]
    for i in reversed(dupes):
        del fee_by_gecko[gid][i]
        del slug_of[gid][i]
    if not fee_by_gecko[gid]:
        del fee_by_gecko[gid]; del slug_of[gid]; meta.pop(gid, None)

print(f"[2] {len(fee_by_gecko)} unique tokens; {resolved_via_parent} adapters resolved "
      f"via parent; skipped {skipped}", flush=True)

FEES = pd.DataFrame({g: pd.concat(v, axis=1).sum(axis=1) for g, v in fee_by_gecko.items()})
FEES = FEES.sort_index()
FEES = FEES[FEES.index >= START]
print(f"[3] fee panel {FEES.shape} {FEES.index[0].date()}..{FEES.index[-1].date()}", flush=True)

# ---------------------------------------------------------------- 2. prices
geckos = sorted(fee_by_gecko)
end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
price_cols = {}
for n, gid in enumerate(geckos):
    chunks, cursor = [], START
    while cursor < end:
        span = min(500, int((end - cursor).days) + 1)
        # searchWidth=4h: wide enough to bridge the ~20% of daily grid points the
        # endpoint otherwise drops, narrow enough that it cannot backfill a price
        # from before a token existed.
        d = get(f"https://coins.llama.fi/chart/coingecko:{gid}"
                f"?start={int(cursor.timestamp())}&span={span}&period=1d&searchWidth=14400",
                f"px2_{gid}_{cursor.date()}")
        pts = (d.get("coins", {}).get(f"coingecko:{gid}", {}) or {}).get("prices", [])
        if pts:
            chunks.append(pd.Series({pd.Timestamp(x["timestamp"], unit="s").normalize(): x["price"]
                                     for x in pts}))
        cursor += pd.Timedelta(days=span)
    if chunks:
        s = pd.concat(chunks)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        if len(s) >= 120:
            price_cols[gid] = s
    if (n + 1) % 25 == 0:
        print(f"    prices {n+1}/{len(geckos)}  ok {len(price_cols)}", flush=True)

PX = pd.DataFrame(price_cols).sort_index()
PX = PX[(PX.index >= START) & (PX.index <= end)]
print(f"[4] price panel {PX.shape} {PX.index[0].date()}..{PX.index[-1].date()}", flush=True)

# benchmarks always present
for extra in ("bitcoin", "ethereum"):
    assert extra in PX.columns, f"missing benchmark {extra}"

OUT.mkdir(parents=True, exist_ok=True)
PX.to_csv(OUT / "prices.csv.gz")
FEES.reindex(columns=PX.columns).to_csv(OUT / "fees.csv.gz")
man = pd.DataFrame([meta[g] for g in PX.columns])
man["adapters"] = man["adapters"].map(lambda a: "|".join(a))
man["price_start"] = [PX[g].first_valid_index().date() for g in PX.columns]
man["price_days"] = [int(PX[g].notna().sum()) for g in PX.columns]
man.to_csv(OUT / "universe.csv", index=False)
print(f"[5] wrote panels to {OUT}", flush=True)
print(man.sort_values("alltime_fees", ascending=False).head(20).to_string(index=False), flush=True)
