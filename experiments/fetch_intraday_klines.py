"""Download 1-minute klines from Binance's public data archives.

data.binance.vision publishes monthly ZIPs of 1m OHLCV per symbol, complete back
to 2017. (The Binance REST API is geo-blocked from this host; these static
archives are not.) Each month is streamed, parsed, appended and the ZIP deleted,
so peak disk stays a few MB.

Two format quirks handled: open_time is milliseconds in older files and
microseconds from 2025 on, and some months carry a CSV header row while most
do not.
"""
from __future__ import annotations
import io, sys, urllib.request, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("/home/user/Bayes-ically-Rich/data/crypto_intraday")
OUT.mkdir(parents=True, exist_ok=True)
BASE = "https://data.binance.vision/data/spot"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_vol", "trades", "taker_base", "taker_quote", "ignore"]
KEEP = ["ts", "open", "high", "low", "close", "volume", "trades"]
UA = {"User-Agent": "crypto-intraday-research/1.0"}


def fetch_zip(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=180) as r:
            return r.read()
    except Exception:
        return None


def parse(blob: bytes) -> pd.DataFrame | None:
    try:
        z = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile:
        return None
    raw = z.read(z.namelist()[0]).decode()
    if not raw.strip():
        return None
    header = 0 if raw.split(",", 1)[0].split("\n")[0].strip().isdigit() else "infer"
    df = pd.read_csv(io.StringIO(raw), header=None if header == 0 else 0)
    df.columns = COLS[:df.shape[1]]
    t = df["open_time"].astype("int64")
    # 16 digits => microseconds (Binance switched in 2025), 13 => milliseconds
    unit = "us" if t.iloc[0] > 1e15 else "ms"
    df["ts"] = pd.to_datetime(t, unit=unit)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float32")
    df["trades"] = df["trades"].astype("int32")
    return df[KEEP]


def months(start: str, end: str):
    return [d.strftime("%Y-%m") for d in pd.date_range(start, end, freq="MS")]


def build(symbol: str, start: str = "2017-08"):
    end = pd.Timestamp.utcnow().tz_localize(None).replace(day=1) - pd.Timedelta(days=1)
    frames, missing = [], []
    for m in months(start, end.strftime("%Y-%m")):
        blob = fetch_zip(f"{BASE}/monthly/klines/{symbol}/1m/{symbol}-1m-{m}.zip")
        d = parse(blob) if blob else None
        if d is None:
            missing.append(m)
        else:
            frames.append(d)
        if len(frames) % 24 == 0 and frames:
            print(f"  {symbol} {m}  cumulative rows {sum(len(f) for f in frames):,}", flush=True)
    # current partial month, from daily files
    for day in pd.date_range(end + pd.Timedelta(days=1),
                             pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(days=1)):
        blob = fetch_zip(f"{BASE}/daily/klines/{symbol}/1m/{symbol}-1m-{day.date()}.zip")
        d = parse(blob) if blob else None
        if d is not None:
            frames.append(d)
    if not frames:
        print(f"{symbol}: NO DATA"); return
    df = pd.concat(frames, ignore_index=True).drop_duplicates("ts").sort_values("ts")
    path = OUT / f"{symbol}_1m.parquet"
    df.to_parquet(path, compression="zstd", index=False)
    gaps = int((df.ts.diff().dt.total_seconds().fillna(60) != 60).sum())
    print(f"{symbol}: {len(df):,} rows  {df.ts.min()} -> {df.ts.max()}  "
          f"{path.stat().st_size/1e6:.0f}MB  gaps={gaps}  months_missing={len(missing)}", flush=True)
    if missing:
        print(f"   missing: {missing[:12]}{' ...' if len(missing) > 12 else ''}", flush=True)


if __name__ == "__main__":
    for sym in (sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]):
        build(sym)
