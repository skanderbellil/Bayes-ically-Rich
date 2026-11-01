"""Download + parse every CTA EOD summary file (2014-01-02 ..) into one parquet per day.

Resumable: days already parsed are skipped. Raw CSVs are not kept (~100 GB).
    python experiments/build_dipbuy_cts.py [--workers 10] [--start 2014-01-01]
Then stack with:  python experiments/build_dipbuy_cts.py --stack
"""
import argparse
import sys
import time
from multiprocessing import Pool

import _bootstrap  # noqa: F401
import pandas as pd

from posterioralpha.dipbuy import cts
from posterioralpha.dipbuy.paths import CTS_DAY_DIR, PANEL_PATH, ensure_dirs


def work(args):
    name, date = args
    out = CTS_DAY_DIR / f"{date:%Y%m%d}.parquet"
    if out.exists():
        return date, "skip", 0.0
    t = time.time()
    for attempt in range(4):
        try:
            df = cts.parse_eod(cts.fetch(name))
            df.insert(0, "date", date)
            df.to_parquet(out, index=False)
            return date, "ok", time.time() - t
        except Exception as e:  # network hiccup -> back off and retry
            err = e
            time.sleep(2 ** (attempt + 1))
    return date, f"FAIL {err}", time.time() - t


def stack():
    files = sorted(CTS_DAY_DIR.glob("*.parquet"))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["sym"] = df.sym.astype("category"); df["prim"] = df.prim.astype("category")
    df.to_parquet(PANEL_PATH, index=False)
    print(f"stacked {len(files)} days, {len(df):,} rows -> {PANEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--start", default="2014-01-01")
    ap.add_argument("--stack", action="store_true")
    a = ap.parse_args()
    ensure_dirs()
    if a.stack:
        stack(); sys.exit()
    lst = cts.list_eod_files()
    lst = lst[lst.date >= a.start]
    print(f"{len(lst)} files {lst.date.min():%Y-%m-%d}..{lst.date.max():%Y-%m-%d}, {lst['size'].sum()/1e9:.1f} GB", flush=True)
    n = 0
    with Pool(a.workers) as p:
        for date, st, dt in p.imap_unordered(work, list(zip(lst.name, lst.date))):
            n += 1
            if st != "skip" and (n % 25 == 0 or st.startswith("FAIL")):
                print(f"[{n}/{len(lst)}] {date:%Y-%m-%d} {st} {dt:.0f}s", flush=True)
    print("done", flush=True)
