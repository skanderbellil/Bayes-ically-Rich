#!/usr/bin/env python3
"""
Fetch & store INTRADAY (10-min) price histories for the geopolitics universe
============================================================================

The daily token-history cache (one mark per calendar day) can't see intraday
wicks — a contract that spiked to the mids and fell back the same day looks flat
at the daily close. This pulls the finest history the Polymarket CLOB serves
(``interval=max`` with ``fidelity=10`` → ~10-minute bars over a market's full
life) for every geopolitics leg in the panel, and archives it into the committed
snapshot so the intraday exit analysis is reproducible from the repo.

Availability note: the CLOB only retains high-fidelity history for a recent
window, so OLD resolved markets return 0 intraday points (we keep whatever is
served and report coverage). Daily remains the fallback for those.

Output:
  data/raw/polymarket/tokenhr_<id>.csv                    (per-token, gitignored)
  data/polymarket_calibration_snapshot/token_intraday_cache_<date>.tar.gz  (committed)

Usage:
  python experiments/fetch_regime_intraday.py
  python experiments/fetch_regime_intraday.py --fidelity 10 --domain geopolitics
"""
from __future__ import annotations
import argparse, glob, subprocess, time, datetime as dt
from pathlib import Path
import pandas as pd
import _bootstrap  # noqa
from posterioralpha.polymarket.fetch import _get, CLOB_URL
from posterioralpha.polymarket.paths import RAW_DIR, ensure_dirs

OUT = "data/polymarket_calibration_snapshot"
PANELS = [
    "data/polymarket_calibration_snapshot/regime_calibration_panel.csv",
    "data/polymarket_calibration_snapshot/regime_calibration_panel_lowvol_1k.csv",
]


def fetch_intraday(token: str, fidelity: int) -> pd.Series:
    d = _get(CLOB_URL, {"market": token, "interval": "max", "fidelity": fidelity})
    h = (d or {}).get("history", []) if isinstance(d, dict) else []
    if not h:
        return pd.Series(dtype=float, name=token)
    df = pd.DataFrame(h)
    df["ts"] = pd.to_datetime(df["t"], unit="s", utc=True)
    return df.set_index("ts")["p"].astype(float).sort_index()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fidelity", type=int, default=10, help="bar size in minutes (10 = finest the CLOB serves)")
    ap.add_argument("--domain", default="geopolitics")
    ap.add_argument("--legs", default="yes,no", help="comma list of legs to fetch")
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()
    ensure_dirs()
    legs = set(args.legs.split(","))

    tokens = {}
    for p in PANELS:
        if not Path(p).exists():
            continue
        P = pd.read_csv(p)
        sub = P[(P.domain == args.domain) & (P.leg.isin(legs))]
        for _, r in sub.iterrows():
            tokens.setdefault(str(r["token"]), (str(r.get("question", ""))[:50], pd.to_datetime(r["res_date"]).date()))
    print("intraday fetch: %d unique %s tokens (legs=%s) @ fidelity=%dmin"
          % (len(tokens), args.domain, args.legs, args.fidelity))

    cov, total_pts, with_data = 0, 0, []
    for i, (tok, (q, res)) in enumerate(tokens.items(), 1):
        cache = RAW_DIR / f"tokenhr_{tok}.csv"
        try:
            s = fetch_intraday(tok, args.fidelity)
        except Exception as e:
            print("  [%d/%d] %-50s ERROR %s" % (i, len(tokens), q, type(e).__name__)); continue
        if len(s):
            s.to_frame(name="p").to_csv(cache)
            cov += 1; total_pts += len(s); with_data.append(tok)
            gap = s.index.to_series().diff().dt.total_seconds().median() / 60
            if i <= 12 or i % 25 == 0:
                print("  [%d/%d] %-50s %5d pts (~%.0fmin, res %s)" % (i, len(tokens), q, len(s), gap, res))
        time.sleep(args.sleep)

    print("\ncoverage: %d/%d tokens have intraday (%d total points, ~%.0f pts/token)"
          % (cov, len(tokens), total_pts, total_pts / max(cov, 1)))

    # archive the intraday cache into the committed snapshot
    stamp = dt.date.today().isoformat()
    tar = f"{OUT}/token_intraday_cache_{stamp}.tar.gz"
    files = glob.glob(str(RAW_DIR / "tokenhr_*.csv"))
    if files:
        # tar the tokenhr_*.csv under a stable prefix matching the daily cache layout
        rel = [str(Path(f).relative_to(RAW_DIR.parent)) for f in files]  # polymarket/tokenhr_*.csv
        subprocess.run(["tar", "czf", tar, "-C", str(RAW_DIR.parent)] + rel, check=False)
        print("archived %d intraday files -> %s" % (len(files), tar))
    else:
        print("no intraday files to archive")


if __name__ == "__main__":
    main()
