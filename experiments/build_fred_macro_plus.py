#!/usr/bin/env python3
"""
Extend the curated FRED panel → ``datasets/fred_macro_plus.csv``.

Adds seven series the base panel lacks, each with its causal publication
lag (business days), JOINED onto the existing cached ``fred_macro.csv`` —
the base file is left untouched so earlier results stay reproducible.

  DGS3MO        3m bill yield — the front end / cuts-priced-in dial
  DEXJPUS       yen per USD — carry/funding stress
  CCSA          continued jobless claims (weekly, 1 week behind ICSA)
  MORTGAGE30US  30y mortgage rate (weekly PMMS)
  BAMLH0A3HYC   ICE BofA CCC-&-lower OAS — the junkiest credit (API ~3y depth)
  UMCSENT       U. Michigan consumer sentiment (monthly)
  PERMIT        building permits (monthly, ~1 month release delay)

Requires FRED_API_KEY in the environment or ``<repo>/.env``.
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import logging

import pandas as pd

from posterioralpha.data import load_fred_macro
from posterioralpha.data.macro import DATASETS_DIR, fetch_fred

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_CSV = DATASETS_DIR / "fred_macro_plus.csv"

# series id -> (publication lag in business days, description)
EXTRA_SERIES = {
    "DGS3MO":       (1,  "3m Treasury bill yield"),
    "DEXJPUS":      (1,  "JPY per USD (carry / funding stress)"),
    "CCSA":         (10, "Continued jobless claims (weekly, +1wk vs ICSA)"),
    "MORTGAGE30US": (2,  "30y fixed mortgage rate (weekly PMMS)"),
    "BAMLH0A3HYC":  (1,  "ICE BofA CCC & lower OAS (API: ~3y depth)"),
    "UMCSENT":      (21, "U. Michigan consumer sentiment (monthly)"),
    "PERMIT":       (25, "Building permits, SAAR (monthly)"),
}


def main():
    base = load_fred_macro()
    idx = base.index
    extra = {}
    for sid, (lag, desc) in EXTRA_SERIES.items():
        s = fetch_fred(sid)
        extra[sid] = s.reindex(idx, method="ffill").shift(lag)
        print(f"{sid:<13} lag {lag:>2}  {s.index.min().date()} → "
              f"{s.index.max().date()}  {desc}")

    panel = base.join(pd.DataFrame(extra, index=idx))
    panel.to_csv(OUT_CSV)
    print(f"\nfred_macro_plus.csv: {panel.shape[0]} days × {panel.shape[1]} "
          f"series ({panel.index[0].date()} → {panel.index[-1].date()})")


if __name__ == "__main__":
    main()
