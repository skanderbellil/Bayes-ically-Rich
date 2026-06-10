#!/usr/bin/env python3
"""
Quant pod simulation: research → hire → monitor → retire, walk-forward.

Each quarterly review, the timing miner runs on data up to that date only;
candidates beating buy & hold on the slice's recent holdout (with bootstrap
+ permutation support) are hired into one of 3 book seats. Sleeves then run
live with frozen parameters and are retired by a monthly risk stop (live DD
> 1.5× research DD), review-time decay detection (trailing-1y live Sharpe
< 0), or two consecutive re-validation failures. Idle seats sit in cash.
Alpaca ETF costs throughout.

The honest headline is the IS → LIVE HAIRCUT table: what each sleeve
claimed at hire vs what it actually delivered while deployed — the direct
measurement of "was the alpha real, and when was it harvested?".

Usage
-----
  python experiments/run_quant_pod.py                 # QQQ, ~20-40 min
  python experiments/run_quant_pod.py --underlying SPY --seed 11
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging

import pandas as pd

from posterioralpha.data import load_etf_universe_prices, load_fred_macro
from posterioralpha.mining.evaluation import sharpe
from posterioralpha.mining.pod import PodConfig, QuantPod

logging.basicConfig(level=logging.WARNING, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "pod"


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    cagr = float(cum.iloc[-1] ** (252 / max(len(r), 1)) - 1)
    return {"sharpe": round(sharpe(r.values), 2), "cagr": round(cagr, 4),
            "max_dd": round(dd, 4)}


def main():
    ap = argparse.ArgumentParser(description="Walk-forward quant pod")
    ap.add_argument("--underlying", default="QQQ")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seats", type=int, default=3)
    args = ap.parse_args()

    prices = load_etf_universe_prices()[args.underlying].dropna()
    macro = load_fred_macro()
    pod = QuantPod(prices, macro=macro,
                   config=PodConfig(seed=args.seed, seats=args.seats))

    print(f"Underlying: {args.underlying} | seats: {args.seats} | quarterly "
          "reviews, monthly risk stops | Alpaca ETF costs")
    res = pod.run()

    book = res["book"]
    bench = prices.pct_change().loc[book.index]
    half = 1.0 * bench / 2  # half-exposure reference (the book averages <1)

    rows = [
        {"strategy": "pod book (3 seats, idle=cash)", **metrics(book)},
        {"strategy": f"{args.underlying} buy & hold", **metrics(bench)},
        {"strategy": f"{args.underlying} half-exposure", **metrics(half)},
    ]
    print("\n" + "=" * 70)
    print(f"BOOK vs BENCHMARK ({book.index[0].date()} → {book.index[-1].date()}"
          " — every decision walk-forward)")
    print("=" * 70)
    print(pd.DataFrame(rows).set_index("strategy").to_string())

    avg_seats = float(res["n_active"].mean())
    print(f"\navg seats filled: {avg_seats:.2f}/{args.seats} | "
          f"hires: {(res['journal']['action'] == 'HIRE').sum()} | "
          f"fires: {(res['journal']['action'] == 'FIRE').sum()} | "
          f"empty-book reviews: {(res['journal']['action'] == 'PASS').sum()}")

    print("\nIS → LIVE HAIRCUT (research claim vs deployed reality)")
    print("=" * 70)
    print(res["haircut"].to_string(index=False))

    print("\nPOD JOURNAL (last 15 events)")
    print(res["journal"].tail(15).to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    res["journal"].to_csv(OUT_DIR / f"journal_{args.underlying}.csv", index=False)
    res["haircut"].to_csv(OUT_DIR / f"haircut_{args.underlying}.csv", index=False)
    book.to_csv(OUT_DIR / f"book_returns_{args.underlying}.csv")
    print(f"\nSaved to {OUT_DIR}/: journal_*, haircut_*, book_returns_*")


if __name__ == "__main__":
    main()
