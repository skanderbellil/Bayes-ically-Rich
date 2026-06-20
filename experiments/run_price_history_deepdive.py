#!/usr/bin/env python3
"""
Polymarket — price-convergence & flow-lead deep dive
====================================================

The wallet-sentiment study showed dollar-weighted sentiment "predicts"
resolution at ~96%, but warned it may be circular: the 3,500-fill cap means
we only see *late* flow, and late dollar-weighting ≈ late price ≈ outcome.

This deep dive breaks the circularity using the **full price history** (CLOB
prices-history, interval=max — no fill cap), and asks two clean questions:

PART A — Price convergence (all resolved markets)
  At each horizon before resolution (7d, 3d, 24h, 6h, 1h), how accurate /
  well-calibrated is the *price itself* as a probability?  This is the curve of
  "how early does the market know the answer", and exposes longshot mispricing.

PART B — Does money LEAD price? (full-fill-coverage markets only)
  For markets where the fill tape reaches back far enough, compute the
  dollar-weighted flow sentiment *as of* an early horizon and compare its
  accuracy to the *contemporaneous price* at that same horizon.  If flow beats
  price → money leads → tradeable.  If they tie → flow is just a price mirror.

Outputs a console report, a convergence-curve PNG, and a per-(market,horizon)
CSV.

Usage
-----
  python experiments/run_price_history_deepdive.py --n-markets 150
  python experiments/run_price_history_deepdive.py --n-markets 80 --no-flow
"""
import argparse
import logging
import time

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.polymarket.fetch import (
    fetch_market_trades,
    fetch_markets,
    fetch_token_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# Horizons before resolution, in DAYS. (interval=max only serves DAILY bars over
# a long market's full life — hourly is empty for long markets — and convergence
# happens over days anyway, so day-granularity is both necessary and sufficient.)
HORIZONS_D = [7, 5, 3, 2, 1]
HLABEL = {7: "7d", 5: "5d", 3: "3d", 2: "2d", 1: "1d"}


def price_at(series: pd.Series, when: pd.Timestamp) -> float | None:
    """Most-recent price at or before `when` (as-of). None if series starts later."""
    s = series[series.index <= when]
    if s.empty:
        return None
    return float(s.iloc[-1])


def flow_sentiment_asof(trades: pd.DataFrame, when: pd.Timestamp) -> float | None:
    """Dollar-weighted YES share of BUY flow using only fills at/<= `when`.

    `when` is a tz-naive date; trade timestamps are coerced to tz-naive for
    comparison. Returns yes_dollar / (yes_dollar + no_dollar), or None.
    """
    if trades.empty:
        return None
    ts = trades["timestamp"].dt.tz_localize(None) if trades["timestamp"].dt.tz else trades["timestamp"]
    sub = trades[ts <= when]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["outcome"] = sub["outcome"].str.upper()
    sub["dollar"] = sub["size"] * sub["price"]
    buy = sub[sub["side"] == "BUY"]
    yes_d = buy[buy["outcome"] == "YES"]["dollar"].sum()
    no_d  = buy[buy["outcome"] == "NO"]["dollar"].sum()
    if (yes_d + no_d) <= 0:
        return None
    return yes_d / (yes_d + no_d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int, default=150)
    ap.add_argument("--min-volume", type=float, default=50_000.0)
    ap.add_argument("--no-flow",    action="store_true", help="skip Part B (flow lead-lag)")
    ap.add_argument("--sleep",      type=float, default=0.15)
    ap.add_argument("--out",        type=str, default="data/paper_trade/price_deepdive.csv")
    ap.add_argument("--plot",       type=str, default="data/paper_trade/price_convergence.png")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  PRICE-CONVERGENCE & FLOW-LEAD DEEP DIVE                  ║
║  when does the market know? does money lead price?        ║
╚══════════════════════════════════════════════════════════╝""")

    print(f"  Fetching {args.n_markets} resolved markets…")
    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} markets with clean 0/1 resolution\n")

    recs = []          # per (market, horizon): price-based
    flow_recs = []     # per (market, horizon): flow vs price (full-coverage only)

    for i, m in enumerate(markets.itertuples(index=False), 1):
        prices = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(prices) < 3:
            continue
        t_res = prices.index.max()
        y = float(m.outcome)

        # Part A: price at each horizon (days before resolution)
        for d in HORIZONS_D:
            p = price_at(prices, t_res - pd.Timedelta(days=d))
            if p is None:
                continue
            recs.append({"question": str(m.question)[:50], "horizon_d": d,
                         "price": p, "outcome": y})

        # Part B: flow lead-lag (only if the tape reaches back before the horizon)
        if not args.no_flow:
            trades = fetch_market_trades(m.condition_id, max_trades=3500, sleep=0)
            if not trades.empty:
                ts = trades["timestamp"].dt.tz_localize(None) if trades["timestamp"].dt.tz else trades["timestamp"]
                oldest = ts.min()
                for d in HORIZONS_D:
                    when = t_res - pd.Timedelta(days=d)
                    if oldest > when:           # tape doesn't cover this far back
                        continue
                    p_price = price_at(prices, when)
                    p_flow  = flow_sentiment_asof(trades, when)
                    if p_price is None or p_flow is None:
                        continue
                    flow_recs.append({"question": str(m.question)[:50], "horizon_d": d,
                                      "p_price": p_price, "p_flow": p_flow, "outcome": y})
            time.sleep(args.sleep)

        if i % 25 == 0:
            logger.info("  processed %d/%d markets (%d price-recs, %d flow-recs)",
                        i, len(markets), len(recs), len(flow_recs))

    df = pd.DataFrame(recs)
    if df.empty:
        print("No price records — aborting.")
        return
    df.to_csv(args.out, index=False)

    # ── PART A report: convergence curve ──────────────────────────────────
    print(f"\n{'═'*74}")
    print(f"  PART A — PRICE AS A FORECAST, BY HORIZON BEFORE RESOLUTION")
    print(f"  ({df['question'].nunique()} markets, {len(df)} obs)")
    print(f"{'═'*74}")
    print(f"  {'horizon':>8} | {'n':>4} | {'dir.acc':>8} | {'Brier':>7} | {'avg price (YES vs NO)':>26}")
    print(f"  {'-'*8}-+-{'-'*4}-+-{'-'*8}-+-{'-'*7}-+-{'-'*26}")
    curve = []
    for h in HORIZONS_D:
        sub = df[df["horizon_d"] == h]
        if sub.empty:
            continue
        p = np.clip(sub["price"].values, 1e-6, 1 - 1e-6)
        yv = sub["outcome"].values
        decided = (p != 0.5)
        acc = np.mean((p[decided] > 0.5) == (yv[decided] == 1)) if decided.any() else np.nan
        brier = np.mean((p - yv) ** 2)
        avg_yes = sub[sub["outcome"] == 1]["price"].mean()
        avg_no  = sub[sub["outcome"] == 0]["price"].mean()
        curve.append({"h": h, "acc": acc, "brier": brier, "n": len(sub),
                      "avg_yes": avg_yes, "avg_no": avg_no})
        print(f"  {HLABEL[h]:>8} | {len(sub):>4} | {acc:>7.1%} | {brier:>7.4f} | "
              f"YES {avg_yes:>5.2f}  /  NO {avg_no:>5.2f}")

    # ── Longshot calibration: do cheap longshots resolve as priced? ───────
    print(f"\n  Longshot calibration (price at 7d/5d horizon):")
    cal = df[df["horizon_d"].isin([7, 5])].copy()
    bins = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01]
    cal["bucket"] = pd.cut(cal["price"], bins, right=False)
    cb = cal.groupby("bucket", observed=True).agg(n=("outcome", "size"),
                                                   resolved_yes=("outcome", "mean"),
                                                   avg_price=("price", "mean"))
    for b, r in cb.iterrows():
        flag = ""
        if r["n"] >= 5:
            gap = r["resolved_yes"] - r["avg_price"]
            flag = "  ← overpriced" if gap < -0.05 else ("  ← underpriced" if gap > 0.05 else "")
        print(f"    price {str(b):>14}: n={int(r['n']):>3}  priced {r['avg_price']:.2f}  "
              f"resolved YES {r['resolved_yes']:.2f}{flag}")

    # ── PART B report: flow vs price at same horizon ──────────────────────
    fdf = pd.DataFrame(flow_recs)
    if not fdf.empty:
        fdf.to_csv(args.out.replace(".csv", "_flow.csv"), index=False)
        print(f"\n{'═'*74}")
        print(f"  PART B — DOES MONEY LEAD PRICE?  (full-coverage markets only)")
        print(f"  {fdf['question'].nunique()} markets, {len(fdf)} matched obs")
        print(f"{'═'*74}")
        print(f"  {'horizon':>8} | {'n':>4} | {'price acc':>10} | {'flow acc':>9} | {'price Brier':>11} | {'flow Brier':>10}")
        print(f"  {'-'*8}-+-{'-'*4}-+-{'-'*10}-+-{'-'*9}-+-{'-'*11}-+-{'-'*10}")
        for h in HORIZONS_D:
            sub = fdf[fdf["horizon_d"] == h]
            if len(sub) < 5:
                continue
            yv = sub["outcome"].values
            pp = np.clip(sub["p_price"].values, 1e-6, 1 - 1e-6)
            pf = np.clip(sub["p_flow"].values,  1e-6, 1 - 1e-6)
            acc_p = np.mean((pp > 0.5) == (yv == 1))
            acc_f = np.mean((pf > 0.5) == (yv == 1))
            br_p  = np.mean((pp - yv) ** 2)
            br_f  = np.mean((pf - yv) ** 2)
            print(f"  {HLABEL[h]:>8} | {len(sub):>4} | {acc_p:>9.1%} | {acc_f:>8.1%} | "
                  f"{br_p:>11.4f} | {br_f:>10.4f}")
        print("\n  If flow acc > price acc at early horizons → money leads price (alpha).")
        print("  If flow ≈ price → dollar sentiment is just a mirror of the quote.")
    elif not args.no_flow:
        print("\n  PART B — no markets had fill coverage reaching the early horizons")
        print("  (3,500-fill cap → high-volume markets only expose late flow).")

    # ── Plot convergence curves ───────────────────────────────────────────
    if curve:
        cdf = pd.DataFrame(curve).sort_values("h", ascending=False)
        xs = [HLABEL[h] for h in cdf["h"]]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(xs, cdf["acc"] * 100, "o-", color="#2b8cbe", lw=2, label="price dir. accuracy")
        ax1.axhline(100 * max(df["outcome"].mean(), 1 - df["outcome"].mean()),
                    ls="--", color="grey", label="always-majority baseline")
        ax1.set_title("Price directional accuracy vs horizon")
        ax1.set_xlabel("time before resolution"); ax1.set_ylabel("accuracy (%)")
        ax1.set_ylim(40, 102); ax1.legend(); ax1.grid(alpha=0.3)

        ax2.plot(xs, cdf["brier"], "o-", color="#e34a33", lw=2)
        ax2.set_title("Price Brier score vs horizon (lower = sharper)")
        ax2.set_xlabel("time before resolution"); ax2.set_ylabel("Brier")
        ax2.grid(alpha=0.3)

        if not fdf.empty:
            fcurve = []
            for h in HORIZONS_D:
                sub = fdf[fdf["horizon_d"] == h]
                if len(sub) < 5:
                    continue
                yv = sub["outcome"].values
                pf = np.clip(sub["p_flow"].values, 1e-6, 1 - 1e-6)
                pp = np.clip(sub["p_price"].values, 1e-6, 1 - 1e-6)
                fcurve.append({"h": h, "acc_f": np.mean((pf > 0.5) == (yv == 1)),
                               "acc_p": np.mean((pp > 0.5) == (yv == 1))})
            if fcurve:
                fc = pd.DataFrame(fcurve).sort_values("h", ascending=False)
                fx = [HLABEL[h] for h in fc["h"]]
                ax1.plot(fx, fc["acc_f"] * 100, "s--", color="#31a354", lw=2,
                         label="flow dir. accuracy (coverage subset)")
                ax1.legend()

        fig.suptitle("Polymarket: how early does price know, and does flow lead it?",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"\n  Convergence plot → {args.plot}")

    print(f"  Per-obs data     → {args.out}")
    print("✓  Deep dive complete.")


if __name__ == "__main__":
    main()
