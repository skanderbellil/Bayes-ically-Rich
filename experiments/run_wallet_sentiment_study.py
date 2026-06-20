#!/usr/bin/env python3
"""
Polymarket — binary vs dollar-weighted wallet sentiment study
=============================================================

Two ways to read a market's crowd:

  * BINARY  (one wallet = one vote): yes_wallets / total_wallets
  * DOLLAR  (one dollar = one vote): yes_vol    / (yes_vol + no_vol)

Question 1: does stripping the dollar value and treating every wallet equally
            predict resolution *differently* (better or worse) than letting
            money talk?

Question 2: is the DISPERSION between the two — dollar_pct − binary_pct —
            itself informative?  A positive dispersion means a few large
            wallets are more bullish than the dispersed crowd (whales loading
            up against retail).  Does that divergence forecast the outcome?

Method
------
Pull N resolved markets (clean 0/1 labels from Gamma), compute both sentiment
measures from the fill tape, then score each as a probabilistic forecast of the
YES resolution.  Compare directional accuracy, Brier score, and log-loss; then
test whether dispersion adds directional alpha on top of the binary signal.

Usage
-----
  python experiments/run_wallet_sentiment_study.py --n-markets 100
  python experiments/run_wallet_sentiment_study.py --n-markets 60 --min-wallets 30
"""
import argparse
import logging
import time

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from tabulate import tabulate

from posterioralpha.polymarket.fetch import fetch_markets, market_wallet_sentiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def _metrics(p: np.ndarray, y: np.ndarray) -> dict:
    """Directional accuracy, Brier score, log-loss for forecasts p vs labels y."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    # Directional: treat p==0.5 as a coin-flip (count as 0.5 correct)
    pred = np.where(p > 0.5, 1, np.where(p < 0.5, 0, -1))
    decided = pred >= 0
    acc = np.mean(pred[decided] == y[decided]) if decided.any() else float("nan")
    brier = np.mean((p - y) ** 2)
    logloss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"n": len(y), "accuracy": acc, "brier": brier, "logloss": logloss,
            "n_decided": int(decided.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",   type=int, default=100, help="resolved markets to pull")
    ap.add_argument("--min-wallets", type=int, default=20,  help="skip markets with fewer total wallets")
    ap.add_argument("--min-volume",  type=float, default=50_000.0, help="market volume screen")
    ap.add_argument("--sleep",       type=float, default=0.2, help="inter-market delay")
    ap.add_argument("--out",         type=str, default="data/paper_trade/wallet_sentiment_study.csv")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  BINARY vs DOLLAR-WEIGHTED WALLET SENTIMENT STUDY         ║
║  one wallet = one vote   vs   one dollar = one vote       ║
╚══════════════════════════════════════════════════════════╝""")

    print(f"  Fetching {args.n_markets} resolved markets (vol ≥ ${args.min_volume:,.0f})…")
    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} markets with clean 0/1 resolution\n")

    rows = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        cid = m.condition_id
        if not cid:
            continue
        s = market_wallet_sentiment(cid)
        total = s["total_wallets"]
        sh_total  = s["yes_shares"] + s["no_shares"]
        dol_total = s["yes_dollar"] + s["no_dollar"]
        if total < args.min_wallets or dol_total <= 0 or sh_total <= 0:
            continue
        p_count  = s["yes_wallets"] / total
        p_shares = s["yes_shares"] / sh_total
        p_dollar = s["yes_dollar"] / dol_total
        rows.append({
            "question":    str(m.question)[:55],
            "outcome":     float(m.outcome),
            "yes_wallets": s["yes_wallets"],
            "no_wallets":  s["no_wallets"],
            "p_count":     round(p_count, 4),
            "p_shares":    round(p_shares, 4),
            "p_dollar":    round(p_dollar, 4),
            "dispersion":  round(p_dollar - p_count, 4),
            "yes_dollar":  s["yes_dollar"],
            "no_dollar":   s["no_dollar"],
        })
        if i % 20 == 0:
            logger.info("  processed %d/%d (%d usable)", i, len(markets), len(rows))
        time.sleep(args.sleep)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No usable markets — try lowering --min-wallets or --min-volume.")
        return
    df.to_csv(args.out, index=False)

    y = df["outcome"].values
    base_rate = y.mean()

    m_count  = _metrics(df["p_count"].values,  y)
    m_shares = _metrics(df["p_shares"].values, y)
    m_dollar = _metrics(df["p_dollar"].values, y)

    print(f"{'═'*72}")
    print(f"  PREDICTIVE POWER  ({len(df)} resolved markets, YES base-rate {base_rate:.1%})")
    print(f"{'═'*72}")
    print(tabulate([
        ["BINARY  (1 wallet = 1 vote)", f"{m_count['accuracy']:.1%}",
         f"{m_count['brier']:.4f}", f"{m_count['logloss']:.4f}", m_count["n_decided"]],
        ["SHARES  (1 share = 1 vote)",  f"{m_shares['accuracy']:.1%}",
         f"{m_shares['brier']:.4f}", f"{m_shares['logloss']:.4f}", m_shares["n_decided"]],
        ["DOLLAR  (1 $ = 1 vote)",      f"{m_dollar['accuracy']:.1%}",
         f"{m_dollar['brier']:.4f}", f"{m_dollar['logloss']:.4f}", m_dollar["n_decided"]],
    ], headers=["signal", "dir.accuracy", "Brier↓", "logloss↓", "n_decided"],
        tablefmt="rounded_grid"))
    print("  (lower Brier / logloss = better-calibrated probability)")

    # ── Dispersion analysis ────────────────────────────────────────────────
    df["disp_sign"] = np.where(df["dispersion"] > 0.02, "whales↑YES",
                       np.where(df["dispersion"] < -0.02, "whales↑NO", "aligned"))
    print(f"\n{'═'*72}")
    print(f"  DISPERSION  =  dollar_pct − binary_pct")
    print(f"  (positive → big wallets more bullish than the headcount crowd)")
    print(f"{'═'*72}")
    grp = df.groupby("disp_sign").agg(
        n=("outcome", "size"),
        yes_rate=("outcome", "mean"),
        avg_p_count=("p_count", "mean"),
        avg_p_dollar=("p_dollar", "mean"),
    ).reset_index()
    print(tabulate([
        [r["disp_sign"], int(r["n"]), f"{r['yes_rate']:.1%}",
         f"{r['avg_p_count']:.1%}", f"{r['avg_p_dollar']:.1%}"]
        for _, r in grp.iterrows()],
        headers=["dispersion bucket", "n", "YES-resolved", "avg binary", "avg dollar"],
        tablefmt="rounded_grid"))

    # Does dispersion add directional alpha on top of binary?
    # When binary is near 50/50 (undecided), does dispersion sign predict outcome?
    coin = df[(df["p_count"] > 0.40) & (df["p_count"] < 0.60)].copy()
    if len(coin) >= 5:
        coin_follow_dollar = ((coin["p_dollar"] > 0.5) == (coin["outcome"] == 1)).mean()
        coin_follow_count  = ((coin["p_count"]  > 0.5) == (coin["outcome"] == 1)).mean()
        print(f"\n  Tie-breaker test — {len(coin)} markets where binary is 40–60% (crowd undecided):")
        print(f"    follow DOLLAR side → {coin_follow_dollar:.1%} correct")
        print(f"    follow BINARY side → {coin_follow_count:.1%} correct")

    # Correlation of dispersion with outcome
    if df["dispersion"].std() > 0:
        corr = np.corrcoef(df["dispersion"], df["outcome"])[0, 1]
        print(f"\n  corr(dispersion, YES-outcome) = {corr:+.3f}")

    # ── Biggest divergences ────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  LARGEST BINARY↔DOLLAR DIVERGENCES")
    print(f"{'═'*72}")
    show = df.reindex(df["dispersion"].abs().sort_values(ascending=False).index).head(12)
    print(tabulate([
        [r["question"][:42], f"{r['p_count']:.0%}", f"{r['p_dollar']:.0%}",
         f"{r['dispersion']:+.0%}", "YES" if r["outcome"] == 1 else "NO"]
        for _, r in show.iterrows()],
        headers=["question", "binary", "dollar", "disp", "resolved"],
        tablefmt="rounded_grid"))

    print(f"\n  Saved → {args.out}")
    print("✓  Study complete.")


if __name__ == "__main__":
    main()
