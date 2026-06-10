#!/usr/bin/env python3
"""
Net liquidity × forward-looking layers — replacing the lagging trend filter.

A 200-day moving average is a backward-looking transform of SPY's own prices —
by the time it crosses, the information is priced.  This study swaps the
confirmation layer for signals that embed *expectations* or *other markets'*
risk pricing:

  vix_ts  : VIX term structure — VIX3M > VIX (contango).  Implied vols are the
            options market's forecast of future risk; backwardation is a
            leading stress signal, not a price lag.
  credit  : HYG vs IEF 21-day relative strength — the credit market's risk
            appetite, which tends to reprice before equities (cross-asset lead).
  liq     : net liquidity expanding (6-month change > 0), publication-lagged.

Tests each layer alone, pairwise votes with liquidity, the three-way vote, and
compares against the lagging-trend vote from run_liquidity_trend.py and SPY.

    python experiments/run_liquidity_predictive.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_net_liquidity, load_etf_universe_prices
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
VIX_CSV = _bootstrap.ROOT / "datasets" / "vix_term_structure.csv"

RF = 0.04
DAILY_RF = RF / 252
H = 126          # 6-month liquidity change horizon
PUB_LAG = 5
CREDIT_WIN = 21  # credit relative-strength window


def load_vix() -> pd.DataFrame:
    """VIX + VIX3M closes; cached to datasets/ (downloads once via yfinance)."""
    if VIX_CSV.exists():
        return pd.read_csv(VIX_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    import yfinance as yf
    v = yf.download(["^VIX", "^VIX3M"], start="2010-01-01",
                    progress=False, auto_adjust=True)["Close"]
    v.columns = [c.replace("^", "") for c in v.columns]
    v = v.dropna(how="all")
    v.to_csv(VIX_CSV, index_label="Date")
    logger.info("cached VIX term structure → %s", VIX_CSV.name)
    return v


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    vix = load_vix()
    spy = px["SPY"]
    idx = (nl.dropna().index
           .intersection(spy.dropna().index)
           .intersection(vix.dropna().index)
           .intersection(px["HYG"].dropna().index)
           .intersection(px["IEF"].dropna().index))
    nl, spy, vix = nl.reindex(idx), spy.reindex(idx), vix.reindex(idx)
    spy_ret = spy.pct_change()

    # ── Signals (all causal: shift(1); liquidity also publication-lagged) ────
    liq = (nl.diff(H) > 0).astype(float).shift(PUB_LAG).shift(1).fillna(0.0)
    # Forward-looking #1: implied-vol term structure in contango
    vix_ts = (vix["VIX3M"] > vix["VIX"]).astype(float).shift(1).fillna(0.0)
    # Forward-looking #2: credit risk appetite (HY outperforming Treasuries)
    rel = (px["HYG"].reindex(idx).pct_change(CREDIT_WIN)
           - px["IEF"].reindex(idx).pct_change(CREDIT_WIN))
    credit = (rel > 0).astype(float).shift(1).fillna(0.0)
    # The lagging baseline being replaced
    trend = (spy > spy.rolling(200).mean()).astype(float).shift(1).fillna(0.0)

    def ret(expo):
        return pd.Series(expo * spy_ret + (1 - expo) * DAILY_RF, index=idx)

    rows = [
        ["VIX term structure only", ret(vix_ts), vix_ts],
        ["Credit (HYG-IEF) only", ret(credit), credit],
        ["Liquidity only", ret(liq), liq],
        ["Vote: liq + VIX-TS", ret(0.5 * (liq + vix_ts)), 0.5 * (liq + vix_ts)],
        ["Vote: liq + credit", ret(0.5 * (liq + credit)), 0.5 * (liq + credit)],
        ["Vote: liq + VIX-TS + credit", ret((liq + vix_ts + credit) / 3),
         (liq + vix_ts + credit) / 3],
        ["Vote: liq + trend (lagging baseline)", ret(0.5 * (liq + trend)),
         0.5 * (liq + trend)],
        ["SPY buy & hold", spy_ret, pd.Series(1.0, index=idx)],
    ]

    print("\n" + "=" * 78)
    print(f"  NET LIQUIDITY × FORWARD-LOOKING LAYERS  ·  "
          f"{idx.min().date()} → {idx.max().date()}")
    print("  predictive layers: VIX term structure, credit (HYG−IEF) — no MA lag")
    print("=" * 78)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = [[name] + [f"{compute_metrics(r.dropna(), rf=RF).get(k, float('nan')):.2f}" for k in keys]
             for name, r, _ in rows]
    print(tabulate(table, headers=["strategy"] + keys, tablefmt="github"))
    print("\n  avg exposure — " + " | ".join(
        f"{n.split(' only')[0] if ' only' in n else n}: {float(e.mean()):.0%}"
        for n, _, e in rows[:3]))
    # signal agreement diagnostics
    for a, b, lab in [(liq, vix_ts, "liq~VIX-TS"), (liq, credit, "liq~credit"),
                      (vix_ts, credit, "VIX-TS~credit")]:
        print(f"  agreement {lab}: {float((a == b).mean()):.0%}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r, _ in rows:
        if " only" in name:        # keep the chart readable: votes + SPY
            continue
        eq = (1 + r.reindex(idx).fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.5 if "buy" in name else 1.2,
                color="black" if "buy" in name else None, alpha=0.85, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title("Net liquidity × forward-looking confirmation on SPY")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_predictive.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
