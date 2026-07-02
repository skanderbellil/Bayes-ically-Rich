#!/usr/bin/env python3
"""
Polymarket — mid-priced YES × smart-flow consensus COMBO backtest
=================================================================

The two live paper-trade books each trade ONE signal:

  * mid-price YES (`run_midprice_yes_backtest.py`): markets priced inside a
    band a few days before resolution resolve Yes more often than priced —
    buy every in-band market, hold to resolution.
  * smart-flow consensus (`run_smart_flow_paper_update.py`): go long open
    tokens that >= N distinct non-MM leaderboard wallets net-bought recently.

This backtests the INTERSECTION historically: enter an in-band market at a
fixed horizon before resolution ONLY when the smart pool's trailing net flow
agrees (>= N distinct net buyers of the Yes token, and more bulls than bears —
No-leg buyers and Yes-leg net sellers count as bears). Four books per
(band, horizon) answer whether flow *ranks* the band:

  midprice   every in-band market                       (the incumbent)
  combo      in-band AND smart consensus                (the candidate)
  anti       in-band AND zero smart net buyers          (the discard pile)
  flow_only  consensus but OUTSIDE the band             (flow without level)

Honesty
-------
* Pool membership is TODAY'S leaderboard (survivorship in *which wallets
  exist*) — but every fill is timestamped, so signal timing at each entry is
  causal given the pool. Same caveat as ORDER_FLOW/SPECIALISTS; the paper
  tracker is the survivorship-free version.
* The trades API returns each wallet's most recent `--max-trades` fills, so
  pool coverage decays going back in time. Entries are dropped unless at
  least `--min-coverage` of the pool has fill history spanning the entry
  (per-month coverage is printed).
* Entry at daily-close price + `--haircut` (no historical book; the new
  `book_snapshots.csv.gz` accumulation will eventually replace the guess).
* Bootstrap CIs cluster by resolution DATE (same-day resolutions are one
  event, not independent evidence).

Usage
-----
  python experiments/run_midprice_smartflow_combo.py
  python experiments/run_midprice_smartflow_combo.py --n-markets 400 --min-buyers 2
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

from posterioralpha.polymarket.fetch import fetch_markets, fetch_token_history
from posterioralpha.polymarket.smartflow_papertrade import smart_pool
from posterioralpha.polymarket.traders import fetch_trader_trades

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = _bootstrap.ROOT / "results" / "polymarket"

BANDS = [(0.10, 0.90), (0.10, 0.30), (0.30, 0.70), (0.70, 0.90)]


def price_at(series: pd.Series, when: pd.Timestamp) -> float | None:
    s = series[series.index <= when]
    return float(s.iloc[-1]) if not s.empty else None


def pool_fills(pool: list[str], max_trades: int, sleep: float) -> pd.DataFrame:
    """Concatenated timestamped fills of every pool wallet (cached per wallet)."""
    frames = []
    for i, w in enumerate(pool, 1):
        df = fetch_trader_trades(w, max_trades=max_trades, sleep=sleep)
        if not df.empty:
            df = df[["timestamp", "conditionId", "asset", "side", "usdcSize"]].copy()
            df["wallet"] = w
            frames.append(df)
        if i % 20 == 0:
            logger.info("  pool fills %d/%d wallets", i, len(pool))
    fills = pd.concat(frames, ignore_index=True)
    fills["signed"] = np.where(fills["side"].str.upper() == "BUY",
                               fills["usdcSize"], -fills["usdcSize"])
    logger.info("pool fills: %d fills, %d wallets, %s → %s",
                len(fills), fills["wallet"].nunique(),
                fills["timestamp"].min().date(), fills["timestamp"].max().date())
    return fills


def flow_at(fills: pd.DataFrame, yes_token: str, condition_id: str,
            t_entry: pd.Timestamp, window_days: int) -> tuple[int, int]:
    """(bulls, bears) among pool wallets over [t_entry - window, t_entry].

    bull: net (BUY−SELL) dollars in the Yes token > 0
    bear: net buyer of the market's other (No) leg, or net seller of Yes.
    """
    w = fills[(fills["timestamp"] > t_entry - pd.Timedelta(days=window_days))
              & (fills["timestamp"] <= t_entry)
              & (fills["conditionId"] == condition_id)]
    if w.empty:
        return 0, 0
    yes = w[w["asset"] == yes_token].groupby("wallet")["signed"].sum()
    no = w[w["asset"] != yes_token].groupby("wallet")["signed"].sum()
    bulls = set(yes[yes > 0].index) | set(no[no < 0].index)
    bears = set(yes[yes < 0].index) | set(no[no > 0].index)
    return len(bulls - bears), len(bears - bulls)


def clustered_ci(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI of mean per-$1 return, resampling resolution DATES."""
    by_day = frame.groupby(frame["t_res"].dt.date)["ret"].mean()
    if len(by_day) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [by_day.sample(len(by_day), replace=True, random_state=rng.integers(2**31)).mean()
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def book_stats(sel: pd.DataFrame, label: str) -> dict:
    if sel.empty:
        return {"book": label, "n": 0}
    rets = sel["ret"].values
    lo, hi = clustered_ci(sel)
    return {"book": label, "n": len(sel), "n_days": sel["t_res"].dt.date.nunique(),
            "win": (sel["outcome"] == 1).mean(), "mean_ret": rets.mean(),
            "ci_lo": lo, "ci_hi": hi,
            "sharpe": rets.mean() / rets.std(ddof=1) if len(rets) > 1 and rets.std(ddof=1) > 0 else float("nan"),
            "total_pnl": rets.sum()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=250)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, nargs="+", default=[7, 3])
    ap.add_argument("--window", type=int, default=7, help="trailing flow window (days)")
    ap.add_argument("--min-buyers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--max-trades", type=int, default=4000, help="fills fetched per wallet")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="min fraction of pool with fill history spanning an entry")
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("""
╔════════════════════════════════════════════════════════════════╗
║  MID-PRICED YES × SMART-FLOW CONSENSUS  ·  combo backtest       ║
║  does smart net flow rank the mid-price band?                   ║
╚════════════════════════════════════════════════════════════════╝""")

    # ── 1. resolved-market universe + price observations ──────────────────
    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets with clean 0/1 outcome")

    recs = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        prices = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(prices) < 3:
            continue
        t_res = prices.index.max()
        for d in args.horizon:
            t_entry = t_res - pd.Timedelta(days=d)
            p = price_at(prices, t_entry)
            if p is None or not (0.0 < p < 1.0):
                continue
            recs.append({"horizon_d": d, "t_entry": t_entry, "t_res": t_res,
                         "price": p, "outcome": float(m.outcome),
                         "yes_token": str(m.yes_token), "condition_id": str(m.condition_id),
                         "question": str(m.question)[:45]})
        if i % 50 == 0:
            logger.info("  priced %d/%d markets (%d obs)", i, len(markets), len(recs))
        time.sleep(args.sleep if i % 5 == 0 else 0)
    obs = pd.DataFrame(recs)
    print(f"  {len(obs)} (market × horizon) observations")

    # ── 2. smart pool + historical fills ──────────────────────────────────
    pool = smart_pool()
    fills = pool_fills(pool, args.max_trades, args.sleep)

    # coverage: a wallet can only signal at t if its (truncated) history spans t
    first_ts = fills.groupby("wallet")["timestamp"].min()
    def coverage(t: pd.Timestamp) -> float:
        return float((first_ts <= t - pd.Timedelta(days=args.window)).mean())
    obs["coverage"] = obs["t_entry"].map(coverage)
    cov_month = obs.groupby(obs["t_entry"].dt.to_period("M"))["coverage"].mean()
    print("\n  pool fill-history coverage by entry month (fraction of pool visible):")
    print("  " + "  ".join(f"{m}:{c:.2f}" for m, c in cov_month.items()))
    kept = obs[obs["coverage"] >= args.min_coverage].copy()
    print(f"  {len(kept)}/{len(obs)} observations kept at coverage ≥ {args.min_coverage:.0%}\n")

    # ── 3. flow signal at each entry ──────────────────────────────────────
    bulls, bears = [], []
    for i, r in enumerate(kept.itertuples(index=False), 1):
        b, s = flow_at(fills, r.yes_token, r.condition_id, r.t_entry, args.window)
        bulls.append(b); bears.append(s)
        if i % 100 == 0:
            logger.info("  flow signal %d/%d", i, len(kept))
    kept["bulls"], kept["bears"] = bulls, bears
    kept["entry"] = np.minimum(kept["price"] + args.haircut, 0.99)
    kept["ret"] = kept["outcome"] / kept["entry"] - 1.0
    touched = (kept["bulls"] + kept["bears"] > 0).mean()
    print(f"  smart pool touched {touched:.0%} of kept observations in the {args.window}d window\n")

    # ── 4. books ───────────────────────────────────────────────────────────
    all_rows, curves = [], {}
    for d in args.horizon:
        sub = kept[kept["horizon_d"] == d]
        for lo_b, hi_b in BANDS:
            band = sub[(sub["price"] >= lo_b) & (sub["price"] < hi_b)]
            if band.empty:
                continue
            for nb in args.min_buyers:
                combo = band[(band["bulls"] >= nb) & (band["bulls"] > band["bears"])]
                anti = band[band["bulls"] == 0]
                flow_only = sub[(~sub.index.isin(band.index))
                                & (sub["bulls"] >= nb) & (sub["bulls"] > sub["bears"])]
                for label, frame in [("midprice", band), (f"combo≥{nb}", combo),
                                     ("anti(0 bulls)", anti), (f"flow_only≥{nb}", flow_only)]:
                    st = book_stats(frame, label)
                    st.update({"horizon_d": d, "band": f"[{lo_b:.2f},{hi_b:.2f})"})
                    all_rows.append(st)
                if (lo_b, hi_b) == BANDS[0] and nb == args.min_buyers[-1]:
                    curves[f"{d}d midprice"] = band
                    curves[f"{d}d combo≥{nb}"] = combo

    res = pd.DataFrame(all_rows)
    res = res[res["n"] > 0]
    for d in args.horizon:
        print(f"\n{'═'*100}\n  horizon = {d}d before resolution   "
              f"(haircut {args.haircut:.2f}, window {args.window}d)\n{'═'*100}")
        v = res[res["horizon_d"] == d]
        cols = ["band", "book", "n", "n_days", "win", "mean_ret", "ci_lo", "ci_hi", "sharpe", "total_pnl"]
        print(v[cols].to_string(index=False,
              formatters={c: (lambda x: f"{x:+.3f}") for c in
                          ("mean_ret", "ci_lo", "ci_hi", "total_pnl")} |
                          {"win": lambda x: f"{x:.2f}", "sharpe": lambda x: f"{x:+.2f}"}))

    out_csv = RESULTS / "midprice_smartflow_combo.csv"
    res.to_csv(out_csv, index=False)

    # equity curves (additive $1-per-trade PnL, time-ordered by resolution)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label, frame in curves.items():
        f = frame.sort_values("t_res")
        ax.plot(f["t_res"], f["ret"].cumsum(), label=f"{label} (n={len(f)})", lw=1.6)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"Mid-price band {BANDS[0]} × smart-flow consensus — additive PnL per $1 trade "
                 f"(haircut {args.haircut:.2f})")
    ax.set_ylabel("cumulative PnL ($ per $1 staked/trade)")
    ax.legend()
    fig.autofmt_xdate(); fig.tight_layout()
    out_png = RESULTS / "midprice_smartflow_combo.png"
    fig.savefig(out_png, dpi=110)
    print(f"\n  results → {out_csv}\n  plot    → {out_png}")


if __name__ == "__main__":
    main()
