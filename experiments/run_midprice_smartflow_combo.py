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
* The trades API caps offset-paging at 3,000 fills (a hyper-active wallet's
  offset window can span less than a DAY), so fills are fetched by walking
  BACK IN TIME with the `end` parameter under a per-wallet request budget
  (`--max-requests`), and the study restricts itself to markets resolved in
  the last `--days-back` days. Entries are dropped unless >= `--min-visible`
  pool wallets have fill history spanning the trailing window (per-month
  visibility is printed).
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

from posterioralpha.polymarket.fetch import _get, fetch_markets, fetch_token_history
from posterioralpha.polymarket.paths import RAW_DIR
from posterioralpha.polymarket.smartflow_papertrade import smart_pool
from posterioralpha.polymarket.traders import ACTIVITY_URL

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS = _bootstrap.ROOT / "results" / "polymarket"
# committed snapshot (cf. data/polymarket_smart_money_snapshot): the per-obs
# signal panel + summary survive fresh clones, unlike results/ and data/raw/
SNAPSHOT = _bootstrap.ROOT / "data" / "polymarket_combo_snapshot"

BANDS = [(0.10, 0.90), (0.10, 0.30), (0.30, 0.70), (0.70, 0.90)]


def price_at(series: pd.Series, when: pd.Timestamp) -> float | None:
    s = series[series.index <= when]
    return float(s.iloc[-1]) if not s.empty else None


def fetch_wallet_fills_window(wallet: str, start: pd.Timestamp,
                              max_requests: int = 60, page: int = 500,
                              sleep: float = 0.12) -> pd.DataFrame:
    """TRADE fills for one wallet back to `start`, paging by TIME, not offset.

    The data-api hard-caps `offset` at 3000 (hyper-active wallets' offset
    window can span less than a day), but accepts an `end` timestamp — so we
    walk backwards: each request asks for the `page` fills before the oldest
    timestamp seen so far, until we reach `start` or the request budget.
    Cached per (wallet, start-date) under data/raw/polymarket/traders_window/.
    """
    cache_dir = RAW_DIR / "traders_window"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{wallet.lower()}_{start.date()}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["timestamp"],
                           dtype={"asset": str, "conditionId": str, "side": str})

    rows, end_ts = [], None
    for _ in range(max_requests):
        params = {"user": wallet.lower(), "type": "TRADE", "limit": page}
        if end_ts is not None:
            params["end"] = end_ts
        batch = _get(ACTIVITY_URL, params)
        if not isinstance(batch, list) or not batch:
            break
        for r in batch:
            rows.append({"timestamp": int(r.get("timestamp") or 0),
                         "conditionId": str(r.get("conditionId") or ""),
                         "asset": str(r.get("asset") or ""),
                         "side": r.get("side") or "",
                         "usdcSize": float(r.get("usdcSize") or 0.0)})
        oldest = min(r["timestamp"] for r in rows)
        if oldest <= start.timestamp() or len(batch) < page:
            break
        end_ts = oldest - 1
        if sleep:
            time.sleep(sleep)

    df = pd.DataFrame(rows).drop_duplicates()
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_localize(None)
        df = df.sort_values("timestamp").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def pool_fills(pool: list[str], start: pd.Timestamp, max_requests: int,
               sleep: float) -> pd.DataFrame:
    """Concatenated windowed fills of every pool wallet (cached per wallet)."""
    frames = []
    for i, w in enumerate(pool, 1):
        df = fetch_wallet_fills_window(w, start, max_requests=max_requests, sleep=sleep)
        if not df.empty:
            df["wallet"] = w
            frames.append(df)
        if i % 10 == 0:
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
    ap.add_argument("--n-markets", type=int, default=1200,
                    help="markets scanned by volume; filtered to recent resolutions")
    ap.add_argument("--min-volume", type=float, default=10_000.0)
    ap.add_argument("--days-back", type=int, default=90,
                    help="only markets RESOLVED within the last N days (fill-history reach)")
    ap.add_argument("--horizon", type=int, nargs="+", default=[7, 3])
    ap.add_argument("--window", type=int, default=7, help="trailing flow window (days)")
    ap.add_argument("--min-buyers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--max-requests", type=int, default=60,
                    help="fill-history request budget per wallet (500 fills each)")
    ap.add_argument("--min-visible", type=int, default=15,
                    help="min pool wallets with history spanning an entry")
    ap.add_argument("--sleep", type=float, default=0.1)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("""
╔════════════════════════════════════════════════════════════════╗
║  MID-PRICED YES × SMART-FLOW CONSENSUS  ·  combo backtest       ║
║  does smart net flow rank the mid-price band?                   ║
╚════════════════════════════════════════════════════════════════╝""")

    # ── 1. recently-resolved universe + price observations ────────────────
    cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=args.days_back)
    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].copy()
    t_close = pd.to_datetime(markets["closed_time"].fillna(markets["end_date"]),
                             errors="coerce", utc=True, format="mixed").dt.tz_localize(None)
    markets = markets[t_close >= cutoff].reset_index(drop=True)
    print(f"  {len(markets)} markets with clean 0/1 outcome resolved since {cutoff.date()}")

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

    if obs.empty:
        print("No observations — aborting."); return

    # ── 2. smart pool + windowed historical fills ─────────────────────────
    pool = smart_pool()
    fills_start = cutoff - pd.Timedelta(days=max(args.horizon) + args.window + 2)
    fills = pool_fills(pool, fills_start, args.max_requests, args.sleep)

    # visibility: a wallet can signal at t only if its fetched history spans
    # the whole trailing window (budget-capped wallets fade out going back)
    first_ts = fills.groupby("wallet")["timestamp"].min()
    def n_visible(t: pd.Timestamp) -> int:
        return int((first_ts <= t - pd.Timedelta(days=args.window)).sum())
    obs["n_visible"] = obs["t_entry"].map(n_visible)
    cov_month = obs.groupby(obs["t_entry"].dt.to_period("M"))["n_visible"].mean()
    print("\n  mean pool wallets visible, by entry month:")
    print("  " + "  ".join(f"{m}:{c:.0f}" for m, c in cov_month.items()))
    kept = obs[obs["n_visible"] >= args.min_visible].copy()
    print(f"  {len(kept)}/{len(obs)} observations kept at ≥{args.min_visible} visible wallets\n")
    if kept.empty:
        print("No observations survive the visibility gate — aborting."); return

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
    if res.empty:
        print("No books had any trades — aborting."); return
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
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    res.to_csv(SNAPSHOT / "combo_summary.csv", index=False)
    kept.to_csv(SNAPSHOT / "combo_obs_panel.csv", index=False)   # per-obs signal panel

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
    fig.savefig(SNAPSHOT / "combo_equity.png", dpi=110)
    print(f"\n  results  → {out_csv}\n  plot     → {out_png}"
          f"\n  snapshot → {SNAPSHOT} (committed)")


if __name__ == "__main__":
    main()
