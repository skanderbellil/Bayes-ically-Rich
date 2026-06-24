#!/usr/bin/env python3
"""
Polymarket weather edge — London daily max-temperature, point-in-time backtest
==============================================================================

Tests whether London "highest temperature" buckets are mispriced relative to a
five-model NWP consensus (``posterioralpha.polymarket.weather``) — the same
*scheduled, objectively-resolved* archetype as the one macro trade that survived
``STRATEGY_SYNTHESIS``, but in a domain uncorrelated to the Fed cycle.

No-lookahead by construction:
  * forecast  = Open-Meteo *historical-forecast archive* for the target date
                (what the models issued then, not the ERA5 reanalysis).
  * decision  = the last CLOB mid at/just before ``end − lead_hours``; the trade
                pays the spread via ``--slippage``.
  * settlement= the market's own resolution (outcomePrices), not ERA5.

Trade rule (london-edge): for each event-day, BUY the single Yes bucket with the
largest positive ``edge = P_model − P_market`` if it clears ``--threshold``.
One trade per day ⇒ event-days are independent ⇒ the bootstrap resamples days.

Usage
-----
  python experiments/run_polymarket_weather_backtest.py
  python experiments/run_polymarket_weather_backtest.py --threshold 0.05 --lead-hours 24
  python experiments/run_polymarket_weather_backtest.py --start 2026-02-05 --end 2026-06-20
  python experiments/run_polymarket_weather_backtest.py --mode all --refresh
"""
from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import weather as wx
from posterioralpha.polymarket.paths import RAW_DIR, RESULTS_DIR, ensure_dirs

logger = logging.getLogger("weather_backtest")

EVENT_CACHE = RAW_DIR / "weather_events"
FC_CACHE = RAW_DIR / "weather_fc"


# ---------------------------------------------------------------------------
# Cached data access
# ---------------------------------------------------------------------------
def _cached_event(d: date, refresh: bool) -> dict | None:
    EVENT_CACHE.mkdir(parents=True, exist_ok=True)
    fp = EVENT_CACHE / f"{d.isoformat()}.json"
    if fp.exists() and not refresh:
        txt = fp.read_text()
        return json.loads(txt) if txt.strip() else None
    ev = wx.fetch_weather_event(d)
    # strip the non-serialisable Bucket object before caching
    if ev:
        for b in ev["buckets"]:
            b.pop("bucket", None)
    fp.write_text(json.dumps(ev) if ev else "")
    return ev


def _cached_forecast(target: str, refresh: bool) -> wx.ForecastConsensus:
    FC_CACHE.mkdir(parents=True, exist_ok=True)
    fp = FC_CACHE / f"{target}.json"
    if fp.exists() and not refresh:
        d = json.loads(fp.read_text())
        return wx.ForecastConsensus(d["target"], d["mu"], d["sigma"], d["n_models"], d["per_model"])
    fc = wx.point_in_time_forecast(target)
    fp.write_text(json.dumps({
        "target": fc.target, "mu": fc.mu, "sigma": fc.sigma,
        "n_models": fc.n_models, "per_model": fc.per_model,
    }))
    return fc


def _decision_price(token: str, end: datetime, decision_ts: datetime) -> float | None:
    """Last CLOB mid at or before the decision timestamp (no lookahead).

    The CLOB rejects a ``startTs`` more than ~14 days before settlement (the
    market's life), so try the widest accepted window first and shrink on miss.
    """
    end_ts = int(end.timestamp())
    s = pd.Series(dtype=float)
    for lookback in (14, 10, 7, 5):
        start_ts = int((end - timedelta(days=lookback)).timestamp())
        s = wx.fetch_token_window(token, start_ts, end_ts, fidelity_minutes=60)
        if not s.empty:
            break
    if s.empty:
        return None
    s = s[s.index <= pd.Timestamp(decision_ts)]
    return float(s.iloc[-1]) if len(s) else None


# ---------------------------------------------------------------------------
# Per-event evaluation
# ---------------------------------------------------------------------------
def evaluate_event(d: date, lead_hours: int, refresh: bool) -> list[dict] | None:
    ev = _cached_event(d, refresh)
    if not ev or not ev.get("end_date"):
        return None
    # only settled events (every bucket has a 0/1 outcome) are scoreable
    if any(b.get("outcome") is None for b in ev["buckets"]):
        return None
    end = datetime.fromisoformat(ev["end_date"].replace("Z", "+00:00"))
    decision_ts = end - timedelta(hours=lead_hours)

    fc = _cached_forecast(ev["date"], refresh)
    if not fc.trustworthy:
        return None

    # rebuild Bucket objects (cache dropped them) and collect decision-time prices
    prices, bks = {}, []
    for b in ev["buckets"]:
        bucket = wx.Bucket(b["question"], b["lo"], b["hi"], b["label"])
        b2 = dict(b); b2["bucket"] = bucket
        bks.append(b2)
        p = _decision_price(b["token_yes"], end, decision_ts)
        if p is not None:
            prices[b["token_yes"]] = p
    if not prices:
        return None
    rows = wx.compute_edges(bks, fc, prices)
    for r in rows:
        r["event"] = ev["date"]
        r["mu"] = fc.mu
        r["sigma"] = fc.sigma
    return rows


# ---------------------------------------------------------------------------
# Bootstrap & metrics
# ---------------------------------------------------------------------------
def event_bootstrap_ci(per_event_pnl: np.ndarray, n_boot: int = 10_000,
                       seed: int = 0) -> tuple[float, float]:
    """95% CI for mean per-event PnL by resampling event-days with replacement."""
    if len(per_event_pnl) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = per_event_pnl[rng.integers(0, len(per_event_pnl),
                                       size=(n_boot, len(per_event_pnl)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise_trades(trades: pd.DataFrame, slippage: float) -> dict:
    """Per-event PnL (one trade/event), with cost, bootstrap CI, hit rate."""
    if trades.empty:
        return {"n": 0}
    entry = trades["p_market"] + slippage          # pay the ask
    pnl = trades["outcome"] - entry                # profit per Yes contract
    pe = pnl.to_numpy()
    lo, hi = event_bootstrap_ci(pe)
    return {
        "n": int(len(pe)),
        "pnl_per_event": float(pe.mean()),
        "ci": (lo, hi),
        "hit": float((pe > 0).mean()),
        "worst": float(pe.min()),
        "best": float(pe.max()),
        "std": float(pe.std(ddof=1)) if len(pe) > 1 else 0.0,
        "total": float(pe.sum()),
    }


def select_trades(edges: pd.DataFrame, threshold: float, mode: str) -> pd.DataFrame:
    """Apply the trade rule. 'best' = max-edge bucket/day; 'all' = every qualifier."""
    qual = edges[edges["edge"] >= threshold]
    if qual.empty:
        return qual
    if mode == "best":
        idx = qual.groupby("event")["edge"].idxmax()
        return qual.loc[idx]
    return qual


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(edges: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """Reliability of the model probability vs market resolution, by p_model bin."""
    df = edges.dropna(subset=["p_model", "outcome"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["bin"] = pd.qcut(df["p_model"], q=min(n_bins, df["p_model"].nunique()),
                        duplicates="drop")
    g = df.groupby("bin", observed=True)
    return pd.DataFrame({
        "n": g.size(),
        "p_model": g["p_model"].mean(),
        "p_market": g["p_market"].mean(),
        "realized": g["outcome"].mean(),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2026-02-05", help="first event-day (YYYY-MM-DD)")
    ap.add_argument("--end", default="2026-06-20", help="last event-day (YYYY-MM-DD)")
    ap.add_argument("--threshold", type=float, default=0.05, help="min edge to trade")
    ap.add_argument("--lead-hours", type=int, default=24,
                    help="decide this many hours before settlement")
    ap.add_argument("--slippage", type=float, default=0.01, help="half-spread paid per entry")
    ap.add_argument("--mode", choices=["best", "all"], default="best",
                    help="'best' = one max-edge bucket/day; 'all' = every qualifier")
    ap.add_argument("--refresh", action="store_true", help="ignore caches, re-pull")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_dirs()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    days = list(daterange(start, end))

    logger.info("Weather edge backtest — London tmax  |  %s → %s  (%d candidate days)",
                args.start, args.end, len(days))
    logger.info("decision lead = %dh, slippage = %.0f bps, threshold = %.0f bps, mode = %s\n",
                args.lead_hours, args.slippage * 1e4, args.threshold * 1e4, args.mode)

    def _run(d):
        try:
            return evaluate_event(d, args.lead_hours, args.refresh)
        except Exception as e:  # one bad day must not kill the run
            logger.debug("skip %s: %r", d, e)
            return None

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for res in ex.map(_run, days):
            if res:
                rows.extend(res)

    if not rows:
        logger.error("No scoreable event-days found in range. Try --refresh or widen dates.")
        return
    edges = pd.DataFrame(rows)
    n_events = edges["event"].nunique()
    logger.info("Scored %d buckets across %d settled event-days.\n", len(edges), n_events)

    # ---- headline strategy ------------------------------------------------
    trades = select_trades(edges, args.threshold, args.mode)
    summ = summarise_trades(trades, args.slippage)

    # ---- benchmarks -------------------------------------------------------
    # (a) buy the market favorite (highest p_market) each day — no model used
    fav_idx = edges.groupby("event")["p_market"].idxmax()
    fav = summarise_trades(edges.loc[fav_idx], args.slippage)
    # (b) buy the model favorite (highest p_model) each day — model, no price filter
    mdl_idx = edges.groupby("event")["p_model"].idxmax()
    mdl = summarise_trades(edges.loc[mdl_idx], args.slippage)

    def _fmt(s):
        if s.get("n", 0) == 0:
            return ["—"] * 6
        lo, hi = s["ci"]
        return [s["n"], f"{s['pnl_per_event']:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]",
                f"{s['hit']:.0%}", f"{s['worst']:+.3f}", f"{s['total']:+.3f}"]

    table = [
        ["edge ≥ thr (model vs market)", *_fmt(summ)],
        ["market favorite (benchmark)", *_fmt(fav)],
        ["model favorite (benchmark)", *_fmt(mdl)],
    ]
    logger.info(tabulate(
        table,
        headers=["book", "n", "PnL/event", "95% CI", "hit", "worst", "total"],
        tablefmt="github"))

    # ---- threshold sweep --------------------------------------------------
    logger.info("\nThreshold sweep (mode=%s):", args.mode)
    sweep = []
    for thr in (0.03, 0.05, 0.08, 0.10, 0.15):
        s = summarise_trades(select_trades(edges, thr, args.mode), args.slippage)
        if s.get("n", 0):
            lo, hi = s["ci"]
            sweep.append([f"{thr:.0%}", s["n"], f"{s['pnl_per_event']:+.4f}",
                          f"[{lo:+.4f}, {hi:+.4f}]", f"{s['hit']:.0%}", f"{s['worst']:+.3f}"])
    logger.info(tabulate(sweep, headers=["thr", "n", "PnL/event", "95% CI", "hit", "worst"],
                         tablefmt="github"))

    # ---- calibration ------------------------------------------------------
    cal = calibration_table(edges)
    if not cal.empty:
        logger.info("\nCalibration (model prob vs realized, all buckets):")
        logger.info(tabulate(cal.round(3), headers="keys", tablefmt="github", showindex=False))

    # ---- persist ----------------------------------------------------------
    out_csv = RESULTS_DIR / "weather_edge_buckets.csv"
    edges.to_csv(out_csv, index=False)
    if not trades.empty:
        tr = trades.copy()
        tr["entry"] = tr["p_market"] + args.slippage
        tr["pnl"] = tr["outcome"] - tr["entry"]
        tr.sort_values("event").to_csv(RESULTS_DIR / "weather_edge_trades.csv", index=False)

        # equity curve + calibration plot
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        eq = tr.sort_values("event")["pnl"].cumsum()
        axes[0].plot(range(len(eq)), eq.to_numpy(), color="#1f9d55")
        axes[0].axhline(0, color="grey", lw=0.8, ls="--")
        axes[0].set_title(f"Cumulative PnL/contract — edge≥{args.threshold:.0%} (n={summ['n']})")
        axes[0].set_xlabel("trade #"); axes[0].set_ylabel("cum PnL (per contract)")
        if not cal.empty:
            axes[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
            axes[1].plot(cal["p_model"], cal["realized"], "o-", color="#1f9d55", label="model")
            axes[1].plot(cal["p_market"], cal["realized"], "s--", color="#c0392b", label="market")
            axes[1].set_title("Calibration: predicted vs realized")
            axes[1].set_xlabel("predicted P(bucket)"); axes[1].set_ylabel("realized freq")
            axes[1].legend()
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "weather_edge.png", dpi=110)
        logger.info("\nSaved: %s , weather_edge_trades.csv , weather_edge.png",
                    out_csv.name)

    # ---- verdict ----------------------------------------------------------
    if summ.get("n", 0):
        lo, hi = summ["ci"]
        verdict = ("✓ CI excludes zero — candidate edge" if lo > 0 else
                   "✗ CI includes zero — not distinguishable from noise")
        logger.info("\nVERDICT: %s  (%d trades, %+.4f/event, 95%% CI [%+.4f, %+.4f])",
                    verdict, summ["n"], summ["pnl_per_event"], lo, hi)


if __name__ == "__main__":
    main()
