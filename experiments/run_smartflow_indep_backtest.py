#!/usr/bin/env python3
"""
Polymarket — smart-flow independence classifier: retrospective replay ("backtest")
====================================================================================

The independence classifier (``posterioralpha.polymarket.smartflow_independence``,
see ``docs/polymarket/SMART_FLOW_INDEPENDENCE.md``) was just shipped as a
*forward*, pre-registered experiment: it has no resolved positions yet. This
script does NOT retune or pre-empt that forward ledger -- it is a separate,
clearly-labelled *exploratory* replay over the frozen wallet-fill cache
(``data/polymarket_smart_money_snapshot/raw_cache_2026-06-23.tar.gz``, 126
wallets, 224,927 fills, right-truncated at 2026-06-23) so we get an early,
honest-but-biased read of whether the independent/cascade split has any
separation at all, without waiting months for the forward ledger to accumulate
40+ resolved positions per class.

**This is exploratory evidence only.** See the mandatory caveats printed at the
end of this script's output and appended to
``docs/polymarket/SMART_FLOW_INDEPENDENCE.md`` -- the wallet pool is itself the
2026-06-23 snapshot (a form of look-ahead / survivorship: these are wallets we
already know existed and had non-empty histories as of the snapshot date), and
entries use the fill price as an ask proxy (no historical order book, so no
spread is captured). The frozen thresholds (24h / 0.05 / 0.20, 2-of-3 rule) are
NOT retuned here regardless of what this replay shows -- a retuned threshold
would be a new experiment on a new ledger, exactly as the pre-registration doc
says.

Method
------
1. Load the 126 cached wallet fill histories from the tarball (no network for
   this step) into ``trades_by_wallet: {wallet: DataFrame}``, shaped exactly
   like ``fetch_trader_trades`` output.
2. Walk day by day (step 1 day) over the replayable range. On each day ``t``,
   using only fills with ``timestamp <= t`` within the trailing 7-day window,
   find outcome tokens where >= 3 distinct *eligible* wallets have a BUY.  A
   wallet counts as an eligible voter on day ``t`` only if its cached history
   starts *before* ``t - 7d`` -- otherwise a wallet whose cache happens to
   start mid-window would look like it "just discovered" the token, which is a
   left-truncation artefact, not a temporal-independence signal.  An event
   fires the first day a token crosses the threshold (one event per token).
   The event's entry price is the fill price of the chronologically LAST
   first-buyer among the buyer set at crossing -- the closest offline proxy to
   "enter when the Nth buyer completes the consensus", since this replay has
   no historical order book and therefore no bid-ask spread (see caveats).
3. Classifies every event by calling the SHIPPED ``independence_features`` /
   ``classify_independence`` verbatim -- no reimplementation -- passing the
   buyer set at crossing and a ``trades_by_wallet`` view filtered to
   ``timestamp <= t`` (no look-ahead) and the new ``asof=t`` parameter (added
   to the shipped module so the trailing-window cutoff is computed from the
   replay instant rather than hardcoded wall-clock ``now``; live behaviour is
   unchanged because ``asof`` defaults to ``None``).
4. Resolves each event's outcome via ``fetch_market_resolution`` (CLOB,
   authoritative), on-disk cached so re-runs are cheap. Applies the same
   0.05-0.90 entry-price band the live scanner uses; out-of-band events are
   recorded but excluded from PnL. pnl = outcome / entry_price - 1 (per-unit,
   fraction-free -- this replay has no bet-sizing concept).
5. Reports per-class counts, win rate, mean/std pnl, a Welch t-stat of the
   independent-minus-cascade pnl difference, and per-component (temporal /
   price / co-movement) pass-vs-fail diagnostics. Writes the event-level table
   to ``data/polymarket_smart_money_snapshot/smartflow_indep_backtest_events.csv``.

Usage
-----
  python experiments/run_smartflow_indep_backtest.py
  python experiments/run_smartflow_indep_backtest.py --sleep 0.3 --refresh-resolution
"""
from __future__ import annotations

import argparse
import io
import tarfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy import stats
from tabulate import tabulate

from posterioralpha.polymarket.fetch import fetch_market_resolution
from posterioralpha.polymarket.smartflow_independence import (
    classify_independence,
    independence_features,
)

SNAPSHOT_DIR = _bootstrap.ROOT / "data" / "polymarket_smart_money_snapshot"
CACHE_TARBALL = SNAPSHOT_DIR / "raw_cache_2026-06-23.tar.gz"
EVENTS_CSV = SNAPSHOT_DIR / "smartflow_indep_backtest_events.csv"
RESOLUTION_CACHE_CSV = SNAPSHOT_DIR / "smartflow_indep_resolution_cache.csv"

_TRADE_COLS = ["timestamp", "conditionId", "asset", "side", "outcomeIndex",
               "price", "size", "usdcSize", "title", "slug"]


# ---------------------------------------------------------------------------
# 1. Load the frozen cache
# ---------------------------------------------------------------------------

def load_cached_trades(tarball: Path = CACHE_TARBALL) -> dict[str, pd.DataFrame]:
    """Read every ``polymarket/traders/trades_0x*.csv`` member straight out of
    the tarball (no extraction to disk) into ``{wallet: DataFrame}``, shaped
    like ``fetch_trader_trades`` output (timestamps parsed, ids kept as str)."""
    trades_by_wallet: dict[str, pd.DataFrame] = {}
    with tarfile.open(tarball, "r:gz") as tf:
        for member in tf.getmembers():
            if not member.name.startswith("polymarket/traders/trades_0x") or not member.name.endswith(".csv"):
                continue
            wallet = member.name.rsplit("trades_", 1)[1][: -len(".csv")]
            fh = tf.extractfile(member)
            if fh is None:
                continue
            df = pd.read_csv(
                io.BytesIO(fh.read()),
                parse_dates=["timestamp"],
                dtype={"asset": str, "conditionId": str, "slug": str, "title": str, "side": str},
            )
            if df.empty:
                continue
            for c in _TRADE_COLS:
                if c not in df.columns:
                    df[c] = ""
            trades_by_wallet[wallet] = df[_TRADE_COLS].sort_values("timestamp").reset_index(drop=True)
    return trades_by_wallet


# ---------------------------------------------------------------------------
# 2. Replay consensus events, day by day
# ---------------------------------------------------------------------------

def replay_consensus_events(
    trades_by_wallet: dict[str, pd.DataFrame],
    window_days: int = 7,
    min_buyers: int = 3,
    min_price: float = 0.05,
    max_price: float = 0.90,
) -> list[dict]:
    """Walk day by day and fire one consensus event per token, the first day
    >= ``min_buyers`` distinct *eligible* wallets have bought it in the
    trailing ``window_days``. Returns a list of event dicts (unsorted classification
    fields not yet attached -- see ``classify_events``)."""
    wallet_start = {w: df["timestamp"].min() for w, df in trades_by_wallet.items()}

    frames = []
    for w, df in trades_by_wallet.items():
        d = df[["timestamp", "asset", "side", "price", "conditionId", "title"]].copy()
        d["wallet"] = w
        frames.append(d)
    all_trades = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    ts_values = all_trades["timestamp"].values

    sorted_starts = sorted(wallet_start.values())
    if len(sorted_starts) < min_buyers:
        return []
    # Earliest day at least `min_buyers` wallets are eligible (their cached
    # history starts before day_t - window_days).
    start_day = (sorted_starts[min_buyers - 1] + timedelta(days=window_days)).normalize() + timedelta(days=1)
    end_day = all_trades["timestamp"].max().normalize()
    days = pd.date_range(start_day, end_day, freq="D")

    events: list[dict] = []
    fired: set[str] = set()
    for d in days:
        t_end = d + timedelta(days=1) - timedelta(microseconds=1)  # inclusive of the whole day
        window_start = t_end - timedelta(days=window_days)
        eligible = {w for w, s in wallet_start.items() if s < window_start}
        if len(eligible) < min_buyers:
            continue
        lo = np.searchsorted(ts_values, np.datetime64(window_start), side="left")
        hi = np.searchsorted(ts_values, np.datetime64(t_end), side="right")
        if hi <= lo:
            continue
        window_df = all_trades.iloc[lo:hi]
        buys = window_df[(window_df["side"] == "BUY") & (window_df["wallet"].isin(eligible))]
        if buys.empty:
            continue
        counts = buys.groupby("asset")["wallet"].nunique()
        crossing = counts[counts >= min_buyers].index
        for tok in crossing:
            if tok in fired:
                continue
            fired.add(tok)
            tok_buys = buys[buys["asset"] == tok].sort_values("timestamp")
            first_per_wallet = tok_buys.groupby("wallet").first()
            # entry price = fill price of the chronologically LAST first-buyer
            # among the buyer set at crossing (closest offline proxy to
            # "enters when the Nth buyer completes consensus" -- no historical
            # order book here, so this is a fill-price proxy, not an ask).
            entry_row = first_per_wallet.loc[first_per_wallet["timestamp"].idxmax()]
            entry_price = float(entry_row["price"])
            buyer_set = set(first_per_wallet.index)
            events.append({
                "token": tok,
                "condition_id": tok_buys.iloc[0]["conditionId"],
                "question": tok_buys.iloc[0]["title"],
                "event_date": d.date().isoformat(),
                "asof": t_end,
                "n_buyers": len(buyer_set),
                "buyers": buyer_set,
                "entry_price": round(entry_price, 4),
                "in_price_band": bool(min_price <= entry_price <= max_price),
            })
    return events


# ---------------------------------------------------------------------------
# 3. Classify each event with the SHIPPED classifier (verbatim, no look-ahead)
# ---------------------------------------------------------------------------

def classify_events(events: list[dict], trades_by_wallet: dict[str, pd.DataFrame],
                     window_days: int = 7) -> None:
    """Attach independence_features + classify_independence to each event dict
    in place. Each wallet's frame is filtered to `timestamp <= asof` before the
    call, so the classifier never sees fills that happened after the event
    fired (no look-ahead)."""
    for e in events:
        asof = e["asof"]
        buyers = e["buyers"]
        filtered = {w: trades_by_wallet[w][trades_by_wallet[w]["timestamp"] <= asof] for w in buyers}
        feats = independence_features(e["token"], buyers, filtered, window_days, asof=asof)
        score, cls = classify_independence(feats)
        e.update(feats)
        e["indep_score"] = score
        e["indep_class"] = cls
        e["temporal_pass"] = feats["first_buy_span_hours"] >= 24.0
        e["price_pass"] = feats["price_chase"] <= 0.05
        e["comovement_pass"] = feats["pairwise_jaccard"] <= 0.20


# ---------------------------------------------------------------------------
# 4. Resolve outcomes, on-disk cached
# ---------------------------------------------------------------------------

def load_resolution_cache(path: Path = RESOLUTION_CACHE_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["condition_id", "token", "closed", "outcome"])
    return pd.read_csv(path, dtype={"condition_id": str, "token": str})


def save_resolution_cache(df: pd.DataFrame, path: Path = RESOLUTION_CACHE_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_events(
    events: list[dict],
    sleep: float = 0.2,
    timeout_s: float = 10.0,
    refresh: bool = False,
    cache_path: Path = RESOLUTION_CACHE_CSV,
) -> tuple[int, int]:
    """Fetch (condition_id, token) -> outcome for every IN-BAND event, caching
    to `cache_path` so re-runs are cheap. Attaches `outcome` / `pnl` /
    `resolution_status` to each event dict in place.

    Returns (n_resolved, n_unresolved_or_failed) among in-band events attempted.
    """
    cache = load_resolution_cache(cache_path)
    cache_idx = {(r.condition_id, r.token): r for r in cache.itertuples(index=False)} if not cache.empty else {}

    n_resolved = 0
    n_unresolved = 0
    new_rows: list[dict] = []
    for e in events:
        if not e["in_price_band"]:
            e["outcome"] = ""
            e["pnl"] = ""
            e["resolution_status"] = "out_of_band"
            continue

        key = (e["condition_id"], e["token"])
        cached = cache_idx.get(key)
        if cached is not None and not refresh and bool(cached.closed):
            outcome = cached.outcome
        else:
            try:
                res = fetch_market_resolution(e["condition_id"])
            except Exception:
                res = None
            outcome = None
            closed = bool(res and res.get("closed"))
            if closed:
                toks = res["tokens"]
                finalised = any(t["winner"] or t["price"] >= 0.99 for t in toks.values())
                t = toks.get(str(e["token"]))
                if finalised and t is not None:
                    outcome = 1.0 if (t["winner"] or t["price"] >= 0.99) else 0.0
            new_rows.append({
                "condition_id": e["condition_id"], "token": e["token"],
                "closed": closed, "outcome": outcome if outcome is not None else "",
            })
            if sleep:
                time.sleep(sleep)

        if outcome is None or outcome == "":
            e["outcome"] = ""
            e["pnl"] = ""
            e["resolution_status"] = "unresolved"
            n_unresolved += 1
        else:
            outcome = float(outcome)
            e["outcome"] = outcome
            e["pnl"] = round(outcome / e["entry_price"] - 1.0, 4)
            e["resolution_status"] = "resolved"
            n_resolved += 1

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        if not cache.empty:
            cache = pd.concat([cache, new_df], ignore_index=True)
            cache = cache.drop_duplicates(subset=["condition_id", "token"], keep="last")
        else:
            cache = new_df
        save_resolution_cache(cache, cache_path)

    return n_resolved, n_unresolved


# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------

def _class_stats(pnls: pd.Series) -> dict:
    n = len(pnls)
    return {
        "n": n,
        "win_rate": float((pnls > 0).mean()) if n else float("nan"),
        "mean": float(pnls.mean()) if n else float("nan"),
        "std": float(pnls.std(ddof=1)) if n > 1 else float("nan"),
    }


def build_report(df: pd.DataFrame) -> str:
    lines = []
    resolved = df[df["resolution_status"] == "resolved"].copy()
    resolved["pnl"] = pd.to_numeric(resolved["pnl"], errors="coerce")

    lines.append(f"Total events fired: {len(df)}")
    lines.append(f"  in price band [0.05, 0.90]: {int(df['in_price_band'].sum())}")
    lines.append(f"  out of band (excluded from PnL): {int((~df['in_price_band']).sum())}")
    lines.append(f"  resolved (in-band): {len(resolved)}")
    lines.append(f"  unresolved/unfetchable (in-band, attrition): "
                  f"{int((df['resolution_status'] == 'unresolved').sum())}")

    lines.append("")
    lines.append("Per indep_class (resolved events only):")
    rows = []
    for cls, g in resolved.groupby("indep_class"):
        s = _class_stats(g["pnl"])
        rows.append([cls, s["n"], f"{s['win_rate']*100:.1f}%" if s["n"] else "--",
                     f"{s['mean']:+.4f}" if s["n"] else "--",
                     f"{s['std']:.4f}" if s["n"] > 1 else "--"])
    lines.append(tabulate(rows, headers=["class", "n resolved", "win rate", "mean pnl", "std pnl"],
                           tablefmt="rounded_grid"))

    indep_pnl = resolved.loc[resolved["indep_class"] == "independent", "pnl"].dropna()
    casc_pnl = resolved.loc[resolved["indep_class"] == "cascade", "pnl"].dropna()
    lines.append("")
    if len(indep_pnl) >= 2 and len(casc_pnl) >= 2:
        t_stat, p_val = stats.ttest_ind(indep_pnl, casc_pnl, equal_var=False)
        lines.append(f"Welch t-stat (independent - cascade), two-sided: t={t_stat:+.3f}  p={p_val:.3f}")
    else:
        lines.append("Welch t-stat: not enough resolved events in one or both classes (need >= 2 each).")

    lines.append("")
    lines.append("Per-component diagnostics (resolved events, pass vs fail, regardless of class):")
    rows = []
    for comp, label in [("temporal_pass", "temporal (span>=24h)"),
                        ("price_pass", "price (chase<=0.05)"),
                        ("comovement_pass", "co-movement (jaccard<=0.20)")]:
        for passed in (True, False):
            g = resolved[resolved[comp] == passed]
            s = _class_stats(g["pnl"])
            rows.append([label, "pass" if passed else "fail", s["n"],
                         f"{s['mean']:+.4f}" if s["n"] else "--",
                         f"{s['std']:.4f}" if s["n"] > 1 else "--"])
    lines.append(tabulate(rows, headers=["component", "result", "n", "mean pnl", "std pnl"],
                           tablefmt="rounded_grid"))

    lines.append("")
    lines.append("indep_score distribution (all fired events, in-band + out-of-band):")
    score_counts = df["indep_score"].value_counts().sort_index()
    lines.append(tabulate([[s, n] for s, n in score_counts.items()],
                           headers=["indep_score", "n events"], tablefmt="rounded_grid"))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=CACHE_TARBALL, help="frozen wallet-fill cache tarball")
    ap.add_argument("--window-days", type=int, default=7, help="consensus + independence trailing window")
    ap.add_argument("--min-buyers", type=int, default=3, help="distinct eligible buyers required to fire")
    ap.add_argument("--min-price", type=float, default=0.05, help="entry price band floor")
    ap.add_argument("--max-price", type=float, default=0.90, help="entry price band ceiling")
    ap.add_argument("--sleep", type=float, default=0.2, help="delay between resolution API calls")
    ap.add_argument("--refresh-resolution", action="store_true", help="refetch even cached resolutions")
    ap.add_argument("--events-csv", type=Path, default=EVENTS_CSV)
    ap.add_argument("--resolution-cache", type=Path, default=RESOLUTION_CACHE_CSV)
    args = ap.parse_args()

    print("""
╔════════════════════════════════════════════════════════════════════╗
║  SMART-FLOW INDEPENDENCE CLASSIFIER  ·  retrospective replay          ║
║  offline replay over the frozen 2026-06-23 wallet-fill cache          ║
║  EXPLORATORY ONLY -- the pre-registered forward ledger is the         ║
║  deciding test; frozen thresholds are NOT retuned here.               ║
╚════════════════════════════════════════════════════════════════════╝""")

    t0 = time.time()
    trades_by_wallet = load_cached_trades(args.cache)
    print(f"  loaded {len(trades_by_wallet)} wallet histories "
          f"({sum(len(d) for d in trades_by_wallet.values())} fills) in {time.time()-t0:.1f}s")

    t0 = time.time()
    events = replay_consensus_events(trades_by_wallet, args.window_days, args.min_buyers,
                                      args.min_price, args.max_price)
    print(f"  replay fired {len(events)} consensus events in {time.time()-t0:.1f}s")

    classify_events(events, trades_by_wallet, args.window_days)
    n_indep = sum(1 for e in events if e["indep_class"] == "independent")
    n_casc = sum(1 for e in events if e["indep_class"] == "cascade")
    print(f"  classified: {n_indep} independent, {n_casc} cascade "
          f"({sum(1 for e in events if e['in_price_band'])} in price band)")

    t0 = time.time()
    n_resolved, n_unresolved = resolve_events(events, sleep=args.sleep, refresh=args.refresh_resolution,
                                               cache_path=args.resolution_cache)
    print(f"  resolved {n_resolved} / unresolved-or-unfetchable {n_unresolved} "
          f"in-band events in {time.time()-t0:.1f}s")

    df = pd.DataFrame([{k: v for k, v in e.items() if k not in ("buyers", "asof")} for e in events])
    df = df.sort_values("event_date").reset_index(drop=True)
    args.events_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.events_csv, index=False)
    print(f"\n  events table -> {args.events_csv}  ({len(df)} rows)")

    print(f"\n{'='*78}")
    print("  REPORT")
    print(f"{'='*78}")
    if df.empty:
        print("  No events fired -- nothing to report.")
    else:
        print(build_report(df))

    print("\n  Caveats (mandatory reading, see also SMART_FLOW_INDEPENDENCE.md):")
    print("  (a) wallet pool = 2026-06-23 snapshot -> look-ahead/survivorship,")
    print("      inflates ABSOLUTE edge for both classes; the between-class")
    print("      separation is the meaningful number, and it shares the bias.")
    print("  (b) entries use fill price as an ask proxy -- no historical order")
    print("      book, so no bid-ask spread is captured (live entries pay the ask).")
    print("  (c) per-wallet history truncation limits breadth further back in time.")
    print("  (d) exploratory evidence only -- the pre-registered forward ledger")
    print("      remains the deciding test. Frozen thresholds were NOT retuned here.")
    print("\n✓  Replay complete.")


if __name__ == "__main__":
    main()
