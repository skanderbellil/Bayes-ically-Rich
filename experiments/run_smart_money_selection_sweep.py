#!/usr/bin/env python3
"""
Smart-money SELECTION-CRITERIA sweep — is it the *selector* that's broken?
=========================================================================

The first wallet test (``run_smart_money_wallet_test.py``) found that ranking the
pool by total point-in-time PnL gives no edge — ``smart_money`` ≤ ``anti_smart``.
But that used *one* selector. The open question this script answers: **is there
ANY selection criterion under which the wallet idea holds?**

It sweeps every selection knob, all on the warm cache (no re-fetch), and for each
config reports the falsification spreads:

  spread_anti = net_Sharpe(smart) − net_Sharpe(anti)   > 0 ⇒ winners beat losers
  spread_all  = net_Sharpe(smart) − net_Sharpe(all)    > 0 ⇒ selection beats no-selection

Aspects swept
-------------
  pool       : which profit-leaderboard windows seed the candidate pool
               {7d} {30d} {all} {7d,30d} {7d,30d,all}
  metric     : point-in-time ranking score (all causal, computed < t)
               pnl_total   — marked PnL to date            (the original)
               pnl_30d     — marked PnL change, trailing 30d (recent form)
               pnl_90d     — marked PnL change, trailing 90d
               roi         — marked PnL / gross $ traded     (efficiency)
               sharpe      — expanding mean/std of daily equity change
  n_follow   : cohort size {3, 5, 10}
  rebalance  : re-selection cadence in days {7, 14, 30}
  screen     : market-maker screen on / off

The weight construction, costs, marking and point-in-time discipline are
identical to ``run_smart_money_follow`` — only the *ranking* and *eligibility*
change, so any difference is attributable to the selection criterion alone.

Artifacts (git-tracked, ``data/polymarket_smart_money_snapshot/``):
  selection_sweep.csv      — every config × {smart,anti,all} metrics + spreads
  selection_sweep_meta.json

Usage
-----
  python experiments/run_smart_money_selection_sweep.py
  python experiments/run_smart_money_selection_sweep.py --quick   # smaller grid
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.polymarket import (
    MMScreen,
    build_token_panel,
    candidate_universe,
    fetch_leaderboard,
    fetch_many_traders,
    fetch_volume_wallets,
    mm_fingerprint,
    select_universe,
)
from posterioralpha.polymarket.paths import REPO_ROOT
from posterioralpha.polymarket.smartmoney import (
    _EPS,
    trader_cash_panel,
    trader_equity_curve,
    trader_position_panel,
)
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_DIR = REPO_ROOT / "data" / "polymarket_smart_money_snapshot"

POOLS = {
    "7d": ("7d",), "30d": ("30d",), "all": ("all",),
    "7d+30d": ("7d", "30d"), "7d+30d+all": ("7d", "30d", "all"),
}
METRICS = ["pnl_total", "pnl_30d", "pnl_90d", "roi", "sharpe"]


# ---------------------------------------------------------------------------
# Point-in-time skill panels (dates × wallet), all causal
# ---------------------------------------------------------------------------

def gross_traded_panel(trades: pd.DataFrame, tokens: list[str],
                       dates: pd.DatetimeIndex) -> pd.Series:
    """Cumulative gross $ traded per wallet on universe tokens (for ROI)."""
    t = trades[trades["asset"].isin(tokens)].copy()
    if t.empty:
        return pd.Series(0.0, index=dates)
    t["day"] = t["timestamp"].dt.normalize()
    daily = t.groupby("day")["usdcSize"].sum()
    return daily.reindex(dates, fill_value=0.0).cumsum()


def build_skill_panels(equity_df: pd.DataFrame, gross_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All ranking scores as causal dates × wallet panels."""
    dret = equity_df.diff()
    return {
        "pnl_total": equity_df,
        "pnl_30d":   equity_df - equity_df.shift(30),
        "pnl_90d":   equity_df - equity_df.shift(90),
        "roi":       equity_df / gross_df.replace(0.0, np.nan),
        "sharpe":    dret.expanding(min_periods=5).mean()
                     / (dret.expanding(min_periods=5).std() + _EPS),
    }


# ---------------------------------------------------------------------------
# Generic walk-forward follow with a pluggable ranker
# ---------------------------------------------------------------------------

def follow(
    positions: dict[str, pd.DataFrame],
    panel: pd.DataFrame,
    skill: pd.DataFrame,
    active: pd.DataFrame,           # dates × wallet bool: has taken real positions by t
    eligible: list[str],
    mode: str,                      # "top" | "bottom" | "all"
    n_follow: int,
    rebalance: int,
    burn_in: int,
    cost: float,
    min_active: int = 3,
    gross: float = 1.0,
) -> pd.Series:
    """Net daily returns of one config. Identical mechanics to run_smart_money_follow,
    only the ranking (skill) and direction (mode) vary."""
    dates, tokens = panel.index, list(panel.columns)
    first_sel = dates[min(burn_in, len(dates) - 1)]
    prev_w = pd.Series(0.0, index=tokens)
    cohort: list[str] = []
    rets = []

    for i, t in enumerate(dates):
        if t >= first_sel and (i - burn_in) % rebalance == 0:
            elig_active = [w for w in eligible if active.at[t, w]]
            s = skill.loc[t, elig_active].dropna()
            if len(s) < min_active:
                cohort = []
            else:
                ranked = s.sort_values(ascending=False)
                if mode == "all":
                    cohort = list(ranked.index)
                elif mode == "bottom":
                    cohort = list(ranked.index[-n_follow:])
                else:
                    cohort = list(ranked.index[:n_follow])

        if cohort:
            agg = pd.Series(0.0, index=tokens)
            for w in cohort:
                agg = agg.add(positions[w].loc[t], fill_value=0.0)
            dollar = agg * panel.loc[t].reindex(tokens).fillna(0.0)
            gsum = float(dollar.abs().sum())
            w_t = dollar / gsum * gross if gsum > _EPS else pd.Series(0.0, index=tokens)
        else:
            w_t = pd.Series(0.0, index=tokens)

        to = float((w_t - prev_w).abs().sum())
        if i + 1 < len(dates):
            dp = (panel.iloc[i + 1] - panel.iloc[i]).reindex(tokens).fillna(0.0)
            pnl = float((w_t * dp).sum())
        else:
            pnl = 0.0
        rets.append(pnl - cost * to)
        prev_w = w_t

    return pd.Series(rets, index=dates)


def net_metrics(r: pd.Series) -> tuple[float, float]:
    sharpe = compute_metrics(r, rf=0.0).get("Sharpe", 0.0)
    pnl = float(r.fillna(0.0).cumsum().iloc[-1])
    return round(sharpe, 4), round(pnl, 4)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-window", type=int, default=60)
    ap.add_argument("--top-tokens", type=int, default=150)
    ap.add_argument("--max-trades", type=int, default=2000)
    ap.add_argument("--burn-in",    type=int, default=45)
    ap.add_argument("--cost",       type=float, default=0.005)
    ap.add_argument("--quick",      action="store_true", help="small grid (n_follow=5, rebalance=7)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_follows  = [5] if args.quick else [3, 5, 10]
    rebalances = [7] if args.quick else [7, 14, 30]

    # ── DATA (cached) ─────────────────────────────────────────────────────────
    full_pool = candidate_universe(profit_windows=("7d", "30d", "all"), per_window=args.per_window)
    traders = fetch_many_traders(full_pool["wallet"].tolist(), max_trades=args.max_trades)
    if not traders:
        raise SystemExit("No trader histories — check network / cache.")
    universe = select_universe(traders, top_k=args.top_tokens)
    panel = build_token_panel(universe, use_cache=True)
    tokens = [t for t in universe if t in panel.columns]
    panel = panel[tokens]
    dates = panel.index

    # ── per-wallet causal panels (built once) ─────────────────────────────────
    logger.info("Building per-wallet position / cash / equity / gross panels …")
    positions, equity, gross = {}, {}, {}
    for w, t in traders.items():
        pos = trader_position_panel(t, tokens, dates)
        cash = trader_cash_panel(t, tokens, dates)
        positions[w] = pos
        equity[w] = trader_equity_curve(pos, cash, panel)
        gross[w] = gross_traded_panel(t, tokens, dates)
    equity_df = pd.DataFrame(equity)
    gross_df  = pd.DataFrame(gross)
    active    = equity_df.abs() > _EPS                      # has taken real positions by t
    skills    = build_skill_panels(equity_df, gross_df)

    # ── MM screen ─────────────────────────────────────────────────────────────
    mm_watch = fetch_volume_wallets("all", limit=150)
    is_mm = {}
    for w, t in traders.items():
        fp = mm_fingerprint(t, panel, asof=None, screen=MMScreen())
        is_mm[w] = bool(fp["is_mm"] or (w in mm_watch and fp["n_trades"] >= 200
                                        and fp["edge_per_dollar"] <= 0.01))
    n_mm = sum(is_mm.values())

    # ── pool-subset membership (which wallets each window combo seeds) ─────────
    pool_members = {}
    for name, windows in POOLS.items():
        cu = candidate_universe(profit_windows=windows, per_window=args.per_window)
        pool_members[name] = [w for w in cu["wallet"].tolist() if w in traders]

    # ── SWEEP ─────────────────────────────────────────────────────────────────
    rows = []
    grid = list(itertools.product(POOLS, METRICS, n_follows, rebalances, ["on", "off"]))
    logger.info("Sweeping %d configs …", len(grid))
    for k, (pool_name, metric, nf, rb, screen) in enumerate(grid, 1):
        members = pool_members[pool_name]
        eligible = [w for w in members if not (screen == "on" and is_mm.get(w, False))]
        if len(eligible) < 6:                              # need room for top-N and bottom-N
            continue
        sk = skills[metric]
        sm_s, sm_p = net_metrics(follow(positions, panel, sk, active, eligible, "top",
                                        nf, rb, args.burn_in, args.cost))
        an_s, an_p = net_metrics(follow(positions, panel, sk, active, eligible, "bottom",
                                        nf, rb, args.burn_in, args.cost))
        al_s, al_p = net_metrics(follow(positions, panel, sk, active, eligible, "all",
                                        nf, rb, args.burn_in, args.cost))
        rows.append({
            "pool": pool_name, "metric": metric, "n_follow": nf, "rebalance": rb,
            "screen": screen, "n_elig": len(eligible),
            "smart_sharpe": sm_s, "anti_sharpe": an_s, "all_sharpe": al_s,
            "smart_pnl": sm_p, "anti_pnl": an_p, "all_pnl": al_p,
            "spread_anti": round(sm_s - an_s, 4), "spread_all": round(sm_s - al_s, 4),
        })
        if k % 30 == 0:
            logger.info("  %d/%d", k, len(grid))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "selection_sweep.csv", index=False)
    (OUT_DIR / "selection_sweep_meta.json").write_text(json.dumps({
        "run_date": pd.Timestamp.utcnow().isoformat(),
        "params": vars(args), "n_configs": len(df),
        "pool_wallets": len(traders), "mm_flagged": n_mm,
        "universe_tokens": len(tokens), "panel_days": len(dates),
        "panel_start": str(dates.min().date()), "panel_end": str(dates.max().date()),
        "pools": {k: len(v) for k, v in pool_members.items()},
    }, indent=2))

    # ── VERDICT ───────────────────────────────────────────────────────────────
    n = len(df)
    holds = df[(df["spread_anti"] > 0) & (df["spread_all"] > 0)]
    print("\n" + "═" * 92)
    print(f"  SELECTION-CRITERIA SWEEP  ·  {n} configs  ·  universe={len(tokens)} tokens, "
          f"{len(dates)} days, pool={len(traders)} ({n_mm} MM)")
    print("═" * 92)
    print(f"  Configs where the wallet idea HOLDS (smart > anti AND smart > all): "
          f"{len(holds)}/{n} ({len(holds)/max(n,1):.0%})")
    print(f"  Mean spread_anti = {df['spread_anti'].mean():+.3f}  "
          f"(smart − anti Sharpe; >0 ⇒ winners beat losers)")
    print(f"  Mean spread_all  = {df['spread_all'].mean():+.3f}  "
          f"(smart − all-leaders Sharpe; >0 ⇒ selection adds value)")

    print("\n  Best 12 configs by spread_anti:")
    best = df.sort_values("spread_anti", ascending=False).head(12)
    print(tabulate(best[["pool", "metric", "n_follow", "rebalance", "screen",
                         "smart_sharpe", "anti_sharpe", "all_sharpe",
                         "spread_anti", "spread_all"]],
                   headers="keys", tablefmt="rounded_grid", showindex=False))

    print("\n  By metric (mean across pool/size/cadence/screen):")
    by_m = df.groupby("metric")[["smart_sharpe", "anti_sharpe", "all_sharpe",
                                 "spread_anti", "spread_all"]].mean().round(3)
    print(tabulate(by_m.reset_index(), headers="keys", tablefmt="rounded_grid", showindex=False))

    print("\n  By pool window combo (mean):")
    by_p = df.groupby("pool")[["smart_sharpe", "spread_anti", "spread_all"]].mean().round(3)
    print(tabulate(by_p.reset_index(), headers="keys", tablefmt="rounded_grid", showindex=False))

    print(f"\n  Artifacts → {OUT_DIR.relative_to(REPO_ROOT)}/selection_sweep.csv")
    print("✓  Selection sweep complete.")


if __name__ == "__main__":
    main()
