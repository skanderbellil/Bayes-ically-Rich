#!/usr/bin/env python3
"""
Polymarket — dip-confirm entry strategy backtest
=================================================

Core idea: instead of entering at H days out, WATCH the market at H days out
and only enter at check_day if the market showed a dip-then-recover pattern
in between.  The dip-recover group shows 67–89%+ win rate vs 48% base.

Three strategies compared at every horizon / threshold combo:

  flat_H      : $1 at H days out, hold all markets to resolution (baseline)
  flat_conf   : $1 at check_day for ALL markets (isolates timing effect)
  dip_confirm : $1 at check_day ONLY for markets that showed dip→recover
                (combines timing + selection filter)

Dip-confirm logic:
  1. Note p0 at day 0 (H days out, must be in band)
  2. Between days 1 and check_day-1 the price must drop ≥ dip_thresh below p0
  3. At check_day the price must have recovered ≥ recovery_floor × p0
     (default recovery_floor=0.95 — doesn't need full recovery, just most of it)
  4. If triggered: enter at path[check_day] + haircut, hold to resolution

Multiple horizons: [5, 7, 10, 14, 21] days
Multiple dip thresholds: [0.05, 0.10, 0.15, 0.20]
check_day fixed at 2 (observe dip day 1, enter on day 2 recovery)

Usage
-----
  python experiments/run_dip_confirm_backtest.py
  python experiments/run_dip_confirm_backtest.py --n-markets 1000 --check-day 2
  python experiments/run_dip_confirm_backtest.py --check-day 3
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
import scipy.stats

from posterioralpha.polymarket.fetch import fetch_markets, fetch_token_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

HORIZONS       = [5, 7, 10, 14, 21]
DIP_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def build_paths(markets, H, lo, hi, sleep):
    paths = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        s = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(s) < 3:
            continue
        t_res = s.index.max()
        t_entry = t_res - pd.Timedelta(days=H)
        grid = pd.date_range(t_entry, t_res, freq="D")
        sp = s.reindex(s.index.union(grid)).sort_index().ffill().reindex(grid).dropna()
        if len(sp) < H + 1:
            continue
        p0 = float(sp.iloc[0])
        if not (lo <= p0 < hi):
            continue
        paths.append({
            "path": sp.values.astype(float),
            "outcome": float(m.outcome),
            "entry_price": p0,
        })
        if i % 50 == 0:
            logger.info("  H=%dd: built %d/%d (%d in band)", H, i, len(markets), len(paths))
        if sleep and i % 5 == 0:
            time.sleep(sleep)
    return paths


def classify(path, p0, check_day, dip_thresh, recovery_floor=0.95):
    """Return (label, conf_price).
    label: 'dip_recover' | 'dip_persist' | 'steady'
    conf_price: price at check_day (entry price if we enter)
    """
    if len(path) <= check_day:
        return "short", None
    # did it dip between day 1 and check_day-1?
    intra = path[1:check_day]
    dipped = any(px < p0 * (1.0 - dip_thresh) for px in intra)
    conf_px = path[check_day]
    recovered = conf_px >= p0 * recovery_floor
    if not dipped:
        return "steady", conf_px
    elif dipped and recovered:
        return "dip_recover", conf_px
    else:
        return "dip_persist", conf_px


def strat_stats(rets):
    rets = np.array(rets)
    if len(rets) == 0:
        return {"n": 0, "win_rate": float("nan"), "mean_ret": float("nan"),
                "sharpe": float("nan"), "tot_pnl": 0.0, "ci_lo": float("nan"), "ci_hi": float("nan")}
    wr = (rets > 0).mean()
    mu = rets.mean()
    sd = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    sh = mu / sd if (sd and sd > 0) else float("nan")
    k = int((rets > 0).sum())
    ci_lo = scipy.stats.beta.ppf(0.025, k + 1, len(rets) - k + 1)
    ci_hi = scipy.stats.beta.ppf(0.975, k + 1, len(rets) - k + 1)
    return {"n": len(rets), "win_rate": round(float(wr), 3),
            "mean_ret": round(float(mu), 4), "sharpe": round(float(sh), 3),
            "tot_pnl": round(float(rets.sum()), 2),
            "ci_lo": round(float(ci_lo), 3), "ci_hi": round(float(ci_hi), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=1000)
    ap.add_argument("--min-volume", type=float, default=20_000.0)
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 0.50])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--check-day", type=int, default=2,
                    help="day to check for recovery (dip observed days 1..check_day-1)")
    ap.add_argument("--recovery-floor", type=float, default=0.95,
                    help="price must recover to ≥ floor×p0 by check_day")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--out", type=str, default="data/paper_trade/dip_confirm_backtest.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/dip_confirm_backtest.png")
    args = ap.parse_args()

    lo, hi = args.band
    CD = args.check_day

    print("""
╔══════════════════════════════════════════════════════════╗
║  DIP-CONFIRM ENTRY STRATEGY BACKTEST                      ║
║  watch at H days, enter only after dip→recover            ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  band [{lo:.2f},{hi:.2f})  ·  check_day={CD}  ·  recovery_floor={args.recovery_floor:.0%}")
    print(f"  horizons: {HORIZONS}  ·  dip thresholds: {[f'{t:.0%}' for t in DIP_THRESHOLDS]}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets\n")

    rows = []
    paths_cache = {}   # H → paths list

    for H in HORIZONS:
        logger.info("Building paths for H=%dd ...", H)
        paths = build_paths(markets, H, lo, hi, args.sleep)
        paths_cache[H] = paths
        if not paths:
            logger.warning("  No paths for H=%d", H)
            continue

        n_all = len(paths)
        base_wr = sum(p["outcome"] for p in paths) / n_all

        # flat_H baseline
        flat_H_rets = []
        for p in paths:
            entry = min(p["entry_price"] + args.haircut, 0.99)
            flat_H_rets.append(p["outcome"] / entry - 1.0)
        flat_H_stats = strat_stats(flat_H_rets)

        for thresh in DIP_THRESHOLDS:
            # classify each market
            dc_rets, fc_rets, labels = [], [], []
            for p in paths:
                label, conf_px = classify(p["path"], p["entry_price"], CD, thresh, args.recovery_floor)
                labels.append(label)
                entry_H = min(p["entry_price"] + args.haircut, 0.99)
                # flat_conf: enter all at check_day price
                if conf_px is not None:
                    entry_c = min(conf_px + args.haircut, 0.99)
                    fc_rets.append(p["outcome"] / entry_c - 1.0)
                # dip_confirm: enter only if dip_recover
                if label == "dip_recover" and conf_px is not None:
                    entry_c = min(conf_px + args.haircut, 0.99)
                    dc_rets.append(p["outcome"] / entry_c - 1.0)

            labels = np.array(labels)
            n_dr = int((labels == "dip_recover").sum())
            n_dp = int((labels == "dip_persist").sum())
            n_st = int((labels == "steady").sum())

            dc_stats = strat_stats(dc_rets)
            fc_stats = strat_stats(fc_rets)

            row = {
                "H": H, "thresh": thresh, "n_all": n_all,
                "base_wr": round(base_wr, 3),
                "n_dip_recover": n_dr, "n_dip_persist": n_dp, "n_steady": n_st,
                # flat_H
                "flat_H_wr": flat_H_stats["win_rate"], "flat_H_sh": flat_H_stats["sharpe"],
                "flat_H_pnl": flat_H_stats["tot_pnl"], "flat_H_mean": flat_H_stats["mean_ret"],
                # flat_conf
                "flat_c_wr": fc_stats["win_rate"], "flat_c_sh": fc_stats["sharpe"],
                "flat_c_pnl": fc_stats["tot_pnl"],
                # dip_confirm
                "dc_n": dc_stats["n"], "dc_wr": dc_stats["win_rate"],
                "dc_sh": dc_stats["sharpe"], "dc_pnl": dc_stats["tot_pnl"],
                "dc_mean": dc_stats["mean_ret"],
                "dc_ci_lo": dc_stats["ci_lo"], "dc_ci_hi": dc_stats["ci_hi"],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # ── Print results ─────────────────────────────────────────────────────────
    for thresh in DIP_THRESHOLDS:
        sub = df[df["thresh"] == thresh]
        print(f"\n  ══ dip threshold: {thresh:.0%} ════════════════════════════════════════════")
        print(f"  {'H':>4} | {'n_all':>6} | {'base%':>6} | {'n_DR':>5} | {'DR win%':>8} | "
              f"{'CI95':>13} | {'DC mean':>8} | {'DC Sh':>7} | {'flat Sh':>7} | {'Δ Sh':>6}")
        print(f"  {'-'*4}-+-{'-'*6}-+-{'-'*6}-+-{'-'*5}-+-{'-'*8}-+-{'-'*13}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")
        for _, r in sub.iterrows():
            if r["dc_n"] < 3:
                ci_str = "too few"
            else:
                ci_str = f"{r['dc_ci_lo']:.0%}–{r['dc_ci_hi']:.0%}"
            dsh = r["dc_sh"] - r["flat_H_sh"] if not np.isnan(r["dc_sh"]) else float("nan")
            print(f"  {int(r['H']):>4} | {int(r['n_all']):>6} | {r['base_wr']:>5.0%} | "
                  f"{int(r['n_dip_recover']):>5} | {r['dc_wr']:>7.0%} | {ci_str:>13} | "
                  f"{r['dc_mean']:>+7.1%} | {r['dc_sh']:>7.2f} | {r['flat_H_sh']:>7.2f} | {dsh:>+5.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Dip-confirm entry strategy — band [{lo:.2f},{hi:.2f}), check_day={CD}",
                 fontsize=13, fontweight="bold")
    colors = ["#06b6d4", "#f59e0b", "#22c55e", "#a78bfa"]

    # Panel 1: DR win rate by horizon for each thresh
    ax = axes[0, 0]
    for thresh, c in zip(DIP_THRESHOLDS, colors):
        sub = df[df["thresh"] == thresh]
        ax.plot(sub["H"], sub["dc_wr"] * 100, "o-", color=c, lw=2,
                label=f"dip≥{thresh:.0%}")
    # base win rate by horizon
    base_by_H = df.groupby("H")["base_wr"].first()
    ax.plot(base_by_H.index, base_by_H.values * 100, "k--", lw=1.5, label="base win rate")
    ax.set_title("Dip-confirm win rate by horizon")
    ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Win rate (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(HORIZONS)

    # Panel 2: DC Sharpe vs flat-H Sharpe
    ax = axes[0, 1]
    flat_by_H = df.groupby("H")["flat_H_sh"].first()
    ax.plot(flat_by_H.index, flat_by_H.values, "k--", lw=2, label="flat hold")
    for thresh, c in zip(DIP_THRESHOLDS, colors):
        sub = df[(df["thresh"] == thresh) & (df["dc_n"] >= 5)]
        ax.plot(sub["H"], sub["dc_sh"], "o-", color=c, lw=2, label=f"dip≥{thresh:.0%}")
    ax.set_title("Sharpe: dip-confirm vs flat hold")
    ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Sharpe ratio")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(HORIZONS)

    # Panel 3: n_dip_recover (how many events) by horizon
    ax = axes[1, 0]
    for thresh, c in zip(DIP_THRESHOLDS, colors):
        sub = df[df["thresh"] == thresh]
        ax.plot(sub["H"], sub["n_dip_recover"], "o-", color=c, lw=2, label=f"dip≥{thresh:.0%}")
    ax.set_title("# dip-recover triggers by horizon")
    ax.set_xlabel("Horizon (days)"); ax.set_ylabel("# markets triggering")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(HORIZONS)

    # Panel 4: CI lo bound of DC win rate vs base win rate
    ax = axes[1, 1]
    for thresh, c in zip(DIP_THRESHOLDS, colors):
        sub = df[(df["thresh"] == thresh) & (df["dc_n"] >= 5)]
        ax.fill_between(sub["H"], sub["dc_ci_lo"] * 100, sub["dc_ci_hi"] * 100,
                        alpha=0.15, color=c)
        ax.plot(sub["H"], sub["dc_wr"] * 100, "o-", color=c, lw=2, label=f"dip≥{thresh:.0%}")
    ax.plot(base_by_H.index, base_by_H.values * 100, "k--", lw=1.5, label="base")
    ax.set_title("DC win rate with 95% CI")
    ax.set_xlabel("Horizon (days)"); ax.set_ylabel("Win rate (%)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xticks(HORIZONS)

    fig.tight_layout()
    fig.savefig(args.plot, dpi=130, bbox_inches="tight")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out}")
    print("✓  Dip-confirm backtest complete.")


if __name__ == "__main__":
    main()
