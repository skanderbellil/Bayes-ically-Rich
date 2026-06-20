#!/usr/bin/env python3
"""
Polymarket — position-sizing schemes for the long-YES band [0.10, 0.90]
=======================================================================

The bracket backtest showed the YES underpricing is largest *per dollar* at the
cheap end (win-rate − price widest at 0.10–0.50) but that's exactly where
variance is worst (median −100%, fat-tailed winners).  This tests how money
*weighting* across the full 0.10–0.90 band changes the risk/return:

  equal        : $1 on every market               (baseline)
  inverse      : stake ∝ 1/price                  (more on cheap longshots)
  sqrt_inverse : stake ∝ 1/sqrt(price)            (gentler inverse)
  proportional : stake ∝ price                    (more on favorites)

All weights are normalised so the *average* stake = $1 (mean(w)=1), so every
scheme deploys the same average capital per trade and total PnL is comparable.
Reports capital-weighted return, per-trade Sharpe, and max drawdown, and plots
time-ordered equity curves per scheme.

Usage
-----
  python experiments/run_yes_weighting_backtest.py --n-markets 250 --horizon 7 3 1
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BAND = (0.10, 0.90)
SCHEMES = ["equal", "inverse", "sqrt_inverse", "proportional"]


def weights(price: np.ndarray, scheme: str) -> np.ndarray:
    if scheme == "equal":
        w = np.ones_like(price)
    elif scheme == "inverse":
        w = 1.0 / price
    elif scheme == "sqrt_inverse":
        w = 1.0 / np.sqrt(price)
    elif scheme == "proportional":
        w = price.copy()
    else:
        raise ValueError(scheme)
    return w / w.mean()           # normalise so mean stake = 1


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def price_at(series: pd.Series, when: pd.Timestamp) -> float | None:
    s = series[series.index <= when]
    return float(s.iloc[-1]) if not s.empty else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=250)
    ap.add_argument("--min-volume", type=float, default=30_000.0)
    ap.add_argument("--horizon", type=int, nargs="+", default=[7, 3, 1])
    ap.add_argument("--haircut", type=float, default=0.02)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--out", type=str, default="data/paper_trade/yes_weighting.csv")
    ap.add_argument("--plot", type=str, default="data/paper_trade/yes_weighting.png")
    args = ap.parse_args()

    print("""
╔══════════════════════════════════════════════════════════╗
║  YES BAND [0.10,0.90] — POSITION-SIZING SCHEMES          ║
║  equal · inverse(1/p) · sqrt_inverse · proportional(p)   ║
╚══════════════════════════════════════════════════════════╝""")
    print(f"  horizons={args.horizon}d  haircut={args.haircut:.2f}  "
          f"markets≤{args.n_markets}  vol≥${args.min_volume:,.0f}\n")

    markets = fetch_markets(n_markets=args.n_markets, min_volume=args.min_volume, closed=True)
    markets = markets[markets["outcome"].isin([0.0, 1.0])].reset_index(drop=True)
    print(f"  {len(markets)} resolved markets\n")

    recs = []
    for i, m in enumerate(markets.itertuples(index=False), 1):
        prices = fetch_token_history(m.yes_token, fidelity_minutes=1440)
        if len(prices) < 3:
            continue
        t_res = prices.index.max()
        for d in args.horizon:
            p = price_at(prices, t_res - pd.Timedelta(days=d))
            if p is None or not (BAND[0] <= p < BAND[1]):
                continue
            recs.append({"horizon_d": d, "price": p, "outcome": float(m.outcome),
                         "t_res": t_res})
        if i % 50 == 0:
            logger.info("  priced %d/%d (%d obs)", i, len(markets), len(recs))
        time.sleep(args.sleep if i % 5 == 0 else 0)

    obs_all = pd.DataFrame(recs)
    if obs_all.empty:
        print("No observations — aborting.")
        return
    obs_all.to_csv(args.out, index=False)

    summary = []
    curves = {}   # (horizon, scheme) -> equity array
    for d in args.horizon:
        obs = obs_all[obs_all["horizon_d"] == d].sort_values("t_res").reset_index(drop=True)
        if len(obs) < 5:
            continue
        price = obs["price"].values
        entry = np.minimum(price + args.haircut, 0.99)
        ret   = obs["outcome"].values / entry - 1.0      # per $1 staked

        print(f"\n{'═'*84}")
        print(f"  HORIZON {d}d — band [0.10,0.90], {len(obs)} markets")
        print(f"{'═'*84}")
        print(f"  {'scheme':>13} | {'cap-wtd ret':>11} | {'tot PnL':>8} | "
              f"{'Sharpe':>7} | {'maxDD':>7} | {'stake range':>16}")
        print(f"  {'-'*13}-+-{'-'*11}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*16}")
        for scheme in SCHEMES:
            w = weights(price, scheme)
            pnl = w * ret                      # per-trade dollar PnL
            cap_ret = pnl.sum() / w.sum()      # capital-weighted return
            sharpe = pnl.mean() / pnl.std(ddof=1) if pnl.std(ddof=1) > 0 else float("nan")
            eq = np.cumsum(pnl)
            mdd = max_drawdown(eq)
            curves[(d, scheme)] = eq
            summary.append({"horizon_d": d, "scheme": scheme, "n": len(obs),
                            "cap_ret": cap_ret, "tot_pnl": pnl.sum(),
                            "sharpe": sharpe, "maxdd": mdd,
                            "stake_min": w.min(), "stake_max": w.max()})
            print(f"  {scheme:>13} | {cap_ret:>+10.1%} | {pnl.sum():>+7.2f} | "
                  f"{sharpe:>7.2f} | {mdd:>7.2f} | {w.min():>5.2f}x–{w.max():>5.2f}x")

    pd.DataFrame(summary).to_csv(args.out.replace(".csv", "_summary.csv"), index=False)

    # ── Plot equity curves at the base horizon ────────────────────────────
    base_h = args.horizon[0]
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = {"equal": "#636363", "inverse": "#e34a33",
            "sqrt_inverse": "#fe9929", "proportional": "#2b8cbe"}
    for scheme in SCHEMES:
        if (base_h, scheme) in curves:
            eq = curves[(base_h, scheme)]
            s = next(x for x in summary if x["horizon_d"] == base_h and x["scheme"] == scheme)
            ax.plot(range(len(eq)), eq, lw=1.9, color=cmap[scheme],
                    label=f"{scheme:>12}  PnL {s['tot_pnl']:+.1f}  Sharpe {s['sharpe']:.2f}  DD {s['maxdd']:.1f}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(f"YES band [0.10,0.90] @ {base_h}d — sizing schemes (mean stake = $1)")
    ax.set_xlabel("trade # (ordered by resolution date)")
    ax.set_ylabel("cumulative PnL ($)")
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.plot, dpi=120, bbox_inches="tight")

    print(f"\n  Plot    → {args.plot}")
    print(f"  Summary → {args.out.replace('.csv', '_summary.csv')}")
    print("\n  Reminder: all schemes deploy the same AVERAGE stake ($1). 'inverse'")
    print("  concentrates capital on cheap longshots — watch Sharpe & maxDD, not")
    print("  just total PnL, to see whether the extra return is worth the variance.")
    print("✓  Weighting backtest complete.")


if __name__ == "__main__":
    main()
