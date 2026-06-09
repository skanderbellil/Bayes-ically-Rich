#!/usr/bin/env python3
"""
Two robustness attacks on the levered 4-layer liquidity vote.

  ATTACK 1 — diversify the underlying.  The 2x-vote on QQQ alone carries a -42%
  drawdown and leans on one index.  Apply the same exposure to a diversified
  high-beta basket (QQQ + EEM + cyclical sectors) and see if the drawdown
  dependence eases while the engine stays hot.

  ATTACK 2 — push history before 2010 (the GFC).  Net liquidity goes back to
  2003, but the vote's credit (HYG) and vol-term-structure (VIX3M) inputs only
  *exist* from 2007, and the soft-sign normalisation needs ~1y to mature — so
  the honest earliest 'live' date of the full strategy is ~late-2008.  We run
  from the true data start and report what the GFC era shows, flagging the
  limitation plainly.

Vote = liquidity × VIX-TS × credit × dollar (research.overlay + dollar layer),
levered 2x with honest financing / trading costs.  Prices fetched from yfinance
(2007→) and cached to datasets/stress_panel.csv.gz for offline reruns.

    python experiments/run_liquidity_stress.py
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
from posterioralpha.data import load_net_liquidity
from posterioralpha.research import liquidity_vote, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
BASKET = ["QQQ", "EEM", "XLY", "XLK", "XLF", "XLE", "XLI"]
TICKERS = sorted(set(["SPY", "HYG", "IEF", "UUP", "^VIX", "^VIX3M"] + BASKET))


def load_panel() -> pd.DataFrame:
    if PANEL_CSV.exists():
        return pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    import yfinance as yf
    raw = yf.download(TICKERS, start="2007-01-01", auto_adjust=True, progress=False)["Close"]
    raw.columns = [c.replace("^", "") for c in raw.columns]
    raw = raw.dropna(how="all")
    raw.to_csv(PANEL_CSV, index_label="Date")
    logger.info("cached stress panel → %s", PANEL_CSV.name)
    return raw


def apply_2xvote(under_ret, vote):
    expo = (2.0 * vote).clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * under_ret
    fin = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - fin + cash - costs, index=under_ret.index), expo


def metrics_row(name, r):
    m = compute_metrics(r.dropna(), rf=RF)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    gfc = (1 + r[(r.index >= "2008-09-01") & (r.index <= "2009-06-30")]).prod() - 1
    y22 = (1 + r[r.index.year == 2022]).prod() - 1
    return [name] + [f"{m.get(k, float('nan')):.2f}" for k in keys] + \
           [f"{gfc:+.0%}", f"{y22:+.0%}"]


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = load_panel()
    idx = nl.dropna().index.intersection(p.dropna(subset=["HYG", "VIX3M", "UUP", "QQQ"]).index)
    nl, p = nl.reindex(idx), p.reindex(idx)

    # 4-layer vote (3 base + dollar)
    base = liquidity_vote(nl, p["VIX"], p["VIX3M"], p["HYG"], p["IEF"])
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    vote = pd.concat([base[["liq", "vix_ts", "credit"]], dollar.rename("d")],
                     axis=1).mean(axis=1)
    live = vote.dropna().index.min()
    logger.info("vote first live (signals matured): %s", live.date())

    qqq_ret = p["QQQ"].pct_change()
    spy_ret = p["SPY"].pct_change()
    basket_ret = p[BASKET].pct_change().mean(axis=1)

    lev_qqq, e_qqq = apply_2xvote(qqq_ret, vote)
    lev_bask, e_bask = apply_2xvote(basket_ret, vote)

    # evaluate only where the vote is live (avoid crediting the warm-up cash)
    ev = idx[idx >= live]
    rets = {
        "SPY buy & hold": spy_ret.reindex(ev),
        "QQQ buy & hold": qqq_ret.reindex(ev),
        "High-beta basket buy & hold": basket_ret.reindex(ev),
        "2×vote on QQQ": lev_qqq.reindex(ev),
        "2×vote on basket (ATTACK 1)": lev_bask.reindex(ev),
    }

    print("\n" + "=" * 92)
    print(f"  STRESS TEST  ·  vote live {live.date()} → {ev.max().date()}  "
          f"(ATTACK 2: earliest the full strategy can exist)")
    print("=" * 92)
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    print(tabulate([metrics_row(n, r) for n, r in rets.items()],
                   headers=["strategy"] + keys + ["GFC", "2022"], tablefmt="github"))
    print(f"\n  ⚠ credit (HYG) & VIX3M only exist from 2007 and soft-sign needs ~1y to "
          f"mature → the Sep-2008 crash predates a live signal; GFC column starts "
          f"{live.date()}.")
    print(f"  avg exposure — QQQ {float(e_qqq.reindex(ev).mean()):.2f}x | "
          f"basket {float(e_bask.reindex(ev).mean()):.2f}x")

    # rolling OOS vs SPY
    def rc(r):
        return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1
    print("\n  Rolling 3y CAGR win vs SPY:")
    for n in ["2×vote on QQQ", "2×vote on basket (ATTACK 1)"]:
        dc = (rc(rets[n]) - rc(rets["SPY buy & hold"])).dropna()
        print(f"    {n:30s} {float((dc > 0).mean()):.0%}  (median {float(dc.median()):+.1%}/yr)")

    # ── Graphs ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    for n, r in rets.items():
        eqc = (1 + r.fillna(0)).cumprod()
        ax.plot(eqc.index, eqc.values, lw=1.5 if "buy" in n else 1.3,
                color={"SPY buy & hold": "black", "QQQ buy & hold": "grey"}.get(n),
                alpha=0.9, label=n)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.set_title(f"Levered 4-layer vote — diversified basket & full history (from {live.date()})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # drawdowns
    ax = axes[1]
    for n in ["SPY buy & hold", "2×vote on QQQ", "2×vote on basket (ATTACK 1)"]:
        eqc = (1 + rets[n].fillna(0)).cumprod()
        dd = eqc / eqc.cummax() - 1
        ax.plot(dd.index, dd.values * 100, lw=1.1, label=n)
    ax.set_ylabel("drawdown (%)"); ax.legend(fontsize=8)
    ax.set_title("Drawdowns — does diversifying the underlying tame the tail?")
    ax.grid(alpha=0.3)
    # exposure with GFC + 2022 shaded
    ax = axes[2]
    ax.plot(e_bask.reindex(ev).index, e_bask.reindex(ev).values, lw=0.8,
            color="tab:purple", label="basket exposure")
    for span, lab in [(("2008-09-01", "2009-06-30"), "GFC"),
                      (("2022-01-01", "2022-12-31"), "2022")]:
        ax.axvspan(pd.Timestamp(span[0]), pd.Timestamp(span[1]),
                   color="red", alpha=0.08)
    ax.axhline(1.0, color="grey", lw=0.6)
    ax.set_ylabel("exposure (x)"); ax.legend(fontsize=8)
    ax.set_title("Exposure (GFC & 2022 shaded)"); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_stress.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
