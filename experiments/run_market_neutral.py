#!/usr/bin/env python3
"""
Market-neutral pure-alpha backtest: beta-hedged residual mean-reversion.

Universe : SPY (market hedge) + TLT, GLD, EEM, VNQ (residual book)
Period   : 2016-01-01 → 2024-12-31  (2011–2015 warm-up consumed by
                                     regression window + BOCPD run-up)

The strategy regresses each non-SPY asset on SPY under a diffuse
Normal-Inverse-Gamma posterior (closed form = rolling OLS), trades a
dollar-neutral contrarian position on the recent-horizon residual
z-score, hedges residual SPY-beta to zero, gates leverage by BOCPD
expected run-length, and vol-targets to 5% annualized.

No fitting, no parameter search — every number is a posterior moment
or a filter output.  Compared against SPY buy-and-hold and the
existing BOCPD-AMR strategy as long-only benchmarks.
"""
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.backtest import run_amr_backtest
from posterioralpha.backtest import MNParams, run_market_neutral
from posterioralpha.validation import compute_metrics
from posterioralpha.data.loaders import load_portfolio_prices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
RF = 0.04

BT_START = "2016-01-01"
BT_END   = "2024-12-31"

COLORS = {
    "market_neutral": "#1565C0",
    "bocpd_amr":      "#00695C",
    "SPY":            "#37474F",
}
LABELS = {
    "market_neutral": "Market-Neutral (beta-hedged residuals)",
    "bocpd_amr":      "BOCPD-AMR (long-only)",
    "SPY":            "SPY Buy & Hold",
}


# ── Data ────────────────────────────────────────────────────────────────────
df = load_portfolio_prices()
returns = df.pct_change().dropna()
spy_bt  = returns["SPY"].loc[BT_START:BT_END].rename("SPY")

print("""
╔═════════════════════════════════════════════════════════════════╗
║   Market-Neutral Pure-Alpha  ·  Beta-Hedged Residuals           ║
║   SPY hedge + TLT / GLD / EEM / VNQ residual book · 2016–2024   ║
╚═════════════════════════════════════════════════════════════════╝""")
logger.info(f"Universe: {list(returns.columns)}")
logger.info(f"Window  : {BT_START} → {BT_END}")


# ── Strategies ──────────────────────────────────────────────────────────────
params = MNParams(
    market         = "SPY",
    reg_window     = 252,
    z_window       = 21,
    rebalance_freq = "W-FRI",
    target_vol     = 0.05,
    vol_window     = 63,
    leverage_cap   = 2.0,
    tc             = 0.0005,
    hazard         = 1 / 252,
)

logger.info("── Market-Neutral ──")
mn = run_market_neutral(returns, params=params)

logger.info("── BOCPD-AMR (reference) ──")
bo = run_amr_backtest(
    returns, strategy="bocpd_amr",
    lookback=252, vol_window=21, target_vol=0.10,
    leverage_cap=1.50, max_weight=0.40, l2_reg=0.001, tc=0.0005,
)

results = {"market_neutral": mn, "bocpd_amr": bo}
for k in results:
    results[k].returns = results[k].returns.loc[BT_START:BT_END]


# ── Metrics ─────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar",
            "Volatility", "Alpha", "Beta", "Info Ratio"]

def fmt(v, k):
    if not np.isfinite(v):
        return "—"
    if k in ("CAGR", "Max DD", "Volatility", "Alpha"):
        return f"{v:.2%}"
    return f"{v:.2f}"

rows = []
for name, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
    rows.append([LABELS[name]] + [fmt(m.get(k, float("nan")), k) for k in COL_KEYS])
spy_m = compute_metrics(spy_bt, rf=RF)
rows.append([LABELS["SPY"]] + [fmt(spy_m.get(k, float("nan")), k) for k in COL_KEYS])

print("\n" + "═" * 110)
print("  RESULTS  ·  2016–2024")
print("═" * 110)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))


# ── Plots ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle(
    "Market-Neutral Pure-Alpha — beta-hedged residual mean-reversion",
    fontsize=14, fontweight="bold",
)
gs    = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])
ax_gt  = fig.add_subplot(gs[1, 2])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# Cumulative wealth
spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color=COLORS["SPY"],
            linestyle="--", linewidth=1.4, label=LABELS["SPY"])
for name, res in results.items():
    cum = (1 + res.returns).cumprod()
    lw  = 2.6 if name == "market_neutral" else 1.5
    ax_cum.plot(cum.index, cum.values, color=COLORS[name],
                linewidth=lw, label=LABELS[name],
                zorder=5 if name == "market_neutral" else 3)
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=3, framealpha=0.9)
ax_cum.grid(alpha=0.25)

# Drawdown
for name, res in results.items():
    cum = (1 + res.returns).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[name],
               linewidth=1.4, label=LABELS[name])
spy_cum = (1 + spy_bt).cumprod()
spy_dd  = (spy_cum - spy_cum.cummax()) / (spy_cum.cummax() + 1e-9)
ax_dd.plot(spy_dd.index, spy_dd.values, color=COLORS["SPY"],
           linewidth=1.2, linestyle="--", label=LABELS["SPY"])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=7)
ax_dd.grid(alpha=0.25)

# Rolling 6-month Sharpe
for name, res in results.items():
    exc = res.returns - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[name],
               linewidth=1.4, label=LABELS[name])
ax_sr.axhline(0, color="black", lw=0.5, ls="--")
ax_sr.axhline(1, color="green", lw=0.4, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold")
ax_sr.legend(fontsize=7)
ax_sr.grid(alpha=0.25)

# BOCPD gate trajectory for the MN strategy (diagnostics)
if mn.lambdas is not None:
    g = mn.lambdas.reindex(spy_bt.index, method="ffill").fillna(0.0)
    ax_gt.plot(g.index, g.values, color=COLORS["market_neutral"], linewidth=1.2)
    ax_gt.set_ylim(-0.05, 1.1)
    ax_gt.set_title("BOCPD Exposure Gate (ERL / window)", fontweight="bold")
    ax_gt.axhline(1.0, color="grey", lw=0.4, ls=":")
    ax_gt.grid(alpha=0.25)
else:
    ax_gt.set_visible(False)

fig.savefig(RESULTS_DIR / "market_neutral.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/market_neutral.png")

# Save returns
out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "market_neutral_returns.csv")
logger.info("Saved: results/market_neutral_returns.csv")

print("\n✓  Done.")
