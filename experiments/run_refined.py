#!/usr/bin/env python3
"""
Refined strategy comparison — new models only, shorter period for speed.

New strategies
--------------
  bocpd_amr   BOCPD-driven adaptive lookback + λ, then AMR + vol targeting
  hmm3_amr    3-state HMM (bull/sideways/bear) × AMR per regime + vol targeting

Benchmarks (re-run on same period for fair comparison)
--------------
  bayesian_hmm   best Bayesian strategy from previous run
  amr            best AMR strategy from previous run
  equal_weight   simplest baseline
  SPY            buy-and-hold

Period: 2016-01-01 → 2024-12-31  (2 years warm-up from 2014)
~120 monthly rebalances  ×  Bayesian strategies
~470 weekly rebalances   ×  AMR strategies
"""
import logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.backtest.bayesian import run_backtest
from posterioralpha.backtest.amr import run_amr_backtest
from posterioralpha.data.loaders import load_portfolio_prices
from posterioralpha.data.synthetic import expand_universe
from posterioralpha.validation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)
RF = 0.04

COLORS = {
    "bocpd_amr":   "#00695C",   # deep teal
    "hmm3_amr":    "#E65100",   # deep orange
    "bayesian_hmm":"#AD1457",
    "amr":         "#B71C1C",
    "equal_weight":"#546E7A",
    "SPY":         "#37474F",
}
LABELS = {
    "bocpd_amr":   "BOCPD-AMR (new)",
    "hmm3_amr":    "HMM3-AMR (new)",
    "bayesian_hmm":"Bayes HMM (prev best)",
    "amr":         "AMR + VolTarget (prev)",
    "equal_weight":"Equal Weight",
    "SPY":         "SPY B&H",
}

# ── Data ──────────────────────────────────────────────────────────────────
df = load_portfolio_prices()
returns = expand_universe(df.pct_change().dropna(), seed=42)

BT_START = "2016-01-01"
BT_END   = "2024-12-31"
spy_all  = returns["SPY"]

# Clip to backtest window for metrics, keep full history for warm-up
spy_bt = spy_all.loc[BT_START:BT_END].rename("SPY")
spy_m  = compute_metrics(spy_bt, rf=RF)

print("""
╔══════════════════════════════════════════════════════════╗
║   Refined Strategies  ·  BOCPD-AMR  ×  HMM3-AMR         ║
║   Period: 2016–2024  (fast run)                          ║
╚══════════════════════════════════════════════════════════╝""")
logger.info(f"Universe: {returns.shape[1]} assets  |  BT: {BT_START} → {BT_END}")

SHARED_AMR  = dict(lookback=252, vol_window=21, target_vol=0.10,
                   leverage_cap=1.50, max_weight=0.35, l2_reg=0.001, tc=0.0005)
SHARED_BAY  = dict(rebalance_freq="ME", min_history=252, recent_window=60,
                   ewma_halflife=30, rf=RF, sensitivity=1.5,
                   max_weight=0.25, tc=0.001)

# ── Run strategies ────────────────────────────────────────────────────────
results = {}

logger.info("── BOCPD-AMR ──")
results["bocpd_amr"] = run_amr_backtest(returns, strategy="bocpd_amr", **SHARED_AMR)

logger.info("── HMM3-AMR ──")
results["hmm3_amr"]  = run_amr_backtest(returns, strategy="hmm3_amr",  **SHARED_AMR)

logger.info("── Bayesian HMM  (benchmark) ──")
results["bayesian_hmm"] = run_backtest(returns, strategy="bayesian_hmm", **SHARED_BAY)

logger.info("── AMR + VolTarget  (benchmark) ──")
results["amr"] = run_amr_backtest(returns, strategy="amr", **SHARED_AMR)

logger.info("── Equal Weight  (baseline) ──")
results["equal_weight"] = run_backtest(returns, strategy="equal_weight", **SHARED_BAY)

# Clip returns to BT window
for k in results:
    r = results[k].returns
    results[k].returns = r.loc[BT_START:BT_END]

# ── Metrics table ─────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{v:.2f}"

rows = []
for name, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
    rows.append([LABELS[name]] + [fmt(m.get(k, 0), k) for k in COL_KEYS])
rows.append(["SPY Buy & Hold"] + [fmt(spy_m.get(k, 0), k) for k in COL_KEYS])

print("\n" + "═"*80)
print("  RESULTS  ·  2016–2024")
print("═"*80)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))

# ── Plots ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.suptitle("BOCPD-AMR  &  HMM3-AMR  ·  2016–2024", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])
ax_lev = fig.add_subplot(gs[1, 2])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# Cumulative
spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color=COLORS["SPY"],
            linewidth=1.5, linestyle="--", label=LABELS["SPY"])
for name, res in results.items():
    cum = res.cumulative
    lw  = 2.8 if name in ("bocpd_amr","hmm3_amr") else 1.4
    ax_cum.plot(cum.index, cum.values, label=LABELS[name],
                color=COLORS[name], linewidth=lw,
                zorder=5 if name in ("bocpd_amr","hmm3_amr") else 3)
ax_cum.set_yscale("log"); ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=3, framealpha=0.9)

# Drawdown
for name, res in results.items():
    cum = res.cumulative; dd = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[name], linewidth=1.5,
               label=LABELS[name])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=7)

# Rolling Sharpe
for name, res in results.items():
    exc = res.returns - RF/252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[name], linewidth=1.5,
               label=LABELS[name])
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=7)

# AMR Leverage for new strategies
for name in ("bocpd_amr", "hmm3_amr"):
    if hasattr(results[name], "leverage"):
        lev = results[name].leverage
        ax_lev.plot(lev.index, lev.values, color=COLORS[name],
                    linewidth=1.4, label=LABELS[name])
ax_lev.axhline(1.0, color="black", lw=0.8, ls="--", label="1× (no leverage)")
ax_lev.axhline(1.5, color="red",   lw=0.6, ls=":",  alpha=0.7, label="1.5× cap")
ax_lev.set_ylim(0, 1.9); ax_lev.set_title("Vol-Targeting Leverage", fontweight="bold")
ax_lev.legend(fontsize=7)

fig.savefig(RESULTS_DIR / "refined_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/refined_comparison.png")

# Save returns
out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "refined_returns.csv")
logger.info("Saved: results/refined_returns.csv")

print("\n✓  Done.")
