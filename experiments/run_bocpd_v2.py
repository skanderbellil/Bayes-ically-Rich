#!/usr/bin/env python3
"""
BOCPD-AMR v2 — four statistical improvements over v1.

Improvements
------------
  1. Multi-asset BOCPD  — SPY (50%) + TLT (30%) + GLD (20%) aggregate signal
                          catches bond/gold regime breaks missed by SPY-only
  2. CVaR objective     — Expected Shortfall at 5% instead of semi-deviation;
                          tail events weighted by severity, not just existence
  3. Dynamic low-vol tilt — vol_rank penalty in objective scales with cp signal;
                            exploits min-var anomaly most aggressively during stress
  4. Adaptive leverage cap — shrinks during transitions, expands in stable regimes

Universe: SPY, TLT, GLD, EEM, VNQ  (real data, no synthetic)
Period  : 2016-01-01 → 2024-12-31
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
from posterioralpha.validation.metrics import compute_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"; RESULTS_DIR.mkdir(exist_ok=True)
RF = 0.04

COLORS = {
    "bocpd_amr_v2": "#00897B",   # teal (upgraded)
    "bocpd_amr":    "#80CBC4",   # lighter teal (original)
    "bayesian_hmm": "#AD1457",   # pink
    "amr":          "#B71C1C",   # deep red
    "SPY":          "#37474F",
}
LABELS = {
    "bocpd_amr_v2": "BOCPD-AMR v2 (new)",
    "bocpd_amr":    "BOCPD-AMR v1 (baseline)",
    "bayesian_hmm": "Bayes HMM",
    "amr":          "AMR + VolTarget",
    "SPY":          "SPY B&H",
}

# ── Data ───────────────────────────────────────────────────────────────────
df = load_portfolio_prices()
returns = df.pct_change().dropna()

BT_START = "2016-01-01"
BT_END   = "2024-12-31"
spy_bt   = returns["SPY"].loc[BT_START:BT_END].rename("SPY")
spy_m    = compute_metrics(spy_bt, rf=RF)

print("""
╔══════════════════════════════════════════════════════════╗
║   BOCPD-AMR v2  ·  Four Statistical Improvements        ║
║   Real data only  ·  2016–2024                           ║
╚══════════════════════════════════════════════════════════╝""")
logger.info(f"Universe: {list(returns.columns)} ({returns.shape[1]} assets)")

SHARED_AMR = dict(lookback=252, vol_window=21, target_vol=0.10,
                  leverage_cap=1.50, max_weight=0.40, l2_reg=0.001, tc=0.0005)
SHARED_BAY = dict(rebalance_freq="ME", min_history=252, recent_window=60,
                  ewma_halflife=30, rf=RF, sensitivity=1.5,
                  max_weight=0.40, tc=0.001)

# ── Strategies ─────────────────────────────────────────────────────────────
results = {}

logger.info("── BOCPD-AMR v2 (all improvements) ──")
results["bocpd_amr_v2"] = run_amr_backtest(returns, strategy="bocpd_amr_v2", **SHARED_AMR)

logger.info("── BOCPD-AMR v1 (baseline) ──")
results["bocpd_amr"]    = run_amr_backtest(returns, strategy="bocpd_amr",    **SHARED_AMR)

logger.info("── Bayesian HMM ──")
results["bayesian_hmm"] = run_backtest(returns, strategy="bayesian_hmm", **SHARED_BAY)

logger.info("── AMR + VolTarget ──")
results["amr"]          = run_amr_backtest(returns, strategy="amr",          **SHARED_AMR)

# Clip to BT window
for k in results:
    results[k].returns = results[k].returns.loc[BT_START:BT_END]

# ── Metrics ────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{v:.2f}"

rows = []
for name, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
    rows.append([LABELS[name]] + [fmt(m.get(k, 0), k) for k in COL_KEYS])
rows.append(["SPY Buy & Hold"] + [fmt(spy_m.get(k, 0), k) for k in COL_KEYS])

print("\n" + "═"*80)
print("  RESULTS  ·  2016–2024  ·  SPY/TLT/GLD/EEM/VNQ")
print("═"*80)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))

# ── Improvement breakdown table ────────────────────────────────────────────
print("\n── Improvement attribution (v2 vs v1) ──")
m_v1 = compute_metrics(results["bocpd_amr"].returns,    benchmark=spy_bt, rf=RF)
m_v2 = compute_metrics(results["bocpd_amr_v2"].returns, benchmark=spy_bt, rf=RF)
delta_rows = []
for k in COL_KEYS:
    v1, v2 = m_v1.get(k, 0), m_v2.get(k, 0)
    delta = v2 - v1
    sign  = "+" if delta > 0 else ""
    fmted = f"{sign}{delta:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{sign}{delta:.2f}"
    delta_rows.append([k, fmt(v1, k), fmt(v2, k), fmted])
print(tabulate(delta_rows, headers=["Metric", "v1", "v2", "Δ"], tablefmt="simple"))

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.suptitle("BOCPD-AMR v2 vs Baselines  ·  2016–2024", fontsize=14, fontweight="bold")
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
    lw  = 2.8 if name == "bocpd_amr_v2" else 1.5
    ax_cum.plot(cum.index, cum.values, label=LABELS[name],
                color=COLORS[name], linewidth=lw,
                zorder=5 if name == "bocpd_amr_v2" else 3)
ax_cum.set_yscale("log"); ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=3, framealpha=0.9)

# Drawdown
for name, res in results.items():
    cum = res.cumulative
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[name], linewidth=1.5, label=LABELS[name])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=7)

# Rolling Sharpe
for name, res in results.items():
    exc = res.returns - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[name], linewidth=1.5, label=LABELS[name])
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=7)

# Leverage comparison (v1 vs v2)
for name in ("bocpd_amr_v2", "bocpd_amr"):
    if hasattr(results[name], "leverage"):
        lev = results[name].leverage
        ax_lev.plot(lev.index, lev.values, color=COLORS[name],
                    linewidth=1.4, label=LABELS[name])
ax_lev.axhline(1.0, color="black", lw=0.8, ls="--", label="1× (no leverage)")
ax_lev.axhline(1.5, color="red",   lw=0.6, ls=":",  alpha=0.7, label="1.5× cap")
ax_lev.set_ylim(0, 2.1)
ax_lev.set_title("Adaptive Leverage (v1 fixed cap vs v2 adaptive)", fontweight="bold")
ax_lev.legend(fontsize=7)

fig.savefig(RESULTS_DIR / "bocpd_v2_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/bocpd_v2_comparison.png")

out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "bocpd_v2_returns.csv")
logger.info("Saved: results/bocpd_v2_returns.csv")

print("\n✓  Done.")
