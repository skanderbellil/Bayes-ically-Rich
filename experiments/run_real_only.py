#!/usr/bin/env python3
"""
Full strategy comparison on real data only — no synthetic assets.

Universe: SPY, TLT, GLD, EEM, VNQ  (5 real ETFs, 2005-01-03 → 2024-12-31)
Period  : 2016-01-01 → 2024-12-31  (2011–2015 used as warm-up)

Strategies
----------
  bayesian_hmm   Bayesian posterior blending + 2-state HMM regime filter
  bocpd_amr      BOCPD adaptive lookback/λ → AMR + vol targeting
  hmm3_amr       3-state HMM per-regime AMR blend + vol targeting
  amr            Core AMR + vol targeting
  equal_weight   Equal-weight baseline
  SPY            Buy-and-hold benchmark
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
    "bocpd_amr":   "#00695C",
    "hmm3_amr":    "#E65100",
    "bayesian_hmm":"#AD1457",
    "amr":         "#B71C1C",
    "equal_weight":"#546E7A",
    "SPY":         "#37474F",
}
LABELS = {
    "bocpd_amr":   "BOCPD-AMR",
    "hmm3_amr":    "HMM3-AMR",
    "bayesian_hmm":"Bayes HMM",
    "amr":         "AMR + VolTarget",
    "equal_weight":"Equal Weight",
    "SPY":         "SPY B&H",
}

# ── Real data only ─────────────────────────────────────────────────────────
df = load_portfolio_prices()
returns = df.pct_change().dropna()   # 5 real ETFs — no synthetic expansion

BT_START = "2016-01-01"
BT_END   = "2024-12-31"
spy_all  = returns["SPY"]
spy_bt   = spy_all.loc[BT_START:BT_END].rename("SPY")
spy_m    = compute_metrics(spy_bt, rf=RF)

print("""
╔══════════════════════════════════════════════════════════╗
║   Real Data Only  ·  SPY · TLT · GLD · EEM · VNQ        ║
║   Period: 2016–2024  (5 ETFs, no synthetic assets)       ║
╚══════════════════════════════════════════════════════════╝""")
logger.info(f"Universe: {returns.shape[1]} assets — {list(returns.columns)}")
logger.info(f"BT window: {BT_START} → {BT_END}")

# With only 5 assets a 25% cap is very tight (min 4 assets at cap to reach 100%).
# Loosen to 0.40 so the optimiser has room to express views.
SHARED_AMR = dict(lookback=252, vol_window=21, target_vol=0.10,
                  leverage_cap=1.50, max_weight=0.40, l2_reg=0.001, tc=0.0005)
SHARED_BAY = dict(rebalance_freq="ME", min_history=252, recent_window=60,
                  ewma_halflife=30, rf=RF, sensitivity=1.5,
                  max_weight=0.40, tc=0.001)

# ── Run strategies ─────────────────────────────────────────────────────────
results = {}

logger.info("── Bayesian HMM ──")
results["bayesian_hmm"] = run_backtest(returns, strategy="bayesian_hmm", **SHARED_BAY)

logger.info("── BOCPD-AMR ──")
results["bocpd_amr"] = run_amr_backtest(returns, strategy="bocpd_amr", **SHARED_AMR)

logger.info("── HMM3-AMR ──")
results["hmm3_amr"]  = run_amr_backtest(returns, strategy="hmm3_amr",  **SHARED_AMR)

logger.info("── AMR + VolTarget ──")
results["amr"] = run_amr_backtest(returns, strategy="amr", **SHARED_AMR)

logger.info("── Equal Weight ──")
results["equal_weight"] = run_backtest(returns, strategy="equal_weight", **SHARED_BAY)

# Clip to BT window
for k in results:
    r = results[k].returns
    results[k].returns = r.loc[BT_START:BT_END]

# ── Metrics table ──────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{v:.2f}"

rows = []
for name, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
    rows.append([LABELS[name]] + [fmt(m.get(k, 0), k) for k in COL_KEYS])
rows.append(["SPY Buy & Hold"] + [fmt(spy_m.get(k, 0), k) for k in COL_KEYS])

print("\n" + "═"*80)
print("  RESULTS  ·  2016–2024  ·  Real assets only (SPY/TLT/GLD/EEM/VNQ)")
print("═"*80)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 14))
fig.suptitle("Real Data Only — SPY · TLT · GLD · EEM · VNQ  ·  2016–2024",
             fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])
ax_wt  = fig.add_subplot(gs[1, 2])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# Cumulative wealth
spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color=COLORS["SPY"],
            linewidth=1.5, linestyle="--", label=LABELS["SPY"])
for name, res in results.items():
    cum = res.cumulative
    lw  = 2.8 if name == "bayesian_hmm" else 1.6
    ax_cum.plot(cum.index, cum.values, label=LABELS[name],
                color=COLORS[name], linewidth=lw,
                zorder=5 if name == "bayesian_hmm" else 3)
ax_cum.set_yscale("log"); ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=3, framealpha=0.9)

# Drawdown
for name, res in results.items():
    cum = res.cumulative
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[name],
               linewidth=1.5, label=LABELS[name])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=7)

# Rolling Sharpe
for name, res in results.items():
    exc = res.returns - RF/252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[name],
               linewidth=1.5, label=LABELS[name])
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=7)

# Average weights of top strategy (Bayes HMM)
best = results["bayesian_hmm"]
if hasattr(best, "weights") and best.weights is not None:
    avg_w = best.weights.mean()
    ax_wt.bar(avg_w.index, avg_w.values,
              color=["#1565C0","#2E7D32","#F9A825","#AD1457","#00838F"])
    ax_wt.yaxis.set_major_formatter(_PCT)
    ax_wt.set_title("Bayes HMM — Avg Weights", fontweight="bold")
    ax_wt.set_xticklabels(avg_w.index, rotation=30, ha="right")
else:
    ax_wt.set_visible(False)

fig.savefig(RESULTS_DIR / "real_only_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/real_only_comparison.png")

# Save returns
out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "real_only_returns.csv")
logger.info("Saved: results/real_only_returns.csv")

print("\n✓  Done.")
