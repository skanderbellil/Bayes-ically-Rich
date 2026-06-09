#!/usr/bin/env python3
"""
PosteriorAlpha + AMR  ·  Full comparison on expanded multi-asset universe
=========================================================================

Universe  : 5 real ETFs  +  9 synthetic factor-derived assets  =  14 total
Benchmark : SPY buy-and-hold
Period    : 2005–2024  (first year used as warm-up)

Strategies
----------
Bayesian family (monthly rebalance)
  bayesian          original hard-window Bayesian shrinkage
  equal_weight      1/N
  min_variance      global min-var
  hist_max_sharpe   max Sharpe on full prior
  bayesian_ewma     EWMA recent + TC penalty           (extension 1)
  bayesian_hmm      HMM regime blend + uncertainty pen (extension 2)
  bayesian_full     all extensions combined             (extension 3)

AMR family (weekly rebalance)
  amr               Asymmetric Min-Risk + vol targeting (NEW)
  amr_no_vt         AMR without vol targeting  (isolates overlay contribution)
  inv_vol           Inverse volatility + vol targeting
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.backtest.bayesian import STRATEGIES, run_all_strategies
from posterioralpha.backtest.amr import AMR_STRATEGIES, run_all_amr, sensitivity_sweep
from posterioralpha.data.loaders import load_portfolio_prices
from posterioralpha.data.synthetic import expand_universe
from posterioralpha.validation.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S", stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RF = 0.04

# ── Colour palette (consistent across all plots) ──────────────────────────
COLORS = {
    # Bayesian
    "bayesian":        "#1565C0",
    "equal_weight":    "#E65100",
    "min_variance":    "#2E7D32",
    "hist_max_sharpe": "#6A1B9A",
    "bayesian_ewma":   "#00838F",
    "bayesian_hmm":    "#AD1457",
    "bayesian_full":   "#F57F17",
    # AMR
    "amr":             "#B71C1C",
    "amr_no_vt":       "#EF9A9A",
    "inv_vol":         "#4E342E",
    # Benchmark
    "SPY":             "#546E7A",
}
LABELS = {
    "bayesian":        "Bayesian v1",
    "equal_weight":    "Equal Weight",
    "min_variance":    "Min Variance",
    "hist_max_sharpe": "Hist Max Sharpe",
    "bayesian_ewma":   "Bayes EWMA+TC",
    "bayesian_hmm":    "Bayes HMM",
    "bayesian_full":   "Bayesian Full",
    "amr":             "AMR + VolTarget",
    "amr_no_vt":       "AMR (no VolTgt)",
    "inv_vol":         "Inv. Vol + VolTgt",
    "SPY":             "SPY B&H",
}
_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# ═══════════════════════════════════════════════════════════════════════════
# 1.  Load & expand data
# ═══════════════════════════════════════════════════════════════════════════
print("""
╔══════════════════════════════════════════════════════════════╗
║   PosteriorAlpha  ×  AMR  ·  Full Strategy Comparison        ║
╚══════════════════════════════════════════════════════════════╝""")

df_raw = load_portfolio_prices()
real_returns = df_raw.pct_change().dropna()

logger.info(f"Real assets  : {real_returns.columns.tolist()}")
returns = expand_universe(real_returns, seed=42)
logger.info(f"Expanded universe: {returns.shape[1]} assets  "
            f"({real_returns.shape[1]} real + {returns.shape[1]-real_returns.shape[1]} synthetic)")
logger.info(f"Period: {returns.index[0].date()} → {returns.index[-1].date()}  "
            f"({len(returns)} days)")

spy_returns = returns["SPY"].rename("SPY")
spy_metrics = compute_metrics(spy_returns, rf=RF)

# ═══════════════════════════════════════════════════════════════════════════
# 2.  Run Bayesian strategies  (monthly, expanded universe)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n── Bayesian strategies (monthly rebalance) ──")
bay_results = run_all_strategies(
    returns,
    rebalance_freq="ME",
    min_history=252,
    recent_window=60,
    ewma_halflife=30,
    rf=RF,
    sensitivity=1.5,
    max_weight=0.25,   # lower cap for 14-asset universe
    tc=0.001,
)

# ═══════════════════════════════════════════════════════════════════════════
# 3.  Run AMR strategies  (weekly, expanded universe)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n── AMR strategies (weekly rebalance) ──")
amr_results = run_all_amr(
    returns,
    lookback=252,
    vol_window=21,
    target_vol=0.10,
    leverage_cap=1.50,
    lam=0.50,
    max_weight=0.35,
    l2_reg=0.001,
    tc=0.0005,
)

# ═══════════════════════════════════════════════════════════════════════════
# 4.  Metrics table
# ═══════════════════════════════════════════════════════════════════════════
all_results = {**bay_results, **amr_results}
col_keys = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{v:.2f}"

rows = []
for name, res in all_results.items():
    m = compute_metrics(res.returns, benchmark=spy_returns, rf=RF)
    rows.append([LABELS.get(name, name)] + [fmt(m.get(k, 0), k) for k in col_keys])

# SPY row
rows.append(["SPY Buy & Hold"] + [fmt(spy_metrics.get(k, 0), k) for k in col_keys])

print("\n" + "═"*78)
print("  STRATEGY COMPARISON  ·  14-Asset Universe  ·  2005–2024")
print("═"*78)
print(tabulate(rows, headers=["Strategy"] + col_keys, tablefmt="rounded_grid"))

# ═══════════════════════════════════════════════════════════════════════════
# 5.  Sensitivity analysis on AMR parameters
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n── AMR sensitivity sweep ──")
sweep_results = {}
for param, values in [
    ("lam",          [0.3, 0.4, 0.5, 0.6, 0.7]),
    ("target_vol",   [0.08, 0.09, 0.10, 0.11, 0.12]),
    ("leverage_cap", [1.25, 1.40, 1.50, 1.60, 1.75]),
]:
    sweep_results[param] = sensitivity_sweep(
        returns, param=param, values=values,
        lookback=252, vol_window=21, max_weight=0.35, l2_reg=0.001, tc=0.0005,
    )

# ═══════════════════════════════════════════════════════════════════════════
# 6.  Plots
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\nGenerating plots …")

# ── 6a. Master equity curve dashboard ─────────────────────────────────────
fig = plt.figure(figsize=(26, 20))
fig.suptitle("PosteriorAlpha × AMR  ·  Full Strategy Comparison  ·  2005–2024",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

ax_cum  = fig.add_subplot(gs[0, :])
ax_sr   = fig.add_subplot(gs[1, :2])
ax_tab  = fig.add_subplot(gs[1, 2])
ax_dd   = fig.add_subplot(gs[2, 0])
ax_lev  = fig.add_subplot(gs[2, 1])
ax_sens = fig.add_subplot(gs[2, 2])
ax_wts_b = fig.add_subplot(gs[3, :2])
ax_wts_a = fig.add_subplot(gs[3, 2])

# Cumulative
spy_cum = (1 + spy_returns).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color=COLORS["SPY"],
            linewidth=1.5, linestyle="--", label=LABELS["SPY"], zorder=2)

for name, res in all_results.items():
    cum = res.cumulative
    lw  = 3.0 if name in ("bayesian_full", "amr") else 1.3
    zo  = 6 if name in ("bayesian_full", "amr") else 3
    ax_cum.plot(cum.index, cum.values, label=LABELS.get(name, name),
                color=COLORS.get(name, "gray"), linewidth=lw, zorder=zo)

ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth  (log scale)", fontweight="bold")
ax_cum.legend(loc="upper left", fontsize=7.5, framealpha=0.92, ncol=3)

# Rolling Sharpe
for name, res in all_results.items():
    exc = res.returns - RF / 252
    rs  = exc.rolling(252).mean() / exc.rolling(252).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, label=LABELS.get(name, name),
               color=COLORS.get(name, "gray"),
               linewidth=2.5 if name in ("bayesian_full", "amr") else 1.0)
ax_sr.axhline(0, color="black", linewidth=0.6, linestyle="--")
ax_sr.axhline(1, color="green", linewidth=0.5, linestyle=":", alpha=0.5)
ax_sr.set_title("Rolling 1-Year Sharpe", fontweight="bold")
ax_sr.legend(fontsize=6.5, ncol=2)

# Metrics table panel
ax_tab.axis("off")
highlight = {"AMR + VolTarget", "Bayesian Full"}
tbl = ax_tab.table(cellText=rows, colLabels=["Strategy"] + col_keys,
                   loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(6.5); tbl.scale(1.0, 1.7)
for (ri, ci), cell in tbl.get_celld().items():
    if ri == 0:
        cell.set_facecolor("#263238"); cell.set_text_props(color="white", fontweight="bold")
    elif ri <= len(rows) and any(h in rows[ri-1][0] for h in highlight):
        cell.set_facecolor("#FFF9C4")
ax_tab.set_title("Performance Metrics", fontweight="bold", pad=12)

# Drawdown
for name, res in all_results.items():
    cum = res.cumulative; mx = cum.cummax()
    dd  = (cum - mx) / (mx + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS.get(name, "gray"),
               linewidth=2.0 if name in ("bayesian_full","amr") else 0.8,
               label=LABELS.get(name, name))
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=6, ncol=2)

# Leverage over time (AMR)
for name, res in amr_results.items():
    if hasattr(res, "leverage") and res.leverage is not None:
        ax_lev.plot(res.leverage.index, res.leverage.values,
                    color=COLORS.get(name, "gray"), linewidth=1.2,
                    label=LABELS.get(name, name))
ax_lev.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1× (unlevered)")
ax_lev.axhline(1.5, color="red", linewidth=0.6, linestyle=":", alpha=0.7, label="1.5× cap")
ax_lev.set_ylim(0, 1.8)
ax_lev.set_title("AMR Vol-Targeting Leverage", fontweight="bold")
ax_lev.set_ylabel("Scale factor")
ax_lev.legend(fontsize=7)

# Sensitivity: Sharpe vs lambda
lam_sweep = sweep_results.get("lam", {})
if lam_sweep:
    lam_vals  = [float(k.split("=")[1]) for k in lam_sweep]
    sharpes   = [compute_metrics(v.returns, rf=RF).get("Sharpe", 0) for v in lam_sweep.values()]
    ax_sens.plot(lam_vals, sharpes, "o-", color=COLORS["amr"], linewidth=2, markersize=6)
    ax_sens.set_xlabel("λ  (upside participation weight)")
    ax_sens.set_ylabel("Annualised Sharpe")
    ax_sens.set_title("AMR Sensitivity: λ vs Sharpe", fontweight="bold")
    ax_sens.axvline(0.5, color="gray", linewidth=0.8, linestyle="--", label="default λ=0.5")
    ax_sens.legend(fontsize=8)

# Bayesian Full weight evolution
if "bayesian_full" in bay_results:
    w_ts = bay_results["bayesian_full"].weights
    real_cols  = [c for c in w_ts.columns if not c.endswith("_s")]
    synth_cols = [c for c in w_ts.columns if c.endswith("_s")]
    colors_list = (
        [COLORS.get(c, "#90CAF9") for c in real_cols] +
        ["#BDBDBD"] * len(synth_cols)
    )
    w_ts.plot.area(ax=ax_wts_b, stacked=True, alpha=0.80,
                   color=colors_list, legend=False)
    ax_wts_b.yaxis.set_major_formatter(_PCT)
    ax_wts_b.set_title("Bayesian Full — Weight Evolution", fontweight="bold")
    ax_wts_b.legend(w_ts.columns.tolist(), loc="upper right",
                    fontsize=6, ncol=3, framealpha=0.7)

# AMR weight evolution (top-10 average)
if "amr" in amr_results:
    avg_w = amr_results["amr"].weights.mean().sort_values(ascending=False)
    top   = min(10, len(avg_w))
    bars  = ax_wts_a.bar(range(top), avg_w.values[:top],
                         color=[COLORS["amr"]] + ["#EF9A9A"]*(top-1), edgecolor="white")
    ax_wts_a.set_xticks(range(top))
    ax_wts_a.set_xticklabels(avg_w.index[:top], rotation=40, ha="right", fontsize=7)
    ax_wts_a.yaxis.set_major_formatter(_PCT)
    ax_wts_a.set_title("AMR — Avg. Top Weights", fontweight="bold")

fig.savefig(RESULTS_DIR / "full_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/full_comparison.png")

# ── 6b. Sensitivity heat-map ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
fig2.suptitle("AMR Sensitivity Analysis", fontsize=14, fontweight="bold")

param_info = [
    ("lam",          "λ",               ["CAGR", "Sharpe", "Sortino"]),
    ("target_vol",   "Target Vol",      ["CAGR", "Sharpe", "Max DD"]),
    ("leverage_cap", "Leverage Cap",    ["CAGR", "Sharpe", "Calmar"]),
]
for ax, (param, xlabel, metric_list) in zip(axes2, param_info):
    sweep = sweep_results.get(param, {})
    if not sweep:
        continue
    x_vals = [float(k.split("=")[1]) for k in sweep]
    for metric in metric_list:
        y_vals = [compute_metrics(v.returns, rf=RF).get(metric, 0) for v in sweep.values()]
        ax.plot(x_vals, y_vals, "o-", linewidth=1.8, markersize=5, label=metric)
    ax.set_xlabel(xlabel); ax.set_title(f"Sensitivity: {xlabel}", fontweight="bold")
    ax.legend(fontsize=8)

fig2.savefig(RESULTS_DIR / "amr_sensitivity.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
logger.info("Saved: results/amr_sensitivity.png")

# ── 6c. AMR dashboard (Bayesian Full vs AMR head-to-head) ─────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 10))
fig3.suptitle("Head-to-Head: Bayesian Full  vs  AMR + VolTarget",
              fontsize=13, fontweight="bold")

hero_names  = ["bayesian_full", "amr", "SPY"]
hero_series = {k: all_results[k].cumulative if k != "SPY" else (1+spy_returns).cumprod()
               for k in hero_names if k in all_results or k == "SPY"}

ax_hh_cum, ax_hh_dd, ax_hh_sr, ax_hh_lev = axes3.flat

for name, cum in hero_series.items():
    ax_hh_cum.plot(cum.index, cum.values, label=LABELS.get(name, name),
                   color=COLORS.get(name, "gray"), linewidth=2.2)
ax_hh_cum.set_yscale("log"); ax_hh_cum.yaxis.set_major_formatter(_DOLLAR)
ax_hh_cum.set_title("Cumulative Wealth"); ax_hh_cum.legend()

for name in ["bayesian_full", "amr"]:
    if name not in all_results: continue
    r   = all_results[name].returns
    cum = (1+r).cumprod(); mx = cum.cummax()
    dd  = (cum - mx) / (mx + 1e-9)
    ax_hh_dd.fill_between(dd.index, dd.values, 0, alpha=0.25, color=COLORS[name])
    ax_hh_dd.plot(dd.index, dd.values, color=COLORS[name], linewidth=1.8,
                  label=LABELS[name])
ax_hh_dd.yaxis.set_major_formatter(_PCT); ax_hh_dd.set_title("Drawdown"); ax_hh_dd.legend()

for name in ["bayesian_full", "amr"]:
    if name not in all_results: continue
    exc = all_results[name].returns - RF/252
    rs  = exc.rolling(252).mean() / exc.rolling(252).std() * np.sqrt(252)
    ax_hh_sr.plot(rs.index, rs.values, color=COLORS[name], linewidth=1.8, label=LABELS[name])
ax_hh_sr.axhline(1, color="green", linestyle=":", linewidth=0.8)
ax_hh_sr.set_title("Rolling 1-Year Sharpe"); ax_hh_sr.legend()

if "amr" in amr_results:
    lev = amr_results["amr"].leverage
    ax_hh_lev.fill_between(lev.index, lev.values, 1.0,
                            where=(lev.values > 1.0), alpha=0.3, color="red",  label="Leveraged")
    ax_hh_lev.fill_between(lev.index, lev.values, 1.0,
                            where=(lev.values <= 1.0), alpha=0.3, color="blue", label="De-risked")
    ax_hh_lev.plot(lev.index, lev.values, color=COLORS["amr"], linewidth=1.2)
    ax_hh_lev.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_hh_lev.set_ylim(0, 1.8); ax_hh_lev.set_title("AMR Leverage Factor"); ax_hh_lev.legend()

plt.tight_layout()
fig3.savefig(RESULTS_DIR / "head_to_head.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
logger.info("Saved: results/head_to_head.png")

# ═══════════════════════════════════════════════════════════════════════════
# 7.  Save returns CSV
# ═══════════════════════════════════════════════════════════════════════════
out = pd.DataFrame({n: r.returns for n, r in all_results.items()})
out["SPY"] = spy_returns
out.to_csv(RESULTS_DIR / "all_strategy_returns.csv")
logger.info("Saved: results/all_strategy_returns.csv")

print("\n✓  Done.  Plots + CSV saved in results/")
