#!/usr/bin/env python3
"""
BOCPD-AMR v4 — ERL-adaptive EWMA halflife.

Core insight (from v3 diagnostic)
----------------------------------
  • ERL credibility discount fights Omega (removed)
  • Kelly guard on short Sharpe windows fires too many false positives (removed)
  • Pure EWMA Omega is self-sufficient — it already reflects market state

Improvement over v3
--------------------
  ERL-adaptive EWMA halflife  (Keating & Shadwick 2002 + Engle/Patton 2001)
  Instead of a fixed 42-day halflife, scale it with ERL:
      halflife = clip(erl / 3,  14, 84)
  • Fresh regime (erl≈10 days): halflife=14 → Omega reacts to today's data quickly
  • Standard regime (erl≈126 days): halflife=42 → matches v3 exactly
  • Long stable regime (erl≈252 days): halflife=84 → stable, uses more history
  This integrates ERL meaningfully: during crashes Omega becomes defensive faster,
  during prolonged bulls Omega is more stable (less noise-driven whipsaw).

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

from src.backtest import run_backtest
from src.amr import run_amr_backtest, compute_continuous_lam
from src.metrics import compute_metrics
from src.regime_models import precompute_bocpd_multi

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
RF = 0.04

COLORS = {
    "bocpd_amr_v4": "#1A237E",   # deep navy (v4)
    "bocpd_amr_v3": "#1B5E20",   # dark green (v3)
    "bocpd_amr_v2": "#00897B",   # teal (v2)
    "bocpd_amr":    "#80CBC4",   # light teal (v1)
    "bayesian_hmm": "#AD1457",
    "SPY":          "#37474F",
}
LABELS = {
    "bocpd_amr_v4": "BOCPD-AMR v4 (adaptive EWMA hl)",
    "bocpd_amr_v3": "BOCPD-AMR v3 (continuous λ)",
    "bocpd_amr_v2": "BOCPD-AMR v2 (CVaR+multi-BOCPD)",
    "bocpd_amr":    "BOCPD-AMR v1 (original)",
    "bayesian_hmm": "Bayes HMM",
    "SPY":          "SPY B&H",
}

# ── Data ───────────────────────────────────────────────────────────────────
df      = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
returns = df.pct_change().dropna()

BT_START = "2016-01-01"
BT_END   = "2024-12-31"
spy_bt   = returns["SPY"].loc[BT_START:BT_END].rename("SPY")
spy_m    = compute_metrics(spy_bt, rf=RF)

print("""
╔══════════════════════════════════════════════════════════════════╗
║   BOCPD-AMR v4  ·  ERL-adaptive EWMA halflife                   ║
║   Real data only  ·  2016–2024                                   ║
╚══════════════════════════════════════════════════════════════════╝""")
logger.info(f"Universe: {list(returns.columns)}")

SHARED_AMR = dict(lookback=252, vol_window=21, target_vol=0.10,
                  leverage_cap=1.50, max_weight=0.40, l2_reg=0.001, tc=0.0005)
SHARED_BAY = dict(rebalance_freq="ME", min_history=252, recent_window=60,
                  ewma_halflife=30, rf=RF, sensitivity=1.5,
                  max_weight=0.40, tc=0.001)

results = {}
logger.info("── BOCPD-AMR v4 ──")
results["bocpd_amr_v4"] = run_amr_backtest(returns, strategy="bocpd_amr_v4", **SHARED_AMR)
logger.info("── BOCPD-AMR v3 ──")
results["bocpd_amr_v3"] = run_amr_backtest(returns, strategy="bocpd_amr_v3", **SHARED_AMR)
logger.info("── Bayesian HMM ──")
results["bayesian_hmm"] = run_backtest(returns, strategy="bayesian_hmm", **SHARED_BAY)
logger.info("── BOCPD-AMR v1 ──")
results["bocpd_amr"]    = run_amr_backtest(returns, strategy="bocpd_amr",    **SHARED_AMR)

for k in results:
    results[k].returns = results[k].returns.loc[BT_START:BT_END]

# ── Metrics ────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]
def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR","Max DD","Volatility") else f"{v:.2f}"

rows = []
all_metrics = {}
for name, res in results.items():
    m = compute_metrics(res.returns, benchmark=spy_bt, rf=RF)
    all_metrics[name] = m
    rows.append([LABELS[name]] + [fmt(m.get(k,0), k) for k in COL_KEYS])
rows.append(["SPY Buy & Hold"] + [fmt(spy_m.get(k,0), k) for k in COL_KEYS])

print("\n" + "═"*90)
print("  RESULTS  ·  2016–2024  ·  SPY/TLT/GLD/EEM/VNQ")
print("═"*90)
print(tabulate(rows, headers=["Strategy"]+COL_KEYS, tablefmt="rounded_grid"))

# ── v3 → v4 delta ──────────────────────────────────────────────────────────
print("\n── Improvement attribution (v3 → v4) ──")
delta_rows = []
for k in COL_KEYS:
    v3 = all_metrics["bocpd_amr_v3"].get(k, 0)
    v4 = all_metrics["bocpd_amr_v4"].get(k, 0)
    d  = v4 - v3
    s  = "+" if d > 0 else ""
    fd = f"{s}{d:.2%}" if k in ("CAGR","Max DD","Volatility") else f"{s}{d:.2f}"
    delta_rows.append([k, fmt(v3,k), fmt(v4,k), fd])
print(tabulate(delta_rows, headers=["Metric","v3","v4","Δ"], tablefmt="simple"))

# ── Reconstruct λ series ───────────────────────────────────────────────────
logger.info("Reconstructing λ and ERL time series …")
cp_arr, erl_arr = precompute_bocpd_multi(
    returns, assets=["SPY","TLT","GLD"],
    hazard=1/252, weights=np.array([0.50,0.30,0.20]),
)
bocpd_erl_s = pd.Series(erl_arr, index=returns.index)

try:
    rebal_dates = returns.resample("W-FRI").last().index
except Exception:
    rebal_dates = returns.resample("W").last().index
lookback = 252; vol_window = 21
rebal_dates = rebal_dates[rebal_dates > returns.index[lookback + vol_window]]
rebal_dates = rebal_dates[(rebal_dates >= BT_START) & (rebal_dates <= BT_END)]

spy_idx = list(returns.columns).index("SPY")
lam_v4_vals, lam_v3_vals, erl_vals, trans_vals = [], [], [], []

for rebal_t in rebal_dates:
    hist = returns.loc[:rebal_t]
    if len(hist) < lookback + vol_window:
        continue
    window_arr = hist.values[-lookback:]
    erl = float(bocpd_erl_s.loc[:rebal_t].iloc[-1])
    cp  = float(pd.Series(cp_arr, index=returns.index).loc[:rebal_t].iloc[-1])

    # v4: ERL-adaptive halflife, no discount
    _hl = float(np.clip(erl / 3.0, 14.0, 84.0))
    lam_v4 = compute_continuous_lam(window_arr, cp=0.0, erl=None,
                                    spy_idx=spy_idx, rf_daily=RF/252,
                                    ewma_halflife=_hl)
    # v3: fixed 42-day halflife, tiny cp discount
    lam_v3 = compute_continuous_lam(window_arr, cp=cp, erl=None,
                                    spy_idx=spy_idx, rf_daily=RF/252,
                                    ewma_halflife=42.0)
    lam_v4_vals.append(lam_v4)
    lam_v3_vals.append(lam_v3)
    erl_vals.append(erl)
    trans_vals.append(float(np.clip(erl / 3.0, 14.0, 84.0)))  # adaptive halflife

idx_r = rebal_dates[:len(lam_v4_vals)]
lam_v4_s  = pd.Series(lam_v4_vals,  index=idx_r)
lam_v3_s  = pd.Series(lam_v3_vals,  index=idx_r)
erl_s     = pd.Series(erl_vals,     index=idx_r)
trans_s   = pd.Series(trans_vals,   index=idx_r)

print(f"\n── λ distribution comparison ──")
stats = [
    ["v4 (adaptive hl)", f"{lam_v4_s.mean():.3f}", f"{lam_v4_s.std():.3f}",
     f"{lam_v4_s.min():.3f}", f"{lam_v4_s.max():.3f}"],
    ["v3 (hl=42, cp)",   f"{lam_v3_s.mean():.3f}", f"{lam_v3_s.std():.3f}",
     f"{lam_v3_s.min():.3f}", f"{lam_v3_s.max():.3f}"],
]
print(tabulate(stats, headers=["Version","Mean λ","Std λ","Min λ","Max λ"], tablefmt="simple"))

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 18))
fig.suptitle("BOCPD-AMR v4 — ERL-adaptive EWMA halflife  ·  2016–2024",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dd   = fig.add_subplot(gs[1, 0])
ax_sr   = fig.add_subplot(gs[1, 1])
ax_lam  = fig.add_subplot(gs[1, 2])
ax_erl  = fig.add_subplot(gs[2, 0])
ax_tr   = fig.add_subplot(gs[2, 1])
ax_scat = fig.add_subplot(gs[2, 2])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color=COLORS["SPY"],
            lw=1.5, ls="--", label=LABELS["SPY"])
for name, res in results.items():
    lw = 2.8 if name == "bocpd_amr_v4" else 1.4
    ax_cum.plot(res.cumulative.index, res.cumulative.values,
                label=LABELS[name], color=COLORS[name], lw=lw,
                zorder=5 if name == "bocpd_amr_v4" else 3)
ax_cum.set_yscale("log"); ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=8, ncol=3, framealpha=0.9)

for name, res in results.items():
    cum = res.cumulative
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[name], lw=1.5, label=LABELS[name])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=6)

for name, res in results.items():
    exc = res.returns - RF/252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[name], lw=1.5, label=LABELS[name])
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=6)

ax_lam.plot(lam_v4_s.index, lam_v4_s.values, color=COLORS["bocpd_amr_v4"],
            lw=1.4, label="v4 λ (adaptive hl)")
ax_lam.plot(lam_v3_s.index, lam_v3_s.values, color=COLORS["bocpd_amr_v3"],
            lw=1.2, ls="--", alpha=0.7, label="v3 λ (hl=42, cp)")
ax_lam.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.5)
ax_lam.fill_between(lam_v4_s.index, 0.5, lam_v4_s.values,
                    where=lam_v4_s.values > 0.5, alpha=0.2, color="green")
ax_lam.fill_between(lam_v4_s.index, 0.5, lam_v4_s.values,
                    where=lam_v4_s.values < 0.5, alpha=0.2, color="red")
ax_lam.set_title("λ: v4 (adaptive hl) vs v3 (fixed hl=42)", fontweight="bold")
ax_lam.legend(fontsize=7); ax_lam.set_ylim(0.05, 0.85)

ax_erl.plot(erl_s.index, erl_s.values, color="#5C6BC0", lw=1.2)
ax_erl.fill_between(erl_s.index, 0, erl_s.values, alpha=0.2, color="#5C6BC0")
ax_erl.axhline(63, color="red", lw=0.8, ls="--", label="63d halflife")
ax_erl.axhline(126, color="orange", lw=0.6, ls=":", label="126d stable")
ax_erl.set_title("BOCPD Expected Run Length", fontweight="bold")
ax_erl.legend(fontsize=7)

ax_tr.plot(trans_s.index, trans_s.values, color="#5C6BC0", lw=1.2)
ax_tr.fill_between(trans_s.index, 0, trans_s.values, alpha=0.25, color="#5C6BC0")
ax_tr.axhline(42, color="green", lw=0.8, ls="--", alpha=0.7, label="hl=42 (v3 fixed)")
ax_tr.set_title("Adaptive EWMA halflife = clip(erl/3, 14, 84)", fontweight="bold")
ax_tr.set_ylabel("Halflife (days)", fontsize=8)
ax_tr.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}d"))
ax_tr.legend(fontsize=7)

ax_scat.scatter(erl_s.values, lam_v4_s.values, c=lam_v4_s.values,
                cmap="RdYlGn", s=12, alpha=0.6, vmin=0.1, vmax=0.8)
ax_scat.axhline(0.5, color="black", lw=0.6, ls=":")
ax_scat.axvline(126, color="orange", lw=0.8, ls="--", alpha=0.6, label="ERL=126 (→hl=42)")
ax_scat.set_xlabel("Expected Run Length", fontsize=8)
ax_scat.set_ylabel("λ", fontsize=8)
ax_scat.set_title("ERL vs λ (v4 adaptive)", fontweight="bold")
ax_scat.legend(fontsize=7)

fig.savefig(RESULTS_DIR / "bocpd_v4_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/bocpd_v4_comparison.png")

out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "bocpd_v4_returns.csv")
logger.info("Saved: results/bocpd_v4_returns.csv")
print("\n✓  Done.")
