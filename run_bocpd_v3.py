#!/usr/bin/env python3
"""
BOCPD-AMR v3 — continuously-calibrated λ via Omega ratio.

Key change over v2: replace the hand-tuned formula
    lam = 0.55 − 0.30 × clip(cp × 30, 0, 1)
with a fully data-driven calibration
    lam = Omega(r, rf) / (1 + Omega)  ×  credibility(cp)

where Omega = mean(max(r − rf, 0)) / mean(max(rf − r, 0)) is computed on the
252-day return window and cp provides a shrinkage toward 0.5 when the regime
just changed (old Omega estimate no longer valid).

Universe: SPY, TLT, GLD, EEM, VNQ  (real data)
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
    "bocpd_amr_v3": "#1B5E20",   # dark green (v3)
    "bocpd_amr_v2": "#00897B",   # teal (v2)
    "bocpd_amr":    "#80CBC4",   # light teal (v1)
    "bayesian_hmm": "#AD1457",
    "SPY":          "#37474F",
}
LABELS = {
    "bocpd_amr_v3": "BOCPD-AMR v3 (continuous λ)",
    "bocpd_amr_v2": "BOCPD-AMR v2 (preset λ)",
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
╔══════════════════════════════════════════════════════════╗
║   BOCPD-AMR v3  ·  Continuous λ via Omega Ratio         ║
║   Real data only  ·  2016–2024                           ║
╚══════════════════════════════════════════════════════════╝""")
logger.info(f"Universe: {list(returns.columns)}")

SHARED_AMR = dict(lookback=252, vol_window=21, target_vol=0.10,
                  leverage_cap=1.50, max_weight=0.40, l2_reg=0.001, tc=0.0005)
SHARED_BAY = dict(rebalance_freq="ME", min_history=252, recent_window=60,
                  ewma_halflife=30, rf=RF, sensitivity=1.5,
                  max_weight=0.40, tc=0.001)

# ── Run strategies ─────────────────────────────────────────────────────────
results = {}

logger.info("── BOCPD-AMR v3 (continuous λ) ──")
results["bocpd_amr_v3"] = run_amr_backtest(returns, strategy="bocpd_amr_v3", **SHARED_AMR)

logger.info("── BOCPD-AMR v2 (preset λ) ──")
results["bocpd_amr_v2"] = run_amr_backtest(returns, strategy="bocpd_amr_v2", **SHARED_AMR)

logger.info("── BOCPD-AMR v1 (original) ──")
results["bocpd_amr"]    = run_amr_backtest(returns, strategy="bocpd_amr",    **SHARED_AMR)

logger.info("── Bayesian HMM ──")
results["bayesian_hmm"] = run_backtest(returns, strategy="bayesian_hmm", **SHARED_BAY)

for k in results:
    results[k].returns = results[k].returns.loc[BT_START:BT_END]

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
print("  RESULTS  ·  2016–2024")
print("═"*80)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))

# ── v2 → v3 delta ─────────────────────────────────────────────────────────
print("\n── λ calibration impact  (v2 preset → v3 continuous) ──")
m_v2 = compute_metrics(results["bocpd_amr_v2"].returns, benchmark=spy_bt, rf=RF)
m_v3 = compute_metrics(results["bocpd_amr_v3"].returns, benchmark=spy_bt, rf=RF)
delta_rows = []
for k in COL_KEYS:
    v2, v3 = m_v2.get(k, 0), m_v3.get(k, 0)
    d = v3 - v2
    s = "+" if d > 0 else ""
    fd = f"{s}{d:.2%}" if k in ("CAGR","Max DD","Volatility") else f"{s}{d:.2f}"
    delta_rows.append([k, fmt(v2, k), fmt(v3, k), fd])
print(tabulate(delta_rows, headers=["Metric","v2","v3","Δ"], tablefmt="simple"))

# ── Reconstruct λ time series for plotting ─────────────────────────────────
logger.info("Reconstructing λ time series for visualisation …")

# Re-compute multi-asset BOCPD signals
_cp_arr, _erl_arr = precompute_bocpd_multi(
    returns, assets=["SPY","TLT","GLD"],
    hazard=1/252, weights=np.array([0.50, 0.30, 0.20]),
)
bocpd_cp_s  = pd.Series(_cp_arr,  index=returns.index)
bocpd_erl_s = pd.Series(_erl_arr, index=returns.index)

# Weekly rebalance grid (mirrors the backtest)
try:
    rebal_dates = returns.resample("W-FRI").last().index
except Exception:
    rebal_dates = returns.resample("W").last().index
lookback = 252; vol_window = 21
rebal_dates = rebal_dates[rebal_dates > returns.index[lookback + vol_window]]
rebal_dates = rebal_dates[(rebal_dates >= BT_START) & (rebal_dates <= BT_END)]

lam_v3_vals, lam_v2_vals, cp_vals = [], [], []
for rebal_t in rebal_dates:
    hist = returns.loc[:rebal_t]
    if len(hist) < lookback + vol_window:
        continue
    window_arr = hist.values[-lookback:]
    cp  = float(bocpd_cp_s.loc[:rebal_t].iloc[-1])

    # v3 continuous — SPY-based Omega with amplified sigmoid
    _spy_idx = list(returns.columns).index("SPY") if "SPY" in returns.columns else 0
    lam_v3_vals.append(compute_continuous_lam(
        window_arr, cp=cp, spy_idx=_spy_idx, rf_daily=RF/252))
    # v2 preset
    lam_v2_vals.append(0.55 - 0.30 * float(np.clip(cp * 30, 0, 1)))
    cp_vals.append(cp)

lam_v3_s = pd.Series(lam_v3_vals, index=rebal_dates[:len(lam_v3_vals)])
lam_v2_s = pd.Series(lam_v2_vals, index=rebal_dates[:len(lam_v2_vals)])
cp_s     = pd.Series(cp_vals,     index=rebal_dates[:len(cp_vals)])

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle("BOCPD-AMR v3 — Continuous λ via Omega Ratio  ·  2016–2024",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dd   = fig.add_subplot(gs[1, 0])
ax_sr   = fig.add_subplot(gs[1, 1])
ax_lam  = fig.add_subplot(gs[1, 2])
ax_cp   = fig.add_subplot(gs[2, 0])
ax_lam2 = fig.add_subplot(gs[2, 1])
ax_scat = fig.add_subplot(gs[2, 2])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# Cumulative wealth
spy_cum = (1 + spy_bt).cumprod()
ax_cum.plot(spy_cum.index, spy_cum.values, color=COLORS["SPY"],
            lw=1.5, ls="--", label=LABELS["SPY"])
for name, res in results.items():
    lw = 2.8 if name == "bocpd_amr_v3" else 1.4
    ax_cum.plot(res.cumulative.index, res.cumulative.values,
                label=LABELS[name], color=COLORS[name], lw=lw,
                zorder=5 if name == "bocpd_amr_v3" else 3)
ax_cum.set_yscale("log"); ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=9, ncol=3, framealpha=0.9)

# Drawdown
for name, res in results.items():
    cum = res.cumulative
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[name], lw=1.5, label=LABELS[name])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=7)

# Rolling Sharpe
for name, res in results.items():
    exc = res.returns - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[name], lw=1.5, label=LABELS[name])
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=7)

# λ time series comparison
ax_lam.plot(lam_v3_s.index, lam_v3_s.values, color=COLORS["bocpd_amr_v3"],
            lw=1.5, label="v3 continuous λ")
ax_lam.plot(lam_v2_s.index, lam_v2_s.values, color=COLORS["bocpd_amr_v2"],
            lw=1.5, ls="--", label="v2 preset λ", alpha=0.8)
ax_lam.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.5, label="λ = 0.5 (neutral)")
ax_lam.set_ylim(0.05, 0.85)
ax_lam.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
ax_lam.set_title("λ over Time: Continuous vs Preset", fontweight="bold")
ax_lam.legend(fontsize=7)

# BOCPD changepoint probability
ax_cp.fill_between(cp_s.index, cp_s.values, alpha=0.5, color="#C62828", label="cp (changepoint prob)")
ax_cp.set_title("BOCPD Changepoint Probability", fontweight="bold")
ax_cp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
ax_cp.legend(fontsize=7)

# λ v3 alone with regime shading
ax_lam2.plot(lam_v3_s.index, lam_v3_s.values, color=COLORS["bocpd_amr_v3"], lw=1.4)
ax_lam2.fill_between(lam_v3_s.index, 0.5, lam_v3_s.values,
                     where=lam_v3_s.values > 0.5, alpha=0.25, color="green",
                     label="Upside leaning (λ > 0.5)")
ax_lam2.fill_between(lam_v3_s.index, 0.5, lam_v3_s.values,
                     where=lam_v3_s.values < 0.5, alpha=0.25, color="red",
                     label="Downside defensive (λ < 0.5)")
ax_lam2.axhline(0.5, color="black", lw=0.8, ls="--")
ax_lam2.set_ylim(0.05, 0.85)
ax_lam2.set_title("Continuous λ — Regime Behaviour", fontweight="bold")
ax_lam2.legend(fontsize=7)

# Scatter: cp vs λ (shows the credibility discount at work)
sc = ax_scat.scatter(cp_s.values, lam_v3_s.values,
                     c=lam_v3_s.values, cmap="RdYlGn",
                     s=12, alpha=0.6, vmin=0.1, vmax=0.8)
# Overlay preset formula for comparison
cp_grid = np.linspace(0, cp_s.max(), 200)
lam_preset = 0.55 - 0.30 * np.clip(cp_grid * 30, 0, 1)
ax_scat.plot(cp_grid, lam_preset, color="navy", lw=1.5, ls="--", label="v2 preset formula")
ax_scat.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.5)
ax_scat.set_xlabel("Changepoint probability (cp)", fontsize=8)
ax_scat.set_ylabel("λ", fontsize=8)
ax_scat.set_title("cp vs λ: Data-Derived vs Preset", fontweight="bold")
ax_scat.legend(fontsize=7)
plt.colorbar(sc, ax=ax_scat, label="λ value")

fig.savefig(RESULTS_DIR / "bocpd_v3_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/bocpd_v3_comparison.png")

out = pd.DataFrame({n: r.returns for n, r in results.items()})
out["SPY"] = spy_bt
out.to_csv(RESULTS_DIR / "bocpd_v3_returns.csv")
logger.info("Saved: results/bocpd_v3_returns.csv")

# ── λ statistics ───────────────────────────────────────────────────────────
print("\n── λ distribution (v3 continuous vs v2 preset) ──")
lam_stats = [
    ["v3 continuous", f"{lam_v3_s.mean():.3f}", f"{lam_v3_s.std():.3f}",
     f"{lam_v3_s.min():.3f}", f"{lam_v3_s.max():.3f}"],
    ["v2 preset",     f"{lam_v2_s.mean():.3f}", f"{lam_v2_s.std():.3f}",
     f"{lam_v2_s.min():.3f}", f"{lam_v2_s.max():.3f}"],
]
print(tabulate(lam_stats, headers=["Version","Mean λ","Std λ","Min λ","Max λ"],
               tablefmt="simple"))

print("\n✓  Done.")
