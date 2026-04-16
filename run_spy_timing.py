#!/usr/bin/env python3
"""
SPY market timing — gate/sizer decomposition.

Why this file was rewritten
---------------------------
In the earlier multi-lever design, three signals all fought for the same job:

    alloc = λ_continuous × vol_target × regime_scalar

When realized vol on the underlying (≈16%) exceeded the 10% target, VT
already cut size to ~0.6×. An overlapping regime scalar multiplied on top,
and a continuous λ shifted the baseline away from 1.0 too. The three levers
were firing against each other, dragging allocation far below what either
risk control alone would prescribe.

The cleaner decomposition
-------------------------
Separate the *direction* decision from the *sizing* decision:

  GATE          Bayesian binary switch. Either we're in the market or we're not.
                Uses the fixed-halflife (42d) EWMA Omega on SPY — the "Bayesian
                fixed window" baseline — and trips on λ > 0.5.

  SIZE LEVERS   Multiplicative scalers, applied only when the gate is open.
                Each lever has a well-defined *job* it does alone.

                  • VT scaler      = clip(σ_target / σ_realized, 0, lev_cap)
                  • Regime scaler  = clip(0.5 + 0.5·min(ERL, 126)/126, 0.5, 1)

Final weight = gate × ∏ scalers,  then capped at the leverage cap.

Strategies
----------
  bnh             SPY buy & hold                     (baseline)
  bayes_only      gate = λ_fixed > 0.5, no levers    (pure Bayesian binary)
  gate_vt         gate × VT scaler
  gate_regime     gate × regime scaler (ERL-based)
  gate_both       gate × VT × regime scaler          (the proposed fix)
  compounded      continuous λ × VT × regime         (old multi-lever design, for contrast)

Common parameters
-----------------
  Rebalance        weekly (Friday)
  Lookback         252 trading days
  Realized vol     21-day trailing SPY
  Target vol       10%
  Leverage cap     1.5×
  Transaction cost 5 bps per unit of effective-weight turnover
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

from src.amr import compute_continuous_lam
from src.metrics import compute_metrics
from src.regime_models import precompute_bocpd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

RF       = 0.04
BT_START = "2016-01-01"
BT_END   = "2024-12-31"
TC       = 0.0005       # 5 bps per unit of weight change
LOOKBACK = 252
VOL_WIN  = 21
TARGET_VOL   = 0.10
LEVERAGE_CAP = 1.50
FIXED_HL     = 42.0     # fixed EWMA halflife for the Bayesian gate
ERL_FULL     = 126.0    # ERL at which the regime scaler reaches 1.0
_ANN = 252
_EPS = 1e-8

# ── Data ───────────────────────────────────────────────────────────────────
df     = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
spy    = df["SPY"].pct_change().dropna()
spy_bt = spy.loc[BT_START:BT_END].rename("SPY")

print("""
╔══════════════════════════════════════════════════════════════╗
║   SPY Market Timing  ·  gate × sizers decomposition         ║
║   Binary Bayesian gate + VT / regime as size multipliers    ║
║   2016–2024                                                  ║
╚══════════════════════════════════════════════════════════════╝""")

# ── Pre-compute BOCPD on SPY ───────────────────────────────────────────────
logger.info("Pre-computing BOCPD on SPY …")
_cp_arr, _erl_arr = precompute_bocpd(spy.values, hazard=1 / 252)
bocpd_erl = pd.Series(_erl_arr, index=spy.index)
bocpd_cp  = pd.Series(_cp_arr,  index=spy.index)

# ── Rebalance grid ─────────────────────────────────────────────────────────
try:
    rebal_dates = spy.resample("W-FRI").last().index
except Exception:
    rebal_dates = spy.resample("W").last().index
rebal_dates = rebal_dates[rebal_dates > spy.index[LOOKBACK + VOL_WIN]]
rebal_dates = rebal_dates[(rebal_dates >= BT_START) & (rebal_dates <= BT_END)]


# ── Lever definitions ──────────────────────────────────────────────────────

def vt_scaler(recent_returns: np.ndarray) -> float:
    """σ_target / σ_realized, capped by leverage cap. Realized from trailing 21d."""
    sigma = float(recent_returns.std() * np.sqrt(_ANN))
    if sigma < _EPS:
        return 1.0
    return float(np.clip(TARGET_VOL / sigma, 0.0, LEVERAGE_CAP))


def regime_scaler(erl: float) -> float:
    """Down-size when the regime is fresh; full size once ERL ≥ 126d."""
    stability = float(np.clip(min(erl, ERL_FULL) / ERL_FULL, 0.0, 1.0))
    return float(np.clip(0.5 + 0.5 * stability, 0.5, 1.0))


# ── Unified backtest loop ──────────────────────────────────────────────────

def run_strategy(weight_fn, label):
    """
    weight_fn(lam_fixed, lam_continuous, erl, recent) -> float in [0, leverage_cap].
    Returns daily return Series and a diagnostic DataFrame keyed by rebal date.
    """
    port_rets, ret_dates = [], []
    diag = []
    prev_w = None

    for i, rebal_t in enumerate(rebal_dates):
        hist = spy.loc[:rebal_t]
        if len(hist) < LOOKBACK + VOL_WIN:
            continue

        window = hist.values[-LOOKBACK:].reshape(-1, 1)   # (T, 1) for compute_continuous_lam
        recent = hist.values[-VOL_WIN:]
        erl    = float(bocpd_erl.loc[:rebal_t].iloc[-1])

        # Fixed-halflife λ  — the Bayesian binary gate signal
        lam_fixed = compute_continuous_lam(
            window, cp=0.0, erl=None,
            spy_idx=0, rf_daily=RF / _ANN,
            ewma_halflife=FIXED_HL,
        )
        # Continuous λ retained for the legacy "compounded" comparison
        lam_cont = lam_fixed   # same call; separate name for clarity at callsite

        w = float(np.clip(weight_fn(lam_fixed, lam_cont, erl, recent),
                          0.0, LEVERAGE_CAP))

        # Transaction cost on effective weight change
        tc_cost = TC * abs(w - (prev_w if prev_w is not None else w))

        # Hold-period returns
        next_t  = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else spy.index[-1]
        period  = spy.loc[rebal_t:next_t].iloc[1:]
        if len(period) == 0:
            continue

        pf = period.values * w
        if len(pf) > 0:
            pf = pf.copy()
            pf[0] -= tc_cost

        port_rets.extend(pf.tolist())
        ret_dates.extend(period.index.tolist())
        diag.append((rebal_t, w, lam_fixed, erl,
                     vt_scaler(recent), regime_scaler(erl)))

        prev_w = w

    ret_s   = pd.Series(port_rets, index=ret_dates, name=label)
    diag_df = pd.DataFrame(diag, columns=["date","weight","lam","erl","vt","regime"])
    diag_df = diag_df.set_index("date")
    return ret_s, diag_df


# ── Strategy definitions ───────────────────────────────────────────────────
#   gate(lam)   = 1 if lam > 0.5 else 0
#   size_both   = gate × vt × regime
#   compounded  = lam × vt × regime  (old design, for contrast)

def _gate(lam):         return 1.0 if lam > 0.5 else 0.0

def w_bayes_only(lam_fix, lam_c, erl, recent):
    return _gate(lam_fix)

def w_gate_vt(lam_fix, lam_c, erl, recent):
    return _gate(lam_fix) * vt_scaler(recent)

def w_gate_regime(lam_fix, lam_c, erl, recent):
    return _gate(lam_fix) * regime_scaler(erl)

def w_gate_both(lam_fix, lam_c, erl, recent):
    return _gate(lam_fix) * vt_scaler(recent) * regime_scaler(erl)

def w_compounded(lam_fix, lam_c, erl, recent):
    # Old multi-lever design: continuous λ ∈ [0.1, 0.8] treated as allocation,
    # multiplied by VT and regime. All three can shrink size simultaneously.
    return lam_c * vt_scaler(recent) * regime_scaler(erl)


logger.info("bayes_only  — binary gate, no levers")
ret_bayes, info_bayes = run_strategy(w_bayes_only, "bayes_only")

logger.info("gate_vt     — binary gate × VT")
ret_gvt,   info_gvt   = run_strategy(w_gate_vt, "gate_vt")

logger.info("gate_regime — binary gate × regime scaler")
ret_greg,  info_greg  = run_strategy(w_gate_regime, "gate_regime")

logger.info("gate_both   — binary gate × VT × regime   (proposed fix)")
ret_both,  info_both  = run_strategy(w_gate_both, "gate_both")

logger.info("compounded  — λ_continuous × VT × regime  (old multi-lever design)")
ret_comp,  info_comp  = run_strategy(w_compounded, "compounded")

# ── Metrics ────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{v:.2f}"

strategies = {
    "bnh":         (spy_bt,    "SPY Buy & Hold"),
    "bayes_only":  (ret_bayes, "Bayes-only  (gate, no levers)"),
    "gate_vt":     (ret_gvt,   "Gate × VT"),
    "gate_regime": (ret_greg,  "Gate × Regime"),
    "gate_both":   (ret_both,  "Gate × VT × Regime  (fix)"),
    "compounded":  (ret_comp,  "λ × VT × Regime  (old)"),
}

spy_m = compute_metrics(spy_bt, rf=RF)
all_m = {}
rows  = []
for key, (ret, label) in strategies.items():
    m = compute_metrics(ret.loc[BT_START:BT_END], benchmark=spy_bt, rf=RF)
    all_m[key] = m
    rows.append([label] + [fmt(m.get(k, 0), k) for k in COL_KEYS])

print("\n" + "═" * 76)
print("  SPY TIMING — gate × sizer decomposition  ·  2016–2024")
print("═" * 76)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))

# ── Attribution ────────────────────────────────────────────────────────────
print("\n── Time allocation ──")
for key in ("bayes_only", "gate_vt", "gate_regime", "gate_both", "compounded"):
    info = {"bayes_only": info_bayes, "gate_vt": info_gvt,
            "gate_regime": info_greg, "gate_both": info_both,
            "compounded": info_comp}[key]
    pct_alloc = info["weight"].mean()
    pct_on    = (info["weight"] > 0).mean()
    print(f"  {strategies[key][1]:32s}  avg wt = {pct_alloc:5.1%}   "
          f"pct on = {pct_on:5.1%}")

print("\n── Lever statistics ──")
diag_ref = info_both   # levers computed in every strategy, same values
print(f"  VT scaler      mean={diag_ref['vt'].mean():.2f}  "
      f"std={diag_ref['vt'].std():.2f}  "
      f"min={diag_ref['vt'].min():.2f}  "
      f"max={diag_ref['vt'].max():.2f}")
print(f"  Regime scaler  mean={diag_ref['regime'].mean():.2f}  "
      f"std={diag_ref['regime'].std():.2f}  "
      f"min={diag_ref['regime'].min():.2f}  "
      f"max={diag_ref['regime'].max():.2f}")
print(f"  λ (fixed 42d)  mean={diag_ref['lam'].mean():.2f}  "
      f"std={diag_ref['lam'].std():.2f}  "
      f"min={diag_ref['lam'].min():.2f}  "
      f"max={diag_ref['lam'].max():.2f}")

# ── Delta vs Bayes-only ────────────────────────────────────────────────────
print("\n── Marginal effect of each lever (vs Bayes-only gate) ──")
bayes_m = all_m["bayes_only"]
delta_rows = []
for key in ("gate_vt", "gate_regime", "gate_both", "compounded"):
    row = [strategies[key][1]]
    for k in COL_KEYS:
        d = all_m[key].get(k, 0) - bayes_m.get(k, 0)
        s = "+" if d > 0 else ""
        row.append(f"{s}{d:.2%}" if k in ("CAGR","Max DD","Volatility") else f"{s}{d:.2f}")
    delta_rows.append(row)
print(tabulate(delta_rows, headers=["Strategy"] + [f"Δ {k}" for k in COL_KEYS],
               tablefmt="simple"))

print("\n── Delta vs SPY Buy & Hold ──")
delta_rows = []
for key in ("bayes_only", "gate_vt", "gate_regime", "gate_both", "compounded"):
    row = [strategies[key][1]]
    for k in COL_KEYS:
        d = all_m[key].get(k, 0) - spy_m.get(k, 0)
        s = "+" if d > 0 else ""
        row.append(f"{s}{d:.2%}" if k in ("CAGR","Max DD","Volatility") else f"{s}{d:.2f}")
    delta_rows.append(row)
print(tabulate(delta_rows, headers=["Strategy"] + [f"Δ {k}" for k in COL_KEYS],
               tablefmt="simple"))

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle("SPY Timing  ·  Binary Bayesian gate with VT / regime size scalers  ·  2016–2024",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dd   = fig.add_subplot(gs[1, 0])
ax_sr   = fig.add_subplot(gs[1, 1])
ax_wt   = fig.add_subplot(gs[1, 2])
ax_vt   = fig.add_subplot(gs[2, 0])
ax_reg  = fig.add_subplot(gs[2, 1])
ax_erl  = fig.add_subplot(gs[2, 2])

COLORS = {
    "bnh":         "#37474F",   # grey
    "bayes_only":  "#1A237E",   # deep navy
    "gate_vt":     "#00695C",   # deep teal
    "gate_regime": "#E65100",   # deep orange
    "gate_both":   "#1B5E20",   # dark green   (the fix)
    "compounded":  "#B71C1C",   # dark red     (old, broken)
}
LWS = {k: (3.0 if k == "gate_both" else (1.5 if k == "bnh" else 2.0))
       for k in COLORS}
LSS = {k: ("--" if k == "bnh" else "-") for k in COLORS}
_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# Cumulative wealth
for key, (ret, label) in strategies.items():
    cum = (1 + ret.loc[BT_START:BT_END]).cumprod()
    ax_cum.plot(cum.index, cum.values, label=label,
                color=COLORS[key], lw=LWS[key], ls=LSS[key])
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=10, framealpha=0.9, ncol=3)

# Drawdown
for key, (ret, label) in strategies.items():
    cum = (1 + ret.loc[BT_START:BT_END]).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[key], lw=1.4, label=label,
               ls=LSS[key])
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=7)

# Rolling Sharpe
for key, (ret, label) in strategies.items():
    exc = ret.loc[BT_START:BT_END] - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[key], lw=1.4, label=label,
               ls=LSS[key])
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=7)

# Effective weight paths for the three lever combos
ax_wt.plot(info_bayes.index, info_bayes["weight"].values * 100,
           color=COLORS["bayes_only"], lw=1.1, label="Bayes-only (0/100)")
ax_wt.plot(info_both.index,  info_both["weight"].values * 100,
           color=COLORS["gate_both"],  lw=1.5, label="Gate × VT × Regime")
ax_wt.plot(info_comp.index,  info_comp["weight"].values * 100,
           color=COLORS["compounded"], lw=1.1, label="Compounded (old)", alpha=0.8)
ax_wt.axhline(100, color="black", lw=0.6, ls="--", alpha=0.5)
ax_wt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
ax_wt.set_title("Effective SPY weight", fontweight="bold")
ax_wt.legend(fontsize=8)

# VT scaler
ax_vt.plot(info_both.index, info_both["vt"].values, color="#00695C", lw=1.2)
ax_vt.fill_between(info_both.index, 0, info_both["vt"].values, alpha=0.2, color="#00695C")
ax_vt.axhline(1.0, color="black", lw=0.6, ls="--")
ax_vt.axhline(LEVERAGE_CAP, color="red", lw=0.5, ls=":", alpha=0.7, label=f"cap {LEVERAGE_CAP}×")
ax_vt.set_title("VT scaler = σ_target / σ_realized", fontweight="bold")
ax_vt.legend(fontsize=7)

# Regime scaler
ax_reg.plot(info_both.index, info_both["regime"].values, color="#E65100", lw=1.2)
ax_reg.fill_between(info_both.index, 0.5, info_both["regime"].values, alpha=0.2, color="#E65100")
ax_reg.set_ylim(0.45, 1.05)
ax_reg.set_title("Regime scaler = f(ERL)", fontweight="bold")

# ERL
ax_erl.plot(info_both.index, info_both["erl"].values, color="#5C6BC0", lw=1.1)
ax_erl.fill_between(info_both.index, 0, info_both["erl"].values, alpha=0.2, color="#5C6BC0")
ax_erl.axhline(ERL_FULL, color="green", lw=0.8, ls="--", alpha=0.7, label=f"ERL={ERL_FULL:.0f}")
ax_erl.set_title("BOCPD Expected Run Length", fontweight="bold")
ax_erl.legend(fontsize=7)

fig.savefig(RESULTS_DIR / "spy_timing.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/spy_timing.png")

out = pd.DataFrame({
    "bnh":         spy_bt,
    "bayes_only":  ret_bayes,
    "gate_vt":     ret_gvt,
    "gate_regime": ret_greg,
    "gate_both":   ret_both,
    "compounded":  ret_comp,
})
out.to_csv(RESULTS_DIR / "spy_timing_returns.csv")
logger.info("Saved: results/spy_timing_returns.csv")

print("\n✓  Done.")
