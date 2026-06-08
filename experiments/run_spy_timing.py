#!/usr/bin/env python3
"""
SPY market timing via BOCPD + EWMA Omega.

Question: can a simple risk-on / risk-off rule on SPY beat buy & hold?

Three strategies, all SPY-only:
  1. SPY Buy & Hold           — baseline
  2. Binary timing            — 100% SPY when λ > 0.5, else cash
  3. Proportional timing      — SPY allocation = clip((λ − 0.25) / 0.50, 0, 1)
                                 0% SPY at λ=0.25, 50% at λ=0.50, 100% at λ=0.75

Signal: EWMA Omega on SPY with ERL-adaptive halflife (v4)
    halflife = clip(erl / 3,  14, 84)
    λ = sigmoid(2.5 × log(Omega_ewma))   clipped to [0.10, 0.80]

Rebalancing: weekly (Friday), same as AMR strategies
Transaction costs: 5 bps per unit of turnover
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
from posterioralpha.research.amr import compute_continuous_lam
from posterioralpha.research.regimes import precompute_bocpd
from posterioralpha.data.loaders import load_portfolio_prices
from posterioralpha.validation.metrics import compute_metrics

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
_ANN     = 252
_EPS     = 1e-8

# ── Data ───────────────────────────────────────────────────────────────────
df     = load_portfolio_prices()
spy    = df["SPY"].pct_change().dropna()
spy_bt = spy.loc[BT_START:BT_END].rename("SPY")

print("""
╔══════════════════════════════════════════════════════════════╗
║   SPY Market Timing  ·  BOCPD + EWMA Omega (ERL-adaptive)   ║
║   Risk-on / Risk-off  ·  2016–2024                           ║
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

# ── Backtest loop ──────────────────────────────────────────────────────────
# Each strategy: store (date → weight) and reconstruct daily returns

def run_timing(threshold_fn, label):
    """
    threshold_fn(lam) -> float in [0, 1]
    Returns daily return Series and rebalance weight Series.
    """
    port_rets, ret_dates = [], []
    weights_out = []
    prev_w = None

    for i, rebal_t in enumerate(rebal_dates):
        hist = spy.loc[:rebal_t]
        if len(hist) < LOOKBACK + VOL_WIN:
            continue

        window = hist.values[-LOOKBACK:].reshape(-1, 1)  # (T, 1) for compute_continuous_lam
        erl    = float(bocpd_erl.loc[:rebal_t].iloc[-1])

        # ERL-adaptive halflife (v4 approach)
        hl  = float(np.clip(erl / 3.0, 14.0, 84.0))
        lam = compute_continuous_lam(
            window, cp=0.0, erl=None,
            spy_idx=0, rf_daily=RF / _ANN,
            ewma_halflife=hl,
        )

        w = float(np.clip(threshold_fn(lam), 0.0, 1.0))

        # Transaction cost on weight change
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
        weights_out.append((rebal_t, w, lam, erl, hl))

        prev_w = w

    ret_s    = pd.Series(port_rets, index=ret_dates, name=label)
    info_df  = pd.DataFrame(weights_out, columns=["date", "weight", "lam", "erl", "hl"])
    info_df  = info_df.set_index("date")
    return ret_s, info_df


logger.info("Running binary timing (λ > 0.5 → 100% SPY) …")
ret_bin,  info_bin  = run_timing(lambda lam: 1.0 if lam > 0.50 else 0.0,
                                 "binary")

logger.info("Running proportional timing …")
ret_prop, info_prop = run_timing(lambda lam: np.clip((lam - 0.25) / 0.50, 0.0, 1.0),
                                 "proportional")

# ── Metrics ────────────────────────────────────────────────────────────────
COL_KEYS = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar", "Volatility"]

def fmt(v, k):
    return f"{v:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{v:.2f}"

strategies = {
    "binary":       (ret_bin,  "Binary (100% / 0%)"),
    "proportional": (ret_prop, "Proportional (0–100%)"),
    "bnh":          (spy_bt,   "SPY Buy & Hold"),
}

spy_m = compute_metrics(spy_bt, rf=RF)
all_m = {}
rows  = []
for key, (ret, label) in strategies.items():
    m = compute_metrics(ret.loc[BT_START:BT_END], benchmark=spy_bt, rf=RF)
    all_m[key] = m
    rows.append([label] + [fmt(m.get(k, 0), k) for k in COL_KEYS])

print("\n" + "═" * 72)
print("  SPY TIMING vs BUY & HOLD  ·  2016–2024")
print("═" * 72)
print(tabulate(rows, headers=["Strategy"] + COL_KEYS, tablefmt="rounded_grid"))

# ── Attribution: time invested ─────────────────────────────────────────────
print("\n── Time allocation ──")
for key, label in [("binary", "Binary"), ("proportional", "Proportional")]:
    info = info_bin if key == "binary" else info_prop
    pct_invested = info["weight"].mean()
    pct_risky    = (info["weight"] > 0).mean()
    print(f"  {label:20s}  avg allocation = {pct_invested:.1%}   "
          f"weeks invested = {pct_risky:.1%}")

# ── v3 regime-timing quality: how often did we miss crashes? ───────────────
print("\n── λ signal statistics ──")
for key, label in [("binary", "Binary"), ("proportional", "Proportional")]:
    info = info_bin if key == "binary" else info_prop
    print(f"  {label}:  λ mean={info['lam'].mean():.3f}  "
          f"std={info['lam'].std():.3f}  "
          f"min={info['lam'].min():.3f}  "
          f"max={info['lam'].max():.3f}")

# ── Improvement vs B&H ─────────────────────────────────────────────────────
print("\n── vs SPY Buy & Hold ──")
delta_rows = []
for key, label in [("binary", "Binary"), ("proportional", "Proportional")]:
    m = all_m[key]
    row = [label]
    for k in COL_KEYS:
        d = m.get(k, 0) - spy_m.get(k, 0)
        s = "+" if d > 0 else ""
        row.append(f"{s}{d:.2%}" if k in ("CAGR","Max DD","Volatility") else f"{s}{d:.2f}")
    delta_rows.append(row)
print(tabulate(delta_rows, headers=["Strategy"] + [f"Δ {k}" for k in COL_KEYS],
               tablefmt="simple"))

# ── Plots ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 16))
fig.suptitle("SPY Market Timing (BOCPD + EWMA Omega)  ·  2016–2024",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dd   = fig.add_subplot(gs[1, 0])
ax_sr   = fig.add_subplot(gs[1, 1])
ax_lam  = fig.add_subplot(gs[1, 2])
ax_wt   = fig.add_subplot(gs[2, 0])
ax_erl  = fig.add_subplot(gs[2, 1])
ax_hl   = fig.add_subplot(gs[2, 2])

COLORS = {
    "binary":      "#1A237E",   # deep navy
    "proportional":"#1B5E20",   # dark green
    "bnh":         "#37474F",   # grey
}
_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# Cumulative wealth
for key, (ret, label) in strategies.items():
    cum = (1 + ret.loc[BT_START:BT_END]).cumprod()
    lw  = 2.5 if key != "bnh" else 1.5
    ls  = "--" if key == "bnh" else "-"
    ax_cum.plot(cum.index, cum.values, label=label, color=COLORS[key], lw=lw, ls=ls)
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth (log scale)", fontweight="bold")
ax_cum.legend(fontsize=10, framealpha=0.9)

# Drawdown
for key, (ret, label) in strategies.items():
    cum = (1 + ret.loc[BT_START:BT_END]).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[key], lw=1.5, label=label,
               ls="--" if key == "bnh" else "-")
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend(fontsize=8)

# Rolling Sharpe
for key, (ret, label) in strategies.items():
    exc = ret.loc[BT_START:BT_END] - RF / 252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=COLORS[key], lw=1.5, label=label,
               ls="--" if key == "bnh" else "-")
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold"); ax_sr.legend(fontsize=8)

# λ over time
ax_lam.plot(info_bin.index, info_bin["lam"].values, color=COLORS["binary"], lw=1.2)
ax_lam.fill_between(info_bin.index, 0.5, info_bin["lam"].values,
                    where=info_bin["lam"].values > 0.5, alpha=0.25, color="green",
                    label="Risk-on (λ > 0.5)")
ax_lam.fill_between(info_bin.index, 0.5, info_bin["lam"].values,
                    where=info_bin["lam"].values < 0.5, alpha=0.25, color="red",
                    label="Risk-off (λ < 0.5)")
ax_lam.axhline(0.5, color="black", lw=0.8, ls="--")
ax_lam.set_ylim(0.0, 0.9); ax_lam.set_title("λ Signal (EWMA Omega)", fontweight="bold")
ax_lam.legend(fontsize=8)

# SPY allocation over time (proportional)
ax_wt.fill_between(info_prop.index, 0, info_prop["weight"].values * 100,
                   alpha=0.6, color=COLORS["proportional"])
ax_wt.axhline(50, color="black", lw=0.6, ls="--", alpha=0.5)
ax_wt.set_ylim(0, 105)
ax_wt.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}%"))
ax_wt.set_title("SPY Allocation (proportional)", fontweight="bold")

# ERL
ax_erl.plot(info_bin.index, info_bin["erl"].values, color="#5C6BC0", lw=1.2)
ax_erl.fill_between(info_bin.index, 0, info_bin["erl"].values, alpha=0.2, color="#5C6BC0")
ax_erl.axhline(42, color="orange", lw=0.8, ls="--", alpha=0.7, label="ERL=42")
ax_erl.axhline(126, color="red", lw=0.6, ls=":", alpha=0.6, label="ERL=126")
ax_erl.set_title("BOCPD Expected Run Length", fontweight="bold")
ax_erl.legend(fontsize=8)

# Adaptive halflife
ax_hl.plot(info_bin.index, info_bin["hl"].values, color="#E65100", lw=1.2)
ax_hl.fill_between(info_bin.index, 14, info_bin["hl"].values, alpha=0.2, color="#E65100")
ax_hl.axhline(42, color="green", lw=0.8, ls="--", alpha=0.7, label="hl=42 (mid)")
ax_hl.set_ylim(10, 90)
ax_hl.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:.0f}d"))
ax_hl.set_title("Adaptive EWMA Halflife = clip(ERL/3, 14, 84)", fontweight="bold")
ax_hl.legend(fontsize=8)

fig.savefig(RESULTS_DIR / "spy_timing.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/spy_timing.png")

out = pd.DataFrame({
    "binary":       ret_bin,
    "proportional": ret_prop,
    "SPY_BnH":      spy_bt,
})
out.to_csv(RESULTS_DIR / "spy_timing_returns.csv")
logger.info("Saved: results/spy_timing_returns.csv")

print("\n✓  Done.")
