#!/usr/bin/env python3
"""
Hurst-bull × 1.5×  —  multi-asset extension with a hard IS/OOS split.

Motivation
----------
The single-asset SPY version was calibrated and evaluated on overlapping
windows. Two robustness moves here:

  1) Apply the adaptive-quantile rule (variant B from run_hurst_bull_adaptive.py)
     *per asset* across the 5-ticker universe: SPY, TLT, GLD, EEM, VNQ. Each
     asset's bull-gate uses its own rolling-Hurst, its own BOCPD-ERL, and its
     own SMA-200 trend filter. Thresholds are rolling self-quantiles, so the
     rule transports without instrument-specific tuning.

  2) Freeze an out-of-sample window (2021-01-01 → 2024-12-31) and report it
     separately from the in-sample window (2008-01-01 → 2020-12-31). No
     parameters are picked by peeking at the OOS window — the adaptive rule
     only ever uses a trailing quantile, so it is causal by construction.

Portfolio rule
--------------
Weekly (Friday close), let B_t = set of assets whose adaptive-bull gate is
on. Each asset in B_t gets weight LEVERAGE / |B_t|, so the gross book is
always LEVERAGE when anything is on, 0 in cash otherwise. (Equal-weight
across signalled assets is the simplest first cut — no vol-scaling.)

Benchmarks
----------
  1) SPY buy-and-hold (the original benchmark)
  2) Equal-weight-5 buy-and-hold, weekly-rebalanced (to separate signal
     alpha from the plain diversification benefit of holding 5 sleeves).
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from tabulate import tabulate

from src.metrics import compute_metrics
from src.regime_models import precompute_bocpd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# ── Config ─────────────────────────────────────────────────────────────────
RF             = 0.04
ASSETS         = ["SPY", "TLT", "GLD", "EEM", "VNQ"]
# Strategy starts after the 756-day adaptive-quantile warmup on ~2005 data.
BT_START       = "2008-01-01"
IS_END         = "2020-12-31"
OOS_START      = "2021-01-01"
BT_END         = "2024-12-31"

HURST_WIN      = 252
DRIFT_WIN      = 200
ADAPT_WIN      = 756    # 3y rolling quantile
HURST_Q        = 0.60
ERL_Q          = 0.25
LEVERAGE       = 1.50   # gross book cap when any asset is signalled
TC             = 0.0005 # 5bp per unit of per-asset weight change
_EPS, _ANN     = 1e-8, 252

# ── Hurst (R/S) ────────────────────────────────────────────────────────────
def rs_hurst(x):
    x = np.asarray(x, dtype=float); N = len(x)
    if N < 20 or not np.all(np.isfinite(x)): return 0.5
    scales = np.unique(np.logspace(np.log10(10), np.log10(N//2), 6).astype(int))
    rs_list, good = [], []
    for n in scales:
        if n < 4 or n > N//2: continue
        K = N // n; vals = []
        for k in range(K):
            seg = x[k*n:(k+1)*n]
            z = np.cumsum(seg - seg.mean())
            R = float(z.max()-z.min()); S = float(seg.std(ddof=1))
            if S > _EPS: vals.append(R/S)
        if vals:
            rs_list.append(float(np.mean(vals))); good.append(int(n))
    if len(good) < 3: return 0.5
    slope, _ = np.polyfit(np.log(good), np.log(rs_list), 1)
    return float(slope)

def rolling_hurst(r, win):
    H = np.full(len(r), np.nan)
    for t in range(win, len(r)+1):
        H[t-1] = rs_hurst(r[t-win:t])
    return H

# ── Data & per-asset signals ───────────────────────────────────────────────
df = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
rets_all = df[ASSETS].pct_change().dropna(how="all")

sig = {}  # asset -> dict of aligned Series: ret, px, sma, H, erl, h_thr, erl_thr
for a in ASSETS:
    logger.info(f"[{a}] computing Hurst / BOCPD / quantiles")
    r   = df[a].pct_change().dropna()
    px  = df[a].reindex(r.index).ffill()
    sma = px.rolling(DRIFT_WIN, min_periods=DRIFT_WIN).mean()
    _, erl = precompute_bocpd(r.values, hazard=1/252)
    erl_s  = pd.Series(erl, index=r.index)
    H      = pd.Series(rolling_hurst(r.values, HURST_WIN), index=r.index)
    # Trailing quantiles — causal by construction (.rolling uses past-only window).
    H_thr   = H.rolling(ADAPT_WIN, min_periods=252).quantile(HURST_Q)
    ERL_thr = erl_s.rolling(ADAPT_WIN, min_periods=252).quantile(ERL_Q)
    sig[a] = {"ret": r, "px": px, "sma": sma, "H": H, "erl": erl_s,
              "h_thr": H_thr, "erl_thr": ERL_thr}

# Common date spine from SPY (all 5 ETFs trade on the same NYSE calendar).
spine = sig["SPY"]["ret"].index
rebal = sig["SPY"]["ret"].resample("W-FRI").last().index
rebal = rebal[(rebal >= BT_START) & (rebal <= BT_END)]

def _safe(s, t, default):
    v = s.loc[:t].iloc[-1] if len(s.loc[:t]) else default
    return float(v) if pd.notna(v) and np.isfinite(v) else default

def is_bull(a, t):
    """Adaptive-quantile bull gate for asset a at time t."""
    s = sig[a]
    h   = _safe(s["H"],       t, 0.5)
    e   = _safe(s["erl"],     t, 1e9)
    p   = _safe(s["px"],      t, 0.0)
    m   = _safe(s["sma"],     t, np.inf)
    h_t = _safe(s["h_thr"],   t, 0.55)
    e_t = _safe(s["erl_thr"], t, 15.0)
    return (h > h_t) and (p > m) and (e > e_t)

# ── Multi-asset backtest ───────────────────────────────────────────────────
def backtest_multi():
    port_ret, port_dates = [], []
    expo_rows = []  # per-rebalance exposure matrix
    prev_w = {a: 0.0 for a in ASSETS}
    flips = 0
    n_bull_hist = []
    for i, t in enumerate(rebal):
        bulls = [a for a in ASSETS if is_bull(a, t)]
        w = {a: 0.0 for a in ASSETS}
        if bulls:
            share = LEVERAGE / len(bulls)
            for a in bulls:
                w[a] = share
        n_bull_hist.append(len(bulls))
        expo_rows.append({"date": t, **w})

        # Transaction-cost drag: sum of per-asset absolute weight changes.
        tc_cost = sum(TC * abs(w[a] - prev_w[a]) for a in ASSETS)
        if any(abs(w[a] - prev_w[a]) > 1e-9 for a in ASSETS):
            flips += 1

        nxt = rebal[i+1] if i+1 < len(rebal) else spine[-1]
        # Slice each asset's return over the holding period, combine.
        period_idx = None
        combined = None
        for a in ASSETS:
            per = sig[a]["ret"].loc[t:nxt].iloc[1:]
            if period_idx is None:
                period_idx = per.index
                combined = np.zeros(len(per))
            # Align to the SPY calendar (all already on NYSE, so identical in practice).
            per_aligned = per.reindex(period_idx).fillna(0.0).values
            combined = combined + per_aligned * w[a]
        if period_idx is None or len(period_idx) == 0:
            prev_w = w; continue
        combined = combined.copy(); combined[0] -= tc_cost
        port_ret.extend(combined.tolist()); port_dates.extend(period_idx.tolist())
        prev_w = w

    expo = pd.DataFrame(expo_rows).set_index("date")
    pr = pd.Series(port_ret, index=port_dates).loc[BT_START:BT_END]
    return pr, flips, expo, pd.Series(n_bull_hist, index=rebal)

# ── Benchmarks ─────────────────────────────────────────────────────────────
def ew5_buy_hold():
    """Equal-weight-5 across ASSETS, rebalanced weekly. No leverage, no TC
    assessed here — it's a passive reference, not a tradable strategy."""
    w = 1.0 / len(ASSETS)
    r = rets_all[ASSETS].fillna(0.0)
    return (r * w).sum(axis=1).loc[BT_START:BT_END]

spy_bh = sig["SPY"]["ret"].loc[BT_START:BT_END].rename("SPY")
ew5_bh = ew5_buy_hold().rename("EW5")

logger.info("Running multi-asset adaptive Hurst-bull …")
ret_ma, flips_ma, expo, n_bull = backtest_multi()

# ── Report ─────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Hurst-bull × 1.5× — MULTI-ASSET (SPY, TLT, GLD, EEM, VNQ)              ║
║                                                                          ║
║  Per-asset adaptive-quantile gate:                                       ║
║    H_a   > q60(H_a | past 3Y)                                            ║
║    ERL_a > q25(ERL_a | past 3Y)                                          ║
║    P_a   > SMA200(P_a)                                                   ║
║  Equal-weight the qualifying assets, gross capped at 1.5×.              ║
║                                                                          ║
║  Windows reported separately:                                            ║
║    IN-SAMPLE      2008-01-01 → 2020-12-31                                ║
║    OUT-OF-SAMPLE  2021-01-01 → 2024-12-31   (no peeking — adaptive rule  ║
║                                                uses only trailing data)  ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def report_window(label, start, end):
    print(f"\n── {label}  ({start} → {end}) ──")
    bench = spy_bh.loc[start:end]
    rows = []
    for nm, rr in [("SPY Buy & Hold",          spy_bh.loc[start:end]),
                   ("EW-5 Buy & Hold",         ew5_bh.loc[start:end]),
                   ("Multi-asset Hurst×1.5×",  ret_ma.loc[start:end])]:
        m = compute_metrics(rr, rf=RF, benchmark=bench if nm != "SPY Buy & Hold" else None)
        rows.append([nm,
                     f"{m['CAGR']:.2%}",    f"{m['Sharpe']:.2f}", f"{m['Sortino']:.2f}",
                     f"{m['Max DD']:.2%}",  f"{m['Calmar']:.2f}", f"{m['Volatility']:.2%}",
                     f"{m.get('Alpha', float('nan')):.2%}" if 'Alpha' in m else "—",
                     f"{m.get('Beta',  float('nan')):.2f}" if 'Beta'  in m else "—"])
    print(tabulate(rows,
                   headers=["Strategy","CAGR","Sharpe","Sortino","MaxDD","Calmar","Vol","α","β"],
                   tablefmt="rounded_grid"))

report_window("IN-SAMPLE",      BT_START,  IS_END)
report_window("OUT-OF-SAMPLE",  OOS_START, BT_END)
report_window("FULL",           BT_START,  BT_END)

# ── Exposure diagnostics ──────────────────────────────────────────────────
print(f"\n── Exposure profile ──")
avg_expo = expo.mean() * 100
print("  Mean weight by asset (%):")
for a in ASSETS:
    print(f"    {a:<5s} {avg_expo[a]:5.1f}%")
print(f"  Mean # of signalled assets per week:  {n_bull.mean():.2f}")
print(f"  Weeks fully in cash (|B|=0):           "
      f"{int((n_bull == 0).sum())} / {len(n_bull)}  "
      f"({(n_bull == 0).mean():.1%})")
print(f"  Weeks fully loaded (|B|=5):            "
      f"{int((n_bull == 5).sum())} / {len(n_bull)}  "
      f"({(n_bull == 5).mean():.1%})")
print(f"  Rebalances that changed the book:      {flips_ma} / {len(rebal)}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Hurst-bull × 1.5×  ·  multi-asset (SPY/TLT/GLD/EEM/VNQ)  ·  IS 2008–2020  ·  OOS 2021–2024",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.25)
ax_c  = fig.add_subplot(gs[0, :])
ax_d  = fig.add_subplot(gs[1, 0])
ax_s  = fig.add_subplot(gs[1, 1])
ax_e  = fig.add_subplot(gs[2, :])

_PCT = mticker.FuncFormatter(lambda x,_: f"{x:.0%}")
_DOL = mticker.FuncFormatter(lambda x,_: f"${x:.2f}")

series = [(spy_bh, "#37474F", "--", "SPY B&H"),
          (ew5_bh, "#2E7D32", ":",  "EW-5 B&H"),
          (ret_ma, "#6A1B9A", "-",  "Multi-asset Hurst×1.5×")]

for r, c, ls, lb in series:
    cum = (1 + r).cumprod()
    ax_c.plot(cum.index, cum.values, color=c, lw=1.6, ls=ls, label=lb)
ax_c.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_c.text(pd.Timestamp(OOS_START), ax_c.get_ylim()[1] * 0.9, "  OOS →",
          fontsize=9, color="k", alpha=0.7)
ax_c.set_yscale("log"); ax_c.yaxis.set_major_formatter(_DOL)
ax_c.set_title("Cumulative wealth (log)"); ax_c.legend()

for r, c, ls, lb in series:
    cum = (1 + r).cumprod(); dd = (cum - cum.cummax()) / cum.cummax()
    ax_d.plot(dd.index, dd.values, color=c, lw=1.1, ls=ls, label=lb)
ax_d.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_d.yaxis.set_major_formatter(_PCT); ax_d.set_title("Drawdown"); ax_d.legend(fontsize=8)

for r, c, ls, lb in series:
    exc = r - RF/_ANN
    rs = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(_ANN)
    ax_s.plot(rs.index, rs.values, color=c, lw=1.0, ls=ls, label=lb)
ax_s.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_s.axhline(0, color="k", lw=0.5, ls="--"); ax_s.axhline(1, color="g", lw=0.4, ls=":")
ax_s.set_title("Rolling 6-month Sharpe"); ax_s.legend(fontsize=8)

# Stacked exposure over time
expo_plot = expo.reindex(expo.index).fillna(0.0)
colors = {"SPY": "#1565C0", "TLT": "#6D4C41", "GLD": "#FBC02D",
          "EEM": "#00838F", "VNQ": "#AD1457"}
ax_e.stackplot(expo_plot.index,
               [expo_plot[a].values for a in ASSETS],
               colors=[colors[a] for a in ASSETS],
               labels=ASSETS, alpha=0.85)
ax_e.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_e.axhline(LEVERAGE, color="k", lw=0.5, ls=":", alpha=0.5)
ax_e.set_title("Per-asset exposure over time (gross cap = 1.5×)")
ax_e.yaxis.set_major_formatter(_PCT); ax_e.legend(ncol=5, fontsize=9, loc="upper left")

fig.savefig(RESULTS_DIR / "hurst_multiasset.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/hurst_multiasset.png")
print("\n✓  Done.")
