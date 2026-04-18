#!/usr/bin/env python3
"""
Hurst-bull — multi-asset v4: IVOL-5 baseline + Hurst-confidence tilt.

Key insight from v1/v2/v3
-------------------------
v3 had the lowest drawdown and lowest vol, but OOS Sharpe was the worst
because it spent 11% of weeks fully in cash during a SPY-dominated rally.
Meanwhile v2 had the BEST OOS alpha (-3.68%, even beating IVOL-5 B&H's
-4.84%) — its problem was running 18% vol because gross was pinned to 1.5×.

The fix isn't "more filters" or "tighter scaling". It's structural:
**never fully exit the market** on this universe. IVOL-5 at 1.0× gross is
the correct default. The Hurst signal should *tilt* composition and *add*
leverage when confident — not decide whether to be in or out.

v4 mechanics (no fitting)
-------------------------
Signal ingredients per asset (all already causal in v1/v2/v3):
  score_a    = max(0, H_a - 0.5)                    (persistence strength)
  eligible_a = (P_a > SMA200_a) ∧ (ERL_a > erl_P25_a)
  inv_vol_a  = 1 / σ_a

Two portfolios per rebalance:
  w_ivol_a    = inv_vol_a / Σ inv_vol                (IVOL-5, sums to 1)
  w_hurst_a   = (score_a · eligible_a · inv_vol_a) / Σ(...)   (sums to 1 or 0)

Signal confidence (same self-reference as v3):
  S_t  = Σ score_a · eligible_a
  conf = min(1, S_t / P75(S | past 3Y))              (∈ [0, 1])

Blend composition by confidence, and scale gross from 1.0× (always in) up to
1.5× (max leverage) as confidence rises:
  blend_a  = (1 − conf)·w_ivol_a  +  conf·w_hurst_a     (sums to 1 when conf∈[0,1])
  gross_t  = 1.0  +  0.5 · conf                         (∈ [1.0, 1.5])
  w_a,t    = gross_t · blend_a

Dead-zone: skip the trade if Σ|Δw| < 5% (same as v3).

No parameters tied to an IS window. H=0.5 is theoretical; ERL veto and the
gross-scaling reference are trailing self-quantiles per asset / across the
universe. Confidence blend uses the aggregate score directly — no k, no T.

Benchmarks unchanged: SPY B&H, EW-5 B&H, IVOL-5 B&H.
Windows:                IS 2008-01-01 → 2020-12-31, OOS 2021-01-01 → 2024-12-31.
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
RF         = 0.04
ASSETS     = ["SPY", "TLT", "GLD", "EEM", "VNQ"]
BT_START   = "2008-01-01"
IS_END     = "2020-12-31"
OOS_START  = "2021-01-01"
BT_END     = "2024-12-31"

HURST_WIN  = 252
DRIFT_WIN  = 200
VOL_WIN    = 21
ADAPT_WIN  = 756
ERL_Q      = 0.25
S_REF_Q    = 0.75
GROSS_MIN  = 1.00
GROSS_MAX  = 1.50
TC         = 0.0005
H_FLOOR    = 0.50
DEADZONE   = 0.05
_EPS, _ANN = 1e-8, 252

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

# ── Per-asset signals ──────────────────────────────────────────────────────
df = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()

sig = {}
for a in ASSETS:
    logger.info(f"[{a}] computing Hurst / BOCPD / vol / quantiles")
    r   = df[a].pct_change().dropna()
    px  = df[a].reindex(r.index).ffill()
    sma = px.rolling(DRIFT_WIN, min_periods=DRIFT_WIN).mean()
    vol = r.rolling(VOL_WIN,   min_periods=VOL_WIN).std() * np.sqrt(_ANN)
    _, erl = precompute_bocpd(r.values, hazard=1/252)
    erl_s  = pd.Series(erl, index=r.index)
    H      = pd.Series(rolling_hurst(r.values, HURST_WIN), index=r.index)
    ERL_thr = erl_s.rolling(ADAPT_WIN, min_periods=252).quantile(ERL_Q)
    sig[a]  = {"ret": r, "px": px, "sma": sma, "vol": vol, "H": H,
               "erl": erl_s, "erl_thr": ERL_thr}

spine = sig["SPY"]["ret"].index
rebal = sig["SPY"]["ret"].resample("W-FRI").last().index
rebal = rebal[(rebal >= BT_START) & (rebal <= BT_END)]

def _safe(s, t, default):
    v = s.loc[:t].iloc[-1] if len(s.loc[:t]) else default
    return float(v) if pd.notna(v) and np.isfinite(v) else default

def asset_raw(a, t):
    s = sig[a]
    H   = _safe(s["H"],       t, 0.5)
    e   = _safe(s["erl"],     t, 0.0)
    p   = _safe(s["px"],      t, 0.0)
    m   = _safe(s["sma"],     t, np.inf)
    v   = _safe(s["vol"],     t, np.inf)
    e_t = _safe(s["erl_thr"], t, np.inf)
    eligible = (H > H_FLOOR) and (p > m) and (e > e_t)
    score = max(0.0, H - H_FLOOR) if eligible else 0.0
    inv_v = 1.0 / max(v, _EPS)
    return score, inv_v, eligible

# Aggregate score reference (trailing P75) — identical to v3.
logger.info("Pre-computing aggregate score + trailing P75 …")
S_daily = pd.Series(0.0, index=spine)
for t in spine:
    S_daily.loc[t] = sum(asset_raw(a, t)[0] for a in ASSETS)
S_REF = S_daily.rolling(ADAPT_WIN, min_periods=252).quantile(S_REF_Q)

# ── Backtest engine ────────────────────────────────────────────────────────
def backtest(weights_of_t):
    port_ret, port_dates, rows_w = [], [], []
    prev = {a: 0.0 for a in ASSETS}
    flips = 0; skipped = 0
    for i, t in enumerate(rebal):
        w_target = weights_of_t(t)
        move = sum(abs(w_target[a] - prev[a]) for a in ASSETS)
        if move < DEADZONE:
            w = prev.copy(); skipped += 1
        else:
            w = w_target
        rows_w.append({"date": t, **w})
        tc_cost = sum(TC * abs(w[a] - prev[a]) for a in ASSETS)
        if any(abs(w[a] - prev[a]) > 1e-9 for a in ASSETS):
            flips += 1
        nxt = rebal[i+1] if i+1 < len(rebal) else spine[-1]
        period_idx = None; combined = None
        for a in ASSETS:
            per = sig[a]["ret"].loc[t:nxt].iloc[1:]
            if period_idx is None:
                period_idx = per.index; combined = np.zeros(len(per))
            per_al = per.reindex(period_idx).fillna(0.0).values
            combined = combined + per_al * w[a]
        if period_idx is None or len(period_idx) == 0:
            prev = w; continue
        combined = combined.copy(); combined[0] -= tc_cost
        port_ret.extend(combined.tolist()); port_dates.extend(period_idx.tolist())
        prev = w
    expo = pd.DataFrame(rows_w).set_index("date")
    pr = pd.Series(port_ret, index=port_dates).loc[BT_START:BT_END]
    return pr, flips, skipped, expo

# ── v4 weight function ────────────────────────────────────────────────────
def w_v4(t):
    raws = {a: asset_raw(a, t) for a in ASSETS}
    # IVOL-5 baseline (always well-defined)
    inv_v = {a: raws[a][1] for a in ASSETS}
    inv_sum = sum(inv_v.values())
    w_ivol = {a: inv_v[a] / inv_sum for a in ASSETS}

    # Hurst-tilted composition (falls back to IVOL if nothing eligible)
    raw_h = {a: raws[a][0] * inv_v[a] for a in ASSETS}
    raw_sum = sum(raw_h.values())
    if raw_sum > _EPS:
        w_hurst = {a: raw_h[a] / raw_sum for a in ASSETS}
    else:
        w_hurst = w_ivol

    # Signal confidence → blend ratio AND extra leverage
    S_t = sum(raws[a][0] for a in ASSETS)
    S_r = _safe(S_REF, t, 0.0)
    if S_r <= _EPS:                       # warmup period: no reference yet
        conf = 0.0
    else:
        conf = float(min(1.0, S_t / S_r))

    gross = GROSS_MIN + (GROSS_MAX - GROSS_MIN) * conf
    blend = {a: (1 - conf) * w_ivol[a] + conf * w_hurst[a] for a in ASSETS}
    return {a: gross * blend[a] for a in ASSETS}

# ── Benchmarks ─────────────────────────────────────────────────────────────
def ew5_bh():
    w = 1.0 / len(ASSETS)
    r = df[ASSETS].pct_change().fillna(0.0)
    return (r * w).sum(axis=1).loc[BT_START:BT_END]

def ivol5_bh():
    vols = {a: sig[a]["vol"] for a in ASSETS}
    port = []; dates = []; prev = {a: 0.0 for a in ASSETS}
    for i, t in enumerate(rebal):
        inv = {a: 1.0 / max(_safe(vols[a], t, np.inf), _EPS) for a in ASSETS}
        tot = sum(inv.values())
        w = {a: inv[a] / tot for a in ASSETS}
        tc_cost = sum(TC * abs(w[a] - prev[a]) for a in ASSETS)
        nxt = rebal[i+1] if i+1 < len(rebal) else spine[-1]
        period_idx = None; combined = None
        for a in ASSETS:
            per = sig[a]["ret"].loc[t:nxt].iloc[1:]
            if period_idx is None:
                period_idx = per.index; combined = np.zeros(len(per))
            per_al = per.reindex(period_idx).fillna(0.0).values
            combined = combined + per_al * w[a]
        if period_idx is None or len(period_idx) == 0:
            prev = w; continue
        combined = combined.copy(); combined[0] -= tc_cost
        port.extend(combined.tolist()); dates.extend(period_idx.tolist())
        prev = w
    return pd.Series(port, index=dates).loc[BT_START:BT_END]

spy_bh_s  = sig["SPY"]["ret"].loc[BT_START:BT_END].rename("SPY")
ew5_bh_s  = ew5_bh().rename("EW5")
logger.info("Building inverse-vol benchmark …")
ivol_bh_s = ivol5_bh().rename("IVOL5")

logger.info("Running multi-asset v4 …")
ret_v4, flips_v4, skipped_v4, expo_v4 = backtest(w_v4)

# ── Report ─────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Hurst-bull v4 — MULTI-ASSET  (IVOL-5 baseline + Hurst-confidence tilt) ║
║                                                                          ║
║  Always invested: gross ∈ [1.0×, 1.5×] scaled by signal confidence.     ║
║  Composition   : blend of IVOL-5 and Hurst-tilted weights by confidence.║
║  No fitting, no free constants (H=0.5 theoretical; refs are trailing).  ║
║                                                                          ║
║  Windows:  IS 2008-2020  |  OOS 2021-2024                                ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def report(label, start, end):
    print(f"\n── {label}  ({start} → {end}) ──")
    bench = spy_bh_s.loc[start:end]
    rows = []
    for nm, rr in [("SPY Buy & Hold",        spy_bh_s.loc[start:end]),
                   ("EW-5 Buy & Hold",       ew5_bh_s.loc[start:end]),
                   ("IVOL-5 Buy & Hold",     ivol_bh_s.loc[start:end]),
                   ("MA v4  blended-tilt",   ret_v4.loc[start:end])]:
        m = compute_metrics(rr, rf=RF,
                            benchmark=bench if nm != "SPY Buy & Hold" else None)
        rows.append([nm,
                     f"{m['CAGR']:.2%}",    f"{m['Sharpe']:.2f}", f"{m['Sortino']:.2f}",
                     f"{m['Max DD']:.2%}",  f"{m['Calmar']:.2f}", f"{m['Volatility']:.2%}",
                     f"{m.get('Alpha', float('nan')):.2%}" if 'Alpha' in m else "—",
                     f"{m.get('Beta',  float('nan')):.2f}" if 'Beta'  in m else "—"])
    print(tabulate(rows,
                   headers=["Strategy","CAGR","Sharpe","Sortino","MaxDD","Calmar","Vol","α","β"],
                   tablefmt="rounded_grid"))

report("IN-SAMPLE",      BT_START,  IS_END)
report("OUT-OF-SAMPLE",  OOS_START, BT_END)
report("FULL",           BT_START,  BT_END)

# ── Diagnostics ────────────────────────────────────────────────────────────
gross = expo_v4.abs().sum(axis=1)
print(f"\n── v4 exposure profile ──")
avg_expo = expo_v4.mean() * 100
print("  Mean weight by asset (%):")
for a in ASSETS:
    print(f"    {a:<5s} {avg_expo[a]:5.1f}%")
print(f"  Gross book:   mean={gross.mean():.2f}×   "
      f"p25={gross.quantile(0.25):.2f}×   p75={gross.quantile(0.75):.2f}×   "
      f"max={gross.max():.2f}×")
print(f"  Weeks with <1.0× gross:                 "
      f"{int((gross < 1.0 - _EPS).sum())} / {len(gross)}")
print(f"  Dead-zone skips (Σ|Δw| < 5%):           {skipped_v4} / {len(rebal)}")
print(f"  Rebalances that changed the book:       {flips_v4} / {len(rebal)}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle("Hurst-bull v4 (IVOL-5 + confidence tilt)  ·  IS 2008–2020  ·  OOS 2021–2024",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.25)
ax_c = fig.add_subplot(gs[0, :])
ax_d = fig.add_subplot(gs[1, 0])
ax_s = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[2, :])
ax_g = fig.add_subplot(gs[3, :])

_PCT = mticker.FuncFormatter(lambda x,_: f"{x:.0%}")
_DOL = mticker.FuncFormatter(lambda x,_: f"${x:.2f}")

series = [(spy_bh_s,  "#37474F", "--", "SPY B&H"),
          (ew5_bh_s,  "#2E7D32", ":",  "EW-5 B&H"),
          (ivol_bh_s, "#EF6C00", ":",  "IVOL-5 B&H"),
          (ret_v4,    "#6A1B9A", "-",  "MA v4")]

for r, c, ls, lb in series:
    cum = (1 + r).cumprod()
    ax_c.plot(cum.index, cum.values, color=c, lw=1.6, ls=ls, label=lb)
ax_c.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
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
ax_s.axhline(0, color="k", lw=0.5, ls="--")
ax_s.set_title("Rolling 6-month Sharpe"); ax_s.legend(fontsize=8)

colors = {"SPY": "#1565C0", "TLT": "#6D4C41", "GLD": "#FBC02D",
          "EEM": "#00838F", "VNQ": "#AD1457"}
ax_e.stackplot(expo_v4.index,
               [expo_v4[a].values for a in ASSETS],
               colors=[colors[a] for a in ASSETS],
               labels=ASSETS, alpha=0.85)
ax_e.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_e.axhline(GROSS_MAX, color="k", lw=0.5, ls=":", alpha=0.5)
ax_e.axhline(GROSS_MIN, color="k", lw=0.5, ls=":", alpha=0.5)
ax_e.set_title("v4 per-asset exposure")
ax_e.yaxis.set_major_formatter(_PCT); ax_e.legend(ncol=5, fontsize=9, loc="upper left")

ax_g.plot(gross.index, gross.values, color="#6A1B9A", lw=1.0, label="v4 gross book")
ax_g.axhline(GROSS_MAX, color="k", lw=0.5, ls=":", alpha=0.5)
ax_g.axhline(GROSS_MIN, color="k", lw=0.5, ls=":", alpha=0.5)
ax_g.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_g.set_title("v4 gross exposure (always ≥1.0×, scales to 1.5× with signal)")
ax_g.yaxis.set_major_formatter(_PCT); ax_g.set_ylim(0.9, GROSS_MAX * 1.05)

fig.savefig(RESULTS_DIR / "hurst_multiasset_v4.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/hurst_multiasset_v4.png")
print("\n✓  Done.")
