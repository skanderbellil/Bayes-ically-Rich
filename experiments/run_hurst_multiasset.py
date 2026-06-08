#!/usr/bin/env python3
"""
Hurst-bull multi-asset v5 — cross-sectional top-K ranking.

What changed vs v1–v4
---------------------
Every prior variant used ABSOLUTE thresholds ("is H_a > 0.5?"). That mixes
two questions — "is this asset trending?" and "is it trending more than its
peers right now?" — and ends up under-exposed on universes where everything
is lukewarm.

v5 uses RELATIVE (cross-sectional) ranking instead:

  1.  For each asset a at time t, compute
         rank_score_a = (H_a - 0.5) · 1[eligible_a]
      where eligible requires P_a > SMA200_a AND ERL_a > own trailing P25.
  2.  If ≥ K assets are eligible (K = 3 = majority of 5), pick the top-K
      by rank_score and weight them inverse-vol, scaled to 1.5× gross.
  3.  Otherwise, fall back to IVOL-5 at 1.0× gross — never cash.

No fitted parameters: K = ⌈N/2⌉ = 3 is the natural majority threshold for
a 5-asset universe; all other inputs (H, σ, ERL, SMA) are causal functions
of the data. No training-set calibration.

Why this might work when v1–v4 didn't
-------------------------------------
Cross-sectional ranking is self-scaling: it doesn't care about the absolute
H-distribution of any instrument, only relative ordering. It also always
produces a full book when 3+ assets qualify — no dilution across noisy 5-way
weights — and falls back to a passive diversified portfolio rather than
cash when the market is confused.

Benchmarks: SPY B&H, EW-5 B&H, IVOL-5 B&H (unchanged).
Windows:    IS 2008-2020, OOS 2021-2024.
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

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.validation import compute_metrics
from posterioralpha.research import precompute_bocpd, rolling_hurst, rs_hurst
from posterioralpha.data.loaders import load_portfolio_prices

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
LEV_FULL   = 1.50     # gross when top-K book is active
LEV_BASE   = 1.00     # gross when fall-back IVOL-5 is active
K_TOP      = 3        # natural majority of 5
TC         = 0.0005
H_FLOOR    = 0.50
DEADZONE   = 0.05
_EPS, _ANN = 1e-8, 252

# Hurst (R/S) primitives are imported from posterioralpha.research.hurst above.

# ── Per-asset signals ──────────────────────────────────────────────────────
df = load_portfolio_prices()

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
    inv_v = 1.0 / max(v, _EPS)
    return H, eligible, inv_v

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

def w_v5(t):
    """Top-K cross-sectional rank, inverse-vol sized, fall back to IVOL-5."""
    raws = {a: asset_raw(a, t) for a in ASSETS}
    eligible = [a for a in ASSETS if raws[a][1]]
    inv_v = {a: raws[a][2] for a in ASSETS}

    if len(eligible) >= K_TOP:
        # Rank eligible assets by H (descending), take top K.
        ranked = sorted(eligible, key=lambda a: raws[a][0], reverse=True)[:K_TOP]
        tot = sum(inv_v[a] for a in ranked)
        w = {a: 0.0 for a in ASSETS}
        for a in ranked:
            w[a] = LEV_FULL * inv_v[a] / tot
        return w

    # Fall-back: IVOL-5 at 1.0× gross.
    tot = sum(inv_v.values())
    return {a: LEV_BASE * inv_v[a] / tot for a in ASSETS}

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

logger.info("Running multi-asset v5 …")
ret_v5, flips_v5, skipped_v5, expo_v5 = backtest(w_v5)

# ── Report ─────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Hurst-bull v5 — MULTI-ASSET  (cross-sectional top-3 ranking)            ║
║                                                                          ║
║  Top-3 eligible assets by H get inv-vol weights at 1.5× gross.           ║
║  If < 3 eligible: fall back to IVOL-5 B&H at 1.0× gross (no cash).       ║
║  Windows: IS 2008-2020  |  OOS 2021-2024                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def report(label, start, end):
    print(f"\n── {label}  ({start} → {end}) ──")
    bench = spy_bh_s.loc[start:end]
    rows = []
    for nm, rr in [("SPY Buy & Hold",        spy_bh_s.loc[start:end]),
                   ("EW-5 Buy & Hold",       ew5_bh_s.loc[start:end]),
                   ("IVOL-5 Buy & Hold",     ivol_bh_s.loc[start:end]),
                   ("MA v5  top-3 rank",     ret_v5.loc[start:end])]:
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
gross = expo_v5.abs().sum(axis=1)
n_names = (expo_v5 > _EPS).sum(axis=1)
print(f"\n── v5 exposure profile ──")
avg_expo = expo_v5.mean() * 100
print("  Mean weight by asset (%):")
for a in ASSETS:
    print(f"    {a:<5s} {avg_expo[a]:5.1f}%")
print(f"  Gross book:   mean={gross.mean():.2f}×   "
      f"p25={gross.quantile(0.25):.2f}×   p75={gross.quantile(0.75):.2f}×   "
      f"max={gross.max():.2f}×")
print(f"  Weeks in top-3 mode (1.5× gross):       "
      f"{int((gross > LEV_BASE + _EPS).sum())} / {len(gross)}")
print(f"  Weeks in fallback IVOL-5 mode (1.0×):   "
      f"{int(((gross > _EPS) & (gross <= LEV_BASE + _EPS)).sum())} / {len(gross)}")
print(f"  Dead-zone skips (Σ|Δw| < 5%):           {skipped_v5} / {len(rebal)}")
print(f"  Rebalances that changed the book:       {flips_v5} / {len(rebal)}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.suptitle("Hurst-bull v5 (cross-sectional top-3)  ·  IS 2008–2020  ·  OOS 2021–2024",
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
          (ret_v5,    "#6A1B9A", "-",  "MA v5")]

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
ax_e.stackplot(expo_v5.index,
               [expo_v5[a].values for a in ASSETS],
               colors=[colors[a] for a in ASSETS],
               labels=ASSETS, alpha=0.85)
ax_e.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_e.axhline(LEV_FULL, color="k", lw=0.5, ls=":", alpha=0.5)
ax_e.axhline(LEV_BASE, color="k", lw=0.5, ls=":", alpha=0.5)
ax_e.set_title("v5 per-asset exposure")
ax_e.yaxis.set_major_formatter(_PCT); ax_e.legend(ncol=5, fontsize=9, loc="upper left")

ax_g.plot(gross.index, gross.values, color="#6A1B9A", lw=1.0, label="v5 gross book")
ax_g.axhline(LEV_FULL, color="k", lw=0.5, ls=":", alpha=0.5)
ax_g.axhline(LEV_BASE, color="k", lw=0.5, ls=":", alpha=0.5)
ax_g.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_g.set_title("v5 gross exposure (1.0× fallback ↔ 1.5× top-3)")
ax_g.yaxis.set_major_formatter(_PCT); ax_g.set_ylim(0.9, LEV_FULL * 1.05)

fig.savefig(RESULTS_DIR / "hurst_multiasset_v5.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/hurst_multiasset_v5.png")
print("\n✓  Done.")
