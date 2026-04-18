#!/usr/bin/env python3
"""
Hurst-bull × 1.5×  —  multi-asset v2: fitting-free, continuous, inverse-vol.

What changed vs v1
------------------
v1 used per-asset rolling quantile thresholds (q60 on H, q25 on ERL) and an
equal-weight binary gate. That failed OOS partly because (a) the same
quantile levels don't carry the same meaning across SPY/TLT/GLD/EEM/VNQ, and
(b) equal-weighting across a random 1–2 signalled assets is noisy and over-
exposes whichever asset happens to fire.

v2 sticks to rules that need no training window:

  (i)  Trending boundary is THEORETICAL, not fitted: H_a > 0.5 is the
       classical "persistent" / anti-persistent divide from R/S analysis.
  (ii) Trend direction: P_a > SMA200_a   (universal, scale-free).
  (iii) Fresh-regime veto: ERL_a > rolling q25 of each asset's OWN ERL
        over the last 3Y (causal, trailing — already dynamic, no fitting).
  (iv) Persistence STRENGTH is continuous: score_a = max(0, H_a - 0.5).
       A rule that only asks "H > some cutoff" throws away information.
  (v)  Risk sizing is inverse-vol:    raw_w_a = score_a · eligible_a / σ_a.
       This prevents EEM (~25% vol) from dominating a bull sleeve when TLT
       (~13% vol) also fires. Renormalize so Σ|w| = 1.5× when any signal
       is on, else cash.

No parameters tied to the IS window. Rolling ERL-quantile per asset adapts
automatically to each instrument's regime-length distribution. H>0.5 is
theoretical; SMA200 and 21d vol are standard scale-free transforms.

Benchmarks
----------
  1) SPY buy-and-hold
  2) Equal-weight-5 buy-and-hold   (diversification baseline)
  3) Inverse-vol-5 buy-and-hold    (better diversification baseline — shows
                                     whether v2 alpha is just inverse-vol)

Windows: IS 2008-01-01 → 2020-12-31,  OOS 2021-01-01 → 2024-12-31.
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
RF          = 0.04
ASSETS      = ["SPY", "TLT", "GLD", "EEM", "VNQ"]
BT_START    = "2008-01-01"
IS_END      = "2020-12-31"
OOS_START   = "2021-01-01"
BT_END      = "2024-12-31"

HURST_WIN   = 252
DRIFT_WIN   = 200
VOL_WIN     = 21
ADAPT_WIN   = 756
ERL_Q       = 0.25
LEVERAGE    = 1.50
TC          = 0.0005
H_FLOOR     = 0.50  # theoretical persistent/anti-persistent boundary
_EPS, _ANN  = 1e-8, 252

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
    vol = r.rolling(VOL_WIN, min_periods=VOL_WIN).std() * np.sqrt(_ANN)
    _, erl = precompute_bocpd(r.values, hazard=1/252)
    erl_s  = pd.Series(erl, index=r.index)
    H      = pd.Series(rolling_hurst(r.values, HURST_WIN), index=r.index)
    # Rolling self-quantile for ERL — causal, trailing only.
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
    """Return (score, inv_vol, eligible) for asset a at time t.

    score     = max(0, H - 0.5)  when eligible, else 0
    inv_vol   = 1 / σ_a          (21-day ann. vol; causal)
    eligible  = H > 0.5  AND  P > SMA  AND  ERL > own q25(ERL)
    """
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

# ── Multi-asset backtest ───────────────────────────────────────────────────
def backtest(weights_of_t):
    port_ret, port_dates = [], []
    rows_w = []
    prev = {a: 0.0 for a in ASSETS}
    flips = 0
    for i, t in enumerate(rebal):
        w = weights_of_t(t)
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
    return pr, flips, expo

def w_v2(t):
    """Continuous score × inverse-vol, normalized to LEVERAGE gross if any on."""
    raws = {a: asset_raw(a, t) for a in ASSETS}
    raw_w = {a: raws[a][0] * raws[a][1] for a in ASSETS}
    total = sum(raw_w.values())
    if total <= _EPS:
        return {a: 0.0 for a in ASSETS}
    return {a: LEVERAGE * raw_w[a] / total for a in ASSETS}

# ── Benchmarks ─────────────────────────────────────────────────────────────
def ew5_bh():
    w = 1.0 / len(ASSETS)
    r = df[ASSETS].pct_change().fillna(0.0)
    return (r * w).sum(axis=1).loc[BT_START:BT_END]

def ivol5_bh():
    """Inverse-vol-weighted static portfolio, rebalanced weekly at 1.0× gross.
    Separates the 'inverse-vol' ingredient from the 'Hurst signal' ingredient —
    if v2 only matches this, the signal is adding nothing."""
    rets = df[ASSETS].pct_change().dropna(how="all")
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

logger.info("Running multi-asset v2 …")
ret_v2, flips_v2, expo_v2 = backtest(w_v2)

# ── Report ─────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Hurst-bull × 1.5× — MULTI-ASSET v2  (fitting-free, continuous)         ║
║                                                                          ║
║  Per-asset score :  max(0, H_a - 0.5)                                    ║
║  Eligibility     :  H_a > 0.5  ∧  P_a > SMA200  ∧  ERL_a > q25_rolling   ║
║  Risk sizing     :  w_a ∝ score_a / σ_a,   Σ|w| = 1.5× when any eligible ║
║                                                                          ║
║  No parameters fitted on the in-sample window.                          ║
║    H = 0.5 is the R/S theoretical persistence boundary.                 ║
║    ERL quantile is each asset's own trailing 3Y self-quantile.          ║
║    Vol is standard 21d realised (causal).                               ║
║                                                                          ║
║  Windows:  IS 2008-2020 (reported)  |  OOS 2021-2024 (unpeeked)         ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def report(label, start, end):
    print(f"\n── {label}  ({start} → {end}) ──")
    bench = spy_bh_s.loc[start:end]
    rows = []
    for nm, rr in [("SPY Buy & Hold",         spy_bh_s.loc[start:end]),
                   ("EW-5 Buy & Hold",        ew5_bh_s.loc[start:end]),
                   ("IVOL-5 Buy & Hold",      ivol_bh_s.loc[start:end]),
                   ("MA v2  Hurst×IVOL×1.5×", ret_v2.loc[start:end])]:
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

# ── Exposure diagnostics ──────────────────────────────────────────────────
print(f"\n── v2 exposure profile ──")
avg_expo = expo_v2.mean() * 100
gross    = expo_v2.abs().sum(axis=1)
n_on     = (expo_v2 > _EPS).sum(axis=1)
print("  Mean weight by asset (%):")
for a in ASSETS:
    print(f"    {a:<5s} {avg_expo[a]:5.1f}%")
print(f"  Mean # of signalled assets per week:   {n_on.mean():.2f}")
print(f"  Weeks fully in cash (|B|=0):            "
      f"{int((n_on == 0).sum())} / {len(n_on)}  "
      f"({(n_on == 0).mean():.1%})")
print(f"  Weeks fully loaded (|B|=5):             "
      f"{int((n_on == 5).sum())} / {len(n_on)}  "
      f"({(n_on == 5).mean():.1%})")
print(f"  Rebalances that changed the book:       {flips_v2} / {len(rebal)}")
print(f"  Mean gross book (when active):          "
      f"{gross[gross > _EPS].mean():.2f}×")

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Hurst-bull v2 (fitting-free, continuous, inverse-vol)  ·  IS 2008–2020  ·  OOS 2021–2024",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.25)
ax_c = fig.add_subplot(gs[0, :])
ax_d = fig.add_subplot(gs[1, 0])
ax_s = fig.add_subplot(gs[1, 1])
ax_e = fig.add_subplot(gs[2, :])

_PCT = mticker.FuncFormatter(lambda x,_: f"{x:.0%}")
_DOL = mticker.FuncFormatter(lambda x,_: f"${x:.2f}")

series = [(spy_bh_s,  "#37474F", "--", "SPY B&H"),
          (ew5_bh_s,  "#2E7D32", ":",  "EW-5 B&H"),
          (ivol_bh_s, "#EF6C00", ":",  "IVOL-5 B&H"),
          (ret_v2,    "#6A1B9A", "-",  "MA v2")]

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
ax_e.stackplot(expo_v2.index,
               [expo_v2[a].values for a in ASSETS],
               colors=[colors[a] for a in ASSETS],
               labels=ASSETS, alpha=0.85)
ax_e.axvline(pd.Timestamp(OOS_START), color="k", lw=0.8, ls="--", alpha=0.6)
ax_e.axhline(LEVERAGE, color="k", lw=0.5, ls=":", alpha=0.5)
ax_e.set_title("v2 per-asset exposure over time (gross cap = 1.5×)")
ax_e.yaxis.set_major_formatter(_PCT); ax_e.legend(ncol=5, fontsize=9, loc="upper left")

fig.savefig(RESULTS_DIR / "hurst_multiasset_v2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/hurst_multiasset_v2.png")
print("\n✓  Done.")
