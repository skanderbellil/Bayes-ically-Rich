#!/usr/bin/env python3
"""
Hurst-bull × 1.5×  —  dynamic parameter selection.

The fixed version of the strategy (H>0.55, ERL>15) is partly instrument-calibrated:
it happens to be near SPY's sample mean H (0.573) and a reasonable ERL veto. To
reduce overfit risk and make the rule transportable to other instruments, swap
the fixed cutoffs for rolling self-quantiles — each cutoff becomes "is the
current reading in the top/bottom X% of its own recent history?"

Five variants compared:

  (A)  FIXED       H>0.55,  ERL>15          (the baseline from prior commits)
  (B)  ADAPTIVE    H > q60(H|past 3Y),   ERL > q25(ERL|past 3Y)
  (C)  WALK-FWD    re-optimize (H_thr, erl_kill) every quarter on last 3Y,
                   apply for the next 13 weeks. Grid Sharpe-maximised.
  (D)  ENSEMBLE    average the weights from 5 (H_thr, erl_kill) configs.
                   Hedges against picking the wrong single threshold.
  (E)  H-ADAPTIVE  only H threshold is adaptive (q-calibrated to match the
                   fixed baseline's historical activity rate); ERL fixed at 15
                   (since adaptive-ERL tends to over-filter on this data).

Everything else held fixed at baseline:
  rolling-Hurst window = 252d
  drift filter         = SPY > SMA200
  leverage             = 1.5×
  transaction cost     = 5bp per unit weight change
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

RF, BT_START, BT_END = 0.04, "2016-01-01", "2024-12-31"
LOOKBACK, VOL_WIN, _ANN, _EPS = 252, 21, 252, 1e-8

HURST_WIN    = 252
DRIFT_WIN    = 200
LEVERAGE     = 1.50
TC           = 0.0005
ADAPT_WIN    = 756    # 3 years of daily data for rolling quantiles
HURST_Q      = 0.60   # "trending" = H above its own 60th percentile (i.e. top 40%)
ERL_Q        = 0.25   # "fresh regime" = ERL below its own 25th percentile

# Hurst (R/S) primitives are imported from posterioralpha.research.hurst above.

# ── Data ───────────────────────────────────────────────────────────────────
df   = load_portfolio_prices()
spy  = df["SPY"].pct_change().dropna()
spy_bt = spy.loc[BT_START:BT_END].rename("SPY")
px   = df["SPY"].reindex(spy.index).ffill()
sma  = px.rolling(DRIFT_WIN, min_periods=DRIFT_WIN).mean()

logger.info("BOCPD …")
_, _erl = precompute_bocpd(spy.values, hazard=1/252)
erl_series = pd.Series(_erl, index=spy.index)

logger.info(f"Rolling Hurst (W={HURST_WIN}) …")
H = pd.Series(rolling_hurst(spy.values, HURST_WIN), index=spy.index)

# Rolling self-quantiles for the adaptive variant — use *past* observations only.
# .rolling(...).quantile() uses a trailing window, so no look-ahead.
H_q60   = H.rolling(ADAPT_WIN, min_periods=252).quantile(HURST_Q)
ERL_q25 = erl_series.rolling(ADAPT_WIN, min_periods=252).quantile(ERL_Q)

# Rebalance grid
rebal = spy.resample("W-FRI").last().index
rebal = rebal[rebal > spy.index[LOOKBACK + VOL_WIN]]
rebal = rebal[(rebal >= BT_START) & (rebal <= BT_END)]

# ── Core loop helper ──────────────────────────────────────────────────────
def backtest(weight_of_t):
    """`weight_of_t(t)` returns the target weight for rebalance date t."""
    port, dates = [], []
    prev = 0.0; flips = 0
    for i, t in enumerate(rebal):
        w = float(weight_of_t(t))
        tc_cost = TC * abs(w - prev)
        if abs(w - prev) > 1e-9: flips += 1
        nxt = rebal[i+1] if i+1 < len(rebal) else spy.index[-1]
        per = spy.loc[t:nxt].iloc[1:]
        if len(per) == 0: continue
        pf = per.values * w; pf = pf.copy(); pf[0] -= tc_cost
        port.extend(pf.tolist()); dates.extend(per.index.tolist())
        prev = w
    return pd.Series(port, index=dates).loc[BT_START:BT_END], flips

def _safe(s, t, default):
    v = s.loc[:t].iloc[-1] if len(s.loc[:t]) else default
    return float(v) if pd.notna(v) and np.isfinite(v) else default

# ── (A) FIXED baseline ─────────────────────────────────────────────────────
def w_fixed(t):
    h = _safe(H,  t, 0.5)
    e = _safe(erl_series, t, 1e9)
    p = _safe(px,  t, 0.0)
    s = _safe(sma, t, np.inf)
    bull = (h > 0.55) and (p > s) and (e > 15.0)
    return LEVERAGE if bull else 0.0

# ── (B) ADAPTIVE self-quantile thresholds ──────────────────────────────────
# If a quantile isn't available yet (warmup), fall back to the fixed cutoff
# — ensures the strategy is well-defined from day one.
def w_adapt(t):
    h = _safe(H,  t, 0.5)
    e = _safe(erl_series, t, 1e9)
    p = _safe(px,  t, 0.0)
    s = _safe(sma, t, np.inf)
    h_thr = _safe(H_q60,   t, 0.55)
    e_thr = _safe(ERL_q25, t, 15.0)
    bull = (h > h_thr) and (p > s) and (e > e_thr)
    return LEVERAGE if bull else 0.0

# ── (C) WALK-FORWARD grid search ───────────────────────────────────────────
# Every 13 weeks (~quarterly), grid-search (H_thr, erl_kill) on the last 3Y
# and apply the Sharpe-best pair for the next quarter. Scoring function uses
# a simplified in-sample backtest over the window only.
H_GRID   = [0.50, 0.53, 0.55, 0.57, 0.60]
ERL_GRID = [5.0, 10.0, 15.0, 20.0, 30.0]
WF_WIN   = 756   # 3Y in days
WF_FREQ  = 13    # re-optimise every 13 rebalances

def _score(h_thr, e_thr, returns, H_, erl_, px_, sma_):
    """Fast vectorised Sharpe on the window given (h_thr, e_thr)."""
    bull = (H_ > h_thr) & (px_ > sma_) & (erl_ > e_thr)
    # daily weight = LEV * bull_on_previous_rebal_day
    w = bull.astype(float) * LEVERAGE
    w = w.shift(1).fillna(0.0)
    pf = returns * w
    if pf.std() < _EPS:
        return -np.inf
    return float((pf.mean() - RF/_ANN) / pf.std() * np.sqrt(_ANN))

# Cache optimised params per-quarter
wf_params = {}
for i, t in enumerate(rebal):
    if i % WF_FREQ != 0:
        continue
    end = t
    start = end - pd.Timedelta(days=int(WF_WIN * 1.5))
    mask = (spy.index >= start) & (spy.index < end)
    if mask.sum() < 252:
        wf_params[t] = (0.55, 15.0); continue
    rr = spy.loc[mask]; HH = H.loc[mask]; ee = erl_series.loc[mask]
    pp = px.loc[mask]; ss = sma.loc[mask]
    best = (0.55, 15.0); best_s = -np.inf
    for h_thr in H_GRID:
        for e_thr in ERL_GRID:
            s_ = _score(h_thr, e_thr, rr, HH, ee, pp, ss)
            if s_ > best_s:
                best_s = s_; best = (h_thr, e_thr)
    wf_params[t] = best

# Forward-fill optimised params between re-optimisation dates.
wf_h = pd.Series({t: wf_params[t][0] for t in wf_params})
wf_e = pd.Series({t: wf_params[t][1] for t in wf_params})
wf_h = wf_h.reindex(rebal).ffill().fillna(0.55)
wf_e = wf_e.reindex(rebal).ffill().fillna(15.0)

def w_wf(t):
    h = _safe(H,  t, 0.5)
    e = _safe(erl_series, t, 1e9)
    p = _safe(px,  t, 0.0)
    s = _safe(sma, t, np.inf)
    h_thr = float(wf_h.loc[t])
    e_thr = float(wf_e.loc[t])
    bull = (h > h_thr) and (p > s) and (e > e_thr)
    return LEVERAGE if bull else 0.0

# ── (D) ENSEMBLE across multiple threshold configs ─────────────────────────
# Rather than commit to one (H_thr, erl_kill) pair, average the binary weights
# from 5 reasonable configs. Each contributes 1/5 of the target weight if its
# bull condition is met — so the effective weight is a count of how many
# configs agree, scaled to [0, 1.5×]. This smooths sensitivity to the exact
# threshold choice without picking a tuned value.
ENSEMBLE_CFGS = [
    (0.50, 10.0),
    (0.53, 15.0),
    (0.55, 15.0),   # the fixed baseline
    (0.57, 20.0),
    (0.60, 25.0),
]
def w_ensemble(t):
    h = _safe(H,  t, 0.5)
    e = _safe(erl_series, t, 1e9)
    p = _safe(px,  t, 0.0)
    s = _safe(sma, t, np.inf)
    if p <= s:   # drift filter is shared — no point ensembling through a clear exit signal
        return 0.0
    votes = sum(1 for (h_thr, e_thr) in ENSEMBLE_CFGS
                if (h > h_thr) and (e > e_thr))
    return LEVERAGE * (votes / len(ENSEMBLE_CFGS))

# ── (E) H-only adaptive, ERL fixed ─────────────────────────────────────────
# Rolling q47(H) is calibrated so the *historical* activity rate matches the
# fixed baseline's — 0.55 corresponds to roughly the 53rd percentile of SPY's
# own rolling Hurst values. This is the minimal adaptive tweak: the method
# auto-calibrates to a new instrument's H distribution while preserving the
# selectivity of the proven baseline. ERL kept at 15 (fixed-ERL won ablation).
H_q47 = H.rolling(ADAPT_WIN, min_periods=252).quantile(0.47)
def w_h_adapt(t):
    h = _safe(H,  t, 0.5)
    e = _safe(erl_series, t, 1e9)
    p = _safe(px,  t, 0.0)
    s = _safe(sma, t, np.inf)
    h_thr = _safe(H_q47, t, 0.55)
    bull = (h > h_thr) and (p > s) and (e > 15.0)
    return LEVERAGE if bull else 0.0

# ── Run all three ──────────────────────────────────────────────────────────
logger.info("Running fixed / adaptive / walk-forward / ensemble / H-adapt …")
ret_fixed, flips_fixed = backtest(w_fixed)
ret_adapt, flips_adapt = backtest(w_adapt)
ret_wf,    flips_wf    = backtest(w_wf)
ret_ens,   flips_ens   = backtest(w_ensemble)
ret_ha,    flips_ha    = backtest(w_h_adapt)

# ── Report ─────────────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║  Hurst-bull × 1.5× — dynamic parameter selection                        ║
║                                                                          ║
║  STRATEGY IN ONE BREATH                                                  ║
║  "Go long SPY with 1.5× leverage only when three agree: trend strength  ║
║   (Hurst) says the market is persisting, the 200-day average says it's  ║
║   heading up, and the Bayesian change-detector says we're not inside a  ║
║   fresh regime break. Otherwise sit in cash."                            ║
║                                                                          ║
║  Five ways to choose the thresholds:                                    ║
║    FIXED     literature-calibrated numbers (H>0.55, ERL>15)             ║
║    ADAPTIVE  both thresholds are rolling self-quantiles (transportable) ║
║    WALK-FWD  re-tune every quarter on the last 3 years of data          ║
║    ENSEMBLE  average the weights of 5 threshold configs (no single peak)║
║    H-ADAPT   adapt H only (calibrated to match fixed), keep ERL at 15   ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

def row(label, ret, flips):
    m = compute_metrics(ret.loc[BT_START:BT_END], rf=RF, benchmark=spy_bt)
    return [label,
            f"{m['CAGR']:.2%}",    f"{m['Sharpe']:.2f}", f"{m['Sortino']:.2f}",
            f"{m['Max DD']:.2%}",  f"{m['Calmar']:.2f}", f"{m['Volatility']:.2%}",
            f"{int(flips)}"]

spy_m = compute_metrics(spy_bt, rf=RF)
rows = [
    ["SPY Buy & Hold",
     f"{spy_m['CAGR']:.2%}",   f"{spy_m['Sharpe']:.2f}", f"{spy_m['Sortino']:.2f}",
     f"{spy_m['Max DD']:.2%}", f"{spy_m['Calmar']:.2f}", f"{spy_m['Volatility']:.2%}", "—"],
    row("(A) FIXED     H>0.55, ERL>15",           ret_fixed, flips_fixed),
    row("(B) ADAPTIVE  H>q60(H), ERL>q25(ERL)",   ret_adapt, flips_adapt),
    row("(C) WALK-FWD  quarterly grid-search",    ret_wf,    flips_wf),
    row("(D) ENSEMBLE  mean of 5 threshold cfgs", ret_ens,   flips_ens),
    row("(E) H-ADAPT   H>q47(H), ERL fixed @15",  ret_ha,    flips_ha),
]
print(tabulate(rows,
               headers=["Strategy","CAGR","Sharpe","Sortino","Max DD","Calmar","Vol","Flips"],
               tablefmt="rounded_grid"))

# ── Subperiod ──────────────────────────────────────────────────────────────
print("\n── Subperiod robustness ──")
SUBS = [("2016-2019 pre-COVID", "2016-01-01", "2019-12-31"),
        ("2020-2024 post",       "2020-01-01", "2024-12-31"),
        ("2020 crash year",      "2020-01-01", "2020-12-31"),
        ("2022 bear year",       "2022-01-01", "2022-12-31")]
for lbl, s, e in SUBS:
    print(f"\n  {lbl}")
    rows = []
    for nm, r in [("SPY", spy_bt), ("fixed", ret_fixed),
                  ("adaptive", ret_adapt), ("walk-fwd", ret_wf),
                  ("ensemble", ret_ens), ("H-adapt", ret_ha)]:
        rr = r.loc[s:e]
        if rr.abs().sum() < 1e-9:
            rows.append([nm, "0.00% (cash)", "—", "—", "0.00%"])
        else:
            m = compute_metrics(rr, rf=RF)
            rows.append([nm, f"{m['CAGR']:.2%}", f"{m['Sharpe']:.2f}",
                         f"{m['Sortino']:.2f}", f"{m['Max DD']:.2%}"])
    print(tabulate(rows, headers=["","CAGR","Sharpe","Sortino","MaxDD"],
                   tablefmt="simple"))

# ── Adaptive threshold diagnostics ─────────────────────────────────────────
print("\n── Adaptive-quantile thresholds (observed at rebalance dates) ──")
h_thr_reb = H_q60.reindex(rebal, method="ffill").dropna()
e_thr_reb = ERL_q25.reindex(rebal, method="ffill").dropna()
print(f"  H  threshold (rolling q60):  mean={h_thr_reb.mean():.3f}  "
      f"min={h_thr_reb.min():.3f}  max={h_thr_reb.max():.3f}  "
      f"(fixed was 0.550)")
print(f"  ERL threshold (rolling q25): mean={e_thr_reb.mean():.1f}  "
      f"min={e_thr_reb.min():.1f}  max={e_thr_reb.max():.1f}  "
      f"(fixed was 15.0)")

print("\n── Walk-forward parameter path ──")
wf_df = pd.DataFrame({"H_thr": wf_h, "erl_kill": wf_e}).drop_duplicates()
print(f"  Re-optimised {len(wf_df)} times; unique (H_thr, erl_kill) pairs: "
      f"{wf_df.drop_duplicates().shape[0]}")
print("  Distribution of optimised values:")
print(f"    H_thr:     " + "  ".join(f"{v:.2f}={int((wf_h == v).sum())}"
                                       for v in sorted(wf_df['H_thr'].unique())))
print(f"    erl_kill:  " + "  ".join(f"{v:.0f}={int((wf_e == v).sum())}"
                                       for v in sorted(wf_df['erl_kill'].unique())))

# ── Plot ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle("Hurst-bull × 1.5×  ·  fixed vs adaptive vs walk-forward  ·  2016-2024",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)
ax_c = fig.add_subplot(gs[0, :])
ax_d = fig.add_subplot(gs[1, 0])
ax_s = fig.add_subplot(gs[1, 1])
ax_ht= fig.add_subplot(gs[2, 0])
ax_et= fig.add_subplot(gs[2, 1])

_PCT = mticker.FuncFormatter(lambda x,_: f"{x:.0%}")
_DOL = mticker.FuncFormatter(lambda x,_: f"${x:.2f}")

S = [(spy_bt,"#37474F","--","SPY"),
     (ret_fixed,"#6D4C41","-","Fixed"),
     (ret_adapt,"#00838F","-","Adaptive"),
     (ret_wf,   "#6A1B9A","-","Walk-fwd")]
for r,c,ls,lb in S:
    cum = (1+r).cumprod()
    ax_c.plot(cum.index, cum.values, color=c, lw=1.6, ls=ls, label=lb)
ax_c.set_yscale("log"); ax_c.yaxis.set_major_formatter(_DOL)
ax_c.set_title("Cumulative wealth (log)"); ax_c.legend()

for r,c,ls,lb in S:
    cum = (1+r).cumprod(); dd = (cum - cum.cummax())/cum.cummax()
    ax_d.plot(dd.index, dd.values, color=c, lw=1.2, ls=ls, label=lb)
ax_d.yaxis.set_major_formatter(_PCT); ax_d.set_title("Drawdown"); ax_d.legend(fontsize=8)

for r,c,ls,lb in S:
    exc = r - RF/252
    rs = exc.rolling(126).mean()/exc.rolling(126).std()*np.sqrt(252)
    ax_s.plot(rs.index, rs.values, color=c, lw=1.1, ls=ls, label=lb)
ax_s.axhline(0, color="k", lw=0.5, ls="--"); ax_s.axhline(1, color="g", lw=0.4, ls=":")
ax_s.set_title("Rolling 6-month Sharpe"); ax_s.legend(fontsize=8)

ax_ht.plot(H.index, H.values, color="#AD1457", lw=0.7, alpha=0.5, label="Rolling H")
ax_ht.plot(H_q60.index, H_q60.values, color="#00838F", lw=1.2, label="Adaptive q60(H)")
ax_ht.plot(wf_h.index, wf_h.values, color="#6A1B9A", lw=1.2, drawstyle="steps-post",
           label="Walk-fwd H_thr")
ax_ht.axhline(0.55, color="#6D4C41", lw=0.8, ls="--", label="Fixed 0.55")
ax_ht.set_title("Hurst threshold: fixed vs adaptive vs walk-fwd")
ax_ht.legend(fontsize=7, ncol=2)

ax_et.plot(erl_series.index, erl_series.values, color="#5C6BC0", lw=0.6, alpha=0.4, label="ERL")
ax_et.plot(ERL_q25.index, ERL_q25.values, color="#00838F", lw=1.2, label="Adaptive q25(ERL)")
ax_et.plot(wf_e.index, wf_e.values, color="#6A1B9A", lw=1.2, drawstyle="steps-post",
           label="Walk-fwd erl_kill")
ax_et.axhline(15, color="#6D4C41", lw=0.8, ls="--", label="Fixed 15")
ax_et.set_ylim(0, 200)
ax_et.set_title("ERL-kill threshold: fixed vs adaptive vs walk-fwd")
ax_et.legend(fontsize=7, ncol=2)

fig.savefig(RESULTS_DIR / "hurst_bull_adaptive.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/hurst_bull_adaptive.png")
print("\n✓  Done.")
