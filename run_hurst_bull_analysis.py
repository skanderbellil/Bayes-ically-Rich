#!/usr/bin/env python3
"""
Hurst-bull × fixed-leverage — deep diagnostics & sensitivity analysis.

Baseline strategy (the winner from run_spy_timing.py):

    w_t = 1.5 × 1{H_252 > 0.55  AND  SPY > SMA200  AND  ERL > 15}

where
    H_252 = rolling 252-day R/S Hurst exponent of SPY returns
    SMA200 = 200-day simple moving average of SPY price
    ERL    = expected run length from BOCPD on SPY returns
    weight cap = 1.5×, transaction cost = 5bp per unit weight change.

This script:
  1.  Confirms baseline includes ERL-kill, then runs a no-ERL counterfactual.
  2.  Ablates each of {Hurst, SMA-drift, ERL} to isolate each component's
      contribution.
  3.  Sweeps the 6 key parameters one-at-a-time (H threshold, H window,
      drift SMA window, ERL-kill threshold, leverage, transaction cost).
      Flat or monotone curves → not overfit; sharp peaks at the baseline
      value → suspicious.
  4.  Deep baseline diagnostics: annual returns, drawdown episodes,
      turnover / trade count, rolling Sharpe.

All results on SPY daily, 2016-01-01 to 2024-12-31, weekly (Fri) rebalance.
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

from src.metrics import compute_metrics
from src.regime_models import precompute_bocpd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)

# ── Baseline config (the winner) ──────────────────────────────────────────
RF       = 0.04
BT_START = "2016-01-01"
BT_END   = "2024-12-31"
LOOKBACK = 252
VOL_WIN  = 21
_ANN     = 252
_EPS     = 1e-8

BASE = dict(
    hurst_win   = 252,    # rolling R/S window
    hurst_thr   = 0.55,   # H threshold for "trending"
    drift_win   = 200,    # SMA window for drift filter (None disables)
    erl_kill    = 15.0,   # veto if ERL < this (None disables)
    leverage    = 1.50,   # fixed leverage while bull signal is on
    tc          = 0.0005, # transaction cost per unit of weight change (5bp)
)

print("""
╔══════════════════════════════════════════════════════════════╗
║   Hurst-bull × 1.5× leverage  ·  diagnostics & sensitivity  ║
║   Ablation → sweeps → overfitting checks → baseline deep-dive║
╚══════════════════════════════════════════════════════════════╝""")
print(f"  Baseline: H>{BASE['hurst_thr']} over {BASE['hurst_win']}d, "
      f"SMA{BASE['drift_win']} drift, ERL≥{BASE['erl_kill']:.0f}, "
      f"{BASE['leverage']:.2f}× lev, TC={BASE['tc']*1e4:.0f}bp\n")


# ── R/S Hurst ──────────────────────────────────────────────────────────────
def rs_hurst(x: np.ndarray) -> float:
    """R/S Hurst; see run_spy_timing.py for references."""
    x = np.asarray(x, dtype=float)
    N = len(x)
    if N < 20 or not np.all(np.isfinite(x)):
        return 0.5
    scales = np.unique(np.logspace(np.log10(10), np.log10(N // 2), 6).astype(int))
    rs_list, good = [], []
    for n in scales:
        if n < 4 or n > N // 2:
            continue
        K = N // n
        vals = []
        for k in range(K):
            seg = x[k * n:(k + 1) * n]
            z   = np.cumsum(seg - seg.mean())
            R   = float(z.max() - z.min())
            S   = float(seg.std(ddof=1))
            if S > _EPS:
                vals.append(R / S)
        if vals:
            rs_list.append(float(np.mean(vals)))
            good.append(int(n))
    if len(good) < 3:
        return 0.5
    slope, _ = np.polyfit(np.log(good), np.log(rs_list), 1)
    return float(slope)


def rolling_hurst(returns: pd.Series, win: int) -> pd.Series:
    r = returns.values
    H = np.full(len(r), np.nan)
    for t in range(win, len(r) + 1):
        H[t - 1] = rs_hurst(r[t - win:t])
    return pd.Series(H, index=returns.index)


# ── Data ───────────────────────────────────────────────────────────────────
df     = pd.read_csv("portfolio_data.csv", parse_dates=["Date"], index_col="Date").sort_index()
spy    = df["SPY"].pct_change().dropna()
spy_bt = spy.loc[BT_START:BT_END].rename("SPY")
spy_price = df["SPY"].reindex(spy.index).ffill()

logger.info("Pre-computing BOCPD on SPY …")
_cp, _erl = precompute_bocpd(spy.values, hazard=1 / 252)
erl_series = pd.Series(_erl, index=spy.index)

# Pre-cache Hurst series for every window we'll need in the sweeps, keyed by window.
HURST_WINS_NEEDED = sorted({BASE["hurst_win"], 126, 189, 252, 378, 504})
logger.info(f"Pre-computing rolling Hurst for windows {HURST_WINS_NEEDED} …")
HURST_CACHE = {w: rolling_hurst(spy, w) for w in HURST_WINS_NEEDED}

SMA_WINS_NEEDED = sorted({BASE["drift_win"], 100, 150, 200, 250, 300})
SMA_CACHE = {w: spy_price.rolling(w, min_periods=w).mean() for w in SMA_WINS_NEEDED}

# Weekly rebalance grid
try:
    rebal_dates = spy.resample("W-FRI").last().index
except Exception:
    rebal_dates = spy.resample("W").last().index
rebal_dates = rebal_dates[rebal_dates > spy.index[LOOKBACK + VOL_WIN]]
rebal_dates = rebal_dates[(rebal_dates >= BT_START) & (rebal_dates <= BT_END)]


# ── Parameterised strategy runner ──────────────────────────────────────────

def run_hurst_bull(hurst_win, hurst_thr, drift_win, erl_kill, leverage, tc):
    """Returns (return series, diagnostics DataFrame, trade count)."""
    H   = HURST_CACHE[hurst_win]
    SMA = SMA_CACHE[drift_win] if drift_win is not None else None

    port_rets, ret_dates, diag = [], [], []
    prev_w = 0.0
    trades = 0

    for i, t in enumerate(rebal_dates):
        h = H.loc[:t].iloc[-1] if pd.notna(H.loc[:t].iloc[-1]) else 0.5
        p = float(spy_price.loc[:t].iloc[-1])
        s = float(SMA.loc[:t].iloc[-1]) if SMA is not None else -np.inf
        e = float(erl_series.loc[:t].iloc[-1])

        bull = (h > hurst_thr)
        if drift_win is not None:
            bull = bull and (p > s)
        if erl_kill is not None:
            bull = bull and (e > erl_kill)

        w = leverage if bull else 0.0

        tc_cost = tc * abs(w - prev_w)
        if abs(w - prev_w) > 1e-9:
            trades += 1

        next_t = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else spy.index[-1]
        period = spy.loc[t:next_t].iloc[1:]
        if len(period) == 0:
            continue
        pf = period.values * w
        pf = pf.copy()
        pf[0] -= tc_cost
        port_rets.extend(pf.tolist())
        ret_dates.extend(period.index.tolist())
        diag.append((t, w, h, p, s, e))
        prev_w = w

    ret = pd.Series(port_rets, index=ret_dates).loc[BT_START:BT_END]
    dg  = pd.DataFrame(diag, columns=["date","weight","hurst","price","sma","erl"]).set_index("date")
    return ret, dg, trades


def metrics_row(ret, label, tag="", trades=None):
    m = compute_metrics(ret, rf=RF, benchmark=spy_bt)
    row = [label, tag,
           f"{m.get('CAGR',0):.2%}",
           f"{m.get('Sharpe',0):.2f}",
           f"{m.get('Sortino',0):.2f}",
           f"{m.get('Max DD',0):.2%}",
           f"{m.get('Calmar',0):.2f}",
           f"{m.get('Volatility',0):.2%}"]
    if trades is not None:
        row.append(f"{trades}")
    return row, m


HEADERS      = ["Config","Setting","CAGR","Sharpe","Sortino","Max DD","Calmar","Vol"]
HEADERS_TR   = HEADERS + ["Trades"]


# ── Baseline + SPY ─────────────────────────────────────────────────────────
print("═" * 78)
print("  1. BASELINE  vs  SPY buy-and-hold")
print("═" * 78)
ret_base, dg_base, trades_base = run_hurst_bull(**BASE)
spy_m = compute_metrics(spy_bt, rf=RF)
rows = []
rows.append(["SPY Buy & Hold","",
             f"{spy_m['CAGR']:.2%}", f"{spy_m['Sharpe']:.2f}", f"{spy_m['Sortino']:.2f}",
             f"{spy_m['Max DD']:.2%}", f"{spy_m['Calmar']:.2f}", f"{spy_m['Volatility']:.2%}", "—"])
base_row, base_m = metrics_row(ret_base, "Hurst-bull × 1.5× (baseline)", "", trades_base)
rows.append(base_row)
print(tabulate(rows, headers=HEADERS_TR, tablefmt="simple"))


# ── Component ablation ─────────────────────────────────────────────────────
print("\n" + "═" * 78)
print("  2. COMPONENT ABLATION  (which pieces are actually earning their keep?)")
print("═" * 78)
ablations = [
    ("Hurst only",                       dict(BASE, drift_win=None, erl_kill=None)),
    ("Hurst + drift (no ERL)",           dict(BASE, erl_kill=None)),
    ("Hurst + ERL (no drift)",           dict(BASE, drift_win=None)),
    ("Hurst + drift + ERL (baseline)",   dict(BASE)),
    ("Drift only (no Hurst, SMA200×1.5)",dict(BASE, hurst_thr=-1.0, erl_kill=None)),
    ("Drift + ERL (no Hurst)",           dict(BASE, hurst_thr=-1.0)),
]
rows = []
for label, cfg in ablations:
    r, _, tr = run_hurst_bull(**cfg)
    row, _   = metrics_row(r, label, "", tr)
    rows.append(row)
print(tabulate(rows, headers=HEADERS_TR, tablefmt="simple"))


# ── Sensitivity sweeps ─────────────────────────────────────────────────────
print("\n" + "═" * 78)
print("  3. SENSITIVITY SWEEPS  (flat ≈ robust, peaked = overfit)")
print("═" * 78)

def sweep(param, values, pretty):
    print(f"\n  3.{pretty['idx']}  {pretty['name']}  (baseline = {BASE[param]})")
    rows = []
    for v in values:
        cfg = dict(BASE); cfg[param] = v
        r, _, _ = run_hurst_bull(**cfg)
        row, _  = metrics_row(r, f"{param}={v}", "")
        rows.append(row)
    print(tabulate(rows, headers=HEADERS[:-1], tablefmt="simple"))

sweep("hurst_thr", [0.50, 0.52, 0.55, 0.57, 0.60, 0.63],
      dict(idx=1, name="Hurst threshold"))
sweep("hurst_win", [126, 189, 252, 378, 504],
      dict(idx=2, name="Hurst window (trading days)"))
sweep("drift_win", [100, 150, 200, 250, 300],
      dict(idx=3, name="Drift-filter SMA window"))
sweep("erl_kill",  [None, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0],
      dict(idx=4, name="ERL-kill threshold"))
sweep("leverage",  [1.00, 1.25, 1.50, 1.75, 2.00],
      dict(idx=5, name="Fixed leverage multiplier"))
sweep("tc",        [0.0, 0.0002, 0.0005, 0.0010, 0.0020, 0.0050],
      dict(idx=6, name="Transaction cost (per unit weight change)"))


# ── Subperiod robustness of baseline ───────────────────────────────────────
print("\n" + "═" * 78)
print("  4. SUBPERIOD ROBUSTNESS  (does the edge persist, or is it one regime?)")
print("═" * 78)
SUBS = [
    ("2016-2019  (pre-COVID bull)",                   "2016-01-01", "2019-12-31"),
    ("2020-2024  (COVID + 2022 bear + 2023-24 rally)","2020-01-01", "2024-12-31"),
    ("2020       (COVID crash)",                      "2020-01-01", "2020-12-31"),
    ("2022       (Fed-hike bear)",                    "2022-01-01", "2022-12-31"),
]
rows = []
for lbl, s, e in SUBS:
    r_base = ret_base.loc[s:e]
    r_spy  = spy_bt.loc[s:e]
    ms = compute_metrics(r_spy, rf=RF)
    # If the strategy never traded in this subperiod, σ=0 blows up Sharpe.
    # Flag it explicitly rather than printing a garbage number.
    if r_base.abs().sum() < 1e-9:
        hb_str = (f"{0:.2%} (in cash) / {ms['CAGR']:.2%}",
                  "— (in cash) / " + f"{ms['Sharpe']:.2f}",
                  "0.00% (in cash) / " + f"{ms['Max DD']:.2%}")
    else:
        mb = compute_metrics(r_base, rf=RF)
        hb_str = (f"{mb['CAGR']:.2%} / {ms['CAGR']:.2%}",
                  f"{mb['Sharpe']:.2f} / {ms['Sharpe']:.2f}",
                  f"{mb['Max DD']:.2%} / {ms['Max DD']:.2%}")
    rows.append([lbl, *hb_str])
print(tabulate(rows,
               headers=["Period","CAGR (Hurst / SPY)","Sharpe (Hurst / SPY)","MaxDD (Hurst / SPY)"],
               tablefmt="simple"))


# ── Annual returns ─────────────────────────────────────────────────────────
print("\n" + "═" * 78)
print("  5. ANNUAL RETURNS")
print("═" * 78)
annual = pd.DataFrame({
    "Hurst-bull×1.5": ret_base.groupby(ret_base.index.year).apply(lambda x: (1+x).prod() - 1),
    "SPY B&H":        spy_bt.groupby(spy_bt.index.year).apply(lambda x: (1+x).prod() - 1),
})
annual["Δ"] = annual["Hurst-bull×1.5"] - annual["SPY B&H"]
print(tabulate(
    [[y] + [f"{v:.2%}" for v in row] for y, row in annual.iterrows()],
    headers=["Year","Hurst-bull×1.5","SPY B&H","Δ"], tablefmt="simple"))


# ── Drawdown episodes ──────────────────────────────────────────────────────
print("\n" + "═" * 78)
print("  6. TOP DRAWDOWNS")
print("═" * 78)

def dd_episodes(ret, label, k=5):
    cum = (1 + ret).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    in_dd = dd < 0
    episodes = []
    start = None
    for d, flag in in_dd.items():
        if flag and start is None:
            start = d
        elif not flag and start is not None:
            seg = dd.loc[start:d]
            episodes.append((start, seg.idxmin(), d, seg.min(), (d - start).days))
            start = None
    if start is not None:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), seg.index[-1], seg.min(), (seg.index[-1]-start).days))
    episodes.sort(key=lambda x: x[3])
    print(f"\n  {label}")
    rows = [[f"{i+1}", e[0].strftime("%Y-%m-%d"), e[1].strftime("%Y-%m-%d"),
             e[2].strftime("%Y-%m-%d"), f"{e[3]:.2%}", f"{e[4]}d"]
            for i, e in enumerate(episodes[:k])]
    print(tabulate(rows, headers=["#","Start","Trough","End","Depth","Duration"],
                   tablefmt="simple"))

dd_episodes(ret_base, "Hurst-bull × 1.5×")
dd_episodes(spy_bt,   "SPY Buy & Hold")


# ── Turnover and trade stats ───────────────────────────────────────────────
print("\n" + "═" * 78)
print("  7. TURNOVER & TRADE STATS")
print("═" * 78)
w = dg_base["weight"]
flips      = (w.diff().abs() > 1e-9).sum()
pct_on     = (w > 0).mean()
pct_full   = (w >= BASE["leverage"] - 1e-9).mean()
avg_hold   = (len(rebal_dates) / max(flips, 1))   # weeks per flip
tc_drag    = flips * BASE["tc"] * BASE["leverage"] / (len(rebal_dates) / 52.0)  # annualised
print(f"  Weekly rebalance count       {len(rebal_dates):>6}")
print(f"  Position flips (0↔on)        {int(flips):>6}")
print(f"  Avg weeks per flip           {avg_hold:>6.1f}")
print(f"  % weeks at full 1.5× on      {pct_full:>6.1%}")
print(f"  % weeks at any positive wt   {pct_on:>6.1%}")
print(f"  Est. annual TC drag          {tc_drag:>6.2%}")


# ── Diagnostic plot ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("Hurst-bull × 1.5× leverage  ·  diagnostics (2016–2024)",
             fontsize=14, fontweight="bold")
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.30)
ax_cum = fig.add_subplot(gs[0, :])
ax_dd  = fig.add_subplot(gs[1, 0])
ax_sr  = fig.add_subplot(gs[1, 1])
ax_wt  = fig.add_subplot(gs[2, 0])
ax_h   = fig.add_subplot(gs[2, 1])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

cum_b = (1 + ret_base).cumprod()
cum_s = (1 + spy_bt).cumprod()
ax_cum.plot(cum_b.index, cum_b.values, color="#6D4C41", lw=2.0, label="Hurst-bull × 1.5×")
ax_cum.plot(cum_s.index, cum_s.values, color="#37474F", lw=1.2, ls="--", label="SPY B&H")
ax_cum.set_yscale("log"); ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative wealth (log)", fontweight="bold"); ax_cum.legend()

for s, c, l in [(ret_base,"#6D4C41","Hurst-bull × 1.5×"), (spy_bt,"#37474F","SPY")]:
    cum = (1+s).cumprod(); dd = (cum - cum.cummax()) / cum.cummax()
    ax_dd.plot(dd.index, dd.values, color=c, lw=1.3, label=l)
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold"); ax_dd.legend()

for s, c, l in [(ret_base,"#6D4C41","Hurst-bull × 1.5×"), (spy_bt,"#37474F","SPY")]:
    exc = s - RF/252
    rs  = exc.rolling(126).mean() / exc.rolling(126).std() * np.sqrt(252)
    ax_sr.plot(rs.index, rs.values, color=c, lw=1.3, label=l)
ax_sr.axhline(0, color="k", lw=0.5, ls="--"); ax_sr.axhline(1, color="g", lw=0.4, ls=":")
ax_sr.set_title("Rolling 6-month Sharpe", fontweight="bold"); ax_sr.legend()

ax_wt.step(dg_base.index, dg_base["weight"], color="#6D4C41", lw=1.2, where="post")
ax_wt.fill_between(dg_base.index, 0, dg_base["weight"], step="post", alpha=0.25, color="#6D4C41")
ax_wt.axhline(BASE["leverage"], color="r", lw=0.5, ls=":", alpha=0.7, label=f"{BASE['leverage']}× cap")
ax_wt.set_title("Effective weight path", fontweight="bold"); ax_wt.legend(fontsize=8)

ax_h.plot(HURST_CACHE[BASE["hurst_win"]].index, HURST_CACHE[BASE["hurst_win"]].values,
          color="#AD1457", lw=0.9)
ax_h.axhline(BASE["hurst_thr"], color="k", lw=0.6, ls="--", label=f"H={BASE['hurst_thr']}")
ax_h.axhline(0.50, color="grey", lw=0.4, ls=":", alpha=0.6)
ax_h.set_title("Rolling Hurst exponent (252d)", fontweight="bold"); ax_h.legend(fontsize=8)

fig.savefig(RESULTS_DIR / "hurst_bull_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/hurst_bull_analysis.png")

print("\n✓  Done.")
