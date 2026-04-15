#!/usr/bin/env python3
"""
Intramonth Momentum Cycle  ·  Backtest
========================================
Tests the hypothesis from Nathan, Suominen & Tasa (2026):
  "US equity momentum returns concentrate in six trading days each month
   (T-9 to T-4 relative to month-end), driven by institutional dash-for-cash."

Universe  : SPY, TLT, GLD, EEM, VNQ  (from portfolio_data.csv)
Signal    : 12-1 month cross-asset momentum (rank by prior return, skip 1m)
Portfolio : long top-2 / short bottom-2  (equal-weighted each leg)
Strategies:
  WML Window   — WML only during days T-9 to T-4, flat otherwise
  WML Outside  — WML on all other days, flat during window
  WML Full     — WML every trading day
  SPY B&H      — passive benchmark
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from tabulate import tabulate

from src.metrics import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────
CSV_PATH      = Path("portfolio_data.csv")
BT_START      = "2007-01-01"   # need ~14m warmup from data start (2005-01)
BT_END        = "2024-12-31"
RF            = 0.04
MOMENTUM_LONG = 252            # ~12 months
MOMENTUM_SKIP = 21             # skip most recent ~1 month
N_LONG        = 2              # number of assets to go long
N_SHORT       = 2              # number of assets to go short
WINDOW_START  = 9              # T-9  (inclusive)
WINDOW_END    = 4              # T-4  (inclusive)

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  Intramonth Momentum Cycle  ·  WML Window (T-9→T-4) vs Full Month    ║
║  Nathan, Suominen & Tasa (2026) — concept test on 5-ETF universe     ║
╚═══════════════════════════════════════════════════════════════════════╝""")

# ── Load data ──────────────────────────────────────────────────────────────────
logger.info(f"Loading {CSV_PATH} …")
prices = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date").sort_index()
rets   = prices.pct_change().dropna()
assets = rets.columns.tolist()          # ['SPY','TLT','GLD','EEM','VNQ']
spy_rets = rets["SPY"].rename("SPY")

logger.info(f"Loaded {len(rets)} trading days  ·  assets: {assets}")


# ── Intramonth window labelling ────────────────────────────────────────────────

def label_window(holding_days: pd.DatetimeIndex) -> np.ndarray:
    """
    Boolean mask: True on days T-WINDOW_START … T-WINDOW_END (inclusive)
    counting back from the last trading day of the holding period.

    Mapping:  T-k  <=>  holding_days[-(k+1)]
    Window k = WINDOW_END(4) … WINDOW_START(9)  →  6 days.
    """
    n    = len(holding_days)
    mask = np.zeros(n, dtype=bool)
    for k in range(WINDOW_END, WINDOW_START + 1):
        pos = -(k + 1)
        if abs(pos) <= n:
            mask[pos] = True
    return mask


# ── Backtest ───────────────────────────────────────────────────────────────────

def run_wml_backtest(rets: pd.DataFrame) -> dict:
    """
    Monthly-rebalanced WML backtest split by the intramonth T-9→T-4 window.

    At each month-end:
      • Rank all assets by their 12-1m cumulative return
      • Long top N_LONG, short bottom N_SHORT (equal-weighted legs)

    For each subsequent trading day:
      • Compute WML return  = mean(winner_rets) − mean(loser_rets)
      • Assign to Window / Outside / Full buckets
    """
    month_ends = rets.resample("ME").last().index
    min_obs    = MOMENTUM_LONG + MOMENTUM_SKIP + 5
    month_ends = month_ends[month_ends > rets.index[min_obs]]

    rec: dict[str, dict] = {
        "WML Window": {}, "WML Outside": {}, "WML Full": {},
        "Winners":    {}, "Losers":      {},
        # per-leg × period for diagnostics
        "win_window": {}, "win_outside": {},
        "los_window": {}, "los_outside": {},
    }

    for i, me in enumerate(month_ends[:-1]):
        # ── Formation ─────────────────────────────────────────────────────────
        hist = rets.loc[:me]
        if len(hist) < min_obs:
            continue

        formation = hist.iloc[-(MOMENTUM_LONG + MOMENTUM_SKIP):-MOMENTUM_SKIP]
        cum_ret   = (1.0 + formation).prod() - 1.0
        cum_ret   = cum_ret.dropna()

        if len(cum_ret) < N_LONG + N_SHORT + 1:
            continue

        ranked  = cum_ret.sort_values()
        losers  = ranked.index[:N_SHORT].tolist()
        winners = ranked.index[-N_LONG:].tolist()

        # ── Holding period ─────────────────────────────────────────────────────
        next_me      = month_ends[i + 1]
        hold_mask    = (rets.index > me) & (rets.index <= next_me)
        holding_days = rets.index[hold_mask]

        bt_mask      = (holding_days >= BT_START) & (holding_days <= BT_END)
        holding_days = holding_days[bt_mask]
        if len(holding_days) == 0:
            continue

        in_window = label_window(holding_days)
        day_rets  = rets.loc[holding_days]

        for j, date in enumerate(holding_days):
            dr    = day_rets.loc[date]
            w_ret = dr[winners].mean()
            l_ret = dr[losers].mean()
            wml   = w_ret - l_ret
            iw    = bool(in_window[j])

            rec["WML Full"][date]    = wml
            rec["WML Window"][date]  = wml  if iw else 0.0
            rec["WML Outside"][date] = 0.0  if iw else wml
            rec["Winners"][date]     = w_ret
            rec["Losers"][date]      = l_ret
            rec["win_window"][date]  = w_ret if iw else 0.0
            rec["win_outside"][date] = 0.0   if iw else w_ret
            rec["los_window"][date]  = l_ret if iw else 0.0
            rec["los_outside"][date] = 0.0   if iw else l_ret

    return {k: pd.Series(v, name=k).sort_index() for k, v in rec.items()}


logger.info("Running WML intramonth backtest …")
strats = run_wml_backtest(rets)
for k in strats:
    strats[k] = strats[k].loc[BT_START:BT_END]

spy_bt = spy_rets.loc[BT_START:BT_END]

n_window_days  = int((strats["WML Window"]  != 0).sum())
n_outside_days = int((strats["WML Outside"] != 0).sum())
n_total_days   = int((strats["WML Full"]    != 0).sum())
logger.info(
    f"Window days: {n_window_days}  |  Outside days: {n_outside_days}  "
    f"|  Total: {n_total_days}"
)


# ── Intramonth calendar: avg return by day position ───────────────────────────
# Map every WML-Full day to its T-k position within its holding month

def build_day_position_df(series: pd.Series) -> pd.DataFrame:
    """For each active day in `series`, record its T-k position."""
    rows = []
    for _, grp in pd.Series(series.index).groupby(series.index.to_period("M")):
        days = sorted(grp.tolist())
        n    = len(days)
        for j, d in enumerate(days):
            k = n - 1 - j          # k=0 → last day (T), k=9 → T-9, etc.
            rows.append({"date": d, "k": k, "ret": series.loc[d]})
    return pd.DataFrame(rows).set_index("date")


pos_df   = build_day_position_df(strats["WML Full"])
avg_by_k = pos_df.groupby("k")["ret"].mean()

# Same for SPY
spy_pos_df   = build_day_position_df(spy_bt.reindex(strats["WML Full"].index).dropna())
spy_avg_by_k = spy_pos_df.groupby("k")["ret"].mean()

# ── SPY + WML Window overlay ───────────────────────────────────────────────────
# WML is market-neutral (long-short); adding it on top of SPY B&H captures any
# alpha while keeping full equity market exposure.
spy_wml = (spy_bt.reindex(strats["WML Window"].index)
                 .add(strats["WML Window"], fill_value=0)
                 .dropna())


# ── Console output ─────────────────────────────────────────────────────────────
print(f"\n{'='*68}")
print(f"  INTRAMONTH WINDOW ANALYSIS  (T-{WINDOW_START} to T-{WINDOW_END} vs. rest of month)")
print(f"{'='*68}")

diag_rows = []
for label, s in [
    (f"WML Window  (T-{WINDOW_START}→T-{WINDOW_END})", strats["WML Window"]),
    ("WML Outside window",                              strats["WML Outside"]),
    ("WML Full month",                                  strats["WML Full"]),
]:
    active = s[s != 0].dropna()
    n      = len(active)
    avg    = active.mean()
    tstat  = avg / (active.std() / n**0.5) if n > 1 else 0.0
    diag_rows.append([
        label,
        str(n),
        f"{avg*100:+.3f}%",
        f"{tstat:+.2f}",
        f"{(active > 0).mean():.1%}",
    ])

print(tabulate(
    diag_rows,
    headers=["Strategy", "Active days", "Avg daily ret", "t-stat", "Hit rate"],
    tablefmt="rounded_grid",
))

print("\n── Winner & loser legs by window period ──")
leg_rows = []
for period_label, wk, lk in [
    (f"Window  (T-{WINDOW_START}→T-{WINDOW_END})", "win_window",  "los_window"),
    ("Outside window",                              "win_outside", "los_outside"),
]:
    w = strats[wk]; w = w[w != 0].dropna()
    l = strats[lk]; l = l[l != 0].dropna()
    if len(w) and len(l):
        leg_rows.append([
            period_label,
            f"{w.mean()*100:+.3f}%",
            f"{l.mean()*100:+.3f}%",
            f"{(w.mean() - l.mean())*100:+.3f}%",
        ])
print(tabulate(
    leg_rows,
    headers=["Period", "Winners avg/day", "Losers avg/day", "WML spread"],
    tablefmt="simple",
))

METRIC_KEYS  = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
EXTRA_KEYS   = ["Alpha", "Beta", "Info Ratio"]

def fmt(v, k):
    if k in ("CAGR", "Max DD", "Volatility", "Alpha"):
        return f"{v:.2%}"
    return f"{v:.2f}"

perf_map = {
    "WML Window":    (strats["WML Window"],  f"WML Window (T-{WINDOW_START}→T-{WINDOW_END})"),
    "WML Outside":   (strats["WML Outside"], "WML Outside window"),
    "WML Full":      (strats["WML Full"],    "WML Full month"),
    "SPY+WML Win":   (spy_wml,              "SPY + WML Window overlay"),
    "SPY B&H":       (spy_bt,               "SPY Buy & Hold"),
}

perf_metrics = {}
for key, (ret, _) in perf_map.items():
    perf_metrics[key] = compute_metrics(ret, benchmark=spy_bt, rf=RF)

# ── Main metrics table ────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print("  PERFORMANCE METRICS  (benchmark = SPY Buy & Hold)")
print(f"{'='*78}")

perf_rows = []
for key, (ret, label) in perf_map.items():
    m = perf_metrics[key]
    perf_rows.append([label] + [fmt(m.get(k, 0.0), k) for k in METRIC_KEYS])

print(tabulate(
    perf_rows,
    headers=["Strategy"] + METRIC_KEYS,
    tablefmt="rounded_grid",
))

# ── Alpha / Beta / Info Ratio ─────────────────────────────────────────────────
print(f"\n{'='*78}")
print("  BENCHMARK-RELATIVE METRICS  (vs SPY Buy & Hold)")
print(f"{'='*78}")

rel_rows = []
for key, (ret, label) in perf_map.items():
    if key == "SPY B&H":
        continue
    m = perf_metrics[key]
    rel_rows.append([
        label,
        fmt(m.get("Alpha", 0.0), "Alpha"),
        f"{m.get('Beta', 0.0):.2f}",
        f"{m.get('Info Ratio', 0.0):.2f}",
    ])

print(tabulate(
    rel_rows,
    headers=["Strategy", "Alpha (ann.)", "Beta", "Info Ratio"],
    tablefmt="rounded_grid",
))

# ── Delta vs SPY ──────────────────────────────────────────────────────────────
print(f"\n{'='*78}")
print("  DELTA vs SPY BUY & HOLD")
print(f"{'='*78}")

spy_m     = perf_metrics["SPY B&H"]
delta_rows = []
for key, (ret, label) in perf_map.items():
    if key == "SPY B&H":
        continue
    m = perf_metrics[key]
    row = [label]
    for k in METRIC_KEYS:
        d  = m.get(k, 0.0) - spy_m.get(k, 0.0)
        s  = "+" if d >= 0 else ""
        row.append(f"{s}{d:.2%}" if k in ("CAGR", "Max DD", "Volatility") else f"{s}{d:.2f}")
    delta_rows.append(row)

print(tabulate(
    delta_rows,
    headers=["Strategy"] + [f"Δ {k}" for k in METRIC_KEYS],
    tablefmt="rounded_grid",
))

print(f"\n{'='*78}")
print("  CUMULATIVE GROWTH OF $1")
print(f"{'='*78}")
for key, (ret, label) in perf_map.items():
    cum    = (1 + ret).prod()
    marker = "  ◀ best" if cum == max((1 + r).prod() for r, _ in perf_map.values()) else ""
    print(f"  {label:42s}  ${cum:.2f}{marker}")


# ── Plots ──────────────────────────────────────────────────────────────────────
COLORS = {
    "WML Window":  "#1A237E",
    "WML Outside": "#B71C1C",
    "WML Full":    "#1B5E20",
    "SPY+WML Win": "#6A1B9A",
    "SPY B&H":     "#37474F",
}

fig = plt.figure(figsize=(22, 18))
fig.suptitle(
    f"Intramonth Momentum Cycle  ·  WML Window (T-{WINDOW_START}→T-{WINDOW_END}) vs. Rest\n"
    "Nathan, Suominen & Tasa (2026) — concept test on 5-ETF universe",
    fontsize=14, fontweight="bold",
)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

ax_cum  = fig.add_subplot(gs[0, :])
ax_dd   = fig.add_subplot(gs[1, 0])
ax_sr   = fig.add_subplot(gs[1, 1])
ax_cal  = fig.add_subplot(gs[1, 2])
ax_bar  = fig.add_subplot(gs[2, 0])
ax_hist = fig.add_subplot(gs[2, 1])
ax_spy  = fig.add_subplot(gs[2, 2])

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")

# ── 1. Cumulative wealth ──────────────────────────────────────────────────────
for key, (ret, label) in perf_map.items():
    cum = (1 + ret).cumprod()
    is_spy_bnh = key == "SPY B&H"
    ax_cum.plot(cum.index, cum.values, label=label,
                color=COLORS[key],
                lw=1.5 if is_spy_bnh else 2.5,
                ls="--" if is_spy_bnh else "-")
ax_cum.set_yscale("log")
ax_cum.yaxis.set_major_formatter(_DOLLAR)
ax_cum.set_title("Cumulative Wealth  (log scale)", fontweight="bold")
ax_cum.legend(fontsize=10, framealpha=0.9)
ax_cum.grid(True, alpha=0.3)

# ── 2. Drawdown ───────────────────────────────────────────────────────────────
for key, (ret, label) in perf_map.items():
    cum = (1 + ret).cumprod()
    dd  = (cum - cum.cummax()) / (cum.cummax() + 1e-9)
    ax_dd.plot(dd.index, dd.values, color=COLORS[key], lw=1.5, label=label,
               ls="--" if key == "SPY B&H" else "-")
ax_dd.yaxis.set_major_formatter(_PCT)
ax_dd.set_title("Drawdown", fontweight="bold")
ax_dd.legend(fontsize=7)
ax_dd.grid(True, alpha=0.3)

# ── 3. Rolling 6-month Sharpe ─────────────────────────────────────────────────
for key, (ret, label) in perf_map.items():
    exc = ret - RF / 252
    rs  = exc.rolling(126).mean() / (exc.rolling(126).std() + 1e-9) * 252**0.5
    ax_sr.plot(rs.index, rs.values, color=COLORS[key], lw=1.5, label=label,
               ls="--" if key == "SPY B&H" else "-")
ax_sr.axhline(0, color="black", lw=0.6, ls="--")
ax_sr.axhline(1, color="green", lw=0.5, ls=":", alpha=0.6)
ax_sr.set_title("Rolling 6-Month Sharpe", fontweight="bold")
ax_sr.legend(fontsize=7)
ax_sr.grid(True, alpha=0.3)

# ── 4. Avg WML return by T-k position ────────────────────────────────────────
k_vals   = avg_by_k.index.values
k_colors = [COLORS["WML Window"] if WINDOW_END <= k <= WINDOW_START
            else "#B0BEC5" for k in k_vals]
ax_cal.bar(k_vals, avg_by_k.values * 100, color=k_colors,
           edgecolor="white", width=0.7)
ax_cal.axhline(0, color="black", lw=0.6)
ax_cal.axvspan(WINDOW_END - 0.5, WINDOW_START + 0.5,
               alpha=0.12, color=COLORS["WML Window"],
               label=f"T-{WINDOW_START} to T-{WINDOW_END} window")
ax_cal.invert_xaxis()
ax_cal.set_xlabel("Trading days before month-end  (T-k)")
ax_cal.set_ylabel("Avg daily WML return (%)")
ax_cal.set_title("Avg WML by Day Position in Month", fontweight="bold")
ax_cal.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
ax_cal.legend(fontsize=7)
ax_cal.grid(True, alpha=0.3, axis="y")

# ── 5. Bar: avg daily WML in window vs outside ────────────────────────────────
win_avg = strats["WML Window"][strats["WML Window"] != 0].mean() * 100
out_avg = strats["WML Outside"][strats["WML Outside"] != 0].mean() * 100
bars = ax_bar.bar(
    [f"T-{WINDOW_START}→T-{WINDOW_END}\n(window)", "Other days\n(outside)"],
    [win_avg, out_avg],
    color=[COLORS["WML Window"], COLORS["WML Outside"]],
    edgecolor="white", width=0.5,
)
ax_bar.axhline(0, color="black", lw=0.6)
for bar, val in zip(bars, [win_avg, out_avg]):
    offset = 0.001 if val >= 0 else -0.003
    ax_bar.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:+.3f}%", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
ax_bar.set_ylabel("Avg daily WML return (%)")
ax_bar.set_title("Avg Daily WML:\nWindow vs. Outside", fontweight="bold")
ax_bar.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
ax_bar.grid(True, alpha=0.3, axis="y")

# ── 6. Return distributions ───────────────────────────────────────────────────
wml_win_act = strats["WML Window"][strats["WML Window"] != 0] * 100
wml_out_act = strats["WML Outside"][strats["WML Outside"] != 0] * 100
ax_hist.hist(wml_win_act, bins=50, alpha=0.6, color=COLORS["WML Window"],
             label=f"Window  μ={wml_win_act.mean():.3f}%", density=True)
ax_hist.hist(wml_out_act, bins=50, alpha=0.6, color=COLORS["WML Outside"],
             label=f"Outside μ={wml_out_act.mean():.3f}%", density=True)
ax_hist.axvline(wml_win_act.mean(), color=COLORS["WML Window"], lw=2, ls="--")
ax_hist.axvline(wml_out_act.mean(), color=COLORS["WML Outside"], lw=2, ls="--")
ax_hist.set_xlabel("Daily WML return (%)")
ax_hist.set_title("Distribution of WML Returns", fontweight="bold")
ax_hist.legend(fontsize=7)
ax_hist.grid(True, alpha=0.3)

# ── 7. SPY avg return by T-k (month-end calendar effect) ─────────────────────
spy_k  = spy_avg_by_k.index.values
spy_cl = [COLORS["WML Window"] if WINDOW_END <= k <= WINDOW_START
          else "#90CAF9" for k in spy_k]
ax_spy.bar(spy_k, spy_avg_by_k.values * 100, color=spy_cl,
           edgecolor="white", width=0.7)
ax_spy.axhline(0, color="black", lw=0.6)
ax_spy.axvspan(WINDOW_END - 0.5, WINDOW_START + 0.5,
               alpha=0.12, color=COLORS["WML Window"],
               label=f"T-{WINDOW_START} to T-{WINDOW_END} window")
ax_spy.invert_xaxis()
ax_spy.set_xlabel("Trading days before month-end  (T-k)")
ax_spy.set_ylabel("Avg daily SPY return (%)")
ax_spy.set_title("SPY Avg Return by Day Position\n(Month-End Calendar Effect)",
                 fontweight="bold")
ax_spy.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:.2f}%"))
ax_spy.legend(fontsize=7)
ax_spy.grid(True, alpha=0.3, axis="y")

fig.savefig(RESULTS_DIR / "intramonth_momentum.png", dpi=150, bbox_inches="tight")
plt.close(fig)
logger.info("Saved: results/intramonth_momentum.png")

# ── Save returns CSV ───────────────────────────────────────────────────────────
out_df = pd.DataFrame({
    "WML_Window":    strats["WML Window"],
    "WML_Outside":   strats["WML Outside"],
    "WML_Full":      strats["WML Full"],
    "SPY_WML_Window": spy_wml,
    "SPY_BnH":        spy_bt,
})
out_df.to_csv(RESULTS_DIR / "intramonth_momentum_returns.csv")
logger.info("Saved: results/intramonth_momentum_returns.csv")

print("\n✓  Done.  Results in results/")
