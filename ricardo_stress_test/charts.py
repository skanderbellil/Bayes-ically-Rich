"""
Charts. Five diagnostics saved as PNGs:
  (i)   cumulative NAV of all variants vs benchmarks
  (ii)  drawdown of the variants
  (iii) per-name contribution waterfall
  (iv)  rolling 20d beta of the L/S book vs SPY
  (v)   within-category dispersion (fragile_middle cumulative lines)

Design: one consistent, colour-blind-safe palette; longs in blue-greens, shorts
in warm tones, benchmarks in grey. Zero chartjunk, labelled axes, dated title.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

# Consistent, distinguishable palette (Okabe-Ito derived).
PAL = {
    "long_only":       "#0072B2",   # blue
    "dollar_neutral":  "#009E73",   # green
    "beta_neutral":    "#CC79A7",   # magenta
    "dollar_neutral_bsaware": "#E69F00",  # orange
    "dollar_neutral_riskparity": "#000000",  # black
    "SPY":             "#7F7F7F",   # grey
    "BLEND(QQQ+XLI)":  "#BBBBBB",   # light grey
    "pos":             "#009E73",
    "neg":             "#D55E00",
}
_PCT = FuncFormatter(lambda x, _: f"{x:.0%}")


def _save(fig, name):
    path = OUT / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_nav(variant_navs, bench_navs, fname="01_cumulative_nav.png"):
    fig, ax = plt.subplots(figsize=(11, 6))
    for k, s in variant_navs.items():
        ax.plot(s.index, s.values - 1.0, label=k, lw=2.2,
                color=PAL.get(k, None))
    for k, s in bench_navs.items():
        ax.plot(s.index, s.values - 1.0, label=k, lw=1.5, ls="--",
                color=PAL.get(k, "#999999"))
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title("Cumulative return: portfolio variants vs benchmarks  (YTD 2026)")
    ax.set_ylabel("Cumulative return"); ax.set_xlabel("")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    return _save(fig, fname)


def chart_drawdown(variant_navs, fname="02_drawdown.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    for k, s in variant_navs.items():
        dd = s / s.cummax() - 1.0
        ax.plot(dd.index, dd.values, label=k, lw=1.8, color=PAL.get(k, None))
        ax.fill_between(dd.index, dd.values, 0, alpha=0.08, color=PAL.get(k, None))
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title("Drawdown by variant")
    ax.set_ylabel("Drawdown"); ax.legend(frameon=False, fontsize=9)
    return _save(fig, fname)


def chart_waterfall(contrib, total, fname="03_contribution_waterfall.png",
                    top_n=None):
    """Waterfall of per-name contributions to L/S PnL (sorted desc)."""
    c = contrib.sort_values(ascending=False)
    if top_n:  # optionally collapse the middle
        c = c
    names = list(c.index)
    vals = c.values
    cum = np.concatenate([[0.0], np.cumsum(vals)])
    fig, ax = plt.subplots(figsize=(max(12, 0.22 * len(names)), 6.5))
    for i, (nm, v) in enumerate(zip(names, vals)):
        color = PAL["pos"] if v >= 0 else PAL["neg"]
        ax.bar(i, v, bottom=cum[i], color=color, width=0.8, edgecolor="white", lw=0.3)
    # running total line
    ax.plot(range(len(names)), cum[1:], color="k", lw=1.0, alpha=0.6,
            marker="o", ms=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(total, color="#0072B2", lw=1.2, ls=":",
               label=f"L/S total = {total:+.1%}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title("Per-name contribution to dollar-neutral L/S PnL (waterfall)")
    ax.set_ylabel("Contribution to L/S return")
    ax.legend(frameon=False)
    return _save(fig, fname)


def chart_rolling_beta(beta_series, window, fname="04_rolling_beta.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(beta_series.index, beta_series.values, lw=1.8, color="#CC79A7")
    ax.axhline(0, color="k", lw=1.0, label="beta-neutral target (0)")
    ax.axhline(beta_series.mean(), color="#0072B2", ls=":", lw=1.2,
               label=f"mean = {beta_series.mean():+.2f}")
    ax.fill_between(beta_series.index, beta_series.values, 0,
                    where=(beta_series.values > 0), alpha=0.08, color="#D55E00")
    ax.fill_between(beta_series.index, beta_series.values, 0,
                    where=(beta_series.values <= 0), alpha=0.08, color="#009E73")
    ax.set_title(f"Rolling {window}d beta of dollar-neutral L/S vs SPY")
    ax.set_ylabel("Beta"); ax.legend(frameon=False, fontsize=9)
    return _save(fig, fname)


def chart_dispersion(cum_df, xstd, category="fragile_middle",
                     fname="05_dispersion.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [2.4, 1]})
    cmap = plt.get_cmap("tab10")
    for i, col in enumerate(cum_df.columns):
        ax1.plot(cum_df.index, cum_df[col].values, lw=1.7,
                 color=cmap(i % 10), label=col)
    ax1.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax1.yaxis.set_major_formatter(_PCT)
    ax1.set_title(f"Cumulative return by name — {category}")
    ax1.set_ylabel("Cumulative return")
    ax1.legend(frameon=False, ncol=2, fontsize=8)

    ax2.bar(range(len(xstd)), xstd.values * 100, color="#0072B2", width=0.6)
    ax2.set_xticks(range(len(xstd)))
    ax2.set_xticklabels(xstd.index, rotation=45, fontsize=8, ha="right")
    ax2.set_title("Cross-sectional std\nof monthly returns")
    ax2.set_ylabel("Std of monthly returns (%)")
    return _save(fig, fname)


def chart_pair(pair_dict, fname="06_pair_spread.png", a="CRWV", b="NBIS"):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = {f"long_{a}_short_{b}": "#0072B2", f"long_{b}_short_{a}": "#D55E00"}
    for k, s in pair_dict.items():
        ax.plot(s.index, s.values, lw=2.0, label=k, color=colors.get(k))
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.yaxis.set_major_formatter(_PCT)
    ax.set_title(f"Pair spread: {a} vs {b} (cumulative)")
    ax.set_ylabel("Cumulative pair return")
    ax.legend(frameon=False)
    return _save(fig, fname)


def chart_bsaware(frag_df, dn_equal_nav, dn_bsaware_nav,
                  fname="08_bsaware_short.png"):
    """Two panels: (left) how the fragility tilt redistributes short weight per
    name; (right) the two dollar-neutral NAVs (equal-weight short vs
    balance-sheet-aware short)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    d = frag_df.sort_values("w_tilt")
    colors = ["#009E73" if v > 0 else "#D55E00" for v in d["w_tilt"]]
    ax1.barh(range(len(d)), d["w_tilt"].values * 100, color=colors)
    ax1.set_yticks(range(len(d)))
    ax1.set_yticklabels(d.index, fontsize=8)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_title("Short-weight tilt vs equal weight\n"
                  "(+green = shorted MORE on fragility; −red = shorted LESS)")
    ax1.set_xlabel("Weight change vs equal (percentage points)")

    ax2.plot(dn_equal_nav.index, dn_equal_nav.values - 1.0, lw=2.2,
             color="#009E73", label="dollar-neutral (equal short)")
    ax2.plot(dn_bsaware_nav.index, dn_bsaware_nav.values - 1.0, lw=2.2,
             color="#0072B2", label="dollar-neutral (bal-sheet-aware short)")
    ax2.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax2.yaxis.set_major_formatter(_PCT)
    ax2.set_title("Does fitting the idea help?\nequal-weight vs fragility-weighted short")
    ax2.set_ylabel("Cumulative return")
    ax2.legend(frameon=False, fontsize=9)
    return _save(fig, fname)


def chart_tier(tier_df, fname="07_tier_attribution.png"):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    order = tier_df.sort_values("tier_return", ascending=True)
    colors = ["#009E73" if v >= 0 else "#D55E00" for v in order["tier_return"]]
    ax.barh(range(len(order)), order["tier_return"].values, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{i}  (n={int(order.loc[i,'n_names'])})" for i in order.index])
    ax.xaxis.set_major_formatter(_PCT)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Long-book tier attribution (standalone equal-weight return, YTD)")
    ax.set_xlabel("Cumulative return")
    for i, v in enumerate(order["tier_return"].values):
        ax.text(v, i, f" {v:+.1%}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    return _save(fig, fname)
