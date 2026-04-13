"""
Visualisation for PosteriorAlpha backtest results.

Two main entry points:
  plot_single_run()      — multi-panel dashboard for one run
  plot_multi_seed()      — bar charts with mean ± std across seeds
  plot_lambda_heatmap()  — λ evolution heatmap across seeds / years
"""
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import BacktestResult
from src.metrics import compute_metrics

warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")

# ── Colours & labels ──────────────────────────────────────────────────────
COLORS = {
    "bayesian":       "#1565C0",
    "equal_weight":   "#E65100",
    "min_variance":   "#2E7D32",
    "hist_max_sharpe":"#6A1B9A",
    "bayesian_ewma":  "#00838F",   # teal
    "bayesian_hmm":   "#AD1457",   # rose
    "bayesian_full":  "#F57F17",   # amber  ← hero of the refined run
    "SPY":            "#C62828",
}
LABELS = {
    "bayesian":       "Bayesian (v1)",
    "equal_weight":   "Equal Weight",
    "min_variance":   "Min Variance",
    "hist_max_sharpe":"Hist. Max Sharpe",
    "bayesian_ewma":  "Bayesian + EWMA + TC",
    "bayesian_hmm":   "Bayesian + HMM",
    "bayesian_full":  "Bayesian Full (all ext.)",
    "SPY":            "S&P 500 (SPY)",
}
HERO = "bayesian_full"     # the 'best' refined strategy

_PCT    = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")


def _rolling_sharpe(returns: pd.Series, window: int = 252, rf: float = 0.04) -> pd.Series:
    exc = returns - rf / 252
    return exc.rolling(window).mean() / exc.rolling(window).std() * np.sqrt(252)


def _drawdown(returns: pd.Series) -> pd.Series:
    cum     = (1.0 + returns).cumprod()
    rollmax = cum.cummax()
    return (cum - rollmax) / (rollmax + 1e-9)


# ---------------------------------------------------------------------------
# Main dashboard
# ---------------------------------------------------------------------------

def plot_single_run(
    results: Dict[str, BacktestResult],
    spy_returns: Optional[pd.Series],
    seed: int = 0,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    8-panel dashboard:
      Row 0 : Cumulative wealth (log)              [full width]
      Row 1 : Rolling 1-yr Sharpe  [2/3]  | Metrics table [1/3]
      Row 2 : Drawdown  |  λ (Bayesian)  |  P(bull) (HMM)
      Row 3 : Average weights — Bayesian Full  [full width]
    """
    fig = plt.figure(figsize=(24, 18))
    title_strat = "Refined" if "bayesian_full" in results else "Original"
    fig.suptitle(
        f"PosteriorAlpha  ·  {title_strat} Strategies  ·  2005–2024",
        fontsize=16, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

    ax_cum  = fig.add_subplot(gs[0, :])
    ax_sr   = fig.add_subplot(gs[1, :2])
    ax_tab  = fig.add_subplot(gs[1, 2])
    ax_dd   = fig.add_subplot(gs[2, 0])
    ax_lam  = fig.add_subplot(gs[2, 1])
    ax_bull = fig.add_subplot(gs[2, 2])
    ax_wts  = fig.add_subplot(gs[3, :])

    hero = HERO if HERO in results else "bayesian"

    # ── Cumulative wealth ──────────────────────────────────────────────────
    for name, res in results.items():
        cum = res.cumulative
        lw  = 3.0 if name == hero else 1.5
        zo  = 6 if name == hero else 3
        ax_cum.plot(cum.index, cum.values, label=LABELS.get(name, name),
                    color=COLORS.get(name, "gray"), linewidth=lw, zorder=zo)

    if spy_returns is not None:
        spy_cum = (1.0 + spy_returns).cumprod()
        ax_cum.plot(spy_cum.index, spy_cum.values, label="S&P 500 (SPY)",
                    color=COLORS["SPY"], linewidth=1.8, linestyle="--", zorder=2)

    ax_cum.set_yscale("log")
    ax_cum.yaxis.set_major_formatter(_DOLLAR)
    ax_cum.set_title("Cumulative Wealth  (log scale, $1 initial)", fontweight="bold")
    ax_cum.legend(loc="upper left", fontsize=8, framealpha=0.92, ncol=2)

    # ── Rolling Sharpe ─────────────────────────────────────────────────────
    for name, res in results.items():
        rs = _rolling_sharpe(res.returns)
        ax_sr.plot(rs.index, rs.values, label=LABELS.get(name, name),
                   color=COLORS.get(name, "gray"),
                   linewidth=2.5 if name == hero else 1.2)

    ax_sr.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_sr.axhline(1, color="green", linewidth=0.5, linestyle=":", alpha=0.6)
    ax_sr.set_title("Rolling 1-Year Sharpe", fontweight="bold")
    ax_sr.set_ylabel("Sharpe")
    ax_sr.legend(fontsize=7, framealpha=0.9, ncol=2)

    # ── Metrics table ──────────────────────────────────────────────────────
    ax_tab.axis("off")
    hdrs = ["Strategy", "CAGR", "Sharpe", "MaxDD", "Calmar"]
    rows = []
    for name, res in results.items():
        m = compute_metrics(res.returns, benchmark=spy_returns)
        rows.append([LABELS.get(name, name)[:20],
                     f"{m.get('CAGR',0):.1%}", f"{m.get('Sharpe',0):.2f}",
                     f"{m.get('Max DD',0):.1%}", f"{m.get('Calmar',0):.2f}"])
    if spy_returns is not None:
        m = compute_metrics(spy_returns)
        rows.append(["SPY Buy & Hold", f"{m.get('CAGR',0):.1%}",
                     f"{m.get('Sharpe',0):.2f}", f"{m.get('Max DD',0):.1%}",
                     f"{m.get('Calmar',0):.2f}"])

    tbl = ax_tab.table(cellText=rows, colLabels=hdrs, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1.0, 1.9)
    for (r_i, c_i), cell in tbl.get_celld().items():
        if r_i == 0:
            cell.set_facecolor("#1565C0"); cell.set_text_props(color="white", fontweight="bold")
        elif r_i <= len(rows) and hero in (rows[r_i-1][0] if rows else ""):
            cell.set_facecolor("#FFF9C4")
        elif r_i <= len(rows) and "Full" in rows[r_i-1][0]:
            cell.set_facecolor("#FFF9C4")
    ax_tab.set_title("Performance Summary", fontweight="bold", pad=14)

    # ── Drawdown ───────────────────────────────────────────────────────────
    for name, res in results.items():
        dd = _drawdown(res.returns)
        ax_dd.fill_between(dd.index, dd.values, 0,
                           alpha=0.15, color=COLORS.get(name, "gray"))
        ax_dd.plot(dd.index, dd.values, color=COLORS.get(name, "gray"),
                   linewidth=2.0 if name == hero else 0.8,
                   label=LABELS.get(name, name))
    ax_dd.yaxis.set_major_formatter(_PCT)
    ax_dd.set_title("Drawdown", fontweight="bold")
    ax_dd.legend(fontsize=6)

    # ── λ evolution (Bayesian strategies) ─────────────────────────────────
    plotted_lam = False
    for name, res in results.items():
        if res.lambdas is not None:
            ax_lam.plot(res.lambdas.index, res.lambdas.values,
                        color=COLORS.get(name, "gray"), linewidth=1.2,
                        label=LABELS.get(name, name), alpha=0.85)
            plotted_lam = True
    if plotted_lam:
        ax_lam.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax_lam.set_ylim(0, 1)
        ax_lam.set_title("λ  (blending weight: 1 = trust recent)", fontweight="bold")
        ax_lam.set_ylabel("λ")
        ax_lam.legend(fontsize=7)
    else:
        ax_lam.text(0.5, 0.5, "No λ data", ha="center", va="center",
                    transform=ax_lam.transAxes, color="gray")
        ax_lam.set_title("λ", fontweight="bold")

    # ── P(bull) from HMM ──────────────────────────────────────────────────
    plotted_bull = False
    for name, res in results.items():
        if res.bull_probs is not None:
            ax_bull.plot(res.bull_probs.index, res.bull_probs.values,
                         color=COLORS.get(name, "gray"), linewidth=1.2,
                         label=LABELS.get(name, name), alpha=0.85)
            plotted_bull = True
    if plotted_bull:
        ax_bull.fill_between(
            res.bull_probs.index, res.bull_probs.values, 0.5,
            where=(res.bull_probs.values > 0.5),
            alpha=0.20, color="green", label="Bull regime")
        ax_bull.fill_between(
            res.bull_probs.index, res.bull_probs.values, 0.5,
            where=(res.bull_probs.values <= 0.5),
            alpha=0.20, color="red", label="Bear regime")
        ax_bull.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax_bull.set_ylim(0, 1)
        ax_bull.set_title("P(Bull Regime)  — HMM posterior", fontweight="bold")
        ax_bull.set_ylabel("P(bull)")
        ax_bull.legend(fontsize=7)
    else:
        ax_bull.text(0.5, 0.5, "Run bayesian_hmm\nor bayesian_full\nto see this",
                     ha="center", va="center", transform=ax_bull.transAxes,
                     fontsize=10, color="gray")
        ax_bull.set_title("P(Bull Regime)  — HMM posterior", fontweight="bold")

    # ── Average weights — hero strategy ────────────────────────────────────
    if hero in results and len(results[hero].weights) > 0:
        # Plot weight time series as stacked area
        w_ts = results[hero].weights
        w_ts.plot.area(ax=ax_wts, stacked=True, alpha=0.75,
                       color=[COLORS.get(c, "gray") for c in w_ts.columns]
                       if all(c in COLORS for c in w_ts.columns)
                       else plt.cm.tab10(np.linspace(0, 1, w_ts.shape[1])))
        ax_wts.set_title(f"Weight Evolution  —  {LABELS.get(hero, hero)}",
                         fontweight="bold")
        ax_wts.set_ylabel("Weight")
        ax_wts.yaxis.set_major_formatter(_PCT)
        ax_wts.legend(loc="upper right", fontsize=8, ncol=w_ts.shape[1])
    else:
        ax_wts.text(0.5, 0.5, "No weight data", ha="center", va="center",
                    transform=ax_wts.transAxes, color="gray")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Multi-seed summary
# ---------------------------------------------------------------------------

def plot_multi_seed(
    all_metrics: Dict[str, List[Dict]],
    spy_metrics: Optional[Dict] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    metric_keys   = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar"]
    metric_labels = ["CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar"]
    pct_metrics   = {"CAGR", "Max DD"}
    invert        = {"Max DD"}

    strategies = list(all_metrics.keys())
    xlabels    = [LABELS.get(s, s) for s in strategies]
    if spy_metrics:
        strategies.append("SPY")
        xlabels.append("S&P 500 (SPY)")

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(24, 6))
    fig.suptitle("Strategy Comparison  ·  All Extensions",
                 fontsize=14, fontweight="bold")

    for ax, key, label in zip(axes, metric_keys, metric_labels):
        means, errs, colors = [], [], []
        for strat in strategies:
            if strat == "SPY":
                v = spy_metrics.get(key, 0) if spy_metrics else 0
                means.append(v); errs.append(0)
                colors.append(COLORS["SPY"])
            else:
                vals = [m[key] for m in all_metrics.get(strat, []) if key in m]
                means.append(np.mean(vals) if vals else 0)
                errs.append(np.std(vals)   if vals else 0)
                colors.append(COLORS.get(strat, "gray"))

        x    = np.arange(len(xlabels))
        bars = ax.bar(x, means, yerr=errs, color=colors,
                      capsize=4, edgecolor="white", alpha=0.88,
                      error_kw={"linewidth": 1.5})
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=7)
        if key in pct_metrics:
            ax.yaxis.set_major_formatter(_PCT)
        if key in invert:
            ax.invert_yaxis()
            ax.set_title(f"{label}  (↓ better)", fontweight="bold")
        for bar, mean, err in zip(bars, means, errs):
            h   = bar.get_height()
            txt = f"{mean:.0%}" if key in pct_metrics else f"{mean:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (err * 0.1 if err else abs(h) * 0.03),
                    txt, ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# λ heatmap
# ---------------------------------------------------------------------------

def plot_lambda_heatmap(
    lambda_series_by_seed: Dict[int, pd.Series],
    save_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    frames = []
    for seed, lams in lambda_series_by_seed.items():
        monthly = lams.resample("ME").median()
        monthly.name = f"Seed {seed}"
        frames.append(monthly)
    if not frames:
        return None

    df          = pd.concat(frames, axis=1).dropna()
    df["Year"]  = df.index.year
    annual      = df.groupby("Year").mean()

    fig, ax = plt.subplots(figsize=(max(6, len(frames) + 2), 5))
    sns.heatmap(annual.T, ax=ax, cmap="RdBu_r", center=0.5,
                vmin=0.0, vmax=1.0, linewidths=0.5,
                annot=True, fmt=".2f", annot_kws={"size": 8},
                cbar_kws={"label": "Median λ  (0 = trust prior, 1 = trust recent)"})
    ax.set_title("Annual Median λ  ·  Regime Sensitivity Over Time",
                 fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
