"""
Visualisation for PosteriorAlpha backtest results.

Two main entry points:
  plot_single_run()      — 6-panel dashboard for one seed
  plot_multi_seed()      — bar charts with mean ± std across seeds
"""
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — works headless
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.backtest import BacktestResult
from src.metrics import compute_metrics

warnings.filterwarnings("ignore")

# ── Style ──────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("tab10")

COLORS = {
    "bayesian":       "#1565C0",   # deep blue  — hero strategy
    "equal_weight":   "#E65100",   # orange
    "min_variance":   "#2E7D32",   # green
    "hist_max_sharpe":"#6A1B9A",   # purple
    "SPY":            "#C62828",   # red  (benchmark)
}
LABELS = {
    "bayesian":       "Bayesian Adaptive (ours)",
    "equal_weight":   "Equal Weight",
    "min_variance":   "Min Variance",
    "hist_max_sharpe":"Hist. Max Sharpe",
    "SPY":            "S&P 500 (SPY)",
}
HERO = "bayesian"

_PCT   = mticker.FuncFormatter(lambda x, _: f"{x:.0%}")
_DOLLAR = mticker.FuncFormatter(lambda x, _: f"${x:.2f}")


# ---------------------------------------------------------------------------
# Helper: rolling Sharpe
# ---------------------------------------------------------------------------

def _rolling_sharpe(returns: pd.Series, window: int = 252, rf: float = 0.04) -> pd.Series:
    rf_d  = rf / 252
    exc   = returns - rf_d
    return exc.rolling(window).mean() / exc.rolling(window).std() * np.sqrt(252)


def _drawdown(returns: pd.Series) -> pd.Series:
    cum      = (1.0 + returns).cumprod()
    roll_max = cum.cummax()
    return (cum - roll_max) / (roll_max + 1e-9)


# ---------------------------------------------------------------------------
# Single-seed 6-panel dashboard
# ---------------------------------------------------------------------------

def plot_single_run(
    results: Dict[str, BacktestResult],
    spy_returns: Optional[pd.Series],
    seed: int,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    6-panel figure:
      [0] Cumulative wealth (log scale)     [full-width]
      [1] Rolling 1-year Sharpe             [2/3 width]
      [2] Performance summary table         [1/3 width]
      [3] Drawdown profiles
      [4] λ evolution  (Bayesian only)
      [5] Avg top-10 Bayesian weights
    """
    fig = plt.figure(figsize=(22, 15))
    fig.suptitle(
        f"PosteriorAlpha  ·  Bayesian Adaptive Portfolio  ·  Seed {seed}",
        fontsize=15, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_cum  = fig.add_subplot(gs[0, :])
    ax_sr   = fig.add_subplot(gs[1, :2])
    ax_tab  = fig.add_subplot(gs[1, 2])
    ax_dd   = fig.add_subplot(gs[2, 0])
    ax_lam  = fig.add_subplot(gs[2, 1])
    ax_wts  = fig.add_subplot(gs[2, 2])

    all_for_table = dict(results)

    # ── Panel 0: Cumulative wealth ─────────────────────────────────────────
    for name, res in results.items():
        cum = res.cumulative
        ax_cum.plot(cum.index, cum.values,
                    label=LABELS.get(name, name),
                    color=COLORS.get(name, "gray"),
                    linewidth=2.8 if name == HERO else 1.6,
                    zorder=5 if name == HERO else 3)

    if spy_returns is not None:
        spy_cum = (1.0 + spy_returns).cumprod()
        ax_cum.plot(spy_cum.index, spy_cum.values,
                    label=LABELS["SPY"], color=COLORS["SPY"],
                    linewidth=1.6, linestyle="--", zorder=2)

    ax_cum.set_yscale("log")
    ax_cum.yaxis.set_major_formatter(_DOLLAR)
    ax_cum.set_title("Cumulative Wealth  (log scale)", fontweight="bold")
    ax_cum.set_ylabel("Portfolio value  ($1 initial)")
    ax_cum.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ── Panel 1: Rolling Sharpe ────────────────────────────────────────────
    for name, res in results.items():
        rs = _rolling_sharpe(res.returns)
        ax_sr.plot(rs.index, rs.values,
                   label=LABELS.get(name, name),
                   color=COLORS.get(name, "gray"),
                   linewidth=2.5 if name == HERO else 1.4)

    ax_sr.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_sr.axhline(1, color="green", linewidth=0.6, linestyle=":", alpha=0.6, label="Sharpe = 1")
    ax_sr.set_title("Rolling 1-Year Sharpe Ratio", fontweight="bold")
    ax_sr.set_ylabel("Sharpe")
    ax_sr.legend(fontsize=8, framealpha=0.9)

    # ── Panel 2: Metrics table ─────────────────────────────────────────────
    ax_tab.axis("off")
    _headers = ["Strategy", "CAGR", "Sharpe", "MaxDD", "Calmar"]
    rows = []

    for name, res in all_for_table.items():
        m = compute_metrics(res.returns, benchmark=spy_returns)
        rows.append([
            LABELS.get(name, name)[:18],
            f"{m.get('CAGR', 0):.1%}",
            f"{m.get('Sharpe', 0):.2f}",
            f"{m.get('Max DD', 0):.1%}",
            f"{m.get('Calmar', 0):.2f}",
        ])

    if spy_returns is not None:
        m = compute_metrics(spy_returns)
        rows.append(["S&P 500 (SPY)",
                     f"{m.get('CAGR',0):.1%}",
                     f"{m.get('Sharpe',0):.2f}",
                     f"{m.get('Max DD',0):.1%}",
                     f"{m.get('Calmar',0):.2f}"])

    tbl = ax_tab.table(cellText=rows, colLabels=_headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.05, 2.0)

    # Highlight header row and hero strategy row
    for (r_idx, c_idx), cell in tbl.get_celld().items():
        if r_idx == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r_idx <= len(rows) and "Bayesian" in rows[r_idx - 1][0]:
            cell.set_facecolor("#E3F2FD")

    ax_tab.set_title("Performance Summary", fontweight="bold", pad=16)

    # ── Panel 3: Drawdowns ─────────────────────────────────────────────────
    for name, res in results.items():
        dd = _drawdown(res.returns)
        ax_dd.fill_between(dd.index, dd.values, 0,
                           alpha=0.20, color=COLORS.get(name, "gray"))
        ax_dd.plot(dd.index, dd.values,
                   color=COLORS.get(name, "gray"),
                   linewidth=1.5 if name == HERO else 0.8,
                   label=LABELS.get(name, name))

    ax_dd.yaxis.set_major_formatter(_PCT)
    ax_dd.set_title("Drawdown", fontweight="bold")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.legend(fontsize=7)

    # ── Panel 4: λ evolution ──────────────────────────────────────────────
    if HERO in results and results[HERO].lambdas is not None:
        lams = results[HERO].lambdas
        ax_lam.plot(lams.index, lams.values, color=COLORS[HERO], linewidth=1.5)
        ax_lam.fill_between(lams.index, lams.values, 0.5,
                            where=(lams.values > 0.5),
                            alpha=0.30, color="#C62828", label="Trend toward recent")
        ax_lam.fill_between(lams.index, lams.values, 0.5,
                            where=(lams.values <= 0.5),
                            alpha=0.30, color="#1565C0", label="Trend toward prior")
        ax_lam.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        ax_lam.set_ylim(0, 1)
        ax_lam.set_ylabel("λ (blending weight)")
        ax_lam.set_title("Regime Sensitivity λ  (Bayesian only)", fontweight="bold")
        ax_lam.legend(fontsize=7)
    else:
        ax_lam.text(0.5, 0.5, "λ available\nfor Bayesian strategy only",
                    ha="center", va="center", transform=ax_lam.transAxes,
                    fontsize=11, color="gray")
        ax_lam.set_title("Regime Sensitivity λ", fontweight="bold")

    # ── Panel 5: Average top-10 weights ───────────────────────────────────
    if HERO in results:
        avg_w = results[HERO].weights.mean().sort_values(ascending=False)
        top_n = min(10, len(avg_w))
        colors = [COLORS[HERO]] + ["#90CAF9"] * (top_n - 1)
        ax_wts.bar(range(top_n), avg_w.values[:top_n], color=colors, edgecolor="white")
        ax_wts.set_xticks(range(top_n))
        ax_wts.set_xticklabels(avg_w.index[:top_n], rotation=40, ha="right", fontsize=7)
        ax_wts.yaxis.set_major_formatter(_PCT)
        ax_wts.set_title("Avg. Bayesian Weights  (top 10)", fontweight="bold")
        ax_wts.set_ylabel("Average weight")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Multi-seed robustness summary
# ---------------------------------------------------------------------------

def plot_multi_seed(
    all_metrics: Dict[str, List[Dict]],
    spy_metrics: Optional[Dict] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar charts (mean ± std across seeds) for 5 key metrics.

    Parameters
    ----------
    all_metrics : {strategy: [metrics_dict_seed0, metrics_dict_seed1, ...]}
    spy_metrics : single metrics dict for the SPY benchmark
    """
    metric_keys   = ["CAGR", "Sharpe", "Sortino", "Max DD", "Calmar"]
    metric_labels = ["CAGR", "Sharpe", "Sortino", "Max Drawdown", "Calmar"]
    pct_metrics   = {"CAGR", "Max DD"}
    invert        = {"Max DD"}         # lower is better → invert y-axis

    strategies = list(all_metrics.keys())
    xlabels    = [LABELS.get(s, s) for s in strategies]
    if spy_metrics:
        strategies.append("SPY")
        xlabels.append("S&P 500 (SPY)")

    fig, axes = plt.subplots(1, len(metric_keys), figsize=(22, 6), sharey=False)
    fig.suptitle(
        "Multi-Seed Robustness  ·  Mean ± Std across random stock samples",
        fontsize=14, fontweight="bold",
    )

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
                      capsize=5, edgecolor="white", alpha=0.88, error_kw={"linewidth": 1.5})

        ax.set_title(label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=7.5)

        if key in pct_metrics:
            ax.yaxis.set_major_formatter(_PCT)

        if key in invert:
            ax.invert_yaxis()
            ax.set_title(f"{label}  (↓ better)", fontweight="bold")

        # Value labels on bars
        for bar, mean in zip(bars, means):
            h   = bar.get_height()
            txt = f"{mean:.0%}" if key in pct_metrics else f"{mean:.2f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    h + (max(errs) * 0.05 if max(errs) else abs(h) * 0.03),
                    txt, ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


# ---------------------------------------------------------------------------
# Lambda heatmap across seeds
# ---------------------------------------------------------------------------

def plot_lambda_heatmap(
    lambda_series_by_seed: Dict[int, pd.Series],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Monthly median λ across seeds shown as a heatmap (years × seeds)."""
    frames = []
    for seed, lams in lambda_series_by_seed.items():
        monthly = lams.resample("ME").median()
        monthly.name = f"Seed {seed}"
        frames.append(monthly)

    if not frames:
        return None

    df    = pd.concat(frames, axis=1).dropna()
    years = df.index.year.unique()

    # Pivot: rows = year, cols = seed
    df["Year"] = df.index.year
    annual = df.groupby("Year").mean()

    fig, ax = plt.subplots(figsize=(max(6, len(frames) + 2), 5))
    sns.heatmap(
        annual.T, ax=ax,
        cmap="RdBu_r", center=0.5, vmin=0.0, vmax=1.0,
        linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size": 8},
        cbar_kws={"label": "Median λ  (0 = trust prior, 1 = trust recent)"},
    )
    ax.set_title("Annual Median λ by Seed  ·  Regime Shifts Over Time",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("Seed")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
