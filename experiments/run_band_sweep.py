#!/usr/bin/env python3
"""
Mid-priced YES — price-band sweep
===================================
Sweeps multiple [lo, hi) price bands using the existing midprice_backtest.csv
(no API calls needed). For each band at each horizon computes hold performance.

Outputs:
  data/paper_trade/band_sweep.csv   — per-band × horizon stats
  data/paper_trade/band_sweep.png   — heatmap + bar chart
"""
import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data/paper_trade")

# ── Bands to sweep ──────────────────────────────────────────────────────────
# Full sweep: whole range + sliding 0.10-wide windows + user-requested bands
BANDS = [
    (0.10, 0.90),   # full "mid" range
    (0.10, 0.50),   # cheap half (live strategy)
    (0.50, 0.90),   # expensive half
    (0.10, 0.20),   # 10s
    (0.20, 0.30),   # 20s
    (0.30, 0.40),   # 30s
    (0.40, 0.50),   # 40s
    (0.50, 0.60),   # 50s
    (0.60, 0.70),   # 60s
    (0.70, 0.80),   # 70s
    (0.80, 0.90),   # 80s
    # user-requested mid windows
    (0.20, 0.40),
    (0.30, 0.50),
    (0.10, 0.30),
    (0.40, 0.70),
]

HORIZONS = [3, 5, 7, 10, 14]
BET = 0.10   # 10% per trade (same as live strategy)


def stats(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return {"n": 0, "win_rate": float("nan"), "mean_ret": float("nan"),
                "sharpe": float("nan"), "tot_pnl": 0.0}
    wr = (rets > 0).mean()
    mu = rets.mean()
    sd = rets.std(ddof=1) if len(rets) > 1 else float("nan")
    sh = mu / sd if (sd and not np.isnan(sd) and sd > 0) else float("nan")
    return {"n": len(rets), "win_rate": round(float(wr), 4),
            "mean_ret": round(float(mu * BET), 4),
            "sharpe": round(float(sh), 3),
            "tot_pnl": round(float(rets.sum() * BET), 2)}


def main() -> None:
    raw = pd.read_csv(DATA / "midprice_backtest.csv")
    print(f"Loaded {len(raw)} rows from midprice_backtest.csv")

    rows = []
    for (lo, hi) in BANDS:
        for h in HORIZONS:
            sub = raw[(raw["horizon_d"] == h) &
                      (raw["price"] >= lo) & (raw["price"] < hi)]
            if len(sub) == 0:
                continue
            rets = np.where(sub["outcome"] == 1,
                            1.0 / sub["price"].values - 1.0,
                            -1.0)
            s = stats(rets)
            s.update({"lo": lo, "hi": hi, "horizon_d": h,
                      "band": f"[{lo:.2f},{hi:.2f})"})
            rows.append(s)

    df = pd.DataFrame(rows).sort_values(["lo", "hi", "horizon_d"])
    df.to_csv(DATA / "band_sweep.csv", index=False)
    print(f"Saved {len(df)} rows → data/paper_trade/band_sweep.csv\n")

    # ── Print table ──────────────────────────────────────────────────────────
    print(f"{'band':>14} | {'h':>3} | {'n':>4} | {'win%':>6} | {'mean$':>7} | {'Sharpe':>7} | {'totPnL':>7}")
    print(f"  {'-'*14}-+-{'-'*3}-+-{'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for _, r in df.iterrows():
        if r["n"] < 5:
            continue
        print(f"  {r['band']:>14} | {int(r['horizon_d']):>3} | {int(r['n']):>4} | "
              f"{r['win_rate']:>5.1%} | {r['mean_ret']:>+6.3f} | "
              f"{r['sharpe']:>7.3f} | {r['tot_pnl']:>+7.2f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    # 1) Sharpe heatmap: bands × horizons (for 0.10-wide sliding windows)
    slide_bands = [(0.10,0.20),(0.20,0.30),(0.30,0.40),(0.40,0.50),
                   (0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90)]
    band_labels = [f"{int(lo*100)}-{int(hi*100)}" for lo, hi in slide_bands]
    matrix = np.full((len(slide_bands), len(HORIZONS)), np.nan)
    for i, (lo, hi) in enumerate(slide_bands):
        for j, h in enumerate(HORIZONS):
            sub = df[(df["lo"] == lo) & (df["hi"] == hi) & (df["horizon_d"] == h)]
            if not sub.empty and sub.iloc[0]["n"] >= 5:
                matrix[i, j] = sub.iloc[0]["sharpe"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Heatmap
    ax = axes[0]
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.3, vmax=0.5)
    ax.set_xticks(range(len(HORIZONS))); ax.set_xticklabels([f"{h}d" for h in HORIZONS])
    ax.set_yticks(range(len(slide_bands))); ax.set_yticklabels(band_labels)
    ax.set_title("Sharpe by price band × horizon\n(mid-priced YES, hold to resolution)")
    ax.set_xlabel("Entry horizon"); ax.set_ylabel("Price band")
    for i in range(len(slide_bands)):
        for j in range(len(HORIZONS)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if abs(v) < 0.4 else "white")
    plt.colorbar(im, ax=ax, label="Sharpe")

    # Bar chart: selected bands at horizon=5
    ax2 = axes[1]
    sel_bands = [(0.10,0.50),(0.10,0.90),(0.10,0.20),(0.20,0.30),(0.30,0.40),
                 (0.40,0.50),(0.20,0.40),(0.10,0.30),(0.30,0.50)]
    h_sel = 5
    labels2, sharpes2, colors2 = [], [], []
    cmap_b = matplotlib.colormaps["Blues"].resampled(len(sel_bands) + 3)
    for k, (lo, hi) in enumerate(sel_bands):
        sub = df[(df["lo"] == lo) & (df["hi"] == hi) & (df["horizon_d"] == h_sel)]
        if sub.empty or sub.iloc[0]["n"] < 5:
            continue
        sh = sub.iloc[0]["sharpe"]
        labels2.append(f"[{lo:.2f},{hi:.2f})")
        sharpes2.append(sh)
        colors2.append(cmap_b(k + 3))

    bars = ax2.barh(labels2, sharpes2, color=colors2, edgecolor="white")
    ax2.axvline(0, color="grey", lw=0.8)
    ax2.set_title(f"Sharpe by band — horizon={h_sel}d\n(hold to resolution, BET=10%)")
    ax2.set_xlabel("Sharpe ratio")
    for bar, v in zip(bars, sharpes2):
        ax2.text(v + 0.005 if v >= 0 else v - 0.005, bar.get_y() + bar.get_height()/2,
                 f"{v:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)

    fig.suptitle("Mid-priced YES: price-band performance sweep", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(DATA / "band_sweep.png", dpi=130, bbox_inches="tight")
    print("\n  Plot → data/paper_trade/band_sweep.png")
    print("✓  Band sweep complete.")


if __name__ == "__main__":
    main()
