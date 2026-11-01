"""Chart from docs/dipbuy/intraday_entries.csv: Q1-2 common stocks, return to the official close by entry.
Left: 2025-01 -> 2026-10 (hourly bars: entries available all window). Right: last ~60 sessions (5-min bars),
every entry on the SAME trades."""
import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy.paths import DOCS_DIR

R = pd.read_csv(DOCS_DIR / "intraday_entries.csv")
R = R[R.liq == "Q1-2"]
left = ["official open (E1)", "Yahoo first trade", "~10:30", "1st-hour VWAP proxy"]
right = ["official open (E1)", "Yahoo first trade", "09:35", "09:45", "VWAP 5m", "VWAP 15m", "VWAP 30m", "~10:30", "1st-hour VWAP proxy"]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), dpi=140, gridspec_kw={"width_ratios": [4, 9]}, sharey=True)
for ax, win_key, order in ((axes[0], "60m", left), (axes[1], "5m", right)):
    W = R[R.window.str.startswith(win_key)]
    ns = {}
    for k, (side, col, lab) in enumerate((("down", "#2a78d6", "gap-down: buy, sell at close"), ("up", "#eb6834", "gap-up: short, cover at close"))):
        x = W[W.side == side].set_index("entry").reindex(order)
        pos = np.arange(len(order)) + (k - 0.5) * 0.38
        ax.bar(pos, x["mean"] * 100, 0.36, color=col, label=lab)
        ax.errorbar(pos, x["mean"] * 100, yerr=1.96 * x["se"] * 100, fmt="none", ecolor="#52514e", lw=0.8)
        ns[side] = int(x.N.max())
    ax.axhline(0, color="#52514e", lw=0.8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
    title = "2025-01 → 2026-10 (hourly bars)" if win_key == "60m" else "Last ~60 sessions only (5-min bars), same trades for every entry"
    ax.set_title(f"{title}\n{ns['down']} dips, {ns['up']} gap-ups", fontsize=9, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("mean per trade to official close, %\n(net 0.10%, 95% CI clustered by date)", fontsize=8)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("US NYSE/NYSE American, Q1–Q2 COMMON STOCKS, official-auction gap signals: return by entry time",
             fontsize=9.5, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(DOCS_DIR / "intraday_entries_2025.png"); print("ok")
