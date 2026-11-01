"""Chart for suspect B: opening-auction size vs retail tickets (XDP sample days 2023-10-02/03)."""
import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.dipbuy.paths import DOCS_DIR

C = pd.read_csv(DOCS_DIR / "capacity_auction.csv", index_col=0)
V = pd.read_csv(DOCS_DIR / "capacity_auction_by_venue.csv", index_col=0)
cols = [str(c) for c in C.columns]
q = np.arange(1, 6)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), dpi=140)
ax = axes[0]
for venue, col, lab in (("NYSE", "#2a78d6", "NYSE-listed"), ("AMEX", "#eb6834", "NYSE American-listed")):
    ax.bar(q + (-0.2 if venue == "NYSE" else 0.2), V.loc[venue].values.astype(float), 0.38, color=col, label=lab)
for tk, ls in ((2000, ":"), (5000, "--"), (20000, "-")):
    ax.axhline(tk, color="#52514e", lw=1, ls=ls); ax.text(5.55, tk * 1.08, f"${tk // 1000}k ticket", va="bottom", fontsize=8, color="#52514e")
ax.set_yscale("log"); ax.set_xticks(q); ax.set_xticklabels([f"Q{i}" for i in q])
ax.set_ylabel("median opening-auction $ value (log)", fontsize=8)
ax.set_title("Whole opening auction vs one retail ticket", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.set_xlim(0.5, 6.3)
ax = axes[1]
for i, (tk, col) in enumerate(((2000, "#1baf7a"), (5000, "#eda100"), (20000, "#6250d6"))):
    y = C.loc[f"share of names where ${tk // 1000}k > 10% of auction"].values.astype(float) * 100
    ax.bar(q + (i - 1) * 0.27, y, 0.25, color=col, label=f"${tk // 1000}k ticket")
ax.set_xticks(q); ax.set_xticklabels([f"Q{i}" for i in q]); ax.set_ylim(0, 105)
ax.set_ylabel("% of stocks where ticket > 10% of the auction", fontsize=8)
ax.set_title("How often a ticket would be >10% of the auction", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=8)
for a in axes:
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.suptitle("Capacity at the open: NYSE / NYSE American common stocks, XDP trade samples 2023-10-02 and 10-03 (n = 4,298 stock-days)",
             fontsize=9, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(DOCS_DIR / "capacity_auction.png"); print("ok")
