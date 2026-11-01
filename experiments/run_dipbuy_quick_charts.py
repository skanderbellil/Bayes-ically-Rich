import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from posterioralpha.dipbuy import backtest as bt, europe, stats
from posterioralpha.dipbuy.paths import RAW_DIR, DOCS_DIR
C = {"E1": "#2a78d6", "E2": "#eb6834", "E3": "#1baf7a"}

# --- chart 1: US in-sample per-trade means by Q
R = pd.read_csv(DOCS_DIR / "per_trade_by_model.csv")
R = R[R.Q.isin(["1", "2", "3", "4", "5"]) & (R.period == "IS 2015-21")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140, sharey=True)
for ax, side in zip(axes, ["gap-down LONG", "gap-up SHORT"]):
    for i, (m, lab, c) in enumerate([("E1_auction", "E1 auction→auction", C["E1"]), ("E2_strict_limit", "E2 strict limit", C["E2"]),
                                     ("E3_other_venue_entry", "E3 other-venue entry", C["E3"])]):
        x = R[(R.side == side) & (R.model == m)].sort_values("Q")
        ax.bar(np.arange(5) + (i - 1) * 0.27, x["mean"] * 100, 0.25, color=c, label=lab)
        ax.errorbar(np.arange(5) + (i - 1) * 0.27, x["mean"] * 100, yerr=1.96 * x["se"] * 100, fmt="none", ecolor="#52514e", lw=0.8)
    ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{q}" for q in range(1, 6)])
    ax.set_title(f"US {side}, 2015–2021 (net 0.10%)", fontsize=10, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("mean open→close per trade, % (95% CI, date-clustered)", fontsize=8); axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(DOCS_DIR / "quick_us_is_by_model.png")

# --- chart 2 + table: Europe 2025-2026 only
raw = pd.read_parquet(RAW_DIR / "europe_raw.parquet"); U = pd.read_parquet(RAW_DIR / "europe_universe.parquet")
idx = pd.read_parquet(RAW_DIR / "europe_index.parquet")
P = europe.build(raw, idx, U)
thr = np.maximum(0.03, 3 * P.sd60)
el = (P.year >= 2025) & P.Q.notna() & (P.prev_close >= 5) & P.sd60.notna() & (P.gap.abs() < 0.5) & ~P.open_bad
rows = []
for side in ("down", "up"):
    S = P[el & ((P.agap <= -thr) if side == "down" else (P.agap >= thr))].copy(); sg = 1 if side == "down" else -1
    S["E1"] = sg * (S.close / S.open - 1) - 0.001
    S["E2"] = S.E1.where((S.low < S.open * (1 - 1e-6)) if side == "down" else (S.high > S.open * (1 + 1e-6)))
    for m in ("E1", "E2"):
        for q, g in S.groupby("Q"):
            r = stats.clustered_mean(g[m], g.date); r.update(side=side, model=m, Q=int(q)); rows.append(r)
    if side == "down": Dn = S
E = pd.DataFrame(rows)
print(E.pivot_table(index=["side", "model"], columns="Q", values="mean").round(4))
print(E.pivot_table(index=["side", "model"], columns="Q", values="t").round(1))
print(E.pivot_table(index=["side", "model"], columns="Q", values="N"))
core = pd.DataFrame({"ret": idx.adjclose.pct_change(), "intra": idx.close / idx.open - 1})
days = core.loc["2025-01-01":].index
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
ax = axes[0]
for i, (side, m) in enumerate([("down", "E1"), ("down", "E2"), ("up", "E1"), ("up", "E2")]):
    x = E[(E.side == side) & (E.model == m)].sort_values("Q")
    ax.bar(np.arange(5) + (i - 1.5) * 0.2, x["mean"] * 100, 0.18, color=C[m], alpha=1 if side == "down" else 0.45,
           label=f"{'gap-down long' if side=='down' else 'gap-up short'} {m}")
ax.axhline(0, color="#52514e", lw=0.8); ax.set_xticks(range(5)); ax.set_xticklabels([f"Q{q}" for q in range(1, 6)])
ax.set_title("Europe small/mid caps, 2025–2026: per-trade net %", fontsize=10, loc="left"); ax.legend(frameon=False, fontsize=7)
ax = axes[1]
spy = (1 + core.ret.reindex(days).fillna(0)).cumprod(); ax.plot(spy.index, spy, color="#8a8984", ls="--", lw=2, label="STOXX 600 ETF")
for m, c in (("E1", C["E1"]), ("E2", C["E2"])):
    ov = (1 + bt.overlay(Dn[Dn.Q <= 2], m, core, days)).cumprod(); ax.plot(ov.index, ov, color=c, lw=2, label=f"core + Q1-2 overlay, {m}")
ax.set_yscale("log"); ax.legend(frameon=False, fontsize=8); ax.set_title("Growth of €1, 2025–2026 (log)", fontsize=10, loc="left")
for a in axes:
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(DOCS_DIR / "quick_europe_2025_26.png")
for m in ("E1", "E2"):
    print(m, stats.perf(bt.overlay(Dn[Dn.Q <= 2], m, core, days)), "| index", stats.perf(core.ret.reindex(days).fillna(0)))
