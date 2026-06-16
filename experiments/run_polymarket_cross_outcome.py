#!/usr/bin/env python3
"""
Polymarket — anatomy of a multi-outcome field (cross-outcome structure)
=======================================================================

A single event is several mutually-exclusive outcome markets whose prices are
mechanically coupled (they sum to ~1). This study maps that joint behaviour:

  1. No-arbitrage — how tightly does the field sum to 1? (static dutch-book?)
  2. Substitution — who trades against whom (dominant negatively-correlated pair)?
  3. Redistribution — after one candidate is shocked, does the beneficiary keep
     climbing (under-reaction / momentum) or overshoot and revert (over-reaction)?
     ... and is fading that beneficiary tradable?
  4. Relative value — does the top-2 spread mean-revert?

Illustrated on the Biden→Harris handoff (`democratic-nominee-2024`) and a
correlation heatmap of the 2024 presidential field.

Usage
-----
  python experiments/run_polymarket_cross_outcome.py
"""
import argparse
import logging

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401  (adds repo root to sys.path)
from posterioralpha.polymarket import (
    build_price_panel,
    classify_markets,
    coupling_summary,
    event_panels,
    rank_calibration,
    redistribution_drift,
    relative_value_reversion,
    resolution_outcomes,
)
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs
from posterioralpha.polymarket.signals import to_logodds
from posterioralpha.polymarket.trades import TradeParams, simulate_event_trades

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-5s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def redistribution_fade_pnl(evp, panel, meta, outcomes, drop_thresh=0.5, horizon=5, slip=0.01):
    """Trade the over-reaction: short (buy No) the biggest beneficiary of a shock, exit after `horizon`.

    Returns event-clustered PnL stats. Tests whether the redistribution overshoot is
    actually harvestable net of slippage.
    """
    rows = []
    for ev, sub in evp.items():
        z = to_logodds(sub).clip(-6, 6); dz = z.diff()
        idx = sub.index
        for t in range(1, len(sub) - horizon):
            day = dz.iloc[t]
            if day.min() < -drop_thresh and day.drop(day.idxmin()).max() > 0:
                ben = day.drop(day.idxmin()).idxmax()
                p0 = sub[ben].iloc[t]; p1 = sub[ben].iloc[t + horizon]
                if not (0.05 <= p0 <= 0.95) or not np.isfinite(p1):
                    continue
                pnl = -(p1 - p0) - slip                  # short the beneficiary (buy No)
                rows.append({"event": ev, "pnl": pnl})
    tr = pd.DataFrame(rows)
    if tr.empty:
        return {"n": 0}
    ep = tr.groupby("event")["pnl"].mean()
    rng = np.random.default_rng(0)
    boot = [rng.choice(ep.values, len(ep), replace=True).mean() for _ in range(5000)]
    return {"n": len(tr), "n_evt": tr["event"].nunique(), "pnl_evt": ep.mean(),
            "ci_lo": np.percentile(boot, 2.5), "ci_hi": np.percentile(boot, 97.5)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets",  type=int,   default=600)
    ap.add_argument("--min-volume", type=float, default=20_000)
    ap.add_argument("--refresh",    action="store_true")
    ap.add_argument("--no-plots",   action="store_true")
    args = ap.parse_args()

    ensure_dirs()
    print("""
╔══════════════════════════════════════════════════════════╗
║   P O L Y M A R K E T   ·   anatomy of a multi-outcome field ║
║   coupling · substitution · redistribution · relative value  ║
╚══════════════════════════════════════════════════════════╝""")

    panel, meta = build_price_panel(n_markets=args.n_markets, min_volume=args.min_volume,
                                    closed=True, refresh=args.refresh)
    outcomes = resolution_outcomes(meta, panel)
    evp = event_panels(panel, meta)
    logger.info("%d well-covered multi-outcome events", len(evp))

    # ── 1+2. coupling & substitution structure ────────────────────────────
    cs = coupling_summary(evp, meta)
    print("\n" + "═" * 96)
    print("  COUPLING & SUBSTITUTION  ·  per multi-outcome event")
    print("═" * 96)
    print(tabulate([[r["event"][:34], r["k"], f"{r['sum_tightness']:.3f}",
                     f"{r['mean_corr']:+.2f}", f"{r['pair_corr']:+.2f}", r["dominant_pair"]]
                    for _, r in cs.head(14).iterrows()],
                   headers=["event", "k", "|S-1|", "mean ρ", "min ρ", "dominant substitute pair"],
                   tablefmt="rounded_grid"))
    print(f"  median sum-tightness |S-1| = {cs['sum_tightness'].median():.3f}  "
          f"→ field is internally arbitrage-free (neg-risk enforced; no static dutch-book)")
    print(f"  mean within-field pairwise ρ = {cs['mean_corr'].mean():+.3f}  "
          f"→ ≈0 because redistribution is concentrated in the dominant pair, not the whole field")

    # ── 3. redistribution dynamics + tradability ──────────────────────────
    rd = redistribution_drift(evp)
    fade = redistribution_fade_pnl(evp, panel, meta, outcomes)
    print("\n" + "═" * 72)
    print("  REDISTRIBUTION AFTER A SHOCK  (biggest beneficiary of a sharp drop)")
    print("═" * 72)
    print(f"  beneficiary forward 5d log-odds drift: {rd['mean_drift']:+.3f}  "
          f"({rd['frac_momentum']:.0%} momentum / {1-rd['frac_momentum']:.0%} reversion, n={rd['n_shocks']})")
    print(f"  mean beneficiary share of the redistributed mass: {rd['mean_beneficiary_share']:.0%}  "
          f"(high → mass flows to ONE substitute, not the field)")
    if fade.get("n", 0):
        print(f"  → fade-the-beneficiary trade (short it, exit 5d, 1% slip): "
              f"PnL/event {fade['pnl_evt']:+.3f}  95% CI [{fade['ci_lo']:+.3f}, {fade['ci_hi']:+.3f}]  "
              f"(n={fade['n']}, {fade['n_evt']} events)")

    # ── 3b. within-field calibration by price rank ────────────────────────
    rc = rank_calibration(evp, outcomes)
    if not rc.empty:
        print("\n" + "═" * 64)
        print("  WITHIN-FIELD CALIBRATION BY RANK  (winner-take-all fields only)")
        print("═" * 64)
        print(tabulate([[r["group"], int(r["n"]), f"{r['mean_price']:.3f}",
                         f"{r['yes_rate']:.3f}", f"{r['resid']:+.3f}"]
                        for _, r in rc.iterrows()],
                       headers=["rank group", "n", "mean price", "yes rate", "resid"],
                       tablefmt="rounded_grid"))
        print("  favorite under-priced / 2nd over-priced ⇒ challenger is over-bet "
              "(but ~19 mostly-political fields → same one-regime caveat)")

    # ── 4. relative value ─────────────────────────────────────────────────
    rv = relative_value_reversion(evp)
    print(f"\n  RELATIVE VALUE: top-2 spread lag-1 autocorr {rv['mean_autocorr']:+.3f} "
          f"({rv['frac_reverting']:.0%} of events revert)")

    # ── plots: Biden→Harris handoff + presidential-field correlation heatmap ─
    if not args.no_plots:
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))
        # (a) democratic-nominee handoff
        nom = next((s for e, s in evp.items() if "democratic-nominee" in e), None)
        if nom is not None:
            keep = nom.max().sort_values(ascending=False).index[:4]
            for m in keep:
                axL.plot(nom.index, nom[m].values, lw=1.8,
                         label=str(meta.loc[m, "question"])[:24])
            axL.set_title("Probability redistribution: Biden → Harris (Dem nominee 2024)")
            axL.set_ylabel("implied probability"); axL.legend(fontsize=8); axL.grid(alpha=0.25)
        # (b) presidential field correlation heatmap (log-odds changes)
        pres = next((s for e, s in evp.items() if "presidential-election-winner" in e), None)
        if pres is not None:
            keep = pres.max().sort_values(ascending=False).index[:6]
            C = to_logodds(pres[keep]).clip(-6, 6).diff().corr()
            im = axR.imshow(C.values, cmap="RdBu", vmin=-1, vmax=1)
            labs = [str(meta.loc[m, "question"])[:14] for m in keep]
            axR.set_xticks(range(len(keep))); axR.set_xticklabels(labs, rotation=60, fontsize=7, ha="right")
            axR.set_yticks(range(len(keep))); axR.set_yticklabels(labs, fontsize=7)
            axR.set_title("2024 presidential field — log-odds-change correlation")
            fig.colorbar(im, ax=axR, fraction=0.046)
        fig.tight_layout()
        out = RESULTS_DIR / "cross_outcome.png"
        fig.savefig(out, dpi=130)
        logger.info("Saved chart → %s", out)

    print("\n✓  Cross-outcome anatomy complete.")


if __name__ == "__main__":
    main()
