#!/usr/bin/env python3
"""
Where does the net-liquidity effect live? — cross-asset sensitivity.

Net liquidity should move *risk-on* assets most and defensives least.  This
study measures, for every ETF in the large cross-asset universe, how sensitive
its returns are to net-liquidity changes, ranks asset classes, and checks the
ranking holds out-of-sample.

  1. Sensitivity = correlation of an asset's forward 6-month return with the
     6-month net-liquidity change (the horizon where the lead peaks), plus the
     contemporaneous correlation.  Ranked per asset and by fd asset class.
  2. Out-of-sample check: rank sensitivity on 2010–2017, then on 2018→ compare
     each asset's mean return in liquidity-expansion vs contraction regimes for
     the most- vs least-sensitive baskets — does the effect persist and
     concentrate where we predicted?

Runs offline on the bundled datasets.

    python experiments/run_net_liquidity_universe.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import (
    load_net_liquidity, load_etf_universe_prices, load_etf_universe_info,
)
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RF = 0.04
DAILY_RF = RF / 252
H = 126               # ~6-month horizon (peak lead from run_net_liquidity.py)
PUB_LAG = 5
SPLIT = "2018-01-01"  # in-sample / out-of-sample boundary
MIN_DAYS = 252 * 5    # require ~5y history to rank an asset


def _corr(a, b):
    m = a.notna() & b.notna()
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else np.nan


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    px = load_etf_universe_prices()
    info = load_etf_universe_info()
    idx = nl.dropna().index.intersection(px.index)
    nl, px = nl.reindex(idx), px.reindex(idx)
    d_nl = nl.diff(H)

    print("\n" + "=" * 78)
    print(f"  NET-LIQUIDITY SENSITIVITY ACROSS {px.shape[1]} ETFs  ·  "
          f"{idx.min().date()} → {idx.max().date()}")
    print(f"  sensitivity = corr(Δ net-liquidity over {H}d, asset return over next {H}d)")
    print("=" * 78)

    # ── 1. Per-asset sensitivity (full-sample, descriptive) ──────────────────
    recs = []
    for c in px.columns:
        r = px[c]
        if r.notna().sum() < MIN_DAYS:
            continue
        lead = _corr(d_nl, r.pct_change(H).shift(-H))
        contemp = _corr(d_nl, r.pct_change(H))
        recs.append((c, contemp, lead))
    S = pd.DataFrame(recs, columns=["symbol", "contemp", "lead"]).set_index("symbol")
    S = S.join(info[["name", "category_group"]]).dropna(subset=["lead"])

    by_class = (S.groupby("category_group")["lead"]
                .agg(["mean", "count"]).sort_values("mean", ascending=False))
    print("\nLiquidity sensitivity by asset class (mean forward-return correlation):")
    print(tabulate(by_class.round(3), headers=["asset class", "mean lead corr", "n"],
                   tablefmt="github"))

    top = S.sort_values("lead", ascending=False).head(12)
    bot = S.sort_values("lead").head(12)
    show = lambda d: tabulate(
        d[["name", "category_group", "contemp", "lead"]].round(3),
        headers=["symbol", "name", "class", "contemp", "lead"], tablefmt="simple")
    print("\nMOST liquidity-sensitive ETFs:\n" + show(top))
    print("\nLEAST / negatively sensitive ETFs:\n" + show(bot))

    # ── 2. Out-of-sample check (rank IS, measure regime spread OOS) ──────────
    is_mask = idx < SPLIT
    is_sens = {}
    for c in px.columns:
        r = px[c]
        if r[is_mask].notna().sum() < MIN_DAYS // 2:
            continue
        is_sens[c] = _corr(d_nl[is_mask], r.pct_change(H).shift(-H)[is_mask])
    is_sens = pd.Series(is_sens).dropna()
    q = is_sens.quantile([0.25, 0.75])
    high = is_sens[is_sens >= q[0.75]].index
    low = is_sens[is_sens <= q[0.25]].index

    rets = px.pct_change()
    # causal liquidity regime: expanding (Δ6m > 0), lagged for publication
    expanding = (d_nl > 0).astype(float).shift(PUB_LAG).shift(1)
    oos = idx >= SPLIT
    exp_oos = expanding[oos] > 0

    def regime_spread(basket):
        b = rets[basket].mean(axis=1)[oos]
        up = b[exp_oos.reindex(b.index).fillna(False)].mean() * 252
        dn = b[~exp_oos.reindex(b.index).fillna(False)].mean() * 252
        return up, dn, up - dn

    hu, hd, hs = regime_spread(high)
    lu, ld, ls = regime_spread(low)
    print("\n" + "=" * 78)
    print(f"  OUT-OF-SAMPLE ({SPLIT}→): annualised return by liquidity regime")
    print(f"  baskets ranked on 2010–2017 sensitivity  ·  high n={len(high)} low n={len(low)}")
    print("=" * 78)
    print(tabulate([
        ["High-sensitivity basket", f"{hu:+.1%}", f"{hd:+.1%}", f"{hs:+.1%}"],
        ["Low-sensitivity basket",  f"{lu:+.1%}", f"{ld:+.1%}", f"{ls:+.1%}"],
    ], headers=["basket", "liquidity expanding", "contracting", "spread"],
        tablefmt="github"))
    print("\n  A larger high-basket spread = the liquidity effect concentrates "
          "where predicted, out-of-sample.")
    print("  (Overlapping windows inflate significance — read correlations as "
          "descriptive ranking, not t-tested alpha.)")

    # ── Plot: sensitivity by asset class ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    bc = by_class.sort_values("mean")
    ax.barh(bc.index, bc["mean"], color=np.where(bc["mean"] >= 0, "tab:green", "tab:red"),
            alpha=0.8)
    ax.axvline(0, color="grey", lw=0.6)
    ax.set_xlabel(f"mean corr( Δ net-liquidity, next-{H}d return )")
    ax.set_title("Net-liquidity sensitivity by asset class (ETF universe)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out = RESULTS_DIR / "net_liquidity_universe.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
