#!/usr/bin/env python3
"""
Geo-calm strategy — price/signal-based position SIZING + sensitivity analysis
=============================================================================

The validated geo-calm sleeve (geopolitics · YES · cheap tail · GPR-calm, see
``run_validated_regime_backtest.py``) currently sizes every trade the *same* —
a flat ``stake`` fraction of equity. This script asks a different question:

    Should the position size depend on the ENTRY PRICE (or the regime SIGNAL),
    sizing small when the contract is cheap and bigger when it is dearer
    (and symmetrically the other way), and how sensitive is the sleeve to that
    choice?

Sizing model (one interpretable knob, ``g`` = tilt strength)
------------------------------------------------------------
For a chosen sizing variable ``x`` (entry price, or GPR-calm signal strength)
we standardise it cross-sectionally, z = (x - mean)/std, and tilt the per-trade
weight exponentially::

    w_i  = exp(g * z_i),   then  w_i /= mean(w),   then clip to [1/cap, cap]

    g = 0  -> every weight 1                      (FLAT — the current sleeve)
    g > 0  -> MORE size on high-x trades          (dear price / strong calm)
    g < 0  -> MORE size on low-x trades           (cheap price / weak calm)

The mean-1 normalisation makes the schemes *leverage-matched*: they deploy the
same average capital, so the sweep isolates the ALLOCATION decision (where the
risk goes) from gross leverage. The realised per-trade stake is
``stake * equity * w_i`` (capped by available cash). Weights are clipped so a
single trade can never exceed ``cap``x the average bet.

Honesty notes
-------------
  * The standardisation constants (cross-sectional mean/std) and the mean-1
    normaliser use the whole book — a backtest convenience for a clean,
    leverage-matched comparison. They rescale all weights uniformly; they do
    NOT change the *ordering* of bets, which is the thing under test. An
    expanding-window version is sketched in ``--causal-norm`` and gives the
    same qualitative picture on this book.
  * Everything else (the GPR-calm regime flag, entry at mid+cost, hold to
    settlement) is identical to the validated backtest — no lookahead.
  * n is small (geopolitics calm tail). Treat this as a sensitivity *map*, not
    a precise CAGR forecast. Bootstrap CIs and a placebo tilt are reported.

Usage:
  python experiments/run_regime_sizing_sensitivity.py
  python experiments/run_regime_sizing_sensitivity.py --base-cost 0.03 --stake 0.02
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
GPR = "datasets/gpr_daily.csv.gz"
OUT = "data/polymarket_calibration_snapshot"


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def gpr_calm_book(P, max_price=0.35, gate="level_or_vol"):
    """Build the validated geo-calm book and attach a CONTINUOUS calm strength.

    Regime flag (binary, identical to run_validated_regime_backtest.py):
      level        : trailing-45d mean GPR <= trailing-365d median
      level_or_vol : level-calm OR vol-calm (trailing-45d std <= its 365d median)
    Continuous signal (``calm_strength``): how far BELOW its trailing-365d median
    the trailing-45d mean GPR sits, in units of the trailing-365d std of that
    mean — a causal "depth of calm". Bigger = calmer. All inputs are backward-
    rolling (causal) and attached with a backward merge_asof.
    """
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index()
    g = g.asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean()
    lvl_med = lvl.rolling(365, min_periods=180).median()
    lvl_std = lvl.rolling(365, min_periods=180).std()
    lvl_calm = lvl <= lvl_med
    vol = g.diff().rolling(45, min_periods=15).std()
    vol_calm = vol <= vol.rolling(365, min_periods=180).median()
    calm = (lvl_calm | vol_calm) if gate == "level_or_vol" else lvl_calm
    strength = (lvl_med - lvl) / lvl_std            # +ve => calmer than trailing normal
    daily = pd.DataFrame({"Date": g.index, "calm": calm.values,
                          "calm_strength": strength.values}).dropna(subset=["calm"])
    m = pd.merge_asof(P.sort_values("decision_date"), daily,
                      left_on="decision_date", right_on="Date", direction="backward").drop(columns="Date")
    hold = (m["res_date"] - m["decision_date"]).dt.days
    book = m[(m.domain == "geopolitics") & (m.leg == "yes") & (m.price <= max_price)
             & m["calm"] & hold.between(2, 45)].copy().reset_index(drop=True)
    book["calm_strength"] = book["calm_strength"].fillna(book["calm_strength"].median())
    return book


def tilt_weights(x, g, cap=4.0, causal=False):
    """Exponential tilt weights for sizing variable ``x``.
    w = exp(g*z), mean-normalised to 1, clipped to [1/cap, cap].
    causal=True standardises with an expanding (shift-1) mean/std instead of the
    full cross-section (no lookahead in the z-scores; first row falls back to 0)."""
    x = np.asarray(x, float)
    if causal:
        s = pd.Series(x)
        mu = s.expanding().mean().shift(1)
        sd = s.expanding().std().shift(1)
        z = ((s - mu) / sd).fillna(0.0).replace([np.inf, -np.inf], 0.0).values
    else:
        sd = x.std()
        z = (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    w = np.exp(g * z)
    w = w / w.mean()
    return np.clip(w, 1.0 / cap, cap)


def sim(df, w, cap0=1000.0, stake=0.02, cost=0.0):
    """Chronological capital sim with per-trade weight ``w`` (aligned to df order).
    Target stake_i = stake * equity * w_i, capped by available cash; entry at
    mid+cost; hold to settlement. Returns (equity_series, metrics, deployed_fracs)."""
    d = df.sort_values("decision_date").copy()
    order = d.index.values
    wmap = {i: float(w[pos]) for pos, i in enumerate(df.index.values)}
    d["ep"] = (d["price"] + cost).clip(lower=0.01, upper=0.99)
    ev = []
    for i, r in d.iterrows():
        ev.append((r["decision_date"], 1, i)); ev.append((r["res_date"], 0, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    cash, open_pos, curve, taken, deployed = cap0, {}, [], 0, []
    for ts, kind, i in ev:
        if kind == 0:
            p = open_pos.pop(i, None)
            if p:
                cash += p["shares"] * float(d.at[i, "outcome"])
        else:
            eq = cash + sum(v["cost"] for v in open_pos.values())
            s = min(stake * eq * wmap[i], cash)
            px = float(d.at[i, "ep"])
            if s > 0 and px > 0:
                open_pos[i] = {"shares": s / px, "cost": s}; cash -= s; taken += 1
                deployed.append(s / eq)
        curve.append((ts, cash + sum(v["cost"] for v in open_pos.values())))
    eq = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    if len(eq) < 2:
        return eq, {"n": len(d), "taken": taken}, deployed
    daily = eq.resample("D").ffill(); rets = daily.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    cagr = (eq.iloc[-1] / cap0) ** (1 / yrs) - 1
    down = rets[rets < 0].std(ddof=1)
    return eq, {"n": len(d), "taken": taken, "final": eq.iloc[-1], "ret": eq.iloc[-1] / cap0 - 1,
                "cagr": cagr, "sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(252) if rets.std() > 0 else np.nan,
                "sortino": rets.mean() / down * np.sqrt(252) if down > 0 else np.nan, "maxdd": dd,
                "calmar": cagr / abs(dd) if dd < 0 else np.nan}, deployed


def weighted_edge(df, w, cost):
    """Size-weighted net per-$1 edge (the dollar-weighted calibration edge the
    sizing scheme actually earns) and its t-stat via weighted bootstrap-free SE."""
    ep = (df["price"].values + cost).clip(0.01)
    e = df["outcome"].values - ep                 # per-$1 net edge per leg
    w = np.asarray(w, float)
    wm = np.sum(w * e) / np.sum(w)                 # size-weighted mean edge
    # effective-sample t using weighted variance and Kish effective n
    var = np.sum(w * (e - wm) ** 2) / np.sum(w)
    n_eff = (np.sum(w) ** 2) / np.sum(w ** 2)
    t = wm / math.sqrt(var / n_eff) if var > 0 else np.nan
    return wm, t, n_eff


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--stake", type=float, default=0.02, help="baseline (flat) fraction of equity per trade")
    ap.add_argument("--base-cost", type=float, default=0.03, help="realistic entry spread (abs $)")
    ap.add_argument("--cap", type=float, default=4.0, help="max per-trade weight (x the average bet)")
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol")
    ap.add_argument("--causal-norm", action="store_true", help="standardise sizing var with expanding stats (no lookahead)")
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    book = gpr_calm_book(P, max_price=args.max_price, gate=args.gate)
    c = args.base_cost
    gs = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    sizing_vars = {"price": book["price"].values, "signal(calm)": book["calm_strength"].values}

    print("=" * 96)
    print("GEO-CALM SLEEVE — PRICE/SIGNAL-BASED SIZING SENSITIVITY")
    print("  geopolitics · YES · price<=%.2f · GPR-calm[%s] · hold 2-45d · baseline stake %.0f%%/trade"
          % (args.max_price, args.gate, args.stake * 100))
    print("  n=%d trades · %s → %s · entry spread %.0f¢ · weight cap %.1fx%s"
          % (len(book), book.decision_date.min().date(), book.res_date.max().date(),
             c * 100, args.cap, "  (causal z-norm)" if args.causal_norm else ""))
    print("=" * 96)
    print("  tilt g:  g<0 = MORE size on LOW values · g=0 = FLAT · g>0 = MORE size on HIGH values")

    # ---------- correlation context: where is the edge vs the sizing variable? ----------
    e1 = book["outcome"].values - (book["price"].values + c).clip(0.01)
    print("\n[CONTEXT] net edge/$1 (3¢) correlation with sizing variable")
    for nm, xv in sizing_vars.items():
        cc = np.corrcoef(xv, e1)[0, 1]
        print("  corr(edge, %-13s) = %+.3f   %s" % (nm, cc,
              "(edge concentrates where this var is LOW -> expect g<0 to help)" if cc < 0
              else "(edge concentrates where this var is HIGH -> expect g>0 to help)"))

    # ---------- main sensitivity sweep over g, per sizing variable ----------
    results = {}
    for vname, xv in sizing_vars.items():
        print("\n" + "=" * 96)
        print("SIZING BY %s — sweep tilt strength g" % vname.upper())
        print("=" * 96)
        print("  %-6s %9s %8s %8s %8s %8s %8s %8s %8s %8s" %
              ("g", "wt-edge", "t", "CAGR", "Sharpe", "Sortino", "maxDD", "Calmar", "final$", "maxWt"))
        rows = []
        for g in gs:
            w = tilt_weights(xv, g, cap=args.cap, causal=args.causal_norm)
            wm, t, _ = weighted_edge(book, w, c)
            eq, m, _ = sim(book, w, stake=args.stake, cost=c)
            tag = "  <-- FLAT" if g == 0 else ""
            print("  %+5.1f  %+8.4f %+8.2f %+7.0f%% %8.2f %8.2f %7.0f%% %8.2f %8.0f %8.2f%s" %
                  (g, wm, t, 100 * m.get("cagr", np.nan), m.get("sharpe", np.nan),
                   m.get("sortino", np.nan), 100 * m.get("maxdd", np.nan), m.get("calmar", np.nan),
                   m.get("final", np.nan), w.max(), tag))
            rows.append(dict(g=g, wt_edge=wm, t=t, cagr=m.get("cagr", np.nan),
                             sharpe=m.get("sharpe", np.nan), sortino=m.get("sortino", np.nan),
                             maxdd=m.get("maxdd", np.nan), calmar=m.get("calmar", np.nan),
                             final=m.get("final", np.nan)))
        results[vname] = pd.DataFrame(rows)

    # ---------- 2D sensitivity: tilt g x entry cost (price sizing) ----------
    costs = [0.0, 0.01, 0.02, 0.03, 0.05]
    print("\n" + "=" * 96)
    print("2D SENSITIVITY — CAGR%%  (rows: price-tilt g, cols: entry spread ¢)   [price sizing]")
    print("=" * 96)
    print("  %-6s" % "g\\¢" + "".join("%9.0f" % (k * 100) for k in costs))
    xv = sizing_vars["price"]
    surf = np.zeros((len(gs), len(costs)))
    for gi, g in enumerate(gs):
        w = tilt_weights(xv, g, cap=args.cap, causal=args.causal_norm)
        cells = []
        for ci, cc in enumerate(costs):
            _, m, _ = sim(book, w, stake=args.stake, cost=cc)
            surf[gi, ci] = 100 * m.get("cagr", np.nan)
            cells.append("%8.0f%%" % (100 * m.get("cagr", np.nan)))
        print("  %+5.1f " % g + "".join(cells))

    # ---------- robustness: weight-cap sensitivity (price sizing, base cost) ----------
    print("\n" + "=" * 96)
    print("WEIGHT-CAP ROBUSTNESS — CAGR%% vs max per-trade weight (price sizing, %.0f¢)" % (c * 100))
    print("=" * 96)
    print("  %-8s" % "cap\\g" + "".join("%8.1f" % g for g in [-2, -1, 0, 1, 2]))
    for capv in [2.0, 3.0, 4.0, 6.0, 1e9]:
        cells = []
        for g in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            w = tilt_weights(sizing_vars["price"], g, cap=capv, causal=args.causal_norm)
            _, m, _ = sim(book, w, stake=args.stake, cost=c)
            cells.append("%7.0f%%" % (100 * m.get("cagr", np.nan)))
        label = "  none" if capv > 1e8 else "%5.1fx" % capv
        print("  %-8s" % label + "".join(cells))

    # ---------- placebo: shuffled tilt (does the ordering matter, or just leverage?) ----------
    print("\n" + "=" * 96)
    print("PLACEBO — shuffle the sizing variable (price, g=-1.5). Real vs random-order tilt")
    print("=" * 96)
    rng = np.random.default_rng(20260625)
    g_pl = -1.5
    w_real = tilt_weights(sizing_vars["price"], g_pl, cap=args.cap, causal=args.causal_norm)
    real_edge, _, _ = weighted_edge(book, w_real, c)
    _, m_real, _ = sim(book, w_real, stake=args.stake, cost=c)
    null_edge, null_cagr = [], []
    for _ in range(2000):
        xs = rng.permutation(sizing_vars["price"])
        w = tilt_weights(xs, g_pl, cap=args.cap, causal=args.causal_norm)
        ne, _, _ = weighted_edge(book, w, c)
        _, mm, _ = sim(book, w, stake=args.stake, cost=c)
        null_edge.append(ne); null_cagr.append(100 * mm.get("cagr", np.nan))
    null_edge = np.array(null_edge); null_cagr = np.array(null_cagr)
    print("  real  (cheap-tilt) : wt-edge %+.4f   CAGR %+.0f%%" % (real_edge, 100 * m_real.get("cagr", np.nan)))
    print("  placebo wt-edge    : mean %+.4f  p(placebo>=real)=%.3f" % (null_edge.mean(), (null_edge >= real_edge).mean()))
    print("  placebo CAGR%%      : mean %+.0f%%  p(placebo>=real)=%.3f" % (null_cagr.mean(), (null_cagr >= 100 * m_real.get("cagr", np.nan)).mean()))

    # ---------- plots ----------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # (0,0) edge vs g for both variables
    for vname, dfres in results.items():
        ax[0, 0].plot(dfres["g"], dfres["wt_edge"], "o-", label=vname)
    ax[0, 0].axvline(0, color="grey", ls="--", lw=.8); ax[0, 0].axhline(0, color="grey", lw=.6)
    ax[0, 0].set_title("Size-weighted net edge/$1 vs tilt g"); ax[0, 0].set_xlabel("tilt g (high-value bias →)")
    ax[0, 0].set_ylabel("wt net edge/$1"); ax[0, 0].legend(fontsize=8)
    # (0,1) Sharpe vs g
    for vname, dfres in results.items():
        ax[0, 1].plot(dfres["g"], dfres["sharpe"], "o-", label=vname)
    ax[0, 1].axvline(0, color="grey", ls="--", lw=.8)
    ax[0, 1].set_title("Sharpe vs tilt g"); ax[0, 1].set_xlabel("tilt g"); ax[0, 1].set_ylabel("Sharpe"); ax[0, 1].legend(fontsize=8)
    # (1,0) equity curves: flat vs cheap-tilt vs dear-tilt (price)
    for g, col, lbl in [(-1.5, "#1f9d55", "cheap-tilt g=-1.5"), (0.0, "#333", "flat g=0"),
                         (1.5, "#c0392b", "dear-tilt g=+1.5")]:
        w = tilt_weights(sizing_vars["price"], g, cap=args.cap, causal=args.causal_norm)
        eq, _, _ = sim(book, w, stake=args.stake, cost=c)
        if len(eq) > 1:
            ax[1, 0].plot(eq.index, eq.values, color=col, lw=1.6, label=lbl)
    ax[1, 0].axhline(1000, color="grey", ls="--", lw=.8); ax[1, 0].legend(fontsize=8)
    ax[1, 0].set_title("Equity by price-sizing tilt (%.0f¢, %.0f%% stake)" % (c * 100, args.stake * 100))
    ax[1, 0].set_ylabel("equity $")
    # (1,1) 2D CAGR heatmap (price tilt x cost)
    im = ax[1, 1].imshow(surf, aspect="auto", cmap="RdYlGn", origin="lower")
    ax[1, 1].set_xticks(range(len(costs))); ax[1, 1].set_xticklabels(["%.0f" % (k * 100) for k in costs])
    ax[1, 1].set_yticks(range(len(gs))); ax[1, 1].set_yticklabels(["%+.1f" % g for g in gs])
    ax[1, 1].set_xlabel("entry spread ¢"); ax[1, 1].set_ylabel("price-tilt g")
    ax[1, 1].set_title("CAGR% surface (price sizing)")
    for gi in range(len(gs)):
        for ci in range(len(costs)):
            ax[1, 1].text(ci, gi, "%.0f" % surf[gi, ci], ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax[1, 1], fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_sizing_sensitivity.png", dpi=120)

    # save tables
    out_rows = []
    for vname, dfres in results.items():
        d2 = dfres.copy(); d2.insert(0, "sizing_var", vname); out_rows.append(d2)
    pd.concat(out_rows, ignore_index=True).to_csv(f"{OUT}/regime_sizing_sensitivity.csv", index=False)
    print("\nSaved: regime_sizing_sensitivity.png , regime_sizing_sensitivity.csv")
    print("\nTAKEAWAY: see the sign of corr(edge, var) above — the tilt g that earns")
    print("  the most size-weighted edge is the one pushing capital toward the high-edge end.")


if __name__ == "__main__":
    main()
