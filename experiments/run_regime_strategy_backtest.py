#!/usr/bin/env python3
"""
Regime-calm disruptive-tail strategy — backtest + hostile audit
===============================================================

Trades the effect found in REGIME_CALIBRATION.md: low-priced YES contracts on
*disruptive*-domain markets are underpriced **during calm regimes** (calibration
error +0.115, t 3.69) and fairly priced after recent shocks. The strategy:

  BUY the cheap (price <= --max-price) YES leg of a disruptive-domain market,
  a fixed lookback before resolution, ONLY when that domain has been calm
  (lagged surprise intensity S below its per-domain median). Hold to settlement,
  stake 1% of equity, pay a 1c haircut.

Built on the committed panel (data/polymarket_calibration_snapshot/), so it is
fast and reproducible. Reports variant comparison + the same hostile audit we ran
on the mid-price book, and writes equity / calibration / regime graphs.

Usage:
  python experiments/run_regime_strategy_backtest.py
  python experiments/run_regime_strategy_backtest.py --max-price 0.35 --side yes
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
OUT = "data/polymarket_calibration_snapshot"
DISRUPTIVE = {"geopolitics", "crypto/financial", "politics", "ai/tech"}

def tstat(x):
    x = np.asarray(x, float)
    return x.mean()/(x.std(ddof=1)/math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan

def sim(df, cap=1000.0, stake=0.01, haircut=0.01):
    """Chronological 1%-of-equity capital sim; hold to resolution. Returns (equity, summary)."""
    d = df.sort_values("decision_date").copy()
    d["ep"] = (d["price"] + haircut).clip(upper=0.99)
    ev = []
    for i, r in d.iterrows():
        ev.append((r["decision_date"], 1, i)); ev.append((r["res_date"], 0, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    cash, open_pos, curve, taken = cap, {}, [], 0
    for ts, kind, i in ev:
        if kind == 0:
            p = open_pos.pop(i, None)
            if p: cash += p["shares"]*float(d.at[i, "outcome"])
        else:
            eq = cash + sum(v["cost"] for v in open_pos.values())
            s = min(stake*eq, cash); px = float(d.at[i, "ep"])
            if s > 0 and px > 0:
                open_pos[i] = {"shares": s/px, "cost": s}; cash -= s; taken += 1
        curve.append((ts, cash + sum(v["cost"] for v in open_pos.values())))
    eq = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    if len(eq) < 2: return eq, {"n": len(d), "taken": taken}
    daily = eq.resample("D").ffill(); rets = daily.pct_change().dropna()
    yrs = max((eq.index[-1]-eq.index[0]).days/365.25, 1e-9)
    dd = ((eq-eq.cummax())/eq.cummax()).min()
    cagr = (eq.iloc[-1]/cap)**(1/yrs)-1
    down = rets[rets < 0].std(ddof=1)
    return eq, {"n": len(d), "taken": taken, "final": eq.iloc[-1], "ret": eq.iloc[-1]/cap-1,
                "cagr": cagr, "sharpe": rets.mean()/rets.std(ddof=1)*np.sqrt(252) if rets.std() > 0 else np.nan,
                "sortino": rets.mean()/down*np.sqrt(252) if down > 0 else np.nan, "maxdd": dd,
                "calmar": cagr/abs(dd) if dd < 0 else np.nan,
                "ulcer": np.sqrt((((eq-eq.cummax())/eq.cummax())**2).mean())}

def edge_row(df):
    ce = (df["outcome"]-df["price"]); r = df["outcome"]/(df["price"]+0.01)-1
    return dict(n=len(df), win=df["outcome"].mean(), edge=ce.mean(), t=tstat(ce),
                mean_ret=r.mean(), tot_ret=r.sum())

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--side", choices=["yes", "any"], default="yes")
    ap.add_argument("--haircut", type=float, default=0.01)
    args = ap.parse_args()
    ensure_dirs()
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    P["calm"] = P.groupby("domain")["S"].transform(lambda s: s <= s.median())

    base = P[P.domain.isin(DISRUPTIVE) & (P.price <= args.max_price)]
    if args.side == "yes":
        base = base[base.leg == "yes"]
    base = base.sort_values("decision_date").reset_index(drop=True)
    calm = base[base.calm]; turb = base[~base.calm]

    print("="*92)
    print("REGIME-CALM DISRUPTIVE-TAIL STRATEGY — backtest + audit")
    print("  filter: domain in %s, price<=%.2f, side=%s, 1%% stake, %.0f¢ haircut"
          % ("disruptive", args.max_price, args.side, args.haircut*100))
    print("="*92)

    # ---- variant comparison ----
    print("\n[VARIANTS] does the CALM filter add value? (this is the whole thesis)")
    for lbl, df in [("ALL disruptive-tail", base), ("CALM only (strategy)", calm), ("TURBULENT only", turb)]:
        _, m = sim(df, haircut=args.haircut); e = edge_row(df)
        print("  %-22s n=%3d  edge/$1=%+.4f (t=%+.2f)  win=%.2f  CAGR=%+6.1f%%  sharpe=%.2f  maxDD=%3.0f%%  final=$%.0f"
              % (lbl, e["n"], e["edge"], e["t"], e["win"], 100*m.get("cagr", np.nan),
                 m.get("sharpe", np.nan), 100*m.get("maxdd", np.nan), m.get("final", np.nan)))

    print("\n[BY DOMAIN] calm-tail edge per disruptive domain")
    for dom, sub in calm.groupby("domain"):
        e = edge_row(sub); print("  %-18s n=%3d  edge/$1=%+.4f (t=%+.2f)  win=%.2f" % (dom, e["n"], e["edge"], e["t"], e["win"]))

    print("\n[PRICE-BAND SWEEP] calm strategy, max-price cutoff")
    for mp in [0.15, 0.25, 0.35, 0.50]:
        sub = P[P.domain.isin(DISRUPTIVE) & (P.leg == "yes") & (P.price <= mp) & P.calm]
        e = edge_row(sub); _, m = sim(sub, haircut=args.haircut)
        print("  price<=%.2f  n=%3d  edge/$1=%+.4f (t=%+.2f)  CAGR=%+6.1f%%  maxDD=%3.0f%%" % (mp, e["n"], e["edge"], e["t"], 100*m.get("cagr", np.nan), 100*m.get("maxdd", np.nan)))

    # ============ HOSTILE AUDIT (on the calm strategy book) ============
    S = calm
    print("\n" + "-"*92 + "\n[AUDIT] on the CALM strategy book (n=%d)\n" % len(S) + "-"*92)

    print("[1] TIME STABILITY (by resolution year)")
    for yr, sub in S.assign(yr=S.res_date.dt.year).groupby("yr"):
        e = edge_row(sub); print("  %d  n=%3d  win=%.2f  edge/$1=%+.4f (t=%+.2f)  tot_ret=%+.1f" % (yr, e["n"], e["win"], e["edge"], e["t"], e["tot_ret"]))

    print("[1b] FIRST vs SECOND HALF")
    mid = S.decision_date.median()
    for lbl, sub in [("FIRST", S[S.decision_date <= mid]), ("SECOND", S[S.decision_date > mid])]:
        e = edge_row(sub); print("  %-6s n=%3d  edge/$1=%+.4f (t=%+.2f)  win=%.2f" % (lbl, e["n"], e["edge"], e["t"], e["win"]))

    print("[2] WALK-FORWARD")
    for cut in ["2025-01-01", "2025-07-01"]:
        ins = S[S.decision_date < cut]; oos = S[S.decision_date >= cut]
        ei, eo = edge_row(ins), edge_row(oos); _, mo = sim(oos, haircut=args.haircut)
        print("  IS<%s n=%3d edge=%+.4f(t=%+.2f) | OOS n=%3d edge=%+.4f(t=%+.2f) ret=%+.0f%% maxDD=%.0f%%"
              % (cut, ei["n"], ei["edge"], ei["t"], eo["n"], eo["edge"], eo["t"], 100*mo.get("ret", np.nan), 100*mo.get("maxdd", np.nan)))

    print("[5] COST STRESS")
    for hc in [0.01, 0.02, 0.03, 0.05]:
        _, m = sim(S, haircut=hc); ne = (S.outcome-(S.price+hc)).mean()
        print("  %2d¢: edge/$1=%+.4f  CAGR=%+6.1f%%  sharpe=%.2f  maxDD=%3.0f%%  final=$%.0f" % (hc*100, ne, 100*m.get("cagr", np.nan), m.get("sharpe", np.nan), 100*m.get("maxdd", np.nan), m.get("final", np.nan)))

    print("[7] BOOTSTRAP (week-clustered) on per-$1 edge")
    rng = np.random.default_rng(0); S2 = S.assign(rwk=S.res_date.dt.to_period("W").astype(str))
    wk = [g.assign(e=g.outcome-g.price) for _, g in S2.groupby("rwk")]
    wm = np.array([g["e"].mean() for g in wk]); ww = np.array([len(g) for g in wk])
    blk = np.array([((wm[ix]*ww[ix]).sum()/ww[ix].sum()) for ix in (rng.integers(0, len(wk), size=(10000, len(wk))))])
    print("  mean=%+.4f  5th=%+.4f  95th=%+.4f  P(edge<=0)=%.3f  (n_weeks=%d)" % (blk.mean(), np.percentile(blk, 5), np.percentile(blk, 95), (blk <= 0).mean(), len(wk)))

    print("[8] WINNER CONCENTRATION (equal-$)")
    g = (S.outcome/(S.price+args.haircut)-1).sort_values(ascending=False).to_numpy(); gross = g[g > 0].sum()
    for n in [1, 5, 10]:
        print("  top %2d = %.0f%% of gross profit" % (n, 100*g[:n].sum()/gross if gross > 0 else 0))

    _, m = sim(S, haircut=args.haircut)
    print("[11] RISK: CAGR=%.1f%% sharpe=%.2f sortino=%.2f calmar=%.2f maxDD=%.0f%% ulcer=%.3f"
          % (100*m.get("cagr", np.nan), m.get("sharpe", np.nan), m.get("sortino", np.nan), m.get("calmar", np.nan), 100*m.get("maxdd", np.nan), m.get("ulcer", np.nan)))

    # ============ GRAPHS ============
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for lbl, df, c in [("calm (strategy)", calm, "#1f9d55"), ("all disruptive-tail", base, "#888"), ("turbulent", turb, "#c0392b")]:
        eq, _ = sim(df, haircut=args.haircut)
        if len(eq) > 1: ax[0].plot(eq.index, eq.values, label=f"{lbl} (n={len(df)})", color=c, lw=1.4)
    ax[0].axhline(1000, color="grey", ls="--", lw=0.8); ax[0].set_title("Equity: calm filter vs not (1pct/trade)")
    ax[0].set_ylabel("equity (USD)"); ax[0].legend(fontsize=8)

    pb = pd.cut(P[P.domain.isin(DISRUPTIVE)].price, [0,.1,.2,.35,.5,.65,.8,1.0])
    cal = P[P.domain.isin(DISRUPTIVE)].groupby(pb, observed=True).agg(price=("price","mean"), real=("outcome","mean"))
    ax[1].plot([0,1],[0,1],"--",color="grey",lw=.8); ax[1].plot(cal.price, cal.real, "o-", color="#1f9d55")
    ax[1].set_title("Calibration: disruptive domains"); ax[1].set_xlabel("price"); ax[1].set_ylabel("realized YES freq")

    doms = ["geopolitics","crypto/financial","politics"]
    cm = [edge_row(calm[calm.domain==d])["edge"] for d in doms]
    tm = [edge_row(turb[turb.domain==d])["edge"] for d in doms]
    x = np.arange(len(doms)); ax[2].bar(x-.2, cm, .4, label="calm", color="#1f9d55"); ax[2].bar(x+.2, tm, .4, label="turbulent", color="#c0392b")
    ax[2].axhline(0, color="grey", lw=.8); ax[2].set_xticks(x); ax[2].set_xticklabels([d.split("/")[0] for d in doms], fontsize=8)
    ax[2].set_title("Tail edge by domain: calm vs turbulent"); ax[2].set_ylabel("edge/$1"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/regime_strategy.png", dpi=120)

    calm.assign(entry=calm.price+args.haircut, pnl=calm.outcome-(calm.price+args.haircut)).to_csv(f"{OUT}/regime_strategy_trades.csv", index=False)
    print("\nSaved: regime_strategy.png , regime_strategy_trades.csv")
    print("✓ done.")

if __name__ == "__main__":
    main()
