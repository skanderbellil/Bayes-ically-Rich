#!/usr/bin/env python3
"""
Validated geo-calm strategy — realistic-cost backtest + portfolio metrics
=========================================================================

Backtests the ONE survivor of the per-domain kill battery (geopolitics, GPR-calm,
cheap YES tail) as an actual portfolio sleeve, with an honest execution-cost model
and a full performance-metric panel — the thing the validation step did not do.

Strategy (identical to the live ledger / validation):
  universe = geopolitics markets, YES leg, decision price (mid, 7d pre-resolution)
             <= --max-price, resolving in [2,45] days,
  regime   = GPR-calm: trailing-45d mean GPR below its trailing-365d median,
             computed causally (no lookahead), and
  sizing   = --stake of equity per trade, hold to settlement (no exits).

Realistic costs:
  * ENTRY AT THE ASK, not the mid. The panel stores the mid; buying YES crosses the
    spread, so effective entry = mid + `c`. On thin geopolitical markets the spread
    IS the dominant cost, so we sweep c = 0 (mid, unrealistic) .. 5c and pick a
    conservative 3c base case. Min tick is 1c, so an absolute haircut models the
    spread better than a relative one for low-priced contracts.
  * NO EXIT COST — held to settlement; winners auto-settle at $1, losers at $0.
    Round-trip cost is the entry spread only (a genuine, creditable advantage).
  * CAPACITY — position capped at --cap-frac of a market's lifetime volume; we
    report the AUM ceiling implied by the thinnest markets. (Cumulative volume is a
    loose upper bound on real depth, so the ceiling is optimistic — flagged.)

Metrics: per-$1 edge + t, win rate, profit factor, total/CAGR, Sharpe, Sortino,
Calmar, max drawdown, Ulcer, avg hold, exposure, and a week-clustered bootstrap
90% CI on the net per-$1 edge.

Usage:
  python experiments/run_validated_regime_backtest.py
  python experiments/run_validated_regime_backtest.py --stake 0.02 --base-cost 0.03
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


def _max_concurrent(df):
    """Max number of positions open simultaneously (overlap of [decision, res] windows)."""
    ev = [(r.decision_date, 1) for r in df.itertuples()] + [(r.res_date, -1) for r in df.itertuples()]
    ev.sort(key=lambda e: (e[0], e[1]))
    cur = mx = 0
    for _, d in ev:
        cur += d; mx = max(mx, cur)
    return mx


def gpr_calm_flag(P, gate="level_or_vol"):
    """Causal GPR-calm flag per leg. Two gates:
      level        : trailing-45d mean GPR <= trailing-365d median (high-purity core)
      level_or_vol : level-calm OR vol-calm, where vol-calm = trailing-45d std of daily
                     GPR <= its trailing-365d median (wider book, ~2x trades).
    Both inputs are backward-rolling (causal) and attached with merge_asof backward."""
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index()
    g = g.asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean()
    lvl_calm = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std()
    vol_calm = vol <= vol.rolling(365, min_periods=180).median()
    calm = (lvl_calm | vol_calm) if gate == "level_or_vol" else lvl_calm
    daily = pd.DataFrame({"Date": g.index, "calm": calm.values}).dropna()
    m = pd.merge_asof(P.sort_values("decision_date"), daily,
                      left_on="decision_date", right_on="Date", direction="backward").drop(columns="Date")
    return m


def sim(df, cap=1000.0, stake=0.01, cost=0.0, cap_frac=None, fixed=None):
    """Chronological capital sim; entry at mid+cost, hold to settlement. Sizing is
    `fixed` dollars/trade if given (flat, non-compounding), else `stake` of equity.
    If cap_frac set, position is also capped at cap_frac * market volume."""
    d = df.sort_values("decision_date").copy()
    d["ep"] = (d["price"] + cost).clip(lower=0.01, upper=0.99)
    ev = []
    for i, r in d.iterrows():
        ev.append((r["decision_date"], 1, i)); ev.append((r["res_date"], 0, i))
    ev.sort(key=lambda e: (e[0], e[1]))
    cash, open_pos, curve, taken, deployed = cap, {}, [], 0, []
    for ts, kind, i in ev:
        if kind == 0:
            p = open_pos.pop(i, None)
            if p:
                cash += p["shares"] * float(d.at[i, "outcome"])
        else:
            eq = cash + sum(v["cost"] for v in open_pos.values())
            s = min(fixed, cash) if fixed is not None else min(stake * eq, cash)
            if cap_frac is not None:
                s = min(s, cap_frac * float(d.at[i, "volume"]))
            px = float(d.at[i, "ep"])
            if s > 0 and px > 0:
                open_pos[i] = {"shares": s / px, "cost": s}; cash -= s; taken += 1
                deployed.append(s / eq)
        curve.append((ts, cash + sum(v["cost"] for v in open_pos.values())))
    eq = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
    eq = eq[~eq.index.duplicated(keep="last")].sort_index()
    if len(eq) < 2:
        return eq, {"n": len(d), "taken": taken}
    daily = eq.resample("D").ffill(); rets = daily.pct_change().dropna()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    dd = ((eq - eq.cummax()) / eq.cummax()).min()
    cagr = (eq.iloc[-1] / cap) ** (1 / yrs) - 1
    down = rets[rets < 0].std(ddof=1)
    return eq, {"n": len(d), "taken": taken, "final": eq.iloc[-1], "ret": eq.iloc[-1] / cap - 1,
                "cagr": cagr, "sharpe": rets.mean() / rets.std(ddof=1) * np.sqrt(252) if rets.std() > 0 else np.nan,
                "sortino": rets.mean() / down * np.sqrt(252) if down > 0 else np.nan, "maxdd": dd,
                "calmar": cagr / abs(dd) if dd < 0 else np.nan,
                "ulcer": np.sqrt((((eq - eq.cummax()) / eq.cummax()) ** 2).mean()),
                "exposure": float(np.mean(deployed)) if deployed else 0.0}


def trade_metrics(df, cost):
    """Per-trade economics at a given entry cost (equal-$ view)."""
    ep = (df["price"] + cost).clip(lower=0.01)
    ret = df["outcome"] / ep - 1.0           # per-$1 return on the leg
    net_edge = df["outcome"] - ep            # per-$1 calibration edge net of cost
    wins, losses = ret[df["outcome"] == 1], ret[df["outcome"] == 0]
    pf = wins.sum() / abs(losses.sum()) if losses.sum() != 0 else np.inf
    return dict(n=len(df), win=df["outcome"].mean(), net_edge=net_edge.mean(),
                t=tstat(net_edge), mean_ret=ret.mean(), profit_factor=pf)


def boot_ci(df, cost, n=10000):
    g = df.assign(rwk=df.res_date.dt.to_period("W").astype(str),
                  e=df.outcome - (df.price + cost).clip(lower=0.01))
    wk = [x for _, x in g.groupby("rwk")]
    wm = np.array([x["e"].mean() for x in wk]); ww = np.array([len(x) for x in wk])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(wk), size=(n, len(wk)))
    blk = (wm[idx] * ww[idx]).sum(1) / ww[idx].sum(1)
    return np.percentile(blk, 5), np.percentile(blk, 95), float((blk <= 0).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-price", type=float, default=0.35)
    ap.add_argument("--stake", type=float, default=0.02, help="fraction of equity per trade")
    ap.add_argument("--base-cost", type=float, default=0.03, help="realistic entry spread (abs, $)")
    ap.add_argument("--cap-frac", type=float, default=0.01, help="max position as frac of market volume")
    ap.add_argument("--fixed-stake", type=float, default=None, help="flat $/trade (non-compounding) instead of %% of equity")
    ap.add_argument("--capital", type=float, default=1000.0, help="starting bankroll for the $ sim")
    ap.add_argument("--gate", choices=["level", "level_or_vol"], default="level_or_vol",
                    help="regime gate: level (high-purity) or level_or_vol (wider book, deployed)")
    args = ap.parse_args()

    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])
    P = gpr_calm_flag(P, gate=args.gate)
    hold = (P["res_date"] - P["decision_date"]).dt.days
    book = P[(P.domain == "geopolitics") & (P.leg == "yes") & (P.price <= args.max_price)
             & P["calm"] & hold.between(2, 45)].copy().reset_index(drop=True)

    print("=" * 90)
    print("VALIDATED GEO-CALM STRATEGY — realistic-cost backtest")
    print("  geopolitics · YES · price<=%.2f · GPR-calm[%s] · hold 2-45d · stake %.0f%%/trade"
          % (args.max_price, args.gate, args.stake * 100))
    print("  n=%d trades · %s → %s" % (len(book), book.decision_date.min().date(), book.res_date.max().date()))
    print("=" * 90)

    print("\n[COST SENSITIVITY] entry = mid + c  (hold to settlement, no exit cost)")
    print("  %-6s %5s %6s %8s %7s %7s %7s %7s %7s %7s" %
          ("cost", "win", "edge/$1", "t", "PF", "CAGR", "Sharpe", "Sortino", "maxDD", "Calmar"))
    for c in [0.0, 0.01, 0.02, 0.03, 0.05]:
        tm = trade_metrics(book, c); _, m = sim(book, stake=args.stake, cost=c)
        tag = "  <-- base" if abs(c - args.base_cost) < 1e-9 else ""
        print("  %4.0f¢  %5.2f %+7.4f %+7.2f %7.2f %+6.1f%% %7.2f %7.2f %5.0f%% %7.2f%s" %
              (c * 100, tm["win"], tm["net_edge"], tm["t"], tm["profit_factor"],
               100 * m.get("cagr", np.nan), m.get("sharpe", np.nan), m.get("sortino", np.nan),
               100 * m.get("maxdd", np.nan), m.get("calmar", np.nan), tag))

    # ---- base-case full panel ----
    c = args.base_cost
    tm = trade_metrics(book, c); eq, m = sim(book, stake=args.stake, cost=c)
    lo, hi, p0 = boot_ci(book, c)
    avg_hold = (book.res_date - book.decision_date).dt.days.mean()
    print("\n" + "-" * 90)
    print("BASE CASE — realistic %.0f¢ entry spread, %.0f%% stake, hold to settlement" % (c * 100, args.stake * 100))
    print("-" * 90)
    print("  trades              %d   (win rate %.1f%%, avg hold %.1f d, exposure %.0f%% of equity)"
          % (tm["n"], 100 * tm["win"], avg_hold, 100 * m.get("exposure", 0)))
    print("  net edge / $1       %+.4f  (t %+.2f)   profit factor %.2f" % (tm["net_edge"], tm["t"], tm["profit_factor"]))
    print("  bootstrap 90%% CI    [%+.4f, %+.4f]   P(edge<=0)=%.3f" % (lo, hi, p0))
    print("  total return        %+.1f%%   CAGR %+.1f%%" % (100 * m.get("ret", np.nan), 100 * m.get("cagr", np.nan)))
    print("  Sharpe %.2f  Sortino %.2f  Calmar %.2f  maxDD %.0f%%  Ulcer %.3f  final $%.0f"
          % (m.get("sharpe", np.nan), m.get("sortino", np.nan), m.get("calmar", np.nan),
             100 * m.get("maxdd", np.nan), m.get("ulcer", np.nan), m.get("final", np.nan)))

    # ---- flat-dollar sizing ----
    if args.fixed_stake:
        fs = args.fixed_stake
        ep = (book["price"] + c).clip(lower=0.01)
        pnl = np.where(book["outcome"] == 1, fs * (1.0 / ep - 1.0), -fs)   # $ per trade
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        eqf, mf = sim(book, cap=args.capital, cost=c, fixed=fs)
        staked = fs * mf.get("taken", len(book))
        print("\n" + "-" * 90)
        print("FLAT-DOLLAR SIZING — $%.0f per trade (non-compounding), %.0f¢ entry spread" % (fs, c * 100))
        print("-" * 90)
        print("  trades funded      %d / %d   (total staked $%.0f, max %d open at once)"
              % (mf.get("taken", 0), len(book), staked, _max_concurrent(book)))
        print("  total P&L          %+.0f $   (avg %+.1f $/trade)" % (pnl.sum(), pnl.mean()))
        print("  ROI on $ staked    %+.1f%%   ROI on $%.0f bankroll  %+.1f%%"
              % (100 * pnl.sum() / staked, args.capital, 100 * (eqf.iloc[-1] / args.capital - 1)))
        print("  avg win  %+.0f $ (n=%d)   avg loss  %+.0f $ (n=%d)   profit factor %.2f"
              % (wins.mean() if len(wins) else 0, len(wins),
                 losses.mean() if len(losses) else 0, len(losses),
                 wins.sum() / abs(losses.sum()) if losses.sum() else float("inf")))
        print("  best trade %+.0f $   worst trade %.0f $   $ maxDD %.0f"
              % (pnl.max(), pnl.min(), (eqf - eqf.cummax()).min()))
        print("  Sharpe %.2f  Sortino %.2f  maxDD %.0f%%  final bankroll $%.0f"
              % (mf.get("sharpe", float("nan")), mf.get("sortino", float("nan")),
                 100 * mf.get("maxdd", float("nan")), eqf.iloc[-1]))

    # ---- capacity ----
    print("\n[CAPACITY] position capped at %.1f%% of market lifetime volume" % (100 * args.cap_frac))
    minv, medv = book.volume.min(), book.volume.median()
    print("  thinnest market volume $%.1fM (median $%.1fM)." % (minv / 1e6, medv / 1e6))
    print("  at %.0f%% stake, the %.0f%%-of-volume cap first bites near AUM ~ $%.1fM (median market)."
          % (args.stake * 100, 100 * args.cap_frac, args.cap_frac * medv / args.stake / 1e6))
    _, mc = sim(book, stake=args.stake, cost=c, cap_frac=args.cap_frac)
    print("  with the cap applied at $1k AUM: CAGR %+.1f%% (cap not binding at this size)" % (100 * mc.get("cagr", np.nan)))
    print("  NOTE: lifetime volume overstates real-time depth, so the ceiling is optimistic.")

    # ---- year split & equity graph ----
    print("\n[YEAR] net edge/$1 at %.0f¢ by resolution year" % (c * 100))
    for yr, sub in book.assign(y=book.res_date.dt.year).groupby("y"):
        tmy = trade_metrics(sub, c)
        print("  %d  n=%2d  win=%.2f  net edge=%+.4f (t=%+.2f)" % (yr, tmy["n"], tmy["win"], tmy["net_edge"], tmy["t"]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for c2, col, lbl in [(0.0, "#888", "mid (0¢)"), (args.base_cost, "#1f9d55", "%.0f¢ realistic" % (args.base_cost*100)),
                         (0.05, "#c0392b", "5¢ stress")]:
        e, _ = sim(book, stake=args.stake, cost=c2)
        if len(e) > 1:
            ax[0].plot(e.index, e.values, color=col, lw=1.5, label=lbl)
    ax[0].axhline(1000, color="grey", ls="--", lw=.8); ax[0].legend(fontsize=8)
    ax[0].set_title("Equity vs entry cost (stake %.0f%%)" % (args.stake*100)); ax[0].set_ylabel("equity $")
    costs = [0, 1, 2, 3, 4, 5]
    edges = [trade_metrics(book, k/100)["net_edge"] for k in costs]
    ax[1].axhline(0, color="grey", lw=.8); ax[1].plot(costs, edges, "o-", color="#1f9d55")
    ax[1].axvline(args.base_cost*100, color="#1f9d55", ls=":", lw=1)
    ax[1].set_title("Net edge/$1 vs entry spread"); ax[1].set_xlabel("entry spread (¢)"); ax[1].set_ylabel("net edge/$1")
    fig.tight_layout(); fig.savefig(f"{OUT}/validated_regime_backtest.png", dpi=120)
    book.assign(entry_cost=c, eff_entry=(book.price+c).round(4),
                pnl_net=(book.outcome-(book.price+c)).round(4)).to_csv(
        f"{OUT}/validated_regime_backtest_trades.csv", index=False)
    print("\nSaved: validated_regime_backtest.png , validated_regime_backtest_trades.csv")


if __name__ == "__main__":
    main()
