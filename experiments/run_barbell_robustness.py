#!/usr/bin/env python3
"""
Barbell robustness: does the levered-equity + diversifier structure survive
removing the hindsight from Epilogue 4?

Three things Epilogue 4 could not claim are tested here:

  1. NO HINDSIGHT SLACK — GLD was picked after seeing the answer. Here the
     slack leg is chosen WALK-FORWARD by one a priori rule, fixed before
     looking: each quarter, among 15 diversifier ETFs (metals, broad
     commodities, ags, treasuries, TIPS, IG credit, FX, T-bills), keep
     those with |corr to the levered leg| <= 0.40 over the trailing 3y,
     rank by trailing 3y Sharpe, hold the top 2 inverse-vol weighted
     (fallback BIL if none qualify). Equity-flavored ETFs (EEM/EFA/VNQ/
     GDX) are excluded a priori — they don't diversify the leg.
  2. OTHER LEVERED LEG — everything re-run on SSO (2x SPY) with the SPY
     council dial, alongside QLD with the QQQ council dial.
  3. THE RIGHT NULLS — each dialed book faces its matched STATIC mix
     (avg-exposure, vol-matched, DD-matched weights) under a joint block
     bootstrap; the static walk-forward barbell faces buy & hold.

Prior art for the structure (so we are not claiming novelty): WisdomTree
GDE (90% equities + 90% gold futures) and the Return Stacked ETFs stack a
diversifier on top of full equity exposure; the documented failure mode is
correlation spikes when both legs fall. What this repo adds is the blinded,
mirror-audited council dial as the drawdown governor.

Run run_council_backtest.py for QQQ and SPY first (council minutes).
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"
COUNCIL_DIR = _bootstrap.ROOT / "results" / "council"
TC = 2e-4

SLACK_POOL = ["GLD", "IAU", "SLV", "DBC", "GSG", "DBA", "TLT", "IEF",
              "SHY", "BIL", "TIP", "LQD", "UUP", "FXY", "FXE"]
LOOKBACK = 756          # 3y trailing window for the selector
MAX_CORR = 0.40         # diversifier requirement vs the levered leg
TOP_K = 2               # inverse-vol blend of the top picks
LEGS = {"QLD": "QQQ", "SSO": "SPY"}   # levered leg -> council underlying


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum / cum.cummax()) - 1).min())
    cagr = float(cum.iloc[-1] ** (252 / len(r)) - 1)
    return {"cagr": round(cagr, 3),
            "vol": round(float(r.std() * np.sqrt(252)), 3),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def walk_forward_slack(leg_ret: pd.Series, pool: pd.DataFrame,
                       vol_floor: float = 0.0) -> pd.Series:
    """Quarterly-selected diversifier basket; weights known at quarter start.

    vol_floor > 0 is SELECTOR #2 (disclosed second look): the slack must
    carry a risk premium (annualized vol >= floor), so T-bills cannot win
    the trailing-Sharpe race — the failure mode of selector #1.
    """
    floor_daily = vol_floor / np.sqrt(252)
    rets = pool.pct_change()
    quarters = leg_ret.resample("QE").last().index
    w = pd.DataFrame(0.0, index=leg_ret.index, columns=pool.columns)
    picks = {}
    for q in quarters:
        hist = rets.loc[:q].iloc[-LOOKBACK:]
        lh = leg_ret.loc[:q].iloc[-LOOKBACK:]
        if len(hist) < LOOKBACK:
            continue
        ok = []
        for c in pool.columns:
            h = hist[c].dropna()
            if len(h) < LOOKBACK * 0.9 or h.std() <= floor_daily:
                continue
            if abs(h.corr(lh)) <= MAX_CORR:
                ok.append((c, h.mean() / h.std()))
        ok.sort(key=lambda x: -x[1])
        chosen = [c for c, _ in ok[:TOP_K]] or ["BIL"]
        iv = 1.0 / hist[chosen].std()
        iv = iv / iv.sum()
        picks[q.date()] = {c: round(float(iv[c]), 2) for c in chosen}
        nxt = w.index[(w.index > q)]
        if len(nxt):
            until = quarters[quarters > q]
            stop = until[0] if len(until) else w.index[-1]
            w.loc[nxt[0]:stop, chosen] = iv.values
    slack_ret = (w.shift(1) * rets).sum(axis=1) \
        - TC * w.diff().abs().sum(axis=1).fillna(0.0)
    return slack_ret, picks


def matched_weight(r_risk, r_slack, target, mode):
    lo, hi = 0.01, 1.0
    for _ in range(40):
        w = (lo + hi) / 2
        mix = w * r_risk + (1 - w) * r_slack
        if mode == "vol":
            hit = float(mix.std()) > target
        else:
            cum = (1 + mix).cumprod()
            hit = float(((cum / cum.cummax()) - 1).min()) < target
        hi, lo = (w, lo) if hit else (hi, w)
    return w


def boot(s, r_risk, r_slack, weights, rng, n_boot=2000):
    X = np.column_stack([s.values, r_risk.values, r_slack.values])
    T = len(X)
    wins = {k: 0 for k in weights}
    for _ in range(n_boot):
        ix, t = np.empty(T, dtype=int), 0
        while t < T:
            s0 = rng.integers(0, T)
            for j in range(rng.geometric(1 / 63)):
                if t >= T:
                    break
                ix[t] = (s0 + j) % T
                t += 1
        R = X[ix]
        a = R[:, 0]
        for k, w in weights.items():
            m = w * R[:, 1] + (1 - w) * R[:, 2]
            wins[k] += (1 + a).prod() > (1 + m).prod()
    return {k: round(v / n_boot, 2) for k, v in wins.items()}


def main():
    uni = load_etf_universe_prices()
    lev = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)
    pool = uni[[c for c in SLACK_POOL if c in uni.columns]]
    rng = np.random.default_rng(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    curves = {}
    for leg_name, under in LEGS.items():
        leg = lev[leg_name].dropna()
        e_per = pd.read_csv(COUNCIL_DIR / f"minutes_{under}.csv", index_col=0,
                            parse_dates=True)["exposure"]
        start = max(e_per.index[0], leg.index[0] + pd.Timedelta(days=LOOKBACK))
        idx = leg.loc[start:].index
        r_leg = leg.pct_change().reindex(idx).fillna(0.0)
        r_under = uni[under].pct_change().reindex(idx).fillna(0.0)

        e = e_per.reindex(leg.index).ffill().shift(1).reindex(idx).fillna(0.0)
        held = e
        w_avg = float(held.mean())
        gld = uni["GLD"].pct_change().reindex(idx).fillna(0.0)
        static_gld = w_avg * r_leg + (1 - w_avg) * gld

        rows = {
            f"{under} buy & hold": stats(r_under),
            f"{leg_name} buy & hold": stats(r_leg),
            f"static {leg_name}/GLD (hindsight ref)": stats(static_gld),
        }
        picks = {}
        for sel, floor in (("sel1", 0.0), ("sel2 vol>=8%", 0.08)):
            slack_ret, picks = walk_forward_slack(leg.pct_change().dropna(),
                                                  pool, vol_floor=floor)
            r_sl = slack_ret.reindex(idx).fillna(0.0)
            dialed = (held * r_leg - TC * held.diff().abs().fillna(0.0)
                      + (1 - held) * r_sl)
            static = w_avg * r_leg + (1 - w_avg) * r_sl
            rows[f"static {leg_name}/WF-slack ({sel})"] = stats(static)
            key = f"council {leg_name} + WF-slack ({sel})"
            rows[key] = stats(dialed)
            weights = {"P_w_avg_mix": w_avg,
                       "P_w_volm": matched_weight(r_leg, r_sl,
                                                  float(dialed.std()), "vol"),
                       "P_w_ddm": matched_weight(r_leg, r_sl,
                                                 rows[key]["max_dd"], "dd")}
            rows[key].update(boot(dialed, r_leg, r_sl, weights, rng))

        print(f"\n══ {leg_name} (council on {under}; span {idx[0].date()} → "
              f"{idx[-1].date()}; avg exposure {w_avg:.2f})")
        print(pd.DataFrame(rows).T.to_string())
        last = list(picks.items())[-6:]
        print("walk-forward slack picks (last 6 quarters):",
              {str(k): v for k, v in last})
        all_rows.update({f"[{leg_name}] {k}": v for k, v in rows.items()})
        if leg_name == "QLD":
            for nm, r in (("QQQ buy & hold", r_under),
                          ("static QLD/GLD (hindsight)", static_gld),
                          (f"static {leg_name}/WF-slack (sel2)", static),
                          (f"council {leg_name} + WF-slack (sel2)", dialed)):
                curves[nm] = r

    pd.DataFrame(all_rows).T.to_csv(OUT_DIR / "barbell_robustness.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, r in curves.items():
        cum = (1 + r.fillna(0)).cumprod()
        style = dict(lw=1.2, ls="--", color="0.4") if "buy & hold" in name \
            else dict(lw=1.5)
        ax1.plot(cum.index, cum.values, label=name, **style)
        ax2.plot(cum.index, (cum / cum.cummax() - 1).values, **style)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Barbell without hindsight: walk-forward slack selection")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "barbell_robustness.png", dpi=140)
    print(f"\nSaved to {OUT_DIR}/: barbell_robustness.csv, "
          "barbell_robustness.png")


if __name__ == "__main__":
    main()
