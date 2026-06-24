#!/usr/bin/env python3
"""
Per-domain validation with a DOMAIN-MATCHED exogenous proxy (no lookahead)
==========================================================================

Each disruptive domain is judged against its *own* exogenous regime indicator —
the right instrument for that kind of "calm vs recently-shocked", not one global
index — then put through a battery of kill tests. Only survivors trade.

  domain            matched proxy                               source
  ----------------  ------------------------------------------  --------------------------
  geopolitics       Caldara-Iacoviello GPR index                datasets/gpr_daily.csv.gz
  crypto/financial  BTC 30d realized volatility                 datasets/btc_usd_daily.csv
  politics          Economic Policy Uncertainty (EPU, daily)    datasets/epu_daily.csv
  ai/tech           (insufficient tail data — auto-killed)      -

Regime, fully causal (no lookahead): for each leg take the proxy's trailing-`win`
value and a trailing-`ref`-day median baseline (both backward rolling — row t sees
only rows <= t), attached with merge_asof(direction="backward"). CALM = recent
proxy below its own running baseline (an MA-crossover regime, relative to the
recent normal so structurally-elevated 2023-26 levels don't mislabel everything).

Kill battery on the calm cheap-YES tail (price<=MAXP) vs the turbulent tail:
  [base]  calm edge/$1 + t
  [gap]   calm-minus-turbulent edge gap (the regime effect itself)
  [oos]   walk-forward edge (decisions >= 2025-07-01)
  [cost]  net edge after 3c / 5c haircut
  [boot]  week-clustered bootstrap P(calm edge<=0)
  [plac]  placebo: reshuffle the calm label 5,000x; is the real gap in the tail?
  [conc]  winner concentration (top-1 share of gross profit)

VERDICT (strict — we are trying to kill it):
  VALIDATED iff  n_calm>=20  and base t>=2  and gap>0  and oos edge>0
                 and net@3c>0  and boot P<=0.10  and placebo p<=0.10  and top1<0.60
  else KILLED (binding reasons listed).

Writes data/polymarket_calibration_snapshot/validated_domains.json — the live
strategy / paper ledger read this, so validation drives what actually trades.

Usage: python experiments/run_regime_domain_validation.py [--max-price 0.35]
"""
from __future__ import annotations
import argparse, json, math
import numpy as np, pandas as pd
import _bootstrap  # noqa

PANEL = "data/polymarket_calibration_snapshot/regime_calibration_panel.csv"
OUT = "data/polymarket_calibration_snapshot/validated_domains.json"
RNG = np.random.default_rng(20260624)


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


# ---------- domain-matched proxy daily series (higher = more turbulent) ----------
def proxy_gpr():
    g = pd.read_csv("datasets/gpr_daily.csv.gz", parse_dates=["date"]).set_index("date")["GPRD"]
    return g.sort_index()


def proxy_btc_vol():
    d = pd.read_csv("datasets/btc_usd_daily.csv", parse_dates=["date"]).set_index("date")["close"].sort_index()
    ret = np.log(d).diff()
    return ret.rolling(30, min_periods=15).std() * math.sqrt(365)  # annualized 30d realized vol


def proxy_epu():
    e = pd.read_csv("datasets/epu_daily.csv", parse_dates=["date"]).set_index("date")["EPU"]
    return e.sort_index()


PROXIES = {
    "geopolitics": ("GPR index", proxy_gpr, 45, 365),
    "crypto/financial": ("BTC 30d realized vol", proxy_btc_vol, 30, 365),
    "politics": ("EPU (daily)", proxy_epu, 45, 365),
}


def attach_calm(P, series, win, ref):
    """Causal CALM flag: trailing-`win` mean of proxy below trailing-`ref` median."""
    s = series.asfreq("D").ffill()
    trail = s.rolling(win, min_periods=max(5, win // 3)).mean()
    base = s.rolling(ref, min_periods=ref // 2).median()
    daily = pd.DataFrame({"Date": s.index, "trail": trail.values, "base": base.values}).dropna()
    m = pd.merge_asof(P.sort_values("decision_date"), daily,
                      left_on="decision_date", right_on="Date", direction="backward")
    m["calm"] = m["trail"] <= m["base"]
    return m


def edge(df, hc=0.0):
    return (df["outcome"] - (df["price"] + hc)).mean(), tstat(df["outcome"] - df["price"]), len(df)


def boot_p(df, n=10000):
    if len(df) < 5:
        return np.nan
    g = df.assign(rwk=df.res_date.dt.to_period("W").astype(str), e=df.outcome - df.price)
    wk = [x for _, x in g.groupby("rwk")]
    wm = np.array([x["e"].mean() for x in wk]); ww = np.array([len(x) for x in wk])
    idx = RNG.integers(0, len(wk), size=(n, len(wk)))
    return float(((wm[idx] * ww[idx]).sum(1) / ww[idx].sum(1) <= 0).mean())


def placebo_p(tail, calm_mask, n=5000):
    ce = (tail.outcome - tail.price).values
    k = int(calm_mask.sum()); m = len(ce)
    if k == 0 or k == m:
        return np.nan, np.nan
    real = ce[calm_mask.values].mean() - ce[~calm_mask.values].mean()
    null = np.empty(n)
    for i in range(n):
        p = RNG.permutation(m); cm = np.zeros(m, bool); cm[p[:k]] = True
        null[i] = ce[cm].mean() - ce[~cm].mean()
    return real, float((null >= real).mean())


def top1_conc(df, hc=0.01):
    g = (df.outcome / (df.price + hc) - 1).sort_values(ascending=False).to_numpy()
    gr = g[g > 0].sum()
    return float(g[0] / gr) if gr > 0 else np.nan


def validate(dom, P, maxp):
    r = {"domain": dom}
    if dom not in PROXIES:
        tail = P[(P.domain == dom) & (P.leg == "yes") & (P.price <= maxp)]
        r.update(proxy="(none)", n_tail=len(tail), verdict="KILLED",
                 reason="no matched proxy / insufficient tail data (n_tail=%d)" % len(tail))
        return r
    pname, pfn, win, ref = PROXIES[dom]
    r["proxy"] = pname
    M = attach_calm(P[(P.domain == dom) & (P.leg == "yes") & (P.price <= maxp)].copy(), pfn(), win, ref)
    M = M[M["trail"].notna()]
    tail, calm, turb = M, M[M.calm], M[~M.calm]
    r["n_tail"], r["n_calm"], r["n_turb"] = len(tail), len(calm), len(turb)
    if len(calm) < 5:
        r.update(verdict="KILLED", reason="too few calm legs (%d)" % len(calm))
        return r
    be, bt, _ = edge(calm)
    te, _, _ = edge(turb)
    oe, ot, on = edge(calm[calm.decision_date >= "2025-07-01"])
    r.update(base_edge=round(be, 4), base_t=round(bt, 2), gap=round(be - te, 4),
             turb_edge=round(te, 4), oos_n=on, oos_edge=round(oe, 4), oos_t=round(ot, 2),
             net_3c=round(edge(calm, 0.03)[0], 4), net_5c=round(edge(calm, 0.05)[0], 4),
             boot_p=round(boot_p(calm), 4), top1=round(top1_conc(calm), 3))
    gapv, pp = placebo_p(tail, M.calm)
    r["placebo_p"] = round(pp, 4) if pp == pp else np.nan

    checks = {
        "n_calm>=20": r["n_calm"] >= 20,
        "base_t>=2": r["base_t"] >= 2.0,
        "gap>0": r["gap"] > 0,
        "oos_edge>0": r["oos_edge"] > 0,
        "net_3c>0": r["net_3c"] > 0,
        "boot_p<=0.10": r["boot_p"] == r["boot_p"] and r["boot_p"] <= 0.10,
        "placebo_p<=0.10": r["placebo_p"] == r["placebo_p"] and r["placebo_p"] <= 0.10,
        "top1<0.60": r["top1"] == r["top1"] and r["top1"] < 0.60,
    }
    failed = [k for k, ok in checks.items() if not ok]
    r["verdict"] = "VALIDATED" if not failed else "KILLED"
    r["reason"] = "survives all kill tests" if not failed else "fails: " + ", ".join(failed)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-price", type=float, default=0.35)
    args = ap.parse_args()
    P = pd.read_csv(PANEL, parse_dates=["decision_date", "res_date"])

    print("=" * 98)
    print("PER-DOMAIN VALIDATION via MATCHED PROXY — calm cheap-YES tail (price<=%.2f). Strict kill battery."
          % args.max_price)
    print("=" * 98)
    results = [validate(d, P, args.max_price) for d in ["geopolitics", "crypto/financial", "politics", "ai/tech"]]

    for r in results:
        print("\n### %-16s  proxy=%-22s [%s]" % (r["domain"], r.get("proxy", "-"), r["verdict"]))
        if "base_edge" not in r:
            print("   %s" % r["reason"]); continue
        print("   n_calm=%-3d n_turb=%-3d | base edge/$1=%+.4f (t=%+.2f)  turb=%+.4f  GAP=%+.4f"
              % (r["n_calm"], r["n_turb"], r["base_edge"], r["base_t"], r["turb_edge"], r["gap"]))
        print("   OOS(>=25-07) n=%-2d edge=%+.4f (t=%+.2f) | net@3c=%+.4f @5c=%+.4f | boot P<=0=%s | placebo p=%s | top1=%.0f%%"
              % (r["oos_n"], r["oos_edge"], r["oos_t"], r["net_3c"], r["net_5c"], r["boot_p"], r["placebo_p"], 100 * r["top1"]))
        print("   -> %s" % r["reason"])

    validated = [r["domain"] for r in results if r["verdict"] == "VALIDATED"]
    print("\n" + "=" * 98)
    print("VALIDATED: %s" % (validated or "(none)"))
    print("KILLED:    %s" % [r["domain"] for r in results if r["verdict"] == "KILLED"])
    print("=" * 98)

    payload = {
        "generated": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "max_price": args.max_price,
        "method": "domain-matched exogenous proxy, causal trailing regime, strict kill battery",
        "proxies": {d: PROXIES[d][0] for d in PROXIES},
        "validated_domains": validated,
        "per_domain": {r["domain"]: r for r in results},
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print("Wrote %s" % OUT)


if __name__ == "__main__":
    main()
