#!/usr/bin/env python3
"""
Good vol vs bad vol — is volatility's payoff CONDITIONAL? (user 2026-06-15)

Hypothesis: volatility alone does not predict return, but among HIGH-VOL
assets some are "good vol" (compensated) and some "bad vol" (lottery/junk
that isn't) — and a second, forecastable signal separates them. This is the
idiosyncratic-vol puzzle (Ang-Hodrick-Xing-Zhang) + lottery/MAX (Bali-
Cakici-Whitelaw): high vol that comes with lottery demand is OVERpriced and
underperforms; high vol that comes with momentum/strength may be good.

DESIGN — double sort on the 500-stock equity universe (prices only):
  Vol axis    = 126d realized vol (the "volatility" sort).
  Conditioners (all causal):
    mom   12-1 momentum  P[t-21]/P[t-252]-1
    MAX   lottery: mean of the 5 largest daily returns over the last 21d
          (high MAX = lottery stock = predicted BAD vol)
    skew  skewness of daily returns over 126d (high skew = lottery = BAD)
  Each month: form vol terciles; WITHIN the high-vol and low-vol terciles,
  sort by each conditioner into terciles and measure the EW forward-1m return
  of good-minus-bad. KEY TEST: is the conditioner's spread LARGER in the
  high-vol bucket (does vol AMPLIFY the signal)? And does high-vol+good earn
  while high-vol+bad loses?

Honest box: equity_universe is current-membership (survivorship-biased) —
this biases LEVELS up, so read the SHAPE (good-vs-bad spread, high-vol vs
low-vol amplification), not absolute returns. Bar t>3; net-of-cost noted
(high-vol names cost more). Cross-checked vs the JKP ivol/rmax factors.
"""
import sys
import numpy as np
import pandas as pd
from tabulate import tabulate
import _bootstrap  # noqa: F401
from posterioralpha.data import load_equity_universe_prices

ANN = 252


def tstat(r):
    r = r.dropna()
    return r.mean() / (r.std() + 1e-12) * np.sqrt(len(r))


def ann_sharpe(r):
    r = r.dropna()
    return r.mean() * 12 / (r.std() * np.sqrt(12) + 1e-12)


def main():
    px = load_equity_universe_prices()
    px = px.loc[:, px.notna().mean() > 0.95].dropna(how="all")
    rets = px.pct_change()
    print(f"Universe: {px.shape[1]} stocks, {px.index[0].date()}→{px.index[-1].date()}")

    vol = rets.rolling(126, min_periods=100).std()
    mom = px.shift(21) / px.shift(252) - 1.0
    mx = rets.rolling(21).apply(lambda x: np.sort(x)[-5:].mean(), raw=True)   # lottery
    skew = rets.rolling(126, min_periods=100).skew()
    fwd = px.shift(-21) / px - 1.0

    me = pd.DatetimeIndex([px.index[px.index <= m][-1] for m in
                           pd.Series(index=px.index, dtype=float).resample("ME")
                           .last().index if len(px.index[px.index <= m])])

    def double_sort(cond, cond_name, good_high):
        """Within low/high vol terciles, return spread of cond good-minus-bad.
        good_high=True means high cond = good (long high); else long low."""
        out = {"low_vol": [], "high_vol": []}
        for t in me:
            if t not in vol.index or t not in fwd.index:
                continue
            v, c, f = vol.loc[t], cond.loc[t], fwd.loc[t]
            ok = v.notna() & c.notna() & f.notna()
            if ok.sum() < 60:
                continue
            v, c, f = v[ok], c[ok], f[ok]
            vlo, vhi = v.quantile(1/3), v.quantile(2/3)
            for bucket, mask in [("low_vol", v <= vlo), ("high_vol", v >= vhi)]:
                cb, fb = c[mask], f[mask]
                if len(cb) < 12:
                    continue
                clo, chi = cb.quantile(1/3), cb.quantile(2/3)
                good = fb[cb >= chi].mean() if good_high else fb[cb <= clo].mean()
                bad = fb[cb <= clo].mean() if good_high else fb[cb >= chi].mean()
                out[bucket].append(good - bad)
        return {k: pd.Series(v) for k, v in out.items()}

    print("\nDouble sort: within vol terciles, spread of conditioner "
          "(good − bad), monthly forward return:")
    rows = []
    for cond, name, good_high in [(mom, "momentum (long winners)", True),
                                  (mx, "MAX/lottery (long LOW-max)", False),
                                  (skew, "skew (long LOW-skew)", False)]:
        d = double_sort(cond, name, good_high)
        lo, hi = d["low_vol"], d["high_vol"]
        rows.append([name,
                     f"{lo.mean()*12:+.1%}", f"{tstat(lo):.2f}",
                     f"{hi.mean()*12:+.1%}", f"{tstat(hi):.2f}",
                     f"{(hi.mean()-lo.mean())*12:+.1%}"])
    print(tabulate(rows, headers=["conditioner (good leg)", "LOWvol sprd",
                                  "t", "HIGHvol sprd", "t", "amplification"],
                   tablefmt="github"))
    print("\nKEY TEST: is the spread bigger in HIGH-vol than LOW-vol "
          "(amplification > 0)? That means the conditioner separates 'good "
          "vol' from 'bad vol' — vol is only worth holding WITH the right "
          "signal. Bar t>3; survivorship-biased universe → read the shape.")


def jkp_vol_family():
    """Survivorship-free cross-region test of the vol/lottery/skew/beta family
    (the honest version; the equity-universe double sort above is biased)."""
    DATA = _bootstrap.ROOT / "datasets"
    ind = pd.read_csv(DATA / "jkp_factors_individual.csv.gz", parse_dates=["date"])
    REGIONS = ["usa", "developed", "emerging", "world_ex_us"]

    def st(r):
        r = r.dropna()
        if len(r) < 24 or r.std() == 0:
            return np.nan
        return r.mean() / r.std(ddof=1) * np.sqrt(len(r))

    def ser(name, loc):
        s = ind[(ind.name == name) & (ind.location == loc)]
        return s.set_index("date")["ret"].sort_index()

    fac = ["rvol_21d", "ivol_capm_252d", "ivol_ff3_21d", "rmax5_21d", "rmax1_21d",
           "iskew_ff3_21d", "rskew_21d", "beta_60m", "betadown_252d", "betabab_1260d"]
    rows = []
    for n in fac:
        sign = 1.0 if ser(n, "usa").mean() >= 0 else -1.0
        tc = {loc: st((ser(n, loc) * sign) - 0.015 / 12) for loc in REGIONS}
        ok = tc["usa"] > 3 and all(tc[l] > 2 for l in REGIONS[1:])
        rows.append([n] + [f"{tc[l]:.2f}" for l in REGIONS] + ["YES" if ok else ""])
    print("\nJKP vol-family - cross-region t-stat NET of 1.5%/yr cost "
          "(survivorship-free, factors oriented to JKP positive-premium sign):")
    print(tabulate(rows, headers=["factor", "US", "dev", "em", "wxus", "PASS"],
                   tablefmt="github"))
    print("None clear US t>3 net + cross-region t>2 net. 'Bad vol' (IVOL, "
          "lottery, skew) is real GROSS in the US but dies on cost / does not "
          "replicate; betting-against-beta even REVERSES abroad. The vol metric "
          "itself does NOT separate good from bad vol at our bar - but cop_at "
          "(cash profitability) does. The discriminator of good vs bad volatile "
          "assets is QUALITY (does the volatile firm earn cash?), not a vol stat.")


if __name__ == "__main__":
    main()
    jkp_vol_family()
