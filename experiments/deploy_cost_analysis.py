#!/usr/bin/env python3
"""Retail deployability cost analysis (Alpaca) for the strategy library.

Answers the question: "after realistic Alpaca retail costs, what — if anything —
can a retail trader actually deploy?"

Key constraints encoded (see posterioralpha.backtest.alpaca_costs):
  - Commission-free US equities/ETFs; the binding cost is the bid-ask half-spread
    paid per entry/exit (turnover-scaled), plus margin interest on leverage.
  - Shorting is ETB-only: small/mid/micro caps are hard-to-borrow and CANNOT be
    shorted on Alpaca. The PEAD long-SHORT edge lives there → not deployable.
    So we test the long-ONLY SUE selection tilt, which needs no shorting.

Outputs three blocks:
  1. Long-only SUE selection edge by liquidity tier, net of spread.
  2. Out-of-sample walk-forward of the liquid composite edge.
  3. Absolute long-only top-quintile vs SPY buy-and-hold.

Requires results/pead/signal_panel.csv (run experiments/run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths
from posterioralpha.backtest.alpaca_costs import AlpacaCosts, HALF_SPREAD_BPS
from posterioralpha.data.loaders import load_portfolio_returns

LIQUID = ["Mega Cap", "Large Cap", "Mid Cap"]   # ETB, retail-tradeable, tight spread
AC = AlpacaCosts()


def load_panel() -> pd.DataFrame:
    p = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
    p["announcement_date"] = pd.to_datetime(p["announcement_date"], utc=True, errors="coerce")
    p = p.dropna(subset=["sue", "ret_t21", "market_cap", "announcement_date"])
    p["season"] = p["announcement_date"].dt.to_period("Q")
    p["year"] = p["announcement_date"].dt.year
    return p


def _ann(q):
    q = np.asarray(q); m = q.mean(); s = q.std(ddof=1)
    return dict(annual=(1 + m) ** 4 - 1, sharpe=np.sqrt(4) * m / s if s > 0 else 0,
                t=np.sqrt(len(q)) * m / s if s > 0 else 0, nq=len(q),
                winq=float((q > 0).mean()))


def net_season_alpha(df, top_q=0.8):
    """Per-quarter (top-quintile minus universe) selection alpha, net of spread."""
    out = []
    for _, g in df.groupby("season"):
        pt, pu, c = [], [], []
        for tier, gt in g.groupby("market_cap"):
            if len(gt) < 5:
                continue
            thr = gt["sue"].quantile(top_q)
            pt.append(gt[gt["sue"] >= thr]["ret_t21"].mean() / 100)
            pu.append(gt["ret_t21"].mean() / 100)
            c.append(AC.roundtrip_cost(tier))
        if pt:
            out.append((np.mean(pt) - np.mean(c)) - np.mean(pu))
    return np.array(out)


def abs_topq(df, top_q=0.8):
    """Per-quarter absolute long-only top-quintile return, net of spread."""
    vals, idx = [], []
    for s, g in df.groupby("season"):
        pt, c = [], []
        for tier, gt in g.groupby("market_cap"):
            if len(gt) < 5:
                continue
            thr = gt["sue"].quantile(top_q)
            pt.append(gt[gt["sue"] >= thr]["ret_t21"].mean() / 100)
            c.append(AC.roundtrip_cost(tier))
        if pt:
            vals.append(np.mean(pt) - np.mean(c)); idx.append(s)
    return pd.Series(vals, index=idx)


def market_neutral_ls(df, tiers=("Mega Cap", "Large Cap"), q=0.2):
    """Per-quarter dollar-neutral long top-SUE / short bottom-SUE within ETB tiers.

    Deployable on Alpaca: both legs are easy-to-borrow large/mega caps ($0 borrow),
    commission-free; each leg charged a round-trip spread. Beta ~ 0. Survivorship
    bias is *conservative* here — delisted bottom-SUE names (the best shorts) are
    absent from the live-only panel, so the realised edge is if anything understated.
    """
    out = []
    for _, g in df.groupby("season"):
        legs = []
        for tier in tiers:
            gt = g[g["market_cap"] == tier]
            if len(gt) < 8:
                continue
            hi, lo = gt["sue"].quantile(1 - q), gt["sue"].quantile(q)
            longs = gt[gt["sue"] >= hi]["ret_t21"].mean() / 100
            shorts = gt[gt["sue"] <= lo]["ret_t21"].mean() / 100
            legs.append((longs - shorts) - 2 * AC.roundtrip_cost(tier))
        if legs:
            out.append(np.mean(legs))
    return np.array(out)


def main():
    p = load_panel()
    liq = p[p["market_cap"].isin(LIQUID)].copy()

    print("=" * 84)
    print("RETAIL DEPLOYABILITY (Alpaca)  ·  PEAD long-only SUE selection tilt")
    print("=" * 84)

    # 1. By tier
    print("\n[1] Long-only selection edge by tier (net of Alpaca spread):")
    print(f"    {'Tier':<11}{'1way bps':>9}{'NetEdge/yr':>11}{'Sharpe':>8}{'t':>6}{'Nq':>5}")
    for tier in LIQUID + ["Small Cap", "Micro Cap"]:
        q = net_season_alpha(p[p["market_cap"] == tier])
        if len(q) < 8:
            continue
        m = _ann(q)
        print(f"    {tier:<11}{HALF_SPREAD_BPS[tier]:>9.0f}{m['annual']:>+11.1%}"
              f"{m['sharpe']:>8.2f}{m['t']:>6.1f}{m['nq']:>5}")

    qc = net_season_alpha(liq); mc = _ann(qc)
    print(f"\n    Liquid composite: {mc['annual']:+.1%}/yr net  Sharpe {mc['sharpe']:.2f}  "
          f"t={mc['t']:.1f}  ({mc['nq']} quarters, {mc['winq']:.0%} positive)")

    # 2. OOS walk-forward
    print("\n[2] Out-of-sample walk-forward (liquid composite, net):")
    for cut in (2011, 2015, 2018):
        a = _ann(net_season_alpha(liq[liq["year"] <= cut]))
        b = _ann(net_season_alpha(liq[liq["year"] > cut]))
        print(f"    train≤{cut}: {a['annual']:+.1%}/yr (t={a['t']:.1f})   "
              f"test>{cut}: {b['annual']:+.1%}/yr (t={b['t']:.1f}, {b['winq']:.0%} win, {b['nq']}q)")

    # 3. Absolute vs SPY
    print("\n[3] Absolute long-only top-quintile vs SPY buy & hold:")
    aq = abs_topq(liq)
    spy = load_portfolio_returns()["SPY"]
    spq = (1 + spy).resample("QE").prod() - 1
    spq.index = spq.index.to_period("Q")
    common = aq.index.intersection(spq.index)
    ta, sa = _ann(aq.loc[common].values), _ann(spq.loc[common].values)
    print(f"    Top-SUE long-only: {ta['annual']:+.1%}/yr  Sharpe {ta['sharpe']:.2f}")
    print(f"    SPY buy & hold:    {sa['annual']:+.1%}/yr  Sharpe {sa['sharpe']:.2f}")
    print(f"    → Excess vs SPY: {ta['annual'] - sa['annual']:+.1%}/yr return, "
          f"{ta['sharpe'] - sa['sharpe']:+.2f} Sharpe  ({len(common)} quarters)")

    # 4. HEADLINE: deployable market-neutral long-short (ETB names only)
    print("\n[4] *** DEPLOYABLE MARKET-NEUTRAL *** long top-SUE / short bottom-SUE,")
    print("    Mega+Large only (ETB, $0 borrow, $0 commission, beta~0):")
    mn = market_neutral_ls(p)
    m = _ann(mn)
    eq = np.cumprod(1 + mn)
    mdd = ((eq - np.maximum.accumulate(eq)) / np.maximum.accumulate(eq)).min()
    print(f"    Net {m['annual']:+.1%}/yr  Sharpe {m['sharpe']:.2f}  t={m['t']:.1f}  "
          f"MaxDD {mdd:+.1%}  ({m['winq']:.0%} of {m['nq']} quarters positive)")
    print("    OOS walk-forward:")
    for cut in (2011, 2015, 2018):
        a = _ann(market_neutral_ls(p[p["year"] <= cut]))
        b = _ann(market_neutral_ls(p[p["year"] > cut]))
        print(f"      train≤{cut}: {a['annual']:+.1%}/yr (t={a['t']:.1f})   "
              f"test>{cut}: {b['annual']:+.1%}/yr (t={b['t']:.1f}, {b['winq']:.0%} win)")


if __name__ == "__main__":
    main()
