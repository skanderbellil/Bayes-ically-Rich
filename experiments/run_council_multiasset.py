#!/usr/bin/env python3
"""
Multi-asset council at retail scale: broad + sector ETF basket, Alpaca costs.

The validated 6-seat macro council sets a monthly risk dial e ∈ [0, 1] from
blinded contexts; the risky sleeve is spread across SPY/QQQ + 9 SPDR sectors
(inverse-vol with a bounded ±50% relative-strength tilt, fit-free); the
de-risked sleeve goes to a configurable slack (cash / TLT+GLD / UUP). Costs
are the Alpaca retail model (commission-free, ~1 bp one-way ETF half-spread,
0.3 bp sell fee) on actual turnover — replicable in a retail account as-is.

The report is built around the user's three requirements:
  HONEST      — ablations (tilt vs no-tilt vs single-asset dial) so the
                sector tilt has to earn its complexity; mirror audit on the
                macro seats; zero fitted parameters anywhere.
  ROBUST      — per-year returns, sub-period Sharpes, rolling 3-year
                Sharpe/CAGR win rates vs SPY.
  REPLICABLE  — 13 liquid ETFs, monthly rebalance, long-only, no margin,
                realized cost drag reported in bps/year.

Usage
-----
  python experiments/run_council_multiasset.py
  python experiments/run_council_multiasset.py --slack tlt-gld
  python experiments/run_council_multiasset.py --seed 11
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging

import numpy as np
import pandas as pd

from posterioralpha.council.multiasset import MultiAssetCouncil
from posterioralpha.data import (
    load_etf_universe_prices,
    load_fred_macro,
    load_net_liquidity,
)
from posterioralpha.mining.evaluation import sharpe

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "council"


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    cum = (1 + r).cumprod()
    dd = float(((cum - cum.cummax()) / cum.cummax()).min())
    cagr = float(cum.iloc[-1] ** (252 / max(len(r), 1)) - 1)
    return {"sharpe": round(sharpe(r.values), 2), "cagr": round(cagr, 4),
            "max_dd": round(dd, 4),
            "calmar": round(cagr / abs(dd), 2) if dd else 0.0}


def rolling_wins(strat: pd.Series, bench: pd.Series, years: int = 3) -> dict:
    """% of rolling N-year windows where the strategy beats the benchmark."""
    w = years * 252
    idx = strat.dropna().index.intersection(bench.dropna().index)
    s, b = strat.loc[idx], bench.loc[idx]
    sh_wins, cg_wins, n = 0, 0, 0
    for start in range(0, len(idx) - w, 21):
        sw, bw = s.iloc[start:start + w], b.iloc[start:start + w]
        sh_wins += sharpe(sw.values) > sharpe(bw.values)
        cg_wins += sw.sum() > bw.sum()
        n += 1
    return {"sharpe_win": round(sh_wins / n, 2) if n else np.nan,
            "cagr_win": round(cg_wins / n, 2) if n else np.nan,
            "windows": n}


def main():
    ap = argparse.ArgumentParser(description="Multi-asset council, Alpaca costs")
    ap.add_argument("--slack", choices=["cash", "tlt-gld", "uup"], default="cash")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    prices = load_etf_universe_prices()
    macro = load_fred_macro()
    liq = load_net_liquidity()["net_liquidity"]

    variants = {
        f"council basket+tilt ({args.slack})": dict(tilt=True, slack=args.slack),
        "council basket no-tilt (cash)":       dict(tilt=False, slack="cash"),
        "council SPY-only dial (cash)":        dict(tilt=False, slack="cash",
                                                    risky=["SPY"]),
        "council basket+tilt (tlt-gld)":       dict(tilt=True, slack="tlt-gld"),
        "council QQQ-only dial (tlt-gld)":     dict(tilt=False, slack="tlt-gld",
                                                    risky=["QQQ"]),
        "council QQQ-only dial (uup)":         dict(tilt=False, slack="uup",
                                                    risky=["QQQ"]),
    }
    # dedupe if --slack already covers a variant
    seen, runs = set(), {}
    for name, kw in variants.items():
        key = tuple(sorted(kw.items(), key=str))
        if str(key) in seen:
            continue
        seen.add(str(key))
        runs[name] = MultiAssetCouncil(prices, macro=macro, liquidity=liq,
                                       seed=args.seed, **kw).run()

    main_name = list(runs)[0]
    main_res = runs[main_name]
    bench = prices["SPY"].pct_change().loc[main_res.returns.index]

    print(f"Basket: SPY QQQ + 9 SPDR sectors | slack: {args.slack} | "
          f"monthly, long-only, no margin, Alpaca ETF costs")
    print(f"Sample: {main_res.returns.index[0].date()} → "
          f"{main_res.returns.index[-1].date()} | realized cost drag: "
          f"{main_res.cost_drag_annual * 1e4:.1f} bps/yr\n")

    # ── Headline + ablations ──────────────────────────────────────────────
    rows = [{"strategy": n, **metrics(r.returns)} for n, r in runs.items()]
    rows.append({"strategy": "SPY buy & hold", **metrics(bench)})
    print("=" * 78)
    print("HEADLINE + ABLATIONS (does each layer earn its keep?)")
    print("=" * 78)
    print(pd.DataFrame(rows).set_index("strategy").to_string())

    # ── Robustness: per-year ──────────────────────────────────────────────
    per_year = pd.DataFrame({
        "council": main_res.returns.groupby(main_res.returns.index.year)
                                   .apply(lambda r: (1 + r).prod() - 1),
        "SPY": bench.groupby(bench.index.year).apply(lambda r: (1 + r).prod() - 1),
    }).round(3)
    per_year["edge"] = (per_year["council"] - per_year["SPY"]).round(3)
    print("\nPER-YEAR RETURNS (robustness across environments)")
    print(per_year.to_string())

    # ── Robustness: sub-periods & rolling windows ─────────────────────────
    subs = {"2010-2014": ("2010", "2014"), "2015-2019": ("2015", "2019"),
            "2020-2022": ("2020", "2022"), "2023-2026": ("2023", "2026")}
    sub_rows = []
    for name, (a, b) in subs.items():
        sub_rows.append({
            "period": name,
            "council_sharpe": round(sharpe(main_res.returns.loc[a:b].values), 2),
            "spy_sharpe": round(sharpe(bench.loc[a:b].values), 2),
        })
    print("\nSUB-PERIOD SHARPES")
    print(pd.DataFrame(sub_rows).set_index("period").to_string())

    wins = rolling_wins(main_res.returns, bench)
    print(f"\nROLLING 3-YEAR WINDOWS vs SPY: Sharpe-win {wins['sharpe_win']:.0%}, "
          f"CAGR-win {wins['cagr_win']:.0%} ({wins['windows']} windows)")

    print("\nMIRROR AUDIT (macro seats)")
    print(main_res.audit.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).set_index("strategy").to_csv(
        OUT_DIR / "multiasset_summary.csv")
    per_year.to_csv(OUT_DIR / "multiasset_per_year.csv")
    main_res.weights.to_csv(OUT_DIR / "multiasset_weights.csv")
    print(f"\nSaved to {OUT_DIR}/: multiasset_summary.csv, multiasset_per_year.csv, "
          "multiasset_weights.csv")


if __name__ == "__main__":
    main()
