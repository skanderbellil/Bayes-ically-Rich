#!/usr/bin/env python3
"""
Council backtest: period-isolated, blinded, domain-specialist decisions.

Each month-end, a council of specialists (rates, credit, volatility, dollar,
liquidity, trend) receives a BLINDED context — no dates, no tickers, no
levels; only z-scores and normalized paths from data available at that
moment — and votes a stance. The confidence-weighted vote becomes next
month's exposure. The engine is walk-forward by construction and reports,
alongside the council, every specialist's solo dial and a MIRROR AUDIT:
each specialist also assesses a sign-flipped copy of every context, and an
honest analyst's stances must invert (anti-symmetry ≈ 1). Anything well
below 1 is using information beyond the supplied numbers — for LLM council
members, that's the lookahead-through-memory tell.

Backends
--------
default — deterministic statistical specialists (offline, fit-free)
--llm   — Claude-backed specialists (needs ANTHROPIC_API_KEY in .env and
          `pip install anthropic`). Uses the Message Batches API (50% cost)
          when the assessment count is large, sync calls otherwise.

Usage
-----
  python experiments/run_council_backtest.py                   # QQQ, offline
  python experiments/run_council_backtest.py --underlying SPY
  python experiments/run_council_backtest.py --llm --llm-model claude-haiku-4-5
  python experiments/run_council_backtest.py --llm --max-periods 12  # cheap demo
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging

import pandas as pd

from posterioralpha.council import CouncilBacktest, CouncilConfig
from posterioralpha.data import load_etf_universe_prices, load_fred_macro, load_net_liquidity

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "council"


def main():
    ap = argparse.ArgumentParser(description="Blinded council backtest")
    ap.add_argument("--underlying", default="QQQ")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tc-bps", type=float, default=2.0)
    ap.add_argument("--llm", action="store_true",
                    help="use Claude-backed specialists (needs ANTHROPIC_API_KEY)")
    ap.add_argument("--llm-model", default="claude-opus-4-8")
    ap.add_argument("--max-periods", type=int, default=0,
                    help="limit to the most recent N periods (cost control for --llm)")
    args = ap.parse_args()

    prices = load_etf_universe_prices()[args.underlying].dropna()
    macro = load_fred_macro()
    liquidity = load_net_liquidity()["net_liquidity"]

    cfg = CouncilConfig(tc=args.tc_bps / 1e4, seed=args.seed)
    bt = CouncilBacktest(prices, macro=macro, liquidity=liquidity, config=cfg)

    print(f"Underlying: {args.underlying} — {len(prices)} days "
          f"({prices.index[0].date()} → {prices.index[-1].date()})")
    print(f"Council domains: {', '.join(bt.domains)}")
    n_periods = len(bt.period_dates())
    print(f"Periods: {n_periods} ({cfg.rebalance} decisions, blinded contexts, "
          "1-day lag, mirror audit on)\n")

    verdicts_cache = None
    keep_ids = None
    if args.llm:
        from posterioralpha.council.llm import LLMSpecialist, run_llm_council_batch

        contexts = [c for _, c in bt.build_contexts()]
        if args.max_periods > 0:
            contexts = contexts[-args.max_periods:]
            keep_ids = {c.period_id for c in contexts}
        n_calls = 2 * len(contexts) * len(bt.domains)  # originals + mirrors
        print(f"LLM council: {args.llm_model}, {n_calls} assessments "
              f"({'batched, 50% pricing' if n_calls > 60 else 'sync'})")

        if n_calls > 60:
            verdicts_cache = run_llm_council_batch(
                contexts, list(bt.domains), model=args.llm_model)
        else:
            verdicts_cache = {}
            for ctx in contexts:
                for variant in (ctx, ctx.mirror()):
                    for d in bt.domains:
                        if d not in variant.views:
                            continue
                        spec = LLMSpecialist(d, model=args.llm_model)
                        verdicts_cache[(variant.period_id, d,
                                        variant.is_mirrored)] = spec.assess(variant)
    result = bt.run(verdicts_cache=verdicts_cache, period_ids=keep_ids)

    print("=" * 70)
    print("PERFORMANCE (council vs solo specialists vs buy & hold)")
    print("=" * 70)
    print(bt.summary(result).to_string())

    print("\nMIRROR AUDIT — anti-symmetry ≈ 1 means stances are driven by the")
    print("supplied numbers; well below 1 means something else (for LLMs:")
    print("likely episode recognition = lookahead through memory)")
    print(result.audit.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.underlying}{'_llm' if args.llm else ''}"
    result.minutes.to_csv(OUT_DIR / f"minutes_{tag}.csv")
    bt.summary(result).to_csv(OUT_DIR / f"summary_{tag}.csv")
    result.audit.to_csv(OUT_DIR / f"audit_{tag}.csv")
    print(f"\nSaved to {OUT_DIR}/: minutes_{tag}.csv, summary_{tag}.csv, audit_{tag}.csv")


if __name__ == "__main__":
    main()
