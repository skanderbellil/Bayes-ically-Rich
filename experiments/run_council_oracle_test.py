#!/usr/bin/env python3
"""
Oracle test: falsify the council's lookahead detector before trusting it.

The mirror audit claims to detect analysts who recognize historical
episodes and recall what happened next (the LLM lookahead-through-memory
failure mode). This experiment injects that exact failure in a controlled
dose — OracleSpecialists blend an honest statistical stance with a
"memory" stance keyed on the period's identity (so it does NOT invert on
mirrored contexts) — and checks three things:

  1. DOSE-RESPONSE — as the leak rises 0 → 1, the mirror audit's
     anti-symmetry must fall from ≈1 toward ≈0 while the council's Sharpe
     inflates. If the audit can't see the dose, it's a useless instrument.
  2. LOCALIZATION — corrupt ONE seat; the audit must flag exactly that
     seat and leave the honest five at ≈1.
  3. REPAIR — drop the seats the audit flags (anti-symmetry < 0.8) and
     re-run; performance must fall back to the honest baseline. That's
     the operational protocol for a future LLM council: audit, drop
     suspects, trust what remains.

Everything is offline — no LLM, no key. The oracle plays the cheater.

Usage
-----
  python experiments/run_council_oracle_test.py
  python experiments/run_council_oracle_test.py --underlying SPY --seed 11
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import argparse
import logging

import pandas as pd

from posterioralpha.council import CouncilBacktest, CouncilConfig
from posterioralpha.council.oracle import OracleSpecialist, build_answer_key
from posterioralpha.council.specialists import StatisticalSpecialist
from posterioralpha.data import (
    load_etf_universe_prices,
    load_fred_macro,
    load_net_liquidity,
)
from posterioralpha.mining.evaluation import sharpe

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = _bootstrap.ROOT / "results" / "council"

def council_sharpe(bt, result) -> float:
    return round(sharpe(result.returns.values), 3)


def main():
    ap = argparse.ArgumentParser(description="Oracle test for the council audit")
    ap.add_argument("--underlying", default="QQQ")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--recall-noise", type=float, default=0.5,
                    help="imperfection of the oracle's memory")
    args = ap.parse_args()

    prices = load_etf_universe_prices()[args.underlying].dropna()
    bt = CouncilBacktest(
        prices,
        macro=load_fred_macro(),
        liquidity=load_net_liquidity()["net_liquidity"],
        config=CouncilConfig(seed=args.seed),
    )
    key = build_answer_key(bt)
    print(f"Underlying: {args.underlying} — {len(bt.period_dates())} periods, "
          f"domains: {', '.join(bt.domains)}\n")

    # ── 1. Dose-response: corrupt the whole council at rising leak ────────
    rows = []
    for leak in (0.0, 0.25, 0.5, 0.75, 1.0):
        council = {
            d: OracleSpecialist(d, key, leak=leak,
                                recall_noise=args.recall_noise, seed=args.seed)
            for d in bt.domains
        }
        res = bt.run(council=council)
        rows.append({
            "leak": leak,
            "council_sharpe": council_sharpe(bt, res),
            "mean_anti_symmetry": round(float(res.audit["anti_symmetry"].mean()), 3),
            "mean_leak_ratio": round(float(res.audit["leak_ratio"].mean()), 3),
            "seats_flagged": int((res.audit["verdict"] == "SUSPECT").sum()),
        })
    dose = pd.DataFrame(rows).set_index("leak")
    print("=" * 72)
    print("1. DOSE-RESPONSE — all 6 seats corrupted at leak level L")
    print("   (audit must fall as Sharpe inflates)")
    print("=" * 72)
    print(dose.to_string())

    # ── 2. Localization: corrupt a single seat ────────────────────────────
    target = "trend"
    council = {d: StatisticalSpecialist(d) for d in bt.domains}
    council[target] = OracleSpecialist(target, key, leak=0.75,
                                       recall_noise=args.recall_noise,
                                       seed=args.seed)
    res_one = bt.run(council=council)
    print("\n" + "=" * 72)
    print(f"2. LOCALIZATION — only '{target}' corrupted (leak=0.75)")
    print("=" * 72)
    print(res_one.audit.to_string())

    # ── 3. Repair: drop flagged seats and re-run ──────────────────────────
    flagged = res_one.audit.index[res_one.audit["verdict"] == "SUSPECT"].tolist()
    clean_domains = [d for d in bt.domains if d not in flagged]

    bt_clean = CouncilBacktest(
        prices,
        macro=load_fred_macro(),
        liquidity=load_net_liquidity()["net_liquidity"],
        config=CouncilConfig(seed=args.seed, domains=tuple(clean_domains)),
    )
    res_repaired = bt_clean.run()

    honest = bt.run()
    print("\n" + "=" * 72)
    print("3. REPAIR — drop flagged seats, re-run with the remainder")
    print("=" * 72)
    comparison = pd.DataFrame([
        {"council": "honest (all 6, no leak)",
         "sharpe": council_sharpe(bt, honest)},
        {"council": f"corrupted ({target} leaking)",
         "sharpe": council_sharpe(bt, res_one)},
        {"council": f"repaired (dropped: {', '.join(flagged) or 'none'})",
         "sharpe": council_sharpe(bt_clean, res_repaired)},
    ]).set_index("council")
    print(comparison.to_string())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dose.to_csv(OUT_DIR / f"oracle_dose_response_{args.underlying}.csv")
    res_one.audit.to_csv(OUT_DIR / f"oracle_localization_{args.underlying}.csv")
    comparison.to_csv(OUT_DIR / f"oracle_repair_{args.underlying}.csv")

    detected = (dose.loc[dose.index > 0, "seats_flagged"] > 0).all()
    localized = set(flagged) == {target}
    print("\nVERDICT:",
          "audit detects every nonzero dose" if detected
          else "⚠️ audit MISSED a nonzero dose",
          "| localization", "exact" if localized else f"⚠️ flagged {flagged}")
    print(f"Saved to {OUT_DIR}/: oracle_dose_response_*, oracle_localization_*, "
          "oracle_repair_*")


if __name__ == "__main__":
    main()
