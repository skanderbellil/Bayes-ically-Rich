#!/usr/bin/env python3
"""
Build the free factor datasets cited in the knowledge base → ``datasets/``.

Sources (docs/knowledge/systematic-equity-strategies.md §4):
  ff      — Kenneth French library: US FF5+Mom (monthly + daily), Europe
            5F+Mom (monthly), 12 industry portfolios (monthly).
            → ff_factors_monthly.csv, ff_factors_daily.csv.gz,
              ff_europe_monthly.csv, ff_industry12_monthly.csv
  jkp     — Jensen-Kelly-Pedersen: 13 theme returns, monthly vw_cap, for
            usa/developed/emerging/world_ex_us. → jkp_factors.csv.gz
  openap  — Chen-Zimmermann: 200+ predictor monthly L/S returns + signal
            documentation (needs gdown: pip install -e .[data]).
            → openap_ls_returns.csv.gz, openap_signal_doc.csv

Network; run once (or to refresh — French updates monthly).
"""
import _bootstrap  # noqa: F401  (repo root on sys.path)

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser(description="Build factor datasets")
    ap.add_argument("--source",
                    choices=["ff", "jkp", "jkp-indiv", "openap", "all"],
                    default="all")
    args = ap.parse_args()

    if args.source in ("ff", "all"):
        from posterioralpha.data.factors import build_ff_factors
        for name, df in build_ff_factors().items():
            print(f"{name}: {df.shape[0]} rows × {df.shape[1]} cols "
                  f"({df.index[0].date()} → {df.index[-1].date()})  "
                  f"cols={list(df.columns)[:8]}")

    if args.source in ("jkp", "all"):
        from posterioralpha.data.factors import build_jkp_factors
        panel = build_jkp_factors()
        print(f"jkp_factors: {len(panel)} rows, "
              f"regions={sorted(panel['location'].unique())}, "
              f"themes={panel['name'].nunique()}, "
              f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    if args.source in ("jkp-indiv", "all"):
        from posterioralpha.data.factors import build_jkp_individual
        panel = build_jkp_individual()
        print(f"jkp_factors_individual: {len(panel)} rows, "
              f"regions={sorted(panel['location'].unique())}, "
              f"factors={panel['name'].nunique()}, "
              f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    if args.source in ("openap", "all"):
        from posterioralpha.data.factors import build_openap
        ls = build_openap()
        print(f"openap_ls_returns: {ls.shape[0]} months × "
              f"{ls.shape[1]} predictors "
              f"({ls.index[0].date()} → {ls.index[-1].date()})")


if __name__ == "__main__":
    main()
