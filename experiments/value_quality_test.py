"""
Honest test of VALUE / QUALITY / PROFITABILITY factors (JKP, 4 regions).
Bar: t>3 in US, replicate (same sign, t>2) in developed/emerging/world_ex_us,
survive ~1-2%/yr cost haircut, hold in both sub-periods (split 2004).
Cross-region replication is the multiple-testing guard. (Agent-authored,
loop 2026-06-14: launched-agents honest-edge hunt.)
"""
import _bootstrap  # noqa
import os
import numpy as np
import pandas as pd

REGIONS = ["usa", "developed", "emerging", "world_ex_us"]
SPLIT = "2004-01-01"
ROOT = _bootstrap.ROOT
ind = pd.read_csv(ROOT / "datasets/jkp_factors_individual.csv.gz", parse_dates=["date"])
thm = pd.read_csv(ROOT / "datasets/jkp_factors.csv.gz", parse_dates=["date"])

CAND_IND = ["be_me","at_be","ni_be","ope_be","ope_bel1","gp_at","gp_atl1",
    "ebit_bev","ebit_sale","ebitda_mev","sale_me","fcf_me","cop_at","cop_atl1",
    "div12m_me","niq_at","niq_be","niq_su","niq_at_chg1","niq_be_chg1","qmj",
    "qmj_prof","qmj_safety","qmj_growth","o_score","z_score"]
CAND_THM = ["value","quality","profitability","profit_growth"]


def stats(r):
    r = r.dropna()
    if len(r) < 24:
        return np.nan, np.nan
    m, s = r.mean(), r.std(ddof=1)
    if s == 0:
        return np.nan, np.nan
    return m / s * np.sqrt(12), m / s * np.sqrt(len(r))


def get_series(src, name, loc):
    sub = src[(src.name == name) & (src.location == loc)]
    return sub.set_index("date")["ret"].sort_index()


def eval_factor(src, name, cost_ann=0.015):
    us = get_series(src, name, "usa")
    sign = 1.0 if us.mean() >= 0 else -1.0
    out = {"name": name, "sign": sign}
    for loc in REGIONS:
        r = get_series(src, name, loc) * sign
        out[f"{loc}_sh"], out[f"{loc}_t"] = stats(r)
        out[f"{loc}_tc"] = stats(r - cost_ann / 12.0)[1]
        out[f"{loc}_tpre"] = stats(r[r.index < SPLIT])[1]
        out[f"{loc}_tpost"] = stats(r[r.index >= SPLIT])[1]
    return out


def full_pass(o):
    if not (o["usa_tc"] > 3):
        return False
    for loc in REGIONS:
        if not (o[f"{loc}_tpre"] > 2 and o[f"{loc}_tpost"] > 2):
            return False
        if loc != "usa" and not (o[f"{loc}_tc"] > 2):
            return False
    return True


results = [eval_factor(ind, n) for n in CAND_IND] + [eval_factor(thm, n) for n in CAND_THM]
winners = [o["name"] for o in results if full_pass(o)]
print("Factors clearing FULL bar (t>3 US net-cost + cross-region t>2 + both sub-periods):")
print("  ", winners or "NONE")
for o in results:
    if o["name"] in winners or o["name"] in ("qmj", "profitability", "value", "quality"):
        print(f"\n{o['name']:12s} (sign {o['sign']:+.0f})  {'FULL PASS' if o['name'] in winners else ''}")
        for loc in REGIONS:
            print(f"  {loc:12s} Sharpe {o[f'{loc}_sh']:5.2f}  t {o[f'{loc}_t']:5.2f}  "
                  f"t@1.5% {o[f'{loc}_tc']:5.2f}  pre04 {o[f'{loc}_tpre']:5.2f}  post04 {o[f'{loc}_tpost']:5.2f}")
