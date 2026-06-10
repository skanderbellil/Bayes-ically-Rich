#!/usr/bin/env python3
"""
Vehicle switching — same exposure, choose the WRAPPER by regime.

Literature check (June 2026): Gayed & Bilello's "Leverage for the Long Run"
(Dow Award 2016) gates leveraged *exposure* by the 200d MA; arXiv 2504.20116
(2025) shows theoretically that daily-reset LETF compounding is governed by
return AUTOCORRELATION (positive → compounding tailwind, negative → drag),
not just volatility.  Neither — nor anything else we found — tests the
implementation question we actually face: given a FIXED exposure dial (the
5-layer vote), deliver the same target exposure through

  margin wrapper : 2×vote × QQQ financed at retail RF+150bp
  LETF wrapper   : weight = vote in QLD + cash   (real fund prints)

and SWITCH wrappers daily by a causal regime conditioner.  The vote is
untouched — this is purely an implementation-layer study.  Three candidate
switches race head-to-head (multiplicity flagged; SPY/SSO confirmation block
per the tug-conditioner lesson):

  G  trend location : QQQ > 200d MA          (Gayed's vol-clustering story)
  A  autocorrelation: rolling 1y lag-1 autocorr > 0   (the arXiv story)
  H  Hurst          : rolling 1y R/S Hurst > 0.5      (repo's own machinery)

Primary metric — before any strategy table — is the conditional WRAPPER
SPREAD: mean(LETF-implementation return − margin-implementation return) by
conditioner state.  If a conditioner can't separate the spread, it can't
help, whatever its equity curve says.

    python experiments/run_vehicle_switch.py
"""
import logging
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_net_liquidity
from posterioralpha.research import rolling_hurst, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"
LEV_CSV = _bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
RETAIL_SPREAD = 0.015
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21


def margin_daily(expo, r_under):
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    return (expo * r_under
            - np.maximum(expo - 1.0, 0.0) * (RF + RETAIL_SPREAD) / 252
            + np.maximum(1.0 - expo, 0.0) * DAILY_RF)


def letf_daily(vote, r_letf):
    w = vote.clip(0.0, 1.0).fillna(0.0)
    return w * r_letf + (1.0 - w) * DAILY_RF


def with_tc(r, expo_like):
    return r - TC * expo_like.diff().abs().fillna(0.0)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    lev = pd.read_csv(LEV_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index)
           .intersection(lev.dropna().index))
    nl, p, lev = nl.reindex(idx), p.reindex(idx), lev.reindex(idx)

    liq = soft_sign(nl.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(p["VIX3M"] / p["VIX"] - 1.0).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)

    def setup(t, letf):
        o, c = panel[f"{t}_Open"].reindex(idx), panel[f"{t}_Close"].reindex(idx)
        r_on, r_id = o / c.shift(1) - 1.0, c / o - 1.0
        tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                         - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
        vote = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                             "dollar": dollar, "tug": tug}).mean(axis=1)
        r_u = p[t].pct_change()
        r_l = lev[letf].pct_change()
        cl = p[t]
        conds = {
            "G  >200d MA":  (cl > cl.rolling(200).mean()).astype(float).shift(1),
            "A  autocorr>0": (r_u.rolling(252).apply(
                lambda x: pd.Series(x).autocorr(1), raw=False) > 0)
                .astype(float).shift(1),
            "H  Hurst>0.5": (pd.Series(rolling_hurst(r_u.fillna(0).values, 252),
                                       index=idx) > 0.5).astype(float).shift(1),
        }
        return vote, r_u, r_l, conds

    vote, r_u, r_l, conds = setup("QQQ", "QLD")
    m_r = margin_daily(2.0 * vote, r_u)        # margin wrapper, pre-TC
    l_r = letf_daily(vote, r_l)                # LETF wrapper, pre-TC
    spread = l_r - m_r                          # wrapper spread, same exposure

    live = vote.first_valid_index()
    print("\n" + "=" * 96)
    print(f"  VEHICLE SWITCHING  ·  same exposure (2×vote on QQQ), wrapper chosen "
          f"by regime  ·  {live.date()} → {idx[-1].date()}")
    print("=" * 96)
    print("  Conditional wrapper spread (LETF − margin, annualized, same exposure):")
    rows = []
    for name, c in conds.items():
        on = float(spread[(c == 1)].mean()) * 252
        off = float(spread[(c == 0)].mean()) * 252
        share = float((c == 1).loc[live:].mean())
        rows.append([name, f"{on:+.2%}", f"{off:+.2%}", f"{on - off:+.2%}",
                     f"{share:.0%}"])
    rows.append(["unconditional", f"{float(spread.loc[live:].mean()) * 252:+.2%}",
                 "—", "—", "100%"])
    print(tabulate(rows, headers=["conditioner", "state ON (use LETF)",
                                  "state OFF (use margin)", "separation",
                                  "% days ON"], tablefmt="github"))

    # ── Strategies ───────────────────────────────────────────────────────────
    strat = {
        "QQQ buy & hold": r_u,
        "margin always (150bp)": with_tc(m_r, 2.0 * vote),
        "QLD always": with_tc(l_r, vote),
    }
    for name, c in conds.items():
        r_sw = pd.Series(np.where(c == 1, l_r, m_r), index=idx)
        # TC on the switch itself: full wrapper swap turns over ~2×vote notional
        swap_cost = TC * 2.0 * vote.fillna(0) * c.diff().abs().fillna(0)
        strat[f"switch: {name}"] = with_tc(r_sw, 2.0 * vote) - swap_cost
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = p["SPY"].pct_change().loc[live:]

    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}"])
    print("\n" + tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin"],
                          tablefmt="github"))

    # ── Confirmation on SPY/SSO (no re-selection) ────────────────────────────
    print("\n" + "=" * 96)
    print("  CONFIRMATION  ·  SPY/SSO, same constructions")
    print("=" * 96)
    vote_s, r_us, r_ls, conds_s = setup("SPY", "SSO")
    m_s = margin_daily(2.0 * vote_s, r_us)
    l_s = letf_daily(vote_s, r_ls)
    spread_s = l_s - m_s
    rows = []
    for name, c in conds_s.items():
        on = float(spread_s[(c == 1)].mean()) * 252
        off = float(spread_s[(c == 0)].mean()) * 252
        rows.append([name, f"{on:+.2%}", f"{off:+.2%}", f"{on - off:+.2%}"])
    rows.append(["unconditional", f"{float(spread_s.loc[live:].mean()) * 252:+.2%}",
                 "—", "—"])
    print(tabulate(rows, headers=["conditioner", "ON", "OFF", "separation"],
                   tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    cum_spread = spread.loc[live:].cumsum()
    ax1.plot(cum_spread.index, cum_spread.values * 100, lw=1.2, color="tab:purple",
             label="cumulative wrapper spread (LETF − margin), pp")
    c = conds["G  >200d MA"].loc[live:]
    ax1.fill_between(cum_spread.index, *ax1.get_ylim(), where=(c == 0),
                     color="red", alpha=0.06, label="below 200d MA (margin state)")
    ax1.axhline(0, color="grey", lw=0.6)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_title("When does the daily-reset wrapper win? (QQQ/QLD, same exposure)")
    for name in ["QLD always", "margin always (150bp)", "switch: G  >200d MA",
                 "switch: A  autocorr>0"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        ax2.plot(eq.index, eq.values, lw=1.2, label=name)
    ax2.set_yscale("log"); ax2.set_ylabel("growth of $1 (log)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_title("Wrapper strategies, identical exposure dial")
    fig.tight_layout()
    out = RESULTS_DIR / "vehicle_switch.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
