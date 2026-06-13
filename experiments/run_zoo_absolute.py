#!/usr/bin/env python3
"""
The absolute-return zoo — value/momentum 50/50 and the all-in stack.

Idea (user prompt: chase absolute return, beta or alpha indifferent)
--------------------------------------------------------------------
Every prior study on this panel asked "does X beat the EW-ALL Sharpe" — an
alpha question. This one asks the AQR question instead: the zoo's meta-
streams are individually mediocre but (if value and momentum here behave
like they do everywhere) NEGATIVELY correlated, so the blend can earn more
per unit of risk than any sleeve. Streams (all causal, monthly, from the
prior studies' exact constructions):

  MOM    12m XS spread (top⅓ − bottom⅓ by trailing 12m)      [Idea 3]
  VAL    LT-reversal spread (long losers over t-60..t-13)     [Idea 10]
  SEAS   same-calendar-month spread (10y, ≥5 obs)             [Idea 10]
  EW     EW of all published factors (the Sharpe-1.3 carry)   [Idea 7]

Blends, pre-declared
--------------------
  VM 50/50          0.5·MOM + 0.5·VAL          (the classic, the question)
  VM inv-vol        causal 36m inverse-vol weights
  VMS inv-vol       + seasonality sleeve
  ALL-IN inv-vol    + the EW carry as a fourth sleeve
  ALL-IN @10% vol   causal 12m vol-targeted (lev cap 3×) — the
                    absolute-return chase, levered to a real budget

Honesty box
-----------
• Gross academic L/S: spreads are 2× levered and turn over sleeves
  monthly; the vol-targeted stack borrows on top. Costs/financing would
  take real bites — this maps the ceiling, not the net.
• Trials: 5 blends (+1 vol-targeted EW reference) = 6, on streams already
  in the ledger.
• Correlations are the load-bearing claim — reported, full and by era.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_ELIGIBLE = 30
SPLIT_YEAR = 2004
VOL_WIN = 36                    # months, causal inverse-vol weights
TARGET_VOL = 0.10
LEV_CAP = 3.0


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "max_dd": np.nan, "n_months": len(r)}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    eq = (1 + r).cumprod()
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": mu / (sd + 1e-12),
            "t_stat": mu / (sd + 1e-12) * np.sqrt(len(r) / 12.0),
            "max_dd": float((eq / eq.cummax() - 1).min()),
            "n_months": len(r)}


def main():
    rets = load_openap_returns()
    doc = load_openap_doc().set_index("Acronym")
    meta = doc.loc[doc.index.intersection(rets.columns), ["Year"]].astype(float)
    rets = rets[meta.index]

    age_y = rets.index.year.values[:, None] - meta.Year.values
    published = pd.DataFrame(age_y >= 1, index=rets.index, columns=rets.columns)
    elig = (published & rets.notna()).shift(1).fillna(False)
    live = elig.sum(axis=1)
    live = live[live >= MIN_ELIGIBLE].index
    rete = rets.where(elig)
    logger.info(f"{rets.shape[1]} predictors; live (≥{MIN_ELIGIBLE} eligible): "
                f"{live[0].date()} → {live[-1].date()}")

    def xs_spread(sig):
        sig = sig.where(elig)
        hi = sig.ge(sig.quantile(2 / 3, axis=1), axis=0)
        lo = sig.le(sig.quantile(1 / 3, axis=1), axis=0)
        return rete.where(hi).mean(axis=1) - rete.where(lo).mean(axis=1)

    seas = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for m in range(1, 13):
        block = rets[rets.index.month == m]
        s = block.shift(1).rolling(10, min_periods=5).mean()
        seas.loc[s.index] = s.values

    streams = pd.DataFrame({
        "MOM": xs_spread(rets.rolling(12).sum().shift(1)),
        "VAL": xs_spread(-rets.rolling(48).sum().shift(13)),
        "SEAS": xs_spread(seas),
        "EW": rete.mean(axis=1),
    }).loc[live]

    print("\nStream correlations (the load-bearing fact), full / ≤2004 / >2004:")
    eras = [streams, streams[streams.index.year <= SPLIT_YEAR],
            streams[streams.index.year > SPLIT_YEAR]]
    pairs = [("MOM", "VAL"), ("MOM", "SEAS"), ("VAL", "SEAS"),
             ("MOM", "EW"), ("VAL", "EW"), ("SEAS", "EW")]
    print(tabulate([[f"{a}/{b}"] + [f"{e[a].corr(e[b]):+.2f}" for e in eras]
                    for a, b in pairs],
                   headers=["pair", "full", f"≤{SPLIT_YEAR}", f">{SPLIT_YEAR}"],
                   tablefmt="github"))

    # causal inverse-vol weights across sleeves
    inv = (1.0 / streams.rolling(VOL_WIN, min_periods=24).std()).shift(1)

    def inv_vol_mix(cols):
        w = inv[cols].div(inv[cols].sum(axis=1), axis=0)
        return (w * streams[cols]).sum(axis=1, min_count=len(cols))

    ports = {
        "MOM alone": streams["MOM"],
        "VAL alone": streams["VAL"],
        "VM 50/50": 0.5 * streams["MOM"] + 0.5 * streams["VAL"],
        "VM inv-vol": inv_vol_mix(["MOM", "VAL"]),
        "VMS inv-vol": inv_vol_mix(["MOM", "VAL", "SEAS"]),
        "ALL-IN inv-vol (+EW)": inv_vol_mix(["MOM", "VAL", "SEAS", "EW"]),
        "EW ALL (reference)": streams["EW"],
    }

    # causal vol targeting of the all-in stack (and EW, for a fair race)
    for tag, base in [("ALL-IN @10% vol", ports["ALL-IN inv-vol (+EW)"]),
                      ("EW @10% vol (reference)", streams["EW"])]:
        lev = (TARGET_VOL / (base.rolling(12).std() * np.sqrt(12))
               ).clip(upper=LEV_CAP).shift(1)
        ports[tag] = base * lev

    rows, raw = [], []
    for name, r in ports.items():
        s = ann_stats(r)
        sub = [ann_stats(r[r.index.year <= SPLIT_YEAR])["sharpe"],
               ann_stats(r[r.index.year > SPLIT_YEAR])["sharpe"]]
        rows.append([name, f"{s['ann_ret']:+.2%}", f"{s['ann_vol']:.2%}",
                     f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}",
                     f"{s['max_dd']:.0%}", f"{sub[0]:.2f} / {sub[1]:.2f}"])
        raw.append({"portfolio": name, **s,
                    "sharpe_pre": sub[0], "sharpe_post": sub[1]})

    print("\nAbsolute-return blends (gross):")
    print(tabulate(rows, headers=["portfolio", "ann ret", "vol", "Sharpe", "t",
                                  "maxDD", f"SR ≤/{'>'}{SPLIT_YEAR}"],
                   tablefmt="github"))

    out = RESULTS_DIR / "zoo_absolute.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ gross ceiling, not net: 2×-levered sleeves, monthly turnover, "
          "and the vol target borrows on top. Financing + costs come off "
          "the ann ret before any of this is real.")


if __name__ == "__main__":
    main()
