#!/usr/bin/env python3
"""
Factor seasonality out of sample — the JKP cross-region kill test.

Idea (the validation step Idea 10 demands)
------------------------------------------
run_zoo_metafactors.py found one t>3 positive on the OpenAP panel:
Keloharju same-calendar-month seasonality at the factor level (α vs EW-ALL
+9.5%/yr, t 3.6). Before that claim stands, it must survive data it was
not found on. The JKP panel is the repo's designated OOS set (roadmap
§4b): 13 theme factors × 4 regions, monthly, 1926→. Same recipe, zero
re-tuning: signal = same-calendar-month mean over the prior 10 years
(≥5 obs), spread = EW top third − bottom third of the 13 themes (4 names
per leg), benchmark = EW of all themes, live where all 13 themes exist.

Read
----
usa overlaps the OpenAP discovery sample (different construction — themes
vs raw predictors — so it's a robustness check); developed / emerging /
world_ex_us are genuinely out of sample. The deciding number is each
region's α vs its own EW benchmark.

Honesty box
-----------
• 13 themes is a thin cross-section (4 per leg) — noisier spreads than
  OpenAP's ~60 per leg; that biases toward a false null, not false
  positive.
• 4 regions × 1 recipe = 4 trials, no parameters searched.
• Gross academic returns, same caveats as the discovery study.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_jkp_factors

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

REGIONS = ["usa", "developed", "emerging", "world_ex_us"]


def ann_stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 24:
        return {"ann_ret": np.nan, "ann_vol": np.nan, "sharpe": np.nan,
                "t_stat": np.nan, "n_months": len(r)}
    mu, sd = r.mean() * 12, r.std() * np.sqrt(12)
    return {"ann_ret": mu, "ann_vol": sd, "sharpe": mu / (sd + 1e-12),
            "t_stat": mu / (sd + 1e-12) * np.sqrt(len(r) / 12.0),
            "n_months": len(r)}


def alpha_vs(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    df = pd.concat([r, bench], axis=1, keys=["r", "b"], sort=False).dropna()
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    beta, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
    return beta[0] * 12, beta[0] / se[0]


def main():
    rows, raw = [], []
    for region in REGIONS:
        rets = load_jkp_factors(region).sort_index()
        rets = rets.loc[rets.notna().all(axis=1).idxmax():]

        seas = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
        for m in range(1, 13):
            block = rets[rets.index.month == m]
            s = block.shift(1).rolling(10, min_periods=5).mean()
            seas.loc[s.index] = s.values

        hi = seas.ge(seas.quantile(2 / 3, axis=1), axis=0)
        lo = seas.le(seas.quantile(1 / 3, axis=1), axis=0)
        valid = seas.notna().sum(axis=1) >= 9
        spread = (rets.where(hi).mean(axis=1)
                  - rets.where(lo).mean(axis=1)).where(valid).dropna()
        bench = rets.mean(axis=1).reindex(spread.index)

        s = ann_stats(spread)
        a, t_a = alpha_vs(spread, bench)
        mid = spread.index[len(spread) // 2]
        halves = [ann_stats(spread.loc[:mid]), ann_stats(spread.loc[mid:])]
        rows.append([region, f"{spread.index[0].date()}→", s["n_months"],
                     f"{s['ann_ret']:+.2%}", f"{s['sharpe']:.2f}",
                     f"{s['t_stat']:.2f}", f"{a:+.2%}", f"{t_a:.2f}",
                     f"{halves[0]['sharpe']:.2f} / {halves[1]['sharpe']:.2f}"])
        raw.append({"region": region, **s, "alpha_ann": a, "alpha_t": t_a,
                    "sharpe_h1": halves[0]["sharpe"],
                    "sharpe_h2": halves[1]["sharpe"]})

    print("\nFactor seasonality OOS — JKP themes, same recipe, no re-tuning:")
    print(tabulate(rows, headers=["region", "live", "months", "ann ret",
                                  "Sharpe", "t", "α vs EW", "t(α)",
                                  "Sharpe h1/h2"], tablefmt="github"))

    out = RESULTS_DIR / "seasonality_oos.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ 13-theme cross-section (4 per leg) — thin legs bias toward a "
          "false null; usa is a robustness check, the other regions are the "
          "true OOS.")


if __name__ == "__main__":
    main()
