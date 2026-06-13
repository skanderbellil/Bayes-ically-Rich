#!/usr/bin/env python3
"""
Factor regimes — BOCPD on every factor in the zoo, regime age as a meta-signal.

Idea (novel; the program's own tool pointed at the factor-of-factors level)
---------------------------------------------------------------------------
The meta-factor literature times factors by their returns (momentum),
calendar (seasonality), valuations, or vol. Nobody, as far as the knowledge
base knows, has put an online Bayesian changepoint posterior on each factor
and traded the zoo on its REGIME AGE. That is this program's signature move
(BOCPD-AMR, per-stock ERL panels — roadmap Ideas 1-2) lifted one level up:

  per factor f, month t:  BOCPD (Normal-Gamma, hazard 1/36 — pre-declared,
  factor performance regimes assumed ~3y) filtered over f's full monthly
  history → expected run length ERL_f(t), changepoint prob CP_f(t).

Three pre-declared questions:
  A. Is regime age priced in the cross-section? stale ⅓ (high ERL) vs
     fresh ⅓ (low ERL), spread = stale − fresh.
  B. Does regime age CONDITION factor momentum? Daniel-Moskowitz says
     momentum dies at regime breaks; then TS momentum (12m>0) restricted
     to stale-regime factors should beat unconditional TS momentum
     (Sharpe ~1.0, the Idea-3 result), and the fresh half should be noise.
  C. Systemic dial: the cross-sectional mean CP (how much of the zoo just
     broke) → forward EW-ALL terciles + IC, with the era split that
     exposed the crowding gauges.

Honesty box
-----------
• Trials: 6 portfolios + 1 dial = 7, on the panel with the triptych's
  ledger behind it. Hazard 1/36, median split for B, terciles for A —
  declared here, not searched.
• ERL is bounded by a factor's history length (long-history factors can
  look staler); the cross-sectional rank each month absorbs most of this,
  and all factors carry decades of pre-publication backfill.
• Gross academic L/S returns; the usual survivorship caveats.
"""
import logging
import sys

import numpy as np
import pandas as pd
from tabulate import tabulate

import _bootstrap  # noqa: F401
from posterioralpha.data import load_openap_doc, load_openap_returns
from posterioralpha.research import precompute_bocpd

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MIN_ELIGIBLE = 30
SPLIT_YEAR = 2004
HAZARD = 1 / 36                 # ~one factor-performance regime per 3 years


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


def alpha_vs(r: pd.Series, bench: pd.Series) -> tuple[float, float, float]:
    df = pd.concat([r, bench], axis=1, keys=["r", "b"], sort=False).dropna()
    X = np.column_stack([np.ones(len(df)), df["b"].values])
    beta, *_ = np.linalg.lstsq(X, df["r"].values, rcond=None)
    resid = df["r"].values - X @ beta
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * resid.var(ddof=2))
    return beta[0] * 12, beta[0] / se[0], beta[1]


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
    bench = rete.mean(axis=1).loc[live]
    logger.info(f"{rets.shape[1]} predictors; live (≥{MIN_ELIGIBLE} eligible): "
                f"{live[0].date()} → {live[-1].date()}")

    # ── BOCPD per factor (causal filter; signal lagged 1m below) ─────────
    logger.info(f"BOCPD per factor (hazard {HAZARD:.4f}) …")
    erl = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    cp = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns)
    for f in rets.columns:
        h = rets[f].dropna()
        c, e = precompute_bocpd(h.values, hazard=HAZARD)
        erl.loc[h.index, f] = e
        cp.loc[h.index, f] = c
    erl_sig = erl.shift(1).where(elig)
    cp_sig = cp.shift(1).where(elig)

    # ── A. regime-age cross-section ───────────────────────────────────────
    stale_m = erl_sig.ge(erl_sig.quantile(2 / 3, axis=1), axis=0)
    fresh_m = erl_sig.le(erl_sig.quantile(1 / 3, axis=1), axis=0)
    stale = rete.where(stale_m).mean(axis=1)
    fresh = rete.where(fresh_m).mean(axis=1)

    # ── B. regime-conditioned momentum ────────────────────────────────────
    mom_pos = rets.rolling(12).sum().shift(1).where(elig) > 0
    erl_med = erl_sig.ge(erl_sig.median(axis=1), axis=0)
    ts_all = rete.where(mom_pos).mean(axis=1)
    ts_stale = rete.where(mom_pos & erl_med).mean(axis=1)
    ts_fresh = rete.where(mom_pos & ~erl_med).mean(axis=1)

    ports = {
        "stale 1/3 (high ERL)": stale,
        "fresh 1/3 (low ERL)": fresh,
        "regime-age spread (stale-fresh)": stale - fresh,
        "TS mom unconditional (12m>0)": ts_all,
        "TS mom & stale regime": ts_stale,
        "TS mom & fresh regime": ts_fresh,
        "EW ALL (benchmark)": rete.mean(axis=1),
    }
    ports = {k: v.loc[live] for k, v in ports.items()}

    rows, raw = [], []
    for name, r in ports.items():
        s = ann_stats(r)
        entry = {"portfolio": name, **s}
        if name != "EW ALL (benchmark)":
            a, t_a, b = alpha_vs(r, ports["EW ALL (benchmark)"])
            entry.update({"alpha_ann": a, "alpha_t": t_a, "beta": b})
            extra = [f"{a:+.2%}", f"{t_a:.2f}", f"{b:.2f}"]
        else:
            extra = ["—", "—", "—"]
        rows.append([name, f"{s['ann_ret']:+.2%}", f"{s['ann_vol']:.2%}",
                     f"{s['sharpe']:.2f}", f"{s['t_stat']:.2f}",
                     f"{s['max_dd']:.0%}"] + extra)
        raw.append(entry)
        for tag, sub in [(f"≤{SPLIT_YEAR}", r[r.index.year <= SPLIT_YEAR]),
                         (f">{SPLIT_YEAR}", r[r.index.year > SPLIT_YEAR])]:
            raw.append({"portfolio": f"{name} {tag}", **ann_stats(sub)})

    print("\nRegime-age portfolios & conditioned momentum (gross):")
    print(tabulate(rows, headers=["portfolio", "ann ret", "vol", "Sharpe", "t",
                                  "maxDD", "α vs EW", "t(α)", "β"],
                   tablefmt="github"))
    sub_rows = [[e["portfolio"], f"{e['ann_ret']:+.2%}", f"{e['sharpe']:.2f}",
                 f"{e['t_stat']:.2f}"]
                for e in raw if str(SPLIT_YEAR) in e["portfolio"]]
    print(f"\nSub-period stability (split {SPLIT_YEAR}):")
    print(tabulate(sub_rows, headers=["portfolio", "ann ret", "Sharpe", "t"],
                   tablefmt="github"))

    # ── B2. dose-response: momentum return by regime-age quintile ─────────
    # (diagnostic for the mechanism + 1 declared trial: the Idea-1 lesson
    # says regime gates bite at EXTREME freshness, so also test the
    # momentum book that only excludes the freshest decile)
    win_erl = erl_sig.where(mom_pos)
    qrows = []
    for q in range(5):
        lo_q = win_erl.quantile(q / 5, axis=1)
        hi_q = win_erl.quantile((q + 1) / 5, axis=1)
        m = win_erl.ge(lo_q, axis=0) & win_erl.le(hi_q, axis=0)
        r = rete.where(m).mean(axis=1).loc[live]
        s = ann_stats(r)
        qrows.append([f"Q{q+1} {'(freshest)' if q == 0 else '(stalest)' if q == 4 else ''}",
                      f"{s['ann_ret']:+.2%}", f"{s['sharpe']:.2f}",
                      f"{s['t_stat']:.2f}"])
        raw.append({"portfolio": f"TS mom ERL Q{q+1}", **s})
    print("\nDose-response inside the momentum book (ERL quintiles of winners):")
    print(tabulate(qrows, headers=["ERL quintile", "ann ret", "Sharpe", "t"],
                   tablefmt="github"))

    not_extreme_fresh = erl_sig.gt(erl_sig.quantile(0.10, axis=1), axis=0)
    ts_x = rete.where(mom_pos & not_extreme_fresh).mean(axis=1).loc[live]
    s = ann_stats(ts_x)
    a, t_a, b = alpha_vs(ts_x, ports["EW ALL (benchmark)"])
    print(f"\nTS mom ex freshest decile: ann {s['ann_ret']:+.2%}  Sharpe "
          f"{s['sharpe']:.2f} (t {s['t_stat']:.2f})  α vs EW {a:+.2%} "
          f"(t {t_a:.2f}, β {b:.2f})")
    raw.append({"portfolio": "TS mom ex freshest decile", **s,
                "alpha_ann": a, "alpha_t": t_a, "beta": b})

    # ── C. systemic changepoint dial ──────────────────────────────────────
    dial = cp_sig.mean(axis=1).loc[live]
    fwd1 = bench.shift(-1)
    erank = dial.expanding(60).apply(lambda x: (x[:-1] < x[-1]).mean(), raw=True)
    terc = pd.cut(erank, [0, 1 / 3, 2 / 3, 1.0001], labels=["lo", "mid", "hi"])
    print("\nSystemic dial: x-sectional mean changepoint prob → forward EW:")
    drows = []
    for lab in ["lo", "mid", "hi"]:
        r = fwd1[(terc == lab) & fwd1.notna()]
        drows.append([lab, len(r), f"{r.mean()*12:+.2%}",
                      f"{r.mean()/(r.std()+1e-12)*np.sqrt(12):.2f}",
                      f"{r.std()*np.sqrt(12):.2%}"])
        raw.append({"portfolio": f"dial {lab}", "ann_ret": r.mean() * 12,
                    "sharpe": r.mean() / (r.std() + 1e-12) * np.sqrt(12),
                    "ann_vol": r.std() * np.sqrt(12), "n_months": len(r)})
    ic = dial.corr(fwd1, method="spearman")
    sub_ic = []
    for tag, m in [(f"≤{SPLIT_YEAR}", dial.index.year <= SPLIT_YEAR),
                   (f">{SPLIT_YEAR}", dial.index.year > SPLIT_YEAR)]:
        i = dial[m].corr(fwd1[m], method="spearman")
        n = (dial[m].notna() & fwd1[m].notna()).sum()
        sub_ic.append(f"{tag} {i:+.3f} (t {i*np.sqrt(n):+.1f})")
    drows.append(["rank IC", "", f"{ic:+.3f} (t {ic*np.sqrt(len(dial)):+.1f})",
                  sub_ic[0], sub_ic[1]])
    print(tabulate(drows, headers=["tercile", "n", "fwd 1m (ann)", "fwd Sharpe",
                                   "fwd vol / era IC"], tablefmt="github"))

    out = RESULTS_DIR / "factor_regimes.csv"
    pd.DataFrame(raw).to_csv(out, index=False)
    logger.info(f"\nSaved {out}")
    print("\n⚠ gross panel; hazard/splits pre-declared, not searched; ERL is "
          "weakly confounded with history length (rank x-sections absorb "
          "most of it).")


if __name__ == "__main__":
    main()
