#!/usr/bin/env python3
"""
The liquidity shadow test — does the champion's last plumbing layer survive?

Four consecutive studies killed four macro-plumbing nowcasts with the same
autopsy: auction demand, funding stress, withheld taxes and dealer inventories
are all real macro information that equities have already priced — and the
dealer study sharpened the method, showing the gauge *loses to its own VIX
shadow* (the price-based proxy embeds the plumbing better than the plumbing
data does).  Every layer that survives in the champion is price-based
(VIX term structure, HYG−IEF, −UUP, tug) — except one.  Net liquidity
(Fed assets − TGA − RRP) is the founding layer of the whole thread and the
last surviving piece of Fed-plumbing data.  It has never been given the test
that killed its four cousins.

  shadow_t   = causal price-only replica of the liquidity signal: rolling OLS
               (trailing 3y window, refit monthly, strictly past data) of
               liq_raw = Δ126d net liquidity (pub-lagged) on six observable
               price features at t — IEF/UUP/SPY 126d returns, HYG−IEF 126d,
               VIX 21d mean, VIX3M/VIX − 1.  Prices have no publication lag,
               so this is exactly "what you could have built without FRED".
  residual_t = liq_raw_t − shadow_t   (the true plumbing component)

  Tests (the house template)
  -----
  1. SPANNING      How much of the liquidity signal is price-spanned?
                   Out-of-sample R² and rolling corr(shadow, raw).
  2. INFORMATION   corr(·, fwd 63d QQQ) + causal-tercile forward returns for
                   the raw signal, the shadow, and the residual — does the
                   plumbing component carry any information of its own?
  3. HEAD-TO-HEAD  The 2×5-layer champion with its liq layer (a) real,
                   (b) shadow-replaced, (c) residual-only, (d) dropped —
                   honest costs, rolling 3y OOS win rates vs SPY.

Either outcome matters: if liquidity fails its shadow, the champion is a
pure price vote (no FRED key, no pub-lag risk) and the nulls become five
with one conclusion; if the residual carries signal, we've isolated the one
place where plumbing genuinely leads prices.

Warm-ups (3y regression + 3y MAD) push the common live date past the GFC —
the head-to-head reads 2010→, the GFC stays out of reach for the shadow.

    python experiments/run_liquidity_shadow.py
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
from posterioralpha.research import soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21
REG_WIN, REG_MIN, REFIT = 756, 252, 21


def causal_shadow(y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Rolling-OLS price replica: coefficients fitted on strictly past data
    (trailing REG_WIN, refit every REFIT days), applied to features at t."""
    yv, Xv = y.values.astype(float), X.values.astype(float)
    valid = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
    out = np.full(len(yv), np.nan)
    beta = None
    for i in range(len(yv)):
        if i % REFIT == 0:
            lo = max(0, i - REG_WIN)
            m = valid[lo:i]
            if m.sum() >= REG_MIN:
                Xw = np.column_stack([np.ones(int(m.sum())), Xv[lo:i][m]])
                beta, *_ = np.linalg.lstsq(Xw, yv[lo:i][m], rcond=None)
        if beta is not None and np.isfinite(Xv[i]).all():
            out[i] = beta[0] + Xv[i] @ beta[1:]
    return pd.Series(out, index=y.index)


def causal_rank(s: pd.Series) -> pd.Series:
    return s.rolling(ROLL, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True)


def run(expo, r_cc):
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * r_cc
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return pd.Series(gross - financing + cash - costs, index=r_cc.index)


def roll_sharpe(r):
    ex = r - DAILY_RF
    return (ex.rolling(ROLL).mean() / ex.rolling(ROLL).std()) * np.sqrt(252)


def roll_cagr(r):
    return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1


def main() -> None:
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = (nl.dropna().index
           .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                         .dropna().index)
           .intersection(panel.dropna().index))
    nl, p = nl.reindex(idx), p.reindex(idx)
    qqq = p["QQQ"]
    r_cc = qqq.pct_change()
    spy_ret = p["SPY"].pct_change()

    # ── 1. The shadow: price-only replica of the liquidity signal ───────────
    liq_raw = nl.diff(H_LIQ).shift(PUB_LAG)
    feats = pd.DataFrame({
        "ief126": p["IEF"].pct_change(H_LIQ),
        "uup126": p["UUP"].pct_change(H_LIQ),
        "credit126": p["HYG"].pct_change(H_LIQ) - p["IEF"].pct_change(H_LIQ),
        "spy126": p["SPY"].pct_change(H_LIQ),
        "vix21": p["VIX"].rolling(21).mean(),
        "vixts": p["VIX3M"] / p["VIX"] - 1.0,
    })
    shadow = causal_shadow(liq_raw, feats)
    residual = liq_raw - shadow
    both = shadow.notna() & liq_raw.notna()
    start = shadow.dropna().index[0]

    sse = float(((liq_raw - shadow)[both] ** 2).sum())
    sst = float(((liq_raw[both] - liq_raw[both].mean()) ** 2).sum())
    r2_oos = 1.0 - sse / sst
    roll_corr = liq_raw.rolling(ROLL, min_periods=252).corr(shadow)

    print("\n" + "=" * 96)
    print(f"  LIQUIDITY SHADOW TEST  ·  {start.date()} → {idx[-1].date()}  "
          f"(causal rolling OLS, {REG_WIN}d window, refit {REFIT}d)")
    print("=" * 96)
    print(f"\n  Spanning: out-of-sample R² {r2_oos:+.2f}   "
          f"corr(shadow, raw) {float(np.corrcoef(shadow[both], liq_raw[both])[0, 1]):+.2f}   "
          f"rolling-3y corr median {float(roll_corr.median()):+.2f}  "
          f"[{float(roll_corr.quantile(0.10)):+.2f}, {float(roll_corr.quantile(0.90)):+.2f}]")

    # ── 2. Information: raw vs shadow vs residual ────────────────────────────
    fwd63 = qqq.pct_change(63).shift(-63)
    print("\n  corr(signal_t, fwd 63d QQQ) — full sample and sub-periods:")
    spans = [("full", start, idx[-1]),
             ("2010-2014", "2010", "2014"), ("2015-2019", "2015", "2019"),
             ("2020-2026", "2020", "2026")]
    rows = []
    for name, s in [("raw liquidity", liq_raw), ("price shadow", shadow),
                    ("residual (plumbing)", residual)]:
        row = [name]
        for _, a, b in spans:
            m = s.notna() & fwd63.notna()
            ss, ff = s[m].loc[a:b], fwd63[m].loc[a:b]
            row.append(f"{float(np.corrcoef(ss, ff)[0, 1]):+.2f}" if len(ss) > 100
                       else "—")
        rows.append(row)
    print(tabulate(rows, headers=["signal"] + [s[0] for s in spans],
                   tablefmt="github"))

    print("\n  Forward 63d QQQ (ann.) by causal signal tercile:")
    rows = []
    for name, s in [("raw liquidity", liq_raw), ("price shadow", shadow),
                    ("residual (plumbing)", residual)]:
        rk = causal_rank(s)
        lo, hi = rk <= 0.33, rk >= 0.67
        mid = ~lo & ~hi & rk.notna()
        row = [name]
        for m in (lo, mid, hi):
            mm = m & fwd63.notna()
            row.append(f"{float(fwd63[mm].mean()) * 4:+.1%}")
        mm_hi, mm_lo = hi & fwd63.notna(), lo & fwd63.notna()
        row.append(f"{(float(fwd63[mm_hi].mean()) - float(fwd63[mm_lo].mean())) * 4:+.1%}")
        row.append(f"{float(fwd63[mm_lo].quantile(0.05)):+.1%}/{float(fwd63[mm_hi].quantile(0.05)):+.1%}")
        rows.append(row)
    print(tabulate(rows, headers=["signal", "low", "mid", "high", "hi−lo",
                                  "5th pct lo/hi"], tablefmt="github"))

    # ── 3. Champion head-to-head: real vs shadow vs residual vs none ─────────
    vix_ts = soft_sign(feats["vixts"]).shift(1)
    credit = soft_sign(p["HYG"].pct_change(CREDIT_WIN)
                       - p["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-p["UUP"].pct_change(63)).shift(1)
    q_o, q_c = panel["QQQ_Open"].reindex(idx), panel["QQQ_Close"].reindex(idx)
    r_on, r_id = q_o / q_c.shift(1) - 1.0, q_c / q_o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    base = pd.DataFrame({"vix_ts": vix_ts, "credit": credit,
                         "dollar": dollar, "tug": tug})
    liq_layers = {"real liq (champion)": soft_sign(liq_raw).shift(1),
                  "liq → price shadow": soft_sign(shadow).shift(1),
                  "liq → residual only": soft_sign(residual).shift(1)}

    votes = {f"2×5-layer · {k}": pd.concat([base, v.rename("liq")], axis=1).mean(axis=1)
             for k, v in liq_layers.items()}
    votes["2×4-layer · no liq"] = base.mean(axis=1)
    # fair attribution: start only when every variant's liq layer is live
    # (the vote mean skips missing layers, so first_valid alone is too early)
    live = max([v.first_valid_index() for v in votes.values()]
               + [v.first_valid_index() for v in liq_layers.values()])

    strat = {"SPY buy & hold": spy_ret, "QQQ buy & hold": r_cc}
    strat.update({k: run(2.0 * v, r_cc) for k, v in votes.items()})
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = spy_ret.loc[live:]

    print(f"\n  Champion head-to-head  ·  QQQ, honest costs  ·  common live {live.date()}:")
    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        m = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        ds = (roll_sharpe(r) - roll_sharpe(spy_live)).dropna()
        table.append([name] + [f"{m.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}",
                        f"{float((ds > 0).mean()):.0%}"])
    print(tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin", "SRwin"],
                   tablefmt="github"))
    print("  win rates vs SPY, rolling 3y windows")

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 13))
    ax = axes[0]
    z = lambda s: (s - s[both].mean()) / s[both].std()
    ax.plot(liq_raw.index, z(liq_raw), lw=1.0, color="tab:blue",
            label="raw Δ6m net liquidity (z)")
    ax.plot(shadow.index, z(shadow), lw=1.0, color="tab:orange",
            label="price shadow (z)")
    ax2 = ax.twinx()
    ax2.plot(roll_corr.index, roll_corr, lw=0.9, color="grey", ls="--",
             label="rolling 3y corr (rhs)")
    ax2.set_ylim(-1, 1)
    ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("The liquidity signal and its causal price-only shadow")

    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.4,
                color="black" if name.startswith("SPY") else
                ("grey" if name.startswith("QQQ") else None), label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Champion with the liq layer real / shadow-replaced / residual-only / dropped")

    ax = axes[2]
    for name in ["QQQ buy & hold", "2×5-layer · real liq (champion)",
                 "2×5-layer · liq → price shadow"]:
        eq = (1 + strat[name].fillna(0)).cumprod()
        dd = eq / eq.cummax() - 1
        ax.plot(dd.index, dd.values, lw=1.1,
                color="grey" if "buy" in name else None, label=name)
    ax.set_ylabel("drawdown"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Drawdowns: does replacing FRED data with its price replica cost anything?")
    fig.tight_layout()
    out = RESULTS_DIR / "liquidity_shadow.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
