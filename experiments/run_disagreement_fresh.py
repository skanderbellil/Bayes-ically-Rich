#!/usr/bin/env python3
"""
"Indecision is the tell" — fresh-sample validation of the inverted gauge.

run_model_disagreement found its hypothesis inverted: HIGH cross-model
dispersion marks strong trends (good times) and 200d-MA flips follow model
AGREEMENT — near transitions every model drifts to neutral, so dispersion
collapses.  The inverted throttle built on that reading (2022 −5%, DD −21%)
was flagged post-hoc and unvalidated: the sign was chosen after seeing the
tables, on a sample that effectively starts 2013.

Four of the five stances are price-only and need no FRED/VIX data:

  trend    soft-signed 126d return          meanrev  soft-signed −21d return
  bocpd    causal 3y rank of BOCPD ERL      hmm      walk-forward HMM3 bull-stance

so the gauge D = std(4 stances) can be rebuilt on SPY 1993→ and QQQ 1999→ —
decades the selection never saw.  Validation design:

  BRIDGE   Recompute the modern window (2013→) with the SAME 4 price-only
           stances: does dropping the liquidity stance change the finding?
  FRESH    SPY 1994-2003 and 2004-2012, QQQ 2000-2012: (a) fwd 6m return by
           causal D tercile (claim: hi > lo), (b) P(200d-MA flip within 21d)
           by tercile (claim: lo > hi), (c) the throttle on a 2×trend dial —
           inverted (×rank D) vs original-sign (×(1−rank D)) vs untouched.

All mappings a-priori; the only fitted object is the walk-forward HMM.

    python experiments/run_disagreement_fresh.py
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
from posterioralpha.research import HMM3, precompute_bocpd, soft_sign
from posterioralpha.validation import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
RANK_WIN = 756
HMM_WIN, HMM_REFIT = 756, 21
FWD, CRASH, FLIP_H = 126, -0.15, 21


def causal_rank(s: pd.Series, win: int = RANK_WIN) -> pd.Series:
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def hmm_stance(ret: pd.Series) -> pd.Series:
    vals = ret.values.reshape(-1, 1)
    out = pd.Series(np.nan, index=ret.index)
    for t in range(HMM_WIN, len(ret), HMM_REFIT):
        window = vals[t - HMM_WIN:t + 1]
        model = HMM3().fit(window)
        p_bull, p_side, _ = model.regime_probs(window)
        out.iloc[t] = p_bull + 0.5 * p_side
    return out.ffill()


def apply_exposure(under_ret: pd.Series, expo: pd.Series) -> pd.Series:
    expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
    gross = expo * under_ret
    financing = np.maximum(expo - 1.0, 0.0) * (RF + BORROW_SPREAD) / 252
    cash = np.maximum(1.0 - expo, 0.0) * DAILY_RF
    costs = TC * expo.diff().abs().fillna(0.0)
    return gross - financing + cash - costs


def build(close: pd.Series) -> dict:
    ret = close.pct_change()
    logger.info("BOCPD ERL …")
    _, erl = precompute_bocpd(ret.fillna(0.0).values)
    logger.info("Walk-forward HMM …")
    stances = pd.DataFrame({
        "trend":   soft_sign(close.pct_change(126)).shift(1),
        "meanrev": soft_sign(-close.pct_change(21)).shift(1),
        "bocpd":   causal_rank(pd.Series(erl, index=close.index)),
        "hmm":     hmm_stance(ret).shift(1),
    })
    full = stances.notna().all(axis=1)
    disagree = stances.std(axis=1).where(full)
    d_rank = causal_rank(disagree)
    state = (close > close.rolling(200).mean()).astype(int)
    flip = state.diff().abs() == 1
    flip_ahead = (flip.shift(-1).rolling(FLIP_H).max()
                  .shift(-(FLIP_H - 1)).fillna(0).astype(bool))
    return {"close": close, "ret": ret, "stances": stances,
            "d_rank": d_rank, "flip_ahead": flip_ahead}


def window_tables(b: dict, a: str, z: str) -> tuple:
    close, d_rank, flip_ahead = b["close"], b["d_rank"], b["flip_ahead"]
    fwd_ret = close.pct_change(FWD).shift(-FWD)
    in_w = pd.Series(False, index=close.index)
    in_w.loc[a:z] = True
    lo = (d_rank <= 0.33) & in_w
    mid = (d_rank > 0.33) & (d_rank < 0.67) & in_w
    hi = (d_rank >= 0.67) & in_w
    ret_rows, flip_rows = [], []
    for lab, m in [("low D", lo), ("mid", mid), ("high D", hi)]:
        mm = m & fwd_ret.notna()
        ret_rows.append([lab, f"{float(fwd_ret[mm].mean()):+.1%}",
                         f"{float(fwd_ret[mm].quantile(0.05)):+.1%}",
                         f"{float((fwd_ret[mm] < CRASH).mean()):.0%}", int(mm.sum())])
        flip_rows.append(f"{float(flip_ahead[m].mean()):.1%}")
    base_flip = float(flip_ahead[in_w & d_rank.notna()].mean())
    return ret_rows, flip_rows, base_flip


def throttle_table(b: dict, a: str, z: str) -> list:
    ret, stances, d_rank = b["ret"], b["stances"], b["d_rank"]
    trend = stances["trend"]
    variants = {
        "B&H": pd.Series(1.0, index=ret.index),
        "2×trend dial": 2.0 * trend,
        "2×trend × rank D (inverted)": 2.0 * trend * d_rank,
        "2×trend × (1−rank D) (orig sign)": 2.0 * trend * (1.0 - d_rank),
    }
    rows = []
    for name, e in variants.items():
        r = apply_exposure(ret, e).loc[a:z]
        m = compute_metrics(r.dropna(), rf=RF)
        rows.append([name, f"{m['CAGR']:.1%}", f"{m['Sharpe']:.2f}",
                     f"{m['Max DD']:.0%}", f"{m['Calmar']:.2f}"])
    return rows


def main() -> None:
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    spy = panel["SPY_Close"].dropna()
    qqq = panel["QQQ_Close"].dropna()

    print("\n" + "=" * 96)
    print("  'INDECISION IS THE TELL' — FRESH-SAMPLE VALIDATION  "
          "(4 price-only stances, D = cross-model std)")
    print("=" * 96)

    runs = [("SPY", build(spy),
             [("FRESH 1994-2003", "1994", "2003"),
              ("FRESH 2004-2012", "2004", "2012"),
              ("BRIDGE 2013-2026", "2013", "2026")]),
            ("QQQ", build(qqq),
             [("FRESH 2000-2012", "2000", "2012"),
              ("BRIDGE 2013-2026", "2013", "2026")])]

    for asset, b, windows in runs:
        for wlab, a, z in windows:
            ret_rows, flip_rows, base_flip = window_tables(b, a, z)
            print(f"\n  ── {asset} · {wlab} "
                  + "─" * max(0, 70 - len(asset) - len(wlab)))
            print("  Forward 6m return by causal D tercile "
                  "(claim: high D = good times):")
            print(tabulate(ret_rows, headers=["tercile", "fwd 6m", "5th pct",
                                              f"P(<{CRASH:.0%})", "days"],
                           tablefmt="github"))
            print(f"  P(200d-MA flip within {FLIP_H}d) lo/mid/hi D: "
                  f"{' / '.join(flip_rows)}   (uncond {base_flip:.1%})  "
                  "(claim: low D = transition risk)")
            print("  Throttle on the 2×trend dial:")
            print(tabulate(throttle_table(b, a, z),
                           headers=["variant", "CAGR", "Sharpe", "MaxDD", "Calmar"],
                           tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    for ax, (asset, b) in zip(axes, [("SPY", runs[0][1]), ("QQQ", runs[1][1])]):
        d = b["stances"].std(axis=1).where(b["stances"].notna().all(axis=1))
        ax.plot(d.index, d.rolling(63).mean(), lw=0.9, color="tab:orange",
                label="disagreement D (63d MA)")
        close = b["close"]
        state = (close > close.rolling(200).mean()).astype(int)
        flips = close.index[(state.diff().abs() == 1) & b["d_rank"].notna()]
        for i, f in enumerate(flips):
            ax.axvline(f, color="tab:red", lw=0.5, alpha=0.35,
                       label="200d-MA flip" if i == 0 else None)
        ax.set_title(f"{asset}: model dispersion vs trend flips — "
                     "does indecision precede transitions out of sample?")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "disagreement_fresh.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
