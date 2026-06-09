#!/usr/bin/env python3
"""
What conditions the tug-of-war regime?  Six hypotheses, double-sorted.

The tug signal (soft-signed 21d overnight-minus-intraday return spread) is
real but episodic: on SPY it earned +36pp/yr of next-day tercile spread in
1994-2003, was DEAD 2004-2013 (-3pp), and revived 2014-2026 (+23pp).  A
signal with decade-scale on/off regimes can't be promoted without knowing the
switch.  Six causal candidate conditioners, tested by double-sort on SPY
(1994→2026, ~8,000 days): within each conditioner tercile (causal 3y rank),
the annualized next-day return spread between high-tug and low-tug days.

  H1 vol regime      VIX level                 (tension matters when risk is up?)
  H2 rate era        3m T-bill yield           (ZIRP changed overnight carry?)
  H3 overnight share rolling 1y var(on)/total  (tug works when risk lives
                                                overnight - macro eras?)
  H4 persistence     rolling 1y lag-1 autocorr of the daily (on-id) spread
                                               (tension must be sticky?)
  H5 trend state     SPY vs 200d MA            (bull/bear asymmetry?)
  H6 self            trailing 3y corr(tug, ret) (is efficacy itself persistent
                                                enough to time?)  + direct
                                                check: corr(trailing efficacy,
                                                forward 1y efficacy)

Multiplicity warning: six candidates on one asset — whatever wins must also
hold on QQQ (checked in the confirmation block) and should be treated as a
lead, not a law.

    python experiments/run_tug_conditioner.py
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
from posterioralpha.research import soft_sign

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
COND_CSV = _bootstrap.ROOT / "datasets" / "tug_conditioners.csv.gz"

RANK_WIN = 756


def causal_rank(s: pd.Series, win: int = RANK_WIN) -> pd.Series:
    return s.rolling(win, min_periods=252).apply(
        lambda x: (x < x[-1]).mean(), raw=True).shift(1)


def build(t: str, panel: pd.DataFrame):
    o, c = panel[f"{t}_Open"].dropna(), panel[f"{t}_Close"].dropna()
    r_on, r_id = o / c.shift(1) - 1.0, c / o - 1.0
    ret = c.pct_change()
    raw = ((1 + r_on).rolling(21).apply(np.prod, raw=True)
           - (1 + r_id).rolling(21).apply(np.prod, raw=True))
    sig = soft_sign(raw).shift(1)
    return r_on, r_id, ret, sig


def tug_spread(ret, sig_rank, mask=None) -> float:
    """Annualized next-day return spread, high-tug minus low-tug tercile."""
    m = sig_rank.notna() if mask is None else (mask & sig_rank.notna())
    hi = ret[m & (sig_rank >= 0.67)].mean()
    lo = ret[m & (sig_rank <= 0.33)].mean()
    return float(hi - lo) * 252


def main() -> None:
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    cond_px = pd.read_csv(COND_CSV, parse_dates=["Date"], index_col="Date").sort_index()

    r_on, r_id, ret, sig = build("SPY", panel)
    idx = ret.index
    vix = cond_px["VIX"].reindex(idx).ffill()
    irx = cond_px["IRX"].reindex(idx).ffill()
    sig_rank = causal_rank(sig)

    spread_d = (r_on - r_id)
    conds = {
        "H1 VIX level":        causal_rank(vix),
        "H2 3m T-bill yield":  causal_rank(irx),
        "H3 overnight share":  causal_rank((r_on.rolling(252).var()
                                            / (r_on.rolling(252).var()
                                               + r_id.rolling(252).var()))),
        "H4 spread persistence": causal_rank(
            spread_d.rolling(252).apply(lambda x: pd.Series(x).autocorr(1), raw=False)),
        "H5 above 200d MA":    (panel["SPY_Close"].reindex(idx)
                                > panel["SPY_Close"].reindex(idx).rolling(200).mean())
                               .astype(float).shift(1),
        "H6 trailing efficacy": causal_rank(sig.rolling(756, min_periods=252).corr(ret)),
    }

    base = tug_spread(ret, sig_rank)
    print("\n" + "=" * 92)
    print(f"  WHAT CONDITIONS THE TUG REGIME?  ·  SPY  ·  "
          f"{idx[0].date()} → {idx[-1].date()}")
    print(f"  unconditional tug tercile spread: {base:+.1%}/yr")
    print("=" * 92)
    rows = []
    for name, c in conds.items():
        if name == "H5 above 200d MA":
            buckets = [("below", c == 0), ("—", pd.Series(False, idx)),
                       ("above", c == 1)]
        else:
            buckets = [("low", c <= 0.33), ("mid", (c > 0.33) & (c < 0.67)),
                       ("high", c >= 0.67)]
        vals = [tug_spread(ret, sig_rank, m) for _, m in buckets]
        rows.append([name] + [f"{v:+.1%}" if not np.isnan(v) else "—" for v in vals]
                    + [f"{vals[-1] - vals[0]:+.1%}" if not (np.isnan(vals[0])
                       or np.isnan(vals[-1])) else "—"])
    print(tabulate(rows, headers=["conditioner (causal)", "low/below", "mid",
                                  "high/above", "high−low"], tablefmt="github"))
    print("  cell = annualized high-tug minus low-tug next-day return inside the bucket")

    # H6 direct persistence check: does trailing efficacy predict forward efficacy?
    eff_trail = sig.rolling(756, min_periods=252).corr(ret).shift(1)
    eff_fwd = sig.rolling(252).corr(ret).shift(-252)
    m = eff_trail.notna() & eff_fwd.notna()
    c_eff = float(np.corrcoef(eff_trail[m], eff_fwd[m])[0, 1])
    print(f"\n  H6 persistence: corr(trailing 3y efficacy, forward 1y efficacy) "
          f"= {c_eff:+.2f}  (n={int(m.sum())}, overlapping)")

    # ── Confirmation block: do the SPY winners hold on QQQ? ─────────────────
    print("\n" + "=" * 92)
    print("  CONFIRMATION ON QQQ (1999→) — same constructions, no re-selection")
    print("=" * 92)
    q_on, q_id, q_ret, q_sig = build("QQQ", panel)
    qidx = q_ret.index
    q_rank = causal_rank(q_sig)
    q_spread = (q_on - q_id)
    q_conds = {
        "H3 overnight share": causal_rank((q_on.rolling(252).var()
                                           / (q_on.rolling(252).var()
                                              + q_id.rolling(252).var()))),
        "H4 spread persistence": causal_rank(
            q_spread.rolling(252).apply(lambda x: pd.Series(x).autocorr(1), raw=False)),
        "H1 VIX level": causal_rank(cond_px["VIX"].reindex(qidx).ffill()),
        "H2 3m T-bill yield": causal_rank(cond_px["IRX"].reindex(qidx).ffill()),
        "H6 trailing efficacy": causal_rank(
            q_sig.rolling(756, min_periods=252).corr(q_ret)),
    }
    rows = []
    qbase = tug_spread(q_ret, q_rank)
    for name, c in q_conds.items():
        lo_ = tug_spread(q_ret, q_rank, c <= 0.33)
        hi_ = tug_spread(q_ret, q_rank, c >= 0.67)
        rows.append([name, f"{lo_:+.1%}", f"{hi_:+.1%}", f"{hi_ - lo_:+.1%}"])
    print(f"  unconditional QQQ tug spread: {qbase:+.1%}/yr")
    print(tabulate(rows, headers=["conditioner", "low tercile", "high tercile",
                                  "high−low"], tablefmt="github"))

    # ── Era attribution: which conditioner explains 2004-2013? ──────────────
    print("\n  Era means of each conditioner rank (does it 'see' the dead decade?):")
    rows = []
    for name, c in conds.items():
        cells = [name]
        for a, b in [("1994", "2003"), ("2004", "2013"), ("2014", "2026")]:
            cells.append(f"{float(c.loc[a:b].mean()):.2f}")
        rows.append(cells)
    print(tabulate(rows, headers=["conditioner", "1994-2003", "2004-2013",
                                  "2014-2026"], tablefmt="github"))

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    eff = sig.rolling(504).corr(ret)
    ax.plot(eff.index, eff.values, lw=1.0, color="tab:blue",
            label="rolling 2y corr(tug stance, next-day SPY ret)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvspan(pd.Timestamp("2004-01-01"), pd.Timestamp("2013-12-31"),
               color="red", alpha=0.07, label="dead decade")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Tug efficacy through time — the regime to be explained")
    ax = axes[1]
    on_share = (r_on.rolling(252).var()
                / (r_on.rolling(252).var() + r_id.rolling(252).var()))
    ax.plot(on_share.index, on_share.values, lw=1.0, color="tab:purple",
            label="overnight share of SPY variance (1y)")
    ax.plot(irx.index, (irx / irx.max()).values, lw=0.8, color="tab:green",
            alpha=0.7, label="3m T-bill (scaled)")
    ax.axvspan(pd.Timestamp("2004-01-01"), pd.Timestamp("2013-12-31"),
               color="red", alpha=0.07)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Candidate conditioners vs the dead decade")
    fig.tight_layout()
    out = RESULTS_DIR / "tug_conditioner.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
