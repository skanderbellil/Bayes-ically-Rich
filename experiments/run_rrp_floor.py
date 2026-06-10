#!/usr/bin/env python3
"""
The RRP floor — does liquidity plumbing change when the buffer is gone?

From 2021 to 2024 the Fed's overnight reverse-repo facility held up to $2.5tn
of money-fund cash.  Mechanically, it acted as a SHOCK ABSORBER: when the TGA
pulled cash in (tax dates, post-ceiling rebuilds), money funds could let RRP
balances run down and reserves were shielded.  By 2025 the facility is ~zero
(our own dataset prints RRP < $2bn) — the buffer is gone, so the same
scheduled drains must now pass through to reserves 1:1.  Commentary on this
is everywhere; we found no quantitative study (June 2026), so we test it
ourselves on data we already hold.

This connects to our own earlier finding: run_liquidity_calendar showed tax
and quarter-end windows had NO equity drag — and credited RRP/Fed offsets.
The mechanism makes a falsifiable prediction: the offset dies with the
buffer.

  Regimes (a-priori threshold: buffer ≥ one typical tax drain, which we
  measured at +$87bn → RRP > $100bn, causal):
    pre-RRP    : before the facility filled (2003 → first cross above $100bn)
    buffered   : RRP > $100bn
    post-buffer: RRP < $100bn after the buffered era

  Tests
  -----
  1. ABSORPTION — per tax event, absorption ratio = −ΔRRP/ΔTGA over [0,+10].
     Buffered: should be material.  Post-buffer: mechanically ~0.
  2. PASS-THROUGH — Δnet-liquidity per $ of ΔTGA in tax windows, by regime.
  3. EQUITY DRAG — QQQ return inside tax windows by regime.  Prediction:
     drag appears only when unbuffered (incl. the 2018-19 reserve-scarcity
     episode that culminated in the Sep-2019 repo spike).
  4. SIGNAL EFFICACY — corr(Δ6m net-liq, fwd 63d QQQ) by regime: shocks that
     pass through should make the liquidity layer MORE informative.

Honest limits: the post-buffer era is short (a handful of tax events) — this
is structural-break forensics plus a live-trading rule, not a backtest.

    python experiments/run_rrp_floor.py
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S", stream=sys.stdout)
logger = logging.getLogger(__name__)
RESULTS_DIR = _bootstrap.ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
NL_CSV = _bootstrap.ROOT / "datasets" / "net_liquidity.csv"
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"

BUF = 100.0          # $bn — one typical tax drain (measured +$87bn mean)
TAX_WIN = (0, 10)    # trading days
H_LIQ, PUB_LAG = 126, 5
FWD = 63


def tax_dates(index):
    years = range(index[0].year, index[-1].year + 1)
    out = []
    for y in years:
        for m in (1, 4, 6, 9):
            i = index.searchsorted(pd.Timestamp(y, m, 15))
            if 0 < i < len(index) - TAX_WIN[1]:
                out.append(index[i])
    return out


def main() -> None:
    raw = pd.read_csv(NL_CSV, parse_dates=["date"], index_col="date").sort_index()
    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = raw.dropna().index.intersection(panel[["QQQ_Close", "SPY_Close"]]
                                          .dropna().index)
    raw = raw.reindex(idx).ffill()
    qqq = panel["QQQ_Close"].reindex(idx)
    spy = panel["SPY_Close"].reindex(idx)
    r_qqq = qqq.pct_change()

    rrp, tga, nl = raw["RRPONTSYD"], raw["WTREGEN"], raw["net_liquidity"]

    # ── Per-day regime: prevailing buffer = causal 63d median RRP ───────────
    # (a single year-end spike must not flip the regime; the rolling median
    # asks whether a buffer was USUALLY there)
    buf_state = (rrp.rolling(63).median() > BUF).shift(1).fillna(False)
    # the buffer ERA = the longest contiguous buffered run (2021→); brief
    # 2014-17 test-phase patches stay "buffered" but don't start the clock
    blocks = (buf_state != buf_state.shift()).cumsum()
    runs = [(g.index[0], g.index[-1]) for _, g in buf_state.groupby(blocks)
            if g.iloc[0]]
    big_start, big_end = max(runs, key=lambda ab: (ab[1] - ab[0]))
    day_regime = pd.Series("pre-buffer (unbuffered)", index=idx)
    day_regime[buf_state] = "buffered (RRP>$100bn)"
    day_regime[(~buf_state) & (idx > big_end)] = "post-buffer (RRP<$100bn)"
    regime_names = ["pre-buffer (unbuffered)", "buffered (RRP>$100bn)",
                    "post-buffer (RRP<$100bn)"]
    print("\n" + "=" * 96)
    print("  THE RRP FLOOR  ·  per-day regime = causal 63d median RRP vs "
          f"${BUF:.0f}bn (≈ one tax drain)")
    print("=" * 96)
    for name in regime_names:
        m = day_regime == name
        if m.any():
            print(f"  {name:28s} {m.idxmax().date()} → "
                  f"{m[m].index[-1].date()}   {int(m.sum())} days, "
                  f"mean RRP ${float(rrp[m].mean()):,.0f}bn")
    # local RRP drift for detrending event-window ΔRRP
    rrp_drift = rrp.diff().rolling(63).median().shift(1).fillna(0.0)

    # ── Tests 1-3: per-tax-event mechanics and equity drag ──────────────────
    events = tax_dates(idx)
    rows = []
    per_regime = {k: {"absorb": [], "pass": [], "ret": [], "dtga": []}
                  for k in regime_names}
    for e in events:
        i = idx.get_loc(e)
        lo, hi = i + TAX_WIN[0], i + TAX_WIN[1]
        d_tga = float(tga.iloc[hi] - tga.iloc[lo])
        if d_tga < 20.0:        # not a real drain event (refund season etc.)
            continue
        ndays = hi - lo
        # detrend: remove the prevailing RRP drift (the 2021-22 fill and
        # 2023-24 drain trends would otherwise dominate the event window)
        d_rrp = float(rrp.iloc[hi] - rrp.iloc[lo]) - float(rrp_drift.loc[e]) * ndays
        d_nl = float(nl.iloc[hi] - nl.iloc[lo])
        ret = float(qqq.iloc[hi] / qqq.iloc[lo] - 1.0)
        name = day_regime.loc[e]
        per_regime[name]["absorb"].append(-d_rrp / d_tga)
        per_regime[name]["pass"].append(d_nl / d_tga)
        per_regime[name]["ret"].append(ret)
        per_regime[name]["dtga"].append(d_tga)

    print("\n  Per-tax-event mechanics (events with ΔTGA > $20bn over [0,+10]):")
    rows = []
    for name, d in per_regime.items():
        if not d["absorb"]:
            continue
        rows.append([name, len(d["absorb"]),
                     f"{np.mean(d['dtga']):+.0f}",
                     f"{np.median(d['absorb']):+.2f}",
                     f"{np.median(d['pass']):+.2f}",
                     f"{np.mean(d['ret']):+.1%}",
                     f"{np.mean([r < 0 for r in d['ret']]):.0%}"])
    print(tabulate(rows, headers=["regime", "n", "ΔTGA $bn",
                                  "RRP absorption\n(−ΔRRP/ΔTGA, med)",
                                  "net-liq pass-through\n(ΔNL/ΔTGA, med)",
                                  "QQQ in window", "% negative"],
                   tablefmt="github"))
    print("  absorption 1.0 = buffer ate the whole drain; pass-through −1.0 = "
          "drain hit liquidity in full")

    # ── Test 3b: ALL days equity drag, tax windows by regime ────────────────
    in_tax = pd.Series(False, index=idx)
    for e in events:
        i = idx.get_loc(e)
        in_tax.iloc[i:i + TAX_WIN[1] + 1] = True
    print("\n  QQQ annualized mean daily return, tax windows vs rest, by regime:")
    rows = []
    for name in regime_names:
        m = day_regime == name
        seg, t = r_qqq[m], in_tax[m]
        if not m.any():
            continue
        rows.append([name, f"{float(seg[t].mean()) * 252:+.1%}",
                     f"{float(seg[~t].mean()) * 252:+.1%}",
                     f"{float(seg[t].mean() - seg[~t].mean()) * 252:+.1%}",
                     int(t.sum())])
    print(tabulate(rows, headers=["regime", "tax windows", "other days",
                                  "drag", "tax days"], tablefmt="github"))

    # ── Test 4: liquidity-signal efficacy by regime ──────────────────────────
    d_nl_sig = nl.diff(H_LIQ).shift(PUB_LAG)
    fwd = qqq.pct_change(FWD).shift(-FWD)
    print("\n  corr(Δ6m net-liquidity, next-63d QQQ return) by regime:")
    rows = []
    for name in regime_names:
        rm = day_regime == name
        x, y = d_nl_sig[rm], fwd[rm]
        m = x.notna() & y.notna()
        c = float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 100 else np.nan
        rows.append([name, f"{c:+.2f}" if not np.isnan(c) else "n<100",
                     int(m.sum())])
    print(tabulate(rows, headers=["regime", "corr", "days"], tablefmt="github"))
    print("  (overlapping windows; post-buffer sample short — descriptive)")

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    ax = axes[0]
    ax.plot(rrp.index, rrp.values, lw=1.0, color="tab:blue", label="RRP ($bn)")
    ax.plot(tga.index, tga.values, lw=1.0, color="tab:red", alpha=0.7,
            label="TGA ($bn)")
    ax.axhline(BUF, color="black", lw=0.8, ls="--", label=f"buffer = ${BUF:.0f}bn")
    ax.fill_between(idx, 0, rrp.max(),
                    where=(day_regime == "buffered (RRP>$100bn)").values,
                    color="green", alpha=0.07)
    ax.fill_between(idx, 0, rrp.max(),
                    where=(day_regime == "post-buffer (RRP<$100bn)").values,
                    color="red", alpha=0.10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("The buffer's life and death (green = buffered, red = post-buffer)")

    ax = axes[1]
    for name, marker, color in [("buffered (RRP>$100bn)", "o", "green"),
                                ("post-buffer (RRP<$100bn)", "s", "red"),
                                ("pre-buffer (unbuffered)", "^", "grey")]:
        d = per_regime[name]
        if d["absorb"]:
            ax.scatter(d["dtga"], d["pass"], marker=marker, color=color,
                       alpha=0.7, label=f"{name} (n={len(d['absorb'])})")
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(-1, color="red", lw=0.6, ls=":")
    ax.set_xlabel("tax-window ΔTGA ($bn)")
    ax.set_ylabel("Δnet-liq / ΔTGA")
    ax.set_title("Pass-through per tax event: 0 = absorbed, −1 = full hit")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    names = [n for n in per_regime if per_regime[n]["ret"]]
    rets = [np.mean(per_regime[n]["ret"]) * 100 for n in names]
    ax.bar(range(len(names)), rets,
           color=["grey" if "pre" in n else ("green" if "buffered" in n else "red")
                  for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(" (")[0] for n in names], fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean QQQ return in tax window (%)")
    ax.set_title("Does the tax drain bite equities once the buffer is gone?")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = RESULTS_DIR / "rrp_floor.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
