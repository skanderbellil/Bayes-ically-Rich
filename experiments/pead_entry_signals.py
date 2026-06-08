#!/usr/bin/env python3
"""Entry-signal combinations for tradeable PEAD market-neutral (selection side).

Exit engineering is exhausted (a plain stop wins); the remaining upside is in WHICH
names to be long/short. Builds an enriched panel from the cached prices — SUE plus each
name's pre-announcement 6-month momentum and monthly volatility — and tests whether
combining signals sharpens the market-neutral book vs ranking on SUE alone.

Ideas tested:
  - SUE only (baseline)                     - momentum only (sanity)
  - SUE + momentum composite z-score (sweep blend weight)
  - SUE with momentum *confirmation* (only trade names whose momentum agrees with SUE)
  - SUE within the low-volatility half (cleaner drift)

All use the realistic t+1 entry, 6% stop, hold-to-next-earnings (63d), Mega+Large,
net of Alpaca costs. Requires data/raw/pead/ (run experiments/run_pead.py first).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.pead import paths, signals
from experiments.pead_dynamic_exit import COST, _stats

HOLD, STOP = 63, 6.0


def build_enriched():
    panel = pd.read_csv(paths.RESULTS_DIR / "signal_panel.csv")
    bucket = panel.drop_duplicates("ticker").set_index("ticker")["market_cap"].to_dict()
    names = [t for t, b in bucket.items() if b in ("Mega Cap", "Large Cap")]
    rows, meta = [], []
    for t in names:
        e = signals.load_earnings_signal(paths.RAW_DIR / f"{t}__earnings.csv")
        px = signals.load_prices(paths.RAW_DIR / f"{t}__prices.csv")
        if e is None or px is None:
            continue
        px = px.sort_index()
        close = px["Close"]; dret = np.log(close).diff()
        # Canonical standardized SUE: surprise / std of the firm's OWN past surprises (causal)
        surp = e["Surprise(%)"]
        firm_std = surp.expanding(min_periods=4).std().shift(1)
        std_sue_ser = surp / firm_std
        for ann, row in e.iterrows():
            sue = row.get("Surprise(%)")
            if pd.isna(sue):
                continue
            fut = px[px.index > ann]
            past = close[close.index <= ann]
            if len(fut) < 3 or len(past) < 130:
                continue
            # 6-month momentum, skipping the last 5 days into the print
            mom6 = np.log(past.iloc[-6] / past.iloc[-126]) * 100
            sig = dret[dret.index <= ann].dropna().iloc[-60:].std() * np.sqrt(20) * 100
            ssue = std_sue_ser.get(ann, np.nan)
            if isinstance(ssue, pd.Series):
                ssue = ssue.iloc[0]
            ssue = float(ssue) if pd.notna(ssue) else np.nan
            entry = fut["Close"].iloc[1]; cl = fut["Close"].values
            # announcement-day reaction (Brandt et al.): pre-print close -> t+1 close.
            # Known at the t+1 entry; the strategy trades the drift AFTER it.
            react = np.log(entry / past.iloc[-1]) * 100
            path = np.full(HOLD + 1, np.nan)
            for d in range(1, min(HOLD, len(fut) - 1) + 1):
                path[d] = np.log(cl[d] / entry) * 100
            rows.append(path)
            meta.append((pd.Timestamp(ann).tz_localize(None), t, sue, ssue, mom6, sig, react))
    P = np.array(rows)
    M = pd.DataFrame(meta, columns=["ann", "ticker", "sue", "std_sue", "mom6", "sigma_m", "react"])
    M["season"] = M["ann"].dt.to_period("Q"); M["year"] = M["ann"].dt.year
    return P, M.reset_index(drop=True)


def _exit(c):
    s = np.where(c <= -STOP)[0]
    return -STOP if len(s) else c[-1]


def leg_ret(idx_members, P, sign):
    out = []
    for i in idx_members:
        c = sign * P[i, 1:HOLD + 1]; c = c[~np.isnan(c)]
        if len(c):
            out.append(_exit(c))
    return np.nanmean(out) if out else np.nan


def _z(x):
    x = np.asarray(x, float); s = x.std()
    return (x - x.mean()) / s if s > 0 else x * 0.0


def run(P, M, years, mode="sue", w=0.5):
    """mode: sue | mom | composite | confirm | lowvol"""
    sp = []
    for _, g in M[M["year"].isin(years)].groupby("season"):
        if len(g) < 10:
            continue
        gg = g.copy()
        gg["zs"], gg["zm"], gg["zr"] = _z(gg["sue"]), _z(gg["mom6"]), _z(gg["react"])
        if mode == "react":
            gg["score"] = gg["zr"]
        elif mode == "react_sue":
            gg["score"] = w * gg["zr"] + (1 - w) * gg["zs"]
        elif mode == "stdsue":
            gg = gg[pd.to_numeric(gg["std_sue"], errors="coerce").notna()]
            if len(gg) < 10:
                continue
            gg["score"] = _z(gg["std_sue"])
        elif mode == "sue":
            gg["score"] = gg["zs"]
        elif mode == "mom":
            gg["score"] = gg["zm"]
        elif mode == "composite":
            gg["score"] = w * gg["zs"] + (1 - w) * gg["zm"]
        elif mode == "confirm":
            # only names whose momentum agrees in sign with the surprise
            gg = gg[np.sign(gg["sue"]) == np.sign(gg["mom6"])]
            if len(gg) < 10:
                continue
            gg["score"] = gg["zs"]
        elif mode == "lowvol":
            gg = gg[gg["sigma_m"] <= gg["sigma_m"].median()]
            if len(gg) < 8:
                continue
            gg["score"] = gg["zs"]
        hi, lo = gg["score"].quantile(0.8), gg["score"].quantile(0.2)
        L = leg_ret(gg[gg["score"] >= hi].index, P, +1.0)
        S = leg_ret(gg[gg["score"] <= lo].index, P, -1.0)
        if np.isfinite(L) and np.isfinite(S):
            sp.append((L + S) / 100 - COST)
    return _stats(np.array(sp))


def main():
    P, M = build_enriched()
    oos, allyrs = list(range(2014, 2027)), list(range(2000, 2027))
    print(f"Enriched panel: {len(M)} positions, Mega+Large, with SUE + 6m momentum + vol\n")
    print(f"  {'Entry signal':<34}{'Full Ann':>9}{'Shrp':>6} | {'OOS Ann':>8}{'Shrp':>6}{'MaxDD':>7}")

    def show(lbl, **kw):
        a, b = run(P, M, allyrs, **kw), run(P, M, oos, **kw)
        print(f"  {lbl:<34}{a[0]:>+9.1%}{a[1]:>6.2f} | {b[0]:>+8.1%}{b[1]:>6.2f}{b[2]:>+7.1%}")

    show("SUE only (raw Surprise%, baseline)", mode="sue")
    show("STANDARDIZED SUE (vs firm history)", mode="stdsue")
    show("ANNOUNCEMENT REACTION (Brandt et al.)", mode="react")
    show("Reaction 0.5 + SUE 0.5", mode="react_sue", w=0.5)
    show("Reaction 0.7 + SUE 0.3", mode="react_sue", w=0.7)
    show("Momentum only", mode="mom")
    show("SUE confirmed by momentum sign", mode="confirm")
    show("SUE within low-vol half", mode="lowvol")
    for w in (0.75, 0.6, 0.5, 0.4):
        show(f"Composite {w:.2f}·SUE + {1-w:.2f}·mom", mode="composite", w=w)


if __name__ == "__main__":
    main()
