#!/usr/bin/env python3
"""
Levered barbell: QLD + slack asset, council-dialed vs static — who earns what?

Epilogue 3 closed the QLD question against a QLD-only static null. But a
book with idle slack has a better null available: park the slack in a
diversifier. This script prices the three layers separately:

  layer 1 — the BARBELL: constant mix of QLD and a slack asset
            (GLD / UUP / TLT), daily-rebalanced, vs QQQ & QLD buy & hold.
            Pure diversification: no dial, no fitting, two tickers.
  layer 2 — the DIAL: the blinded statistical council's monthly exposure
            (run_council_backtest.py — run it first) drives the QLD weight;
            slack = 1 − e. Tested against the *static mix* with (a) the
            dial's own average exposure, (b) vol-matched and (c) DD-matched
            static weights — the dial only earns its complexity where it
            beats the matched static barbell, not merely static QLD.
  layer 3 — bootstrap: joint stationary block bootstrap P-values for each
            comparison.

Caveats stated up front: GLD's 2014–2026 decade (and especially 2024–26)
was exceptional — the barbell's edge is partly a gold-decade artifact;
UUP was the slack asset pre-validated by the multiasset-council thread,
GLD is a post-hoc pick among four candidates (mild selection). Daily-reset
2× decay is priced in throughout (real QLD data).
"""
import _bootstrap  # noqa: F401  (adds repo root to sys.path, loads .env)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from posterioralpha.data import load_etf_universe_prices

OUT_DIR = _bootstrap.ROOT / "results" / "pod_policies"
COUNCIL_MINUTES = _bootstrap.ROOT / "results" / "council" / "minutes_QQQ.csv"
TC = 2e-4
SLACKS = ["GLD", "UUP", "TLT"]


def stats(r: pd.Series) -> dict:
    cum = (1 + r).cumprod()
    dd = float(((cum / cum.cummax()) - 1).min())
    cagr = float(cum.iloc[-1] ** (252 / len(r)) - 1)
    return {"cagr": round(cagr, 3),
            "vol": round(float(r.std() * np.sqrt(252)), 3),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 2),
            "max_dd": round(dd, 3), "calmar": round(cagr / abs(dd), 2),
            "wealth": round(float(cum.iloc[-1]), 1)}


def council_book(qld: pd.Series, slack: pd.Series,
                 e_per: pd.Series) -> pd.Series:
    r = qld.pct_change()
    e = e_per.reindex(qld.index).ffill().fillna(0.0)
    held = e.shift(1).fillna(0.0)
    net = held * r.fillna(0.0) - TC * held.diff().abs().fillna(0.0)
    rs = slack.pct_change().reindex(qld.index)
    sl = 1.0 - held
    net = net + sl * rs.fillna(0.0) - TC * sl.diff().abs().fillna(0.0)
    return net.loc[e_per.index[0]:]


def matched_weight(r_risk, r_slack, target, mode):
    lo, hi = 0.01, 1.0
    for _ in range(40):
        w = (lo + hi) / 2
        mix = w * r_risk + (1 - w) * r_slack
        if mode == "vol":
            hit = float(mix.std()) > target
        else:
            cum = (1 + mix).cumprod()
            hit = float(((cum / cum.cummax()) - 1).min()) < target
        hi, lo = (w, lo) if hit else (hi, w)
    return w


def boot_pvalues(s, r_risk, r_slack, weights, rng, n_boot=2000):
    X = np.column_stack([s.values, r_risk.values, r_slack.values])
    T = len(X)
    wins = {k: 0 for k in weights}
    wins["sharpe"] = 0
    for _ in range(n_boot):
        ix, t = np.empty(T, dtype=int), 0
        while t < T:
            s0 = rng.integers(0, T)
            for j in range(rng.geometric(1 / 63)):
                if t >= T:
                    break
                ix[t] = (s0 + j) % T
                t += 1
        R = X[ix]
        a = R[:, 0]
        for k, w in weights.items():
            m = w * R[:, 1] + (1 - w) * R[:, 2]
            wins[k] += (1 + a).prod() > (1 + m).prod()
            if k == "avg":
                wins["sharpe"] += (a.mean() / a.std()
                                   > m.mean() / m.std())
    return {k: v / n_boot for k, v in wins.items()}


def main():
    uni = load_etf_universe_prices()
    qld = pd.read_csv(_bootstrap.ROOT / "datasets" / "levered_etfs.csv.gz",
                      index_col=0, parse_dates=True)["QLD"].dropna()
    e_per = pd.read_csv(COUNCIL_MINUTES, index_col=0,
                        parse_dates=True)["exposure"]
    rng = np.random.default_rng(7)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    span_start = e_per.index[0]
    idx = qld.loc[span_start:].index
    r_qld = qld.pct_change().reindex(idx).fillna(0.0)
    r_qqq = uni["QQQ"].pct_change().reindex(idx).fillna(0.0)

    rows, curves = {}, {"QQQ buy & hold": r_qqq, "QLD buy & hold": r_qld}
    rows["QQQ buy & hold"] = stats(r_qqq)
    rows["QLD buy & hold"] = stats(r_qld)

    for slack_name in SLACKS:
        r_sl = uni[slack_name].pct_change().reindex(idx).fillna(0.0)
        s = council_book(qld, uni[slack_name], e_per).reindex(idx).fillna(0.0)
        e = e_per.reindex(qld.index).ffill().shift(1).reindex(idx)
        w_avg = float(e.mean())

        mix = w_avg * r_qld + (1 - w_avg) * r_sl
        rows[f"static {w_avg:.2f} QLD / {1 - w_avg:.2f} {slack_name}"] = stats(mix)
        st = stats(s)
        rows[f"council QLD + slack {slack_name}"] = st

        weights = {"avg": w_avg,
                   "vol": matched_weight(r_qld, r_sl, float(s.std()), "vol"),
                   "dd": matched_weight(r_qld, r_sl, st["max_dd"], "dd")}
        p = boot_pvalues(s, r_qld, r_sl, weights, rng)
        rows[f"council QLD + slack {slack_name}"].update({
            "P_sh_vs_mix": round(p["sharpe"], 2),
            "P_w_vs_mix": round(p["avg"], 2),
            "P_w_volm": round(p["vol"], 2),
            "P_w_ddm": round(p["dd"], 2)})
        if slack_name == "GLD":
            curves[f"static {w_avg:.2f}/{1 - w_avg:.2f} QLD/GLD"] = mix
            curves["council QLD + slack GLD"] = s

    df = pd.DataFrame(rows).T
    print(f"LEVERED BARBELL ({idx[0].date()} → {idx[-1].date()}; council "
          "exposure from run_council_backtest.py, mirror-audit CLEAN)")
    print("=" * 100)
    print(df.to_string())
    print("\nP_* columns: bootstrap P(council variant beats the STATIC MIX) — "
          "avg-exposure, vol-matched and\nDD-matched weights. The dial earns "
          "its keep only where these are decisive; the barbell itself is\n"
          "judged against the buy & hold rows.")
    df.to_csv(OUT_DIR / "levered_barbell.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, r in curves.items():
        cum = (1 + r).cumprod()
        style = dict(lw=1.2, ls="--", color="0.4") if "buy & hold" in name \
            else dict(lw=1.5)
        ax1.plot(cum.index, cum.values, label=name, **style)
        ax2.plot(cum.index, (cum / cum.cummax() - 1).values, **style)
    ax1.set_yscale("log")
    ax1.set_ylabel("growth of $1 (log)")
    ax1.set_title("Levered barbell: QLD/GLD static vs council-dialed, "
                  "vs buy & hold")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("drawdown")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "levered_barbell.png", dpi=140)
    print(f"\nSaved to {OUT_DIR}/: levered_barbell.csv, levered_barbell.png")


if __name__ == "__main__":
    main()
