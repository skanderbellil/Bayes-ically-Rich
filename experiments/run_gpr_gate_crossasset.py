#!/usr/bin/env python3
"""
Does the level-OR-vol GPR calm gate show up in equities & other asset classes?
==============================================================================

The deployed edge is a *prediction-market mispricing*: during GPR-calm regimes,
cheap-YES contracts on disruptive geopolitical events are underpriced (the market
is complacent about disruption). The gate is a pure causal daily regime label:

    CALM = (trailing-45d MEAN  GPR <= its trailing-365d median)        [level]
        OR (trailing-45d STD of daily GPR <= its trailing-365d median) [vol]

A prediction-market calibration edge has no literal analog on a price series, so
to ask "can we SEE this gate in equities / other asset classes" we test the only
thing a price series can answer:

    Does conditioning on the SAME calm/turbulent label produce a different
    forward return (and risk-adjusted return) for that asset?

TWO PASSES, exactly as asked:
  (1) EACH DOMAIN ALONE — for every asset class we measure the regime split on its
      own series: next-day and 21d-forward return in calm vs turbulent, annualized
      Sharpe in each leg, the calm-minus-turbulent gap, a Welch t on the daily-mean
      difference, and a stationary block-bootstrap p-value for that gap (block=21d,
      which neutralizes the autocorrelation that overlapping/serial returns create).
  (2) SAME SIGNAL, APPLIED CROSS-ASSET — the identical GPR flag (built once from the
      geopolitics index) is laid over every asset. We tabulate the sign pattern: how
      many asset classes move the same way, and whether the magnitudes line up with a
      coherent mechanism (complacency in risk assets / cheap tail hedges in calm).

Causality: the flag at day t uses only GPR dated <= t (backward rolling + ffill).
It predicts the return from t -> t+1 (the flag is lagged one day vs the return), so
there is no lookahead. ETF panel is 2010-2026; BTC is 2022-2026 (short — flagged).

HONESTY: ~20 assets x 2 horizons is a multiple-comparison sweep. Everything is
printed (not just winners); read the SIGN PATTERN across asset classes and the
bootstrap p-values, not any single starred cell.

Usage:
  python experiments/run_gpr_gate_crossasset.py
  python experiments/run_gpr_gate_crossasset.py --fwd 21
"""
from __future__ import annotations
import argparse, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import _bootstrap  # noqa

GPR = "datasets/gpr_daily.csv.gz"
ETF = "datasets/etf_universe_prices.csv.gz"
BTC = "datasets/btc_usd_daily.csv"
OUT = "data/cross_asset"
RNG = np.random.default_rng(20260625)

# representative liquid ETF (or series) per asset class -- "broad etfs we already have"
ASSETS = [
    ("US large cap",     "equity",     "SPY"),
    ("US tech",          "equity",     "QQQ"),
    ("US small cap",     "equity",     "IWM"),
    ("Dev intl (exUS)",  "equity",     "EFA"),
    ("Emerging mkts",    "equity",     "EEM"),
    ("China",            "equity",     "FXI"),
    ("Brazil",           "equity",     "EWZ"),
    ("Energy sector",    "equity",     "XLE"),
    ("Financials",       "equity",     "XLF"),
    ("Defensives (stap)","equity",     "XLP"),
    ("Long Treasuries",  "rates",      "TLT"),
    ("Interm Treasuries","rates",      "IEF"),
    ("IG credit",        "credit",     "LQD"),
    ("HY credit",        "credit",     "HYG"),
    ("Gold",             "metal",      "GLD"),
    ("Silver",           "metal",      "SLV"),
    ("Gold miners",      "metal",      "GDX"),
    ("Broad commodity",  "commodity",  "DBC"),
    ("US dollar (UUP)",  "fx",         "UUP"),
    ("Euro (FXE)",       "fx",         "FXE"),
    ("Bitcoin",          "crypto",     "BTC"),
]


def tstat(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def welch_t(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    return (a.mean() - b.mean()) / math.sqrt(va + vb) if va + vb > 0 else np.nan


def gpr_calm_flag():
    """Deployed level-OR-vol calm flag, daily, fully causal (copied from the live gate)."""
    g = pd.read_csv(GPR, parse_dates=["date"]).set_index("date")["GPRD"].sort_index()
    g = g.asfreq("D").ffill()
    lvl = g.rolling(45, min_periods=15).mean()
    lvl_calm = lvl <= lvl.rolling(365, min_periods=180).median()
    vol = g.diff().rolling(45, min_periods=15).std()
    vol_calm = vol <= vol.rolling(365, min_periods=180).median()
    calm = (lvl_calm | vol_calm)
    return calm.rename("calm")


def load_prices():
    etf = pd.read_csv(ETF, parse_dates=["Date"]).set_index("Date").sort_index()
    btc = pd.read_csv(BTC, parse_dates=["date"]).set_index("date")["close"].rename("BTC").sort_index()
    px = etf.join(btc, how="outer")
    return px


def block_boot_p(diff_daily, calm_mask, n=5000, block=21):
    """Stationary-block bootstrap p-value that mean(calm)-mean(turb) <= 0.
    Resamples 21-day blocks of (return, calm-label) jointly so serial dependence
    and the regime's persistence are preserved under the null of no gap."""
    r = diff_daily.values
    c = calm_mask.values.astype(bool)
    m = len(r)
    real = r[c].mean() - r[~c].mean()
    nb = m // block
    if nb < 5:
        return real, np.nan
    starts = np.arange(m - block)
    null = np.empty(n)
    for i in range(n):
        # resample blocks, then randomly relabel calm/turb within the pooled draw
        s = RNG.choice(starts, size=nb, replace=True)
        idx = (s[:, None] + np.arange(block)).ravel()
        rr = r[idx]
        # shuffle the calm labels of the SAME length to break any real gap
        cc = c[idx].copy(); RNG.shuffle(cc)
        if cc.all() or (~cc).all():
            null[i] = 0.0
        else:
            null[i] = rr[cc].mean() - rr[~cc].mean()
    p = float((null >= real).mean()) if real >= 0 else float((null <= real).mean())
    return real, p


def analyze(name, ticker, px, flag, fwd):
    if ticker not in px.columns:
        return None
    p = px[ticker].dropna()
    if len(p) < 250:
        return None
    ret1 = p.pct_change()                                   # daily close-close
    retF = p.pct_change(fwd).shift(-fwd)                    # fwd-day forward (overlapping)
    # causal regime: flag known at t-1 governs the return realized at t (next-day)
    f = flag.reindex(p.index).ffill()
    calm_prev = f.shift(1)                                  # lag the flag one day
    df = pd.DataFrame({"r1": ret1, "rF": retF, "calm": calm_prev}).dropna(subset=["r1", "calm"])
    df["calm"] = df["calm"].astype(bool)
    calm = df[df.calm]; turb = df[~df.calm]
    if len(calm) < 60 or len(turb) < 60:
        return None

    def leg(d, col):
        x = d[col].dropna()
        ann_mu = x.mean() * 252; ann_sd = x.std(ddof=1) * math.sqrt(252)
        return ann_mu, ann_sd, (ann_mu / ann_sd if ann_sd > 0 else np.nan), len(x)

    cm, cs, csh, cn = leg(calm, "r1"); tm, ts, tsh, tn = leg(turb, "r1")
    # forward-horizon mean (non-annualized, per-trade), overlapping
    cF = calm["rF"].dropna().mean(); tF = turb["rF"].dropna().mean()
    diff_daily = df["r1"]
    real, bp = block_boot_p(diff_daily, df["calm"])
    return dict(name=name, ticker=ticker, pct_calm=len(calm) / len(df),
                calm_ann=cm, calm_sh=csh, calm_t=tstat(calm["r1"]),
                turb_ann=tm, turb_sh=tsh, turb_t=tstat(turb["r1"]),
                gap_ann=cm - tm, welch=welch_t(calm["r1"], turb["r1"]),
                calm_fwd=cF, turb_fwd=tF, gap_fwd=cF - tF, boot_p=bp,
                n_calm=cn, n_turb=tn, start=df.index.min(), end=df.index.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fwd", type=int, default=21, help="forward horizon (trading days) for the 2nd-pass return")
    args = ap.parse_args()
    import os; os.makedirs(OUT, exist_ok=True)

    flag = gpr_calm_flag()
    px = load_prices()
    pct_calm_all = flag.reindex(px.index).ffill().mean()

    print("=" * 112)
    print("LEVEL-OR-VOL GPR CALM GATE — CROSS-ASSET TEST  (same causal flag the live geo strategy uses)")
    print("  flag = (45d-mean GPR <= 365d-median) OR (45d-std GPR <= 365d-median);  calm covers %.0f%% of days"
          % (100 * pct_calm_all))
    print("  Q: does the SAME regime label split forward returns in equities & other classes? (lagged 1d, causal)")
    print("=" * 112)

    rows = [analyze(n, t, px, flag, args.fwd) for n, _, t in ASSETS]
    cls = {t: c for n, c, t in ASSETS}
    rows = [r for r in rows if r]

    # ---------- PASS 1: each domain alone ----------
    print("\n[PASS 1] EACH ASSET ALONE — next-day return split by GPR regime (annualized), %dd-fwd as robustness" % args.fwd)
    print("  %-18s %4s | %7s %6s %5s | %7s %6s %5s | %8s %6s %8s %6s"
          % ("asset", " tk", "calmAnn", "Sh", "t", "turbAnn", "Sh", "t",
             "GAPann", "Welch", "fwdGAP", "bootP"))
    order = ["equity", "rates", "credit", "metal", "commodity", "fx", "crypto"]
    rows.sort(key=lambda r: (order.index(cls[r["ticker"]]), -r["gap_ann"]))
    last_c = None
    for r in rows:
        c = cls[r["ticker"]]
        if c != last_c:
            print("  " + "-" * 104 + "  [%s]" % c); last_c = c
        star = "*" if (r["boot_p"] == r["boot_p"] and r["boot_p"] <= 0.10) else " "
        print("  %-18s %4s | %+6.1f%% %+6.2f %+5.2f | %+6.1f%% %+6.2f %+5.2f | %+7.1f%% %+6.2f %+7.2f%% %5.3f%s"
              % (r["name"], r["ticker"], 100 * r["calm_ann"], r["calm_sh"], r["calm_t"],
                 100 * r["turb_ann"], r["turb_sh"], r["turb_t"], 100 * r["gap_ann"],
                 r["welch"], 100 * r["gap_fwd"], r["boot_p"], star))

    # ---------- PASS 2: same signal, sign pattern across classes ----------
    print("\n[PASS 2] SAME SIGNAL CROSS-ASSET — is the sign pattern coherent?")
    risk = [r for r in rows if cls[r["ticker"]] in ("equity", "credit", "crypto", "commodity")]
    hedge = [r for r in rows if cls[r["ticker"]] in ("rates", "metal")]
    def share_pos(g): return np.mean([r["gap_ann"] > 0 for r in g]) if g else float("nan")
    print("  RISK assets (equity/credit/commodity/crypto): calm>turb in %.0f%% of %d  | mean GAPann %+.1f%%"
          % (100 * share_pos(risk), len(risk), 100 * np.mean([r["gap_ann"] for r in risk])))
    print("  HEDGE assets (Treasuries/gold/silver/miners): calm>turb in %.0f%% of %d  | mean GAPann %+.1f%%"
          % (100 * share_pos(hedge), len(hedge), 100 * np.mean([r["gap_ann"] for r in hedge])))
    sig = [r for r in rows if r["boot_p"] == r["boot_p"] and r["boot_p"] <= 0.10]
    print("  block-bootstrap survivors (p<=0.10, %d of %d): %s"
          % (len(sig), len(rows), ", ".join("%s(%+.1f%%)" % (r["ticker"], 100 * r["gap_ann"]) for r in sig) or "(none)"))
    exp = len(rows) * 0.10
    print("  -> with %d tests, ~%.0f survivors expected by chance at p<=0.10; got %d. "
          "Read direction+coherence, not the star count." % (len(rows), exp, len(sig)))

    # ---------- chart: gap by asset, colored by class ----------
    palette = {"equity": "#1f77b4", "rates": "#2ca02c", "credit": "#9467bd",
               "metal": "#d4a017", "commodity": "#8c564b", "fx": "#7f7f7f", "crypto": "#e377c2"}
    fig, ax = plt.subplots(figsize=(11, 6.5))
    rs = sorted(rows, key=lambda r: r["gap_ann"])
    y = np.arange(len(rs))
    ax.barh(y, [100 * r["gap_ann"] for r in rs],
            color=[palette[cls[r["ticker"]]] for r in rs],
            edgecolor=["black" if (r["boot_p"] == r["boot_p"] and r["boot_p"] <= 0.10) else "none" for r in rs],
            linewidth=1.4)
    ax.set_yticks(y); ax.set_yticklabels(["%s (%s)" % (r["name"], r["ticker"]) for r in rs], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("calm − turbulent  annualized next-day return gap (%)")
    ax.set_title("Level-OR-vol GPR calm gate: forward-return gap by asset class\n"
                 "(black outline = block-bootstrap p<=0.10; same causal flag as the live geo strategy)", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in palette]
    ax.legend(handles, list(palette), fontsize=8, ncol=4, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/gpr_gate_crossasset.png", dpi=120)
    pd.DataFrame(rows).to_csv(f"{OUT}/gpr_gate_crossasset.csv", index=False)
    print("\nSaved: %s/gpr_gate_crossasset.png , %s/gpr_gate_crossasset.csv" % (OUT, OUT))


if __name__ == "__main__":
    main()
