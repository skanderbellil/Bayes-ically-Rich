#!/usr/bin/env python3
"""
Auction-demand surprises — trading the liquidity calendar's OUTCOMES.

run_liquidity_calendar traded scheduled dates; this trades scheduled
*surprises*.  Every Treasury coupon auction prints a demand scorecard at
~1pm ET (bid-to-cover, indirect-bidder share).  Commentary treats weak
prints ("tails") as risk-off triggers; academic work models auction risk
premia — but we found no systematic public backtest of demand surprises →
forward risk-asset returns (June 2026).  Free data: TreasuryDirect
FiscalData API, cached in datasets/treasury_auctions.csv.gz (1,941 coupon
auctions, 2003→2026).

  Surprise per auction (causal): z-score of bid-to-cover and of the
  indirect-bidder share vs the trailing 8 auctions of the SAME base tenor;
  composite = mean of the two z's.  Reopenings map to their base tenor.

  Tests
  -----
  1. MECHANICAL CHECK — weak long-end demand should hit bonds same-day:
     TLT day-0 return by surprise tercile (if this fails, stop believing
     anything downstream).
  2. EQUITY TRANSMISSION — QQQ day-0 and drift [+1,+5], [+1,+21] by
     surprise tercile, long end (10y+) vs belly (2-7y).
  3. AGGREGATE LAYER — trailing 63d mean composite surprise (causal) as an
     "auction demand health" signal: forward-return terciles, and added as a
     6th layer to the vote on QQQ with honest costs.
  4. REGIME HOOK — day-0 sensitivity in the post-buffer era (2025-09→):
     descriptive only, n is small.

    python experiments/run_auction_surprise.py
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
AUC_CSV = _bootstrap.ROOT / "datasets" / "treasury_auctions.csv.gz"
PANEL_CSV = _bootstrap.ROOT / "datasets" / "overnight_panel.csv.gz"
STRESS_CSV = _bootstrap.ROOT / "datasets" / "stress_panel.csv.gz"
PORT_CSV = _bootstrap.ROOT / "datasets" / "portfolio_data.csv"

RF = 0.04
DAILY_RF = RF / 252
BORROW_SPREAD = 0.005
TC = 0.0002
LEV_CAP = 2.0
ROLL = 756
TRAIL = 8           # same-tenor auctions for the surprise baseline
H_LIQ, PUB_LAG, CREDIT_WIN = 126, 5, 21


def base_tenor(term: str) -> int:
    """'9-Year 10-Month' → 10;  '2-Year' → 2  (reopenings → base tenor)."""
    y = int(term.split("-")[0])
    return {1: 2, 4: 5, 6: 7, 9: 10, 19: 20, 29: 30}.get(y, y)


def main() -> None:
    auc = pd.read_csv(AUC_CSV, parse_dates=["auction_date"])
    auc = auc[auc.security_type.isin(["Note", "Bond"])].dropna(
        subset=["bid_to_cover_ratio"]).sort_values("auction_date")
    auc["tenor"] = auc["security_term"].map(base_tenor)
    auc["ind_share"] = auc["indirect_bidder_accepted"] / auc["total_accepted"]

    # causal same-tenor z-scores
    for col, z in [("bid_to_cover_ratio", "z_b2c"), ("ind_share", "z_ind")]:
        g = auc.groupby("tenor")[col]
        mu = g.transform(lambda s: s.shift(1).rolling(TRAIL).mean())
        sd = g.transform(lambda s: s.shift(1).rolling(TRAIL).std())
        auc[z] = (auc[col] - mu) / sd
    auc["surprise"] = auc[["z_b2c", "z_ind"]].mean(axis=1)
    auc = auc.dropna(subset=["surprise"])
    logger.info("Scored auctions: %d (%s → %s)", len(auc),
                auc.auction_date.min().date(), auc.auction_date.max().date())

    panel = pd.read_csv(PANEL_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    port = pd.read_csv(PORT_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    idx = panel["QQQ_Close"].dropna().index
    r_qqq = panel["QQQ_Close"].pct_change().reindex(idx)
    r_tlt = port["TLT"].pct_change().reindex(idx)

    # map auctions to trading days; same-day close reaction (results ~1pm ET)
    auc["t"] = [idx[idx.searchsorted(d)] if idx.searchsorted(d) < len(idx)
                else pd.NaT for d in auc.auction_date]
    auc = auc.dropna(subset=["t"])

    def drift(r, t, a, b):
        i = idx.get_loc(t)
        if i + b >= len(idx):
            return np.nan
        return float((1 + r.iloc[i + a:i + b + 1]).prod() - 1)

    long_end = auc[auc.tenor >= 10]
    belly = auc[auc.tenor <= 7]

    print("\n" + "=" * 96)
    print(f"  AUCTION DEMAND SURPRISES  ·  {len(auc)} scored coupon auctions  ·  "
          f"{auc.auction_date.min().date()} → {auc.auction_date.max().date()}")
    print("=" * 96)

    for label, sub in [("LONG END (≥10y)", long_end), ("BELLY (2-7y)", belly)]:
        q = sub["surprise"].rank(pct=True)
        rows = []
        for lab, m in [("weak demand (bottom ⅓)", q <= 1 / 3),
                       ("mid", (q > 1 / 3) & (q < 2 / 3)),
                       ("strong demand (top ⅓)", q >= 2 / 3)]:
            s = sub[m]
            tlt0 = np.nanmean([drift(r_tlt, t, 0, 0) for t in s.t])
            qqq0 = np.nanmean([drift(r_qqq, t, 0, 0) for t in s.t])
            qqq5 = np.nanmean([drift(r_qqq, t, 1, 5) for t in s.t])
            qqq21 = np.nanmean([drift(r_qqq, t, 1, 21) for t in s.t])
            rows.append([lab, f"{tlt0:+.2%}", f"{qqq0:+.2%}", f"{qqq5:+.2%}",
                         f"{qqq21:+.2%}", len(s)])
        print(f"\n  {label}:")
        print(tabulate(rows, headers=["surprise tercile", "TLT day0",
                                      "QQQ day0", "QQQ [+1,+5]",
                                      "QQQ [+1,+21]", "n"], tablefmt="github"))
    print("  (tercile rank is full-sample for display; the strategy below uses "
          "only causal information)")

    # ── Aggregate layer ──────────────────────────────────────────────────────
    daily = (auc.set_index("t")["surprise"].groupby(level=0).mean()
             .reindex(idx))
    agg = daily.rolling(63, min_periods=8).mean().shift(1)   # demand health
    fwd21 = panel["QQQ_Close"].reindex(idx).pct_change(21).shift(-21)
    rk = agg.rank(pct=True)
    print("\n  Aggregate 63d auction-demand health → next-21d QQQ:")
    rows = []
    for lab, m in [("weak", rk <= 1 / 3), ("mid", (rk > 1 / 3) & (rk < 2 / 3)),
                   ("strong", rk >= 2 / 3)]:
        rows.append([lab, f"{float(fwd21[m].mean()) * 12:+.1%}", int(m.sum())])
    m = agg.notna() & fwd21.notna()
    c = float(np.corrcoef(agg[m], fwd21[m])[0, 1])
    print(tabulate(rows, headers=["demand health", "fwd 21d QQQ (ann.)", "days"],
                   tablefmt="github"))
    print(f"  corr(agg surprise, fwd 21d QQQ) = {c:+.3f}")

    # ── As a 6th vote layer (honest costs) ───────────────────────────────────
    nl = load_net_liquidity()["net_liquidity"]
    p = pd.read_csv(STRESS_CSV, parse_dates=["Date"], index_col="Date").sort_index()
    vidx = (nl.dropna().index
            .intersection(p[["SPY", "QQQ", "HYG", "IEF", "UUP", "VIX", "VIX3M"]]
                          .dropna().index)
            .intersection(panel.dropna().index))
    nlv, pv = nl.reindex(vidx), p.reindex(vidx)
    liq = soft_sign(nlv.diff(H_LIQ).shift(PUB_LAG)).shift(1)
    vix_ts = soft_sign(pv["VIX3M"] / pv["VIX"] - 1.0).shift(1)
    credit = soft_sign(pv["HYG"].pct_change(CREDIT_WIN)
                       - pv["IEF"].pct_change(CREDIT_WIN)).shift(1)
    dollar = soft_sign(-pv["UUP"].pct_change(63)).shift(1)
    o, cqv = panel["QQQ_Open"].reindex(vidx), panel["QQQ_Close"].reindex(vidx)
    r_on, r_id = o / cqv.shift(1) - 1.0, cqv / o - 1.0
    tug = soft_sign(((1 + r_on).rolling(21).apply(np.prod, raw=True)
                     - (1 + r_id).rolling(21).apply(np.prod, raw=True))).shift(1)
    auc_layer = soft_sign(agg.reindex(vidx))      # already lagged via agg
    s5 = pd.DataFrame({"liq": liq, "vix_ts": vix_ts, "credit": credit,
                       "dollar": dollar, "tug": tug})
    v5 = s5.mean(axis=1)
    v6 = pd.concat([s5, auc_layer.rename("auction")], axis=1).mean(axis=1)
    rq = pv["QQQ"].pct_change()

    def run(expo):
        expo = expo.clip(0.0, LEV_CAP).fillna(0.0)
        return (expo * rq - np.maximum(expo - 1, 0) * (RF + BORROW_SPREAD) / 252
                + np.maximum(1 - expo, 0) * DAILY_RF
                - TC * expo.diff().abs().fillna(0))

    live = max(v6.first_valid_index(), v5.first_valid_index())
    strat = {"QQQ buy & hold": rq, "2×5-layer vote": run(2 * v5),
             "2×6-layer (+auction)": run(2 * v6)}
    strat = {k: v.loc[live:] for k, v in strat.items()}
    spy_live = pv["SPY"].pct_change().loc[live:]

    def roll_cagr(r):
        return (1 + r).rolling(ROLL).apply(np.prod, raw=True) ** (252 / ROLL) - 1

    keys = ["CAGR", "Volatility", "Sharpe", "Sortino", "Max DD", "Calmar"]
    table = []
    for name, r in strat.items():
        mm = compute_metrics(r.dropna(), rf=RF)
        ret22 = (1 + r[r.index.year == 2022]).prod() - 1
        dc = (roll_cagr(r) - roll_cagr(spy_live)).dropna()
        table.append([name] + [f"{mm.get(k, float('nan')):.2f}" for k in keys]
                     + [f"{ret22:+.0%}", f"{float((dc > 0).mean()):.0%}"])
    print("\n" + tabulate(table, headers=["strategy"] + keys + ["2022", "CAGRwin"],
                          tablefmt="github"))
    corr5 = pd.concat([s5, auc_layer.rename("auction")], axis=1).corr()["auction"]
    print("  corr(auction layer, others): "
          + "  ".join(f"{k} {v:+.2f}" for k, v in corr5.drop("auction").items()))

    # ── Regime hook: post-buffer day-0 sensitivity ───────────────────────────
    post = auc[auc.t >= pd.Timestamp("2025-09-18")]
    pre = auc[auc.t < pd.Timestamp("2025-09-18")]
    print("\n  Day-0 QQQ sensitivity to surprise (per unit z), pre vs post-buffer:")
    for lab, s in [("pre 2025-09", pre), ("post-buffer", post)]:
        d0 = np.array([drift(r_qqq, t, 0, 0) for t in s.t])
        z = s["surprise"].values
        m = ~np.isnan(d0)
        beta = (np.polyfit(z[m], d0[m], 1)[0] if m.sum() > 10 else np.nan)
        print(f"    {lab:12s} beta {beta * 1e4:+.1f} bp/z   (n={int(m.sum())})")

    # ── Graph ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    ax = axes[0]
    ax.plot(agg.index, agg.values, lw=1.0, color="tab:blue",
            label="63d aggregate auction-demand surprise (z)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Aggregate Treasury auction demand health")
    ax = axes[1]
    for name, r in strat.items():
        eq = (1 + r.fillna(0)).cumprod()
        ax.plot(eq.index, eq.values, lw=1.3,
                color="grey" if "buy" in name else None, label=name)
    ax.set_yscale("log"); ax.set_ylabel("growth of $1 (log)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Does the auction layer add anything to the vote?")
    fig.tight_layout()
    out = RESULTS_DIR / "auction_surprise.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    logger.info("Saved: %s", out)
    print("\n✓  Done.")


if __name__ == "__main__":
    main()
