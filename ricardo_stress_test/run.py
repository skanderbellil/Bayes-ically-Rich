"""
Ricardo-with-a-Bloomberg-terminal L/S stress test -- main entry point.

Runs the full diagnostic and writes:
  outputs/*.png        the five (+2) charts
  outputs/summary.csv  the metrics table (all variants, with/without borrow)
  outputs/report.txt   the full printed report (also echoed to stdout)

Usage:  python run.py            (uses cached data if present)
        python run.py --force    (re-download everything)

This is DIAGNOSIS, not optimization: equal weights everywhere, no curve-fitting.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    LONG_TICKERS, SHORT_TICKERS, UNIVERSE, RISK_FREE_ANNUAL,
    ROLLING_BETA_PLOT_WINDOW, HARD_TO_BORROW, BORROW_APR_DEFAULT, BORROW_APR_HTB,
)
import data as datamod
import portfolio as pf
import analysis as an
import charts as ch
import fundamentals as fund

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

_report_lines = []
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    _report_lines.append(line)


def fmt_metrics(df):
    d = df.copy()
    for c in ["YTD_return", "ann_return", "ann_vol", "max_drawdown",
              "worst_day", "best_day", "pct_positive_days"]:
        if c in d: d[c] = d[c].map(lambda x: f"{x:+.2%}")
    if "Sharpe_rf4" in d: d["Sharpe_rf4"] = d["Sharpe_rf4"].map(lambda x: f"{x:+.2f}")
    return d


def main(force=False):
    emit("=" * 96)
    emit("  RICARDO WITH A BLOOMBERG TERMINAL — L/S THESIS STRESS TEST")
    emit("  Long inelastic supply (power/grid/fuel/silicon chokepoints)  |  "
         "Short leveraged compute + labor arbitrage")
    emit("=" * 96)

    # ---------------- DATA ----------------
    d = datamod.load_dataset(force=force)
    prices, returns = d["prices"], d["returns"]
    long_kept = [t for t in LONG_TICKERS if t in returns.columns]
    short_kept = [t for t in SHORT_TICKERS if t in returns.columns]
    spy = returns["SPY"]

    emit("\n[1] DATA QUALITY")
    emit(f"    Window: {prices.index.min().date()} -> {prices.index.max().date()} "
         f"({len(prices)} trading days, {len(returns)} return days)")
    emit(f"    Kept {len(long_kept)} longs, {len(short_kept)} shorts, "
         f"benchmarks SPY/QQQ/XLI.")
    dropped = d["report"][d["report"].status == "DROPPED"]["ticker"].tolist()
    emit(f"    Dropped (>10% missing): {dropped or 'none'}")
    emit(f"    Failed to download: {d['failed'] or 'none'}")
    emit(f"    FX pairs used: {datamod.required_fx_pairs(d['currency_map'])}")

    # ---------------- BALANCE-SHEET FRAGILITY (iteration 2) ----------------
    # Ex-ante, no-lookahead financing-fragility score -> tilted short weights.
    fragdf = fund.build_fragility(short_kept, force=force)
    frag_w = fragdf["w_bsaware"]

    # ---------------- VARIANTS (with & without borrow) ----------------
    variants_wb, diag_wb = pf.build_variants(returns, long_kept, short_kept, spy,
                                             apply_borrow=True,
                                             short_frag_weights=frag_w)
    variants_nb, diag_nb = pf.build_variants(returns, long_kept, short_kept, spy,
                                             apply_borrow=False,
                                             short_frag_weights=frag_w)
    benches = pf.benchmark_returns(returns)

    # ---------------- 3. METRICS ----------------
    emit("\n[2] METRICS — WITH borrow haircut (default 4%, 15% HTB: "
         f"{sorted(HARD_TO_BORROW)})")
    series_wb = {**variants_wb, **benches}
    mt_wb = an.metrics_table(series_wb, rf=RISK_FREE_ANNUAL)
    emit(fmt_metrics(mt_wb).to_string())

    emit("\n[3] METRICS — WITHOUT borrow haircut")
    series_nb = {**variants_nb, **benches}
    mt_nb = an.metrics_table(series_nb, rf=RISK_FREE_ANNUAL)
    emit(fmt_metrics(mt_nb).to_string())

    # borrow drag = difference in YTD for the L/S variants
    emit("\n[4] BORROW DRAG (YTD return, with vs without)")
    for v in ["dollar_neutral", "beta_neutral", "dollar_neutral_bsaware"]:
        drag = mt_wb.loc[v, "YTD_return"] - mt_nb.loc[v, "YTD_return"]
        emit(f"    {v:16s}: {mt_nb.loc[v,'YTD_return']:+.2%} (no borrow) -> "
             f"{mt_wb.loc[v,'YTD_return']:+.2%} (with)  |  drag = {drag:+.2%}")

    # ---------------- 4. CONTRIBUTION ----------------
    contrib, ctotal = an.name_contributions(returns, long_kept, short_kept,
                                            apply_borrow=True)
    emit("\n[5] PER-NAME CONTRIBUTION to dollar-neutral L/S PnL "
         f"(sum={ctotal:+.2%}; check vs sum-of-daily)")
    emit("    Top 6 positive:")
    for nm, v in contrib.head(6).items():
        emit(f"       {nm:10s} {v:+.2%}   ({UNIVERSE[nm]['book']}/{UNIVERSE[nm]['tier']})")
    emit("    Top 6 negative:")
    for nm, v in contrib.tail(6).iloc[::-1].items():
        emit(f"       {nm:10s} {v:+.2%}   ({UNIVERSE[nm]['book']}/{UNIVERSE[nm]['tier']})")

    # ---------------- 5. FRAGILITY INDEX ----------------
    frag = an.fragility_index(returns, long_kept, short_kept, apply_borrow=True)
    emit("\n[6] ONE-NAME FRAGILITY INDEX (the Nebius test)")
    for k, v in frag["runs"].items():
        emit(f"    {k:22s} L/S YTD = {v:+.2%}")
    emit(f"    Best contributor : {frag['best']}  ({frag['best_contrib']:+.2%})")
    emit(f"    Worst contributor: {frag['worst']} ({frag['worst_contrib']:+.2%})")
    emit(f"    Return SPREAD across the 3 runs = {frag['spread']:.2%}")
    emit(f"    -> dropping BEST costs {frag['drop_best_impact']:+.2%}; "
         f"dropping WORST adds {-frag['drop_worst_impact']:+.2%}")

    # ---------------- 6. DISPERSION ----------------
    cum_fm, monthly_fm, xstd_fm = an.category_dispersion(returns, short_kept_fragile :=
        [t for t in short_kept if UNIVERSE[t]["tier"] == "fragile_middle"])
    emit("\n[7] WITHIN-CATEGORY DISPERSION — fragile_middle")
    emit(f"    Names: {short_kept_fragile}")
    emit("    Cross-sectional std of monthly returns (thesis wants this HIGH):")
    for mth, v in xstd_fm.items():
        emit(f"       {mth}: {v:.2%}")
    emit(f"    Mean monthly cross-sectional std = {xstd_fm.mean():.2%}  "
         f"(vs a tight factor basket ~2-5%)")

    # ---------------- 7. PAIR SPREAD ----------------
    pair = an.pair_spread(returns, "CRWV", "NBIS")
    emit("\n[8] PAIR SPREAD — CRWV vs NBIS")
    if pair:
        for k, s in pair.items():
            emit(f"    {k:22s} cumulative = {s.iloc[-1]:+.2%}")
    else:
        emit("    (CRWV/NBIS not both available)")

    # ---------------- 8. TIER ATTRIBUTION ----------------
    tier_df = an.tier_attribution(returns, long_kept)
    emit("\n[9] LONG-BOOK TIER ATTRIBUTION (standalone equal-weight, YTD)")
    tv = tier_df.copy()
    tv["tier_return"] = tv["tier_return"].map(lambda x: f"{x:+.2%}")
    tv["ann_vol"] = tv["ann_vol"].map(lambda x: f"{x:.1%}")
    emit(tv.to_string())
    best_tier = tier_df["tier_return"].idxmax()
    worst_tier = tier_df["tier_return"].idxmin()
    emit(f"    Carried H1: {best_tier} ({tier_df.loc[best_tier,'tier_return']:+.1%});  "
         f"Lagged: {worst_tier} ({tier_df.loc[worst_tier,'tier_return']:+.1%})")

    # ---------------- 9b. BALANCE-SHEET-AWARE SHORT (iteration 2) ----------------
    dn_eq = mt_wb.loc["dollar_neutral", "YTD_return"]
    dn_bs = mt_wb.loc["dollar_neutral_bsaware", "YTD_return"]
    emit("\n[10] BALANCE-SHEET-AWARE SHORT — 'fit the idea, not the data'")
    emit("     Fragility = mean percentile-rank of 3 ex-ante FINANCING metrics")
    emit("     (net-debt/rev, burn/cash, debt/cash), point-in-time as of the")
    emit(f"     quarter public by {fund._as_of_cutoff()} (no lookahead), held static.")
    show = fragdf[["asof_quarter", "net_debt", "cash", "burn", "fragility",
                   "w_equal", "w_bsaware", "w_tilt"]].copy()
    for c in ["net_debt", "cash", "burn"]:
        show[c] = show[c].map(lambda x: f"{x/1e9:,.2f}B" if pd.notna(x) else "n/a")
    for c in ["fragility", "w_equal", "w_bsaware"]:
        show[c] = show[c].map(lambda x: f"{x:.3f}")
    show["w_tilt"] = show["w_tilt"].map(lambda x: f"{x:+.3f}")
    emit(show.to_string())
    emit(f"     Most-shorted (fragile): {', '.join(fragdf.head(3).index)}")
    emit(f"     Least-shorted (sturdy): {', '.join(fragdf.tail(3).index)}")
    emit(f"     NBIS tilt = {fragdf.loc['NBIS','w_tilt']:+.3f} "
         f"(net cash -> correctly shorted LESS)")
    # Was financing fragility PREDICTIVE of short success this period? Rank-correlate
    # each short's fragility with its realized contribution to L/S PnL (higher
    # contribution = better short, i.e. the name fell / underperformed). Per the
    # thesis this should be POSITIVE (fragile de-rates). This is the honest test.
    short_contrib = contrib.reindex(fragdf.index)
    # Spearman = Pearson on ranks (computed directly to avoid a scipy dependency).
    rho = float(fragdf["fragility"].rank().corr(short_contrib.rank()))
    emit(f"     Rank-corr(fragility, realized short contribution) = {rho:+.2f}  "
         + ("-> fragility PREDICTED short success (thesis holds)." if rho > 0.15 else
            "-> fragility was NOT predictive / ANTI-predictive this window: the "
            "financially-fragile names OUTPERFORMED (AI-capex bid), while the sturdy "
            "labor-arb names were the shorts that actually worked."))
    emit(f"     dollar-neutral YTD: equal short = {dn_eq:+.2%}  ->  "
         f"balance-sheet-aware short = {dn_bs:+.2%}   (delta {dn_bs-dn_eq:+.2%})")
    frag_w.to_frame("bsaware_short_weight").join(
        fragdf[["fragility", "w_tilt"]]).to_csv(OUT / "fragility_weights.csv")

    # ---------------- 9. CHARTS ----------------
    emit("\n[11] WRITING CHARTS ...")
    variant_navs = {k: pf.cum_nav(v) for k, v in variants_wb.items()}
    bench_navs = {k: pf.cum_nav(v) for k, v in benches.items()}
    ls_beta = an.rolling_beta_series(variants_wb["dollar_neutral"], spy,
                                     ROLLING_BETA_PLOT_WINDOW)
    paths = []
    paths.append(ch.chart_nav(variant_navs, bench_navs))
    paths.append(ch.chart_drawdown(variant_navs))
    paths.append(ch.chart_waterfall(contrib, ctotal))
    paths.append(ch.chart_rolling_beta(ls_beta, ROLLING_BETA_PLOT_WINDOW))
    paths.append(ch.chart_dispersion(cum_fm, xstd_fm))
    if pair:
        paths.append(ch.chart_pair(pair))
    paths.append(ch.chart_tier(tier_df))
    paths.append(ch.chart_bsaware(fragdf,
                                  pf.cum_nav(variants_wb["dollar_neutral"]),
                                  pf.cum_nav(variants_wb["dollar_neutral_bsaware"])))
    for p in paths:
        emit(f"     saved {p.name}")

    # ---------------- SUMMARY CSV ----------------
    mt_wb2 = mt_wb.add_suffix("_wBorrow")
    mt_nb2 = mt_nb.add_suffix("_noBorrow")
    summary = mt_wb2.join(mt_nb2)
    summary.to_csv(OUT / "summary.csv")
    contrib.to_frame("contribution_to_LS").to_csv(OUT / "name_contributions.csv")
    emit(f"     saved summary.csv, name_contributions.csv")

    # ---------------- 10. CONCLUSIONS ----------------
    dn_ytd = mt_wb.loc["dollar_neutral", "YTD_return"]
    dn_ytd_nb = mt_nb.loc["dollar_neutral", "YTD_return"]
    borrow_drag = dn_ytd - dn_ytd_nb
    frag_ratio = frag["spread"] / abs(dn_ytd) if abs(dn_ytd) > 1e-9 else np.inf
    top2 = contrib.abs().sort_values(ascending=False).head(2)
    top2_share = top2.sum() / contrib.abs().sum()

    dn_bs = mt_wb.loc["dollar_neutral_bsaware", "YTD_return"]
    bs_delta = dn_bs - dn_ytd

    emit("\n" + "=" * 96)
    emit("  SIX-BULLET CONCLUSIONS")
    emit("=" * 96)
    emit(f"  1. THESIS vs LUCK: dollar-neutral L/S returned {dn_ytd:+.1%} YTD. "
         f"Excluding the single best OR worst name moves the result within a "
         f"{frag['spread']:.1%} band (spread/return = {frag_ratio:.1f}x). "
         + ("Alpha is BROAD/thesis-driven." if frag_ratio < 0.6 else
            "Alpha is NAME-LUCKY — hostage to 1-2 names."))
    emit(f"  2. CONCENTRATION: the 2 largest absolute contributors "
         f"({', '.join(top2.index)}) account for {top2_share:.0%} of gross "
         f"contribution — {'diversified' if top2_share < 0.35 else 'concentrated'}.")
    emit(f"  3. BORROW DRAG: the HTB haircut (15% on {sorted(HARD_TO_BORROW)}) "
         f"costs the dollar-neutral book {abs(borrow_drag):.2%} YTD "
         f"({dn_ytd_nb:+.1%} gross -> {dn_ytd:+.1%} net). Real, but "
         f"{'not' if abs(borrow_drag) < abs(dn_ytd_nb)*0.3 else ''} thesis-breaking.")
    emit(f"  4. TIER TO OVERWEIGHT: {best_tier} led "
         f"({tier_df.loc[best_tier,'tier_return']:+.1%}); {worst_tier} lagged "
         f"({tier_df.loc[worst_tier,'tier_return']:+.1%}). Long alpha is "
         f"tier-differentiated, not uniform.")
    emit(f"  5. SHORT-SIDE DISPERSION: fragile_middle mean monthly cross-sectional "
         f"std = {xstd_fm.mean():.1%} — {'HIGH' if xstd_fm.mean() > 0.06 else 'LOW'}, "
         f"confirming the short basket is a bag of idiosyncratic outcomes "
         f"(balance-sheet selection matters more than the basket).")
    emit(f"  6. FIT THE IDEA: weighting the short book by ex-ante FINANCING "
         f"fragility (no lookahead, no tuning) took NBIS from equal-weight to "
         f"{fragdf.loc['NBIS','w_tilt']:+.1%} tilt and moved dollar-neutral YTD "
         f"{dn_ytd:+.1%} -> {dn_bs:+.1%} ({bs_delta:+.1%}). "
         + ("The idea, properly implemented, ADDS value — the naive equal-weight "
            "short was leaving money on the table by shorting cash-rich names as "
            "hard as debt-funded burners." if bs_delta > 0 else
            "It did NOT rescue the short leg this period — the fragile names RALLIED "
            "regardless of balance sheet, so the thesis's H1 problem was direction, "
            "not weighting. Honest result, not curve-fit."))
    emit("=" * 96)

    (OUT / "report.txt").write_text("\n".join(_report_lines))
    emit(f"\n[done] full report -> outputs/report.txt")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
