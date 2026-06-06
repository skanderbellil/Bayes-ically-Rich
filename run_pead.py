#!/usr/bin/env python3
"""Broad PEAD test: analyst-surprise SUE across market-cap buckets.

Orchestration:
  1. Load US primary-listed universe from FinanceDatabase.
  2. Stratified sample per cap bucket (tractable yfinance pull).
  3. Fetch earnings surprises + adjusted prices (checkpoint/resume).
  4. Build signal panel aligned to announcement dates.
  5. Fama-MacBeth IC test: overall + per cap bucket, drift (t+1, t+21, t+63).
  6. Report: IC, t-stats, long-short, power, and honest survivorship caveat.

Survivalship note: yfinance returns zero data for delisted tickers, so the
panel is live-names only (0 delisted in 151k+). This is a structural limit
of available APIs, not a design choice. We report this explicitly.
"""

from pathlib import Path

import pandas as pd

from src.pead import universe, fetch, signals, fama_macbeth


def main():
    # --- Paths ---
    equities_csv = "financial_data/equities.csv"
    raw_dir = "data/raw/pead"
    ledger_path = f"{raw_dir}/fetch_ledger.csv"

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # --- Load universe ---
    print("=" * 80)
    print("PEAD: Post-Earnings-Announcement-Drift Broad Test")
    print("=" * 80)
    print("\n[1] Loading universe...")
    univ = universe.load_live_universe(equities_csv)
    print(f"  Loaded {len(univ)} US primary-listed names.")
    print(f"  Market cap buckets:", dict(univ["market_cap"].value_counts()))

    # --- Stratified sample: 150 per bucket for tractable pull ---
    print("\n[2] Stratified sampling (150 per cap bucket)...")
    sample = universe.stratified_sample(univ, per_bucket=150, seed=42)
    tickers = sorted(sample.index.tolist())
    print(f"  Sampled {len(tickers)} names across cap buckets.")
    print(f"  Buckets represented: {sample['market_cap'].unique()}")

    # --- Fetch prices + earnings (checkpoint/resume) ---
    print(f"\n[3] Fetching earnings surprises + adjusted prices...")
    print(f"  (This may take 5-10min; will resume on restart if interrupted.)")
    ledger = fetch.fetch_universe(tickers, raw_dir, ledger_path, sleep=0.4)

    ok_count = (ledger["status"] == "OK").sum()
    cache_count = (ledger["status"] == "CACHED").sum()
    no_earn = (ledger["status"] == "NO_EARNINGS").sum()
    errors = ledger["status"].str.startswith("ERROR").sum()

    print(f"\n  Fetch complete:")
    print(f"    OK (new):      {ok_count}")
    print(f"    CACHED:        {cache_count}")
    print(f"    NO_EARNINGS:   {no_earn}")
    print(f"    ERRORS:        {errors}")

    tickers_with_data = ledger[ledger["status"] == "OK"].index.tolist()
    if not tickers_with_data:
        print("  No tickers with earnings data. Stopping.")
        return

    # --- Build signal panel ---
    print(f"\n[4] Building signal panel from {len(tickers_with_data)} tickers...")
    panel = signals.build_signal_panel(tickers_with_data, raw_dir, raw_dir)

    # Ensure announcement_date is datetime
    panel["announcement_date"] = pd.to_datetime(
        panel["announcement_date"], utc=True, errors="coerce"
    )
    panel = panel.dropna(subset=["announcement_date"])

    print(f"  Panel shape: {panel.shape}")
    print(f"  Unique tickers: {panel['ticker'].nunique()}")
    print(f"  Date range: {panel['announcement_date'].min().date()} to "
          f"{panel['announcement_date'].max().date()}")

    # Merge cap buckets back in for stratified tests.
    panel = panel.merge(
        sample[["market_cap"]].rename_axis("ticker").reset_index(),
        on="ticker",
        how="left",
    )
    print(f"  Cap buckets in panel: {panel['market_cap'].unique()}")

    # --- Fama-MacBeth tests ---
    print(f"\n[5] Fama-MacBeth IC tests...")

    # Overall test.
    print("\n  === Overall (all cap buckets) ===")
    fm_overall = fama_macbeth.fama_macbeth_ic(panel)
    _print_fm_result(fm_overall)

    # Per cap bucket.
    print("\n  === Per Market-Cap Bucket ===")
    fm_by_cap = fama_macbeth.test_by_cap_bucket(panel, cap_col="market_cap")
    for cap_bucket in universe.CAP_BUCKETS:
        if cap_bucket in fm_by_cap:
            print(f"\n    {cap_bucket}:")
            _print_fm_result(fm_by_cap[cap_bucket], indent=6)

    # --- Save results ---
    print(f"\n[6] Saving results...")
    results_dir = "results/pead"
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Panel CSV.
    panel_out = Path(results_dir) / "signal_panel.csv"
    panel.to_csv(panel_out, index=False)
    print(f"  Panel: {panel_out}")

    # FM results (one row per test).
    fm_rows = []
    for name, res in [("overall", fm_overall)] + [
        (f"{cap}", fm_by_cap[cap]) for cap in universe.CAP_BUCKETS if cap in fm_by_cap
    ]:
        fm_rows.append({
            "test": name,
            "n_seasons": res.n_seasons,
            "avg_n_per_season": res.avg_n_per_season,
            "ic_t1_mean": res.mean_ic_t1,
            "ic_t1_std": res.std_ic_t1,
            "ic_t1_tstat": res.t_stat_t1,
            "ic_t1_pval": res.pval_t1,
            "ic_t21_mean": res.mean_ic_t21,
            "ic_t21_std": res.std_ic_t21,
            "ic_t21_tstat": res.t_stat_t21,
            "ic_t21_pval": res.pval_t21,
            "ls_annual_ret": res.long_short_annual_return,
            "ls_annual_pval": res.long_short_pvalue,
        })

    fm_df = pd.DataFrame(fm_rows)
    fm_out = Path(results_dir) / "fama_macbeth_results.csv"
    fm_df.to_csv(fm_out, index=False)
    print(f"  FM results: {fm_out}")

    # --- Honest closing statement ---
    print("\n" + "=" * 80)
    print("SURVIVORSHIP CAVEAT")
    print("=" * 80)
    print("""
This test uses live US equity names (151k+ tickers from FinanceDatabase).
yfinance cannot retrieve data for delisted (dead) tickers — attempts return
empty data. Thus the panel is survivorship-biased: it includes no bankrupt,
delisted, or acquired-out names that left the market before 2026.

Long-short returns are likely BIASED UPWARD:
  - Negative SUE on a company that filed for bankruptcy is missing.
  - The empirical long-short return estimates the live-name sample only.

To quantify the bias, a full PEAD test would require:
  - PIT delisted-ticker data (e.g., SEC Edgar, paid vendors)
  - Careful alignment of delisting dates to announcement history
  - Walk-forward testing with a moving delisted name map

See src/pead/universe.py and src/pead/fetch.py for implementation notes.
""")

    print("=" * 80)
    print("Done.")
    return fm_overall, fm_by_cap


def _print_fm_result(res: fama_macbeth.FamaMacbethResult, indent: int = 2):
    """Pretty-print a Fama-MacBeth result."""
    sp = " " * indent

    print(f"{sp}Signal: {res.signal_name}")
    print(f"{sp}Seasons: {res.n_seasons}, avg N/season: {res.avg_n_per_season:.1f}")
    print(f"{sp}")
    print(f"{sp}  t+1 (announcement day):")
    print(f"{sp}    IC mean: {res.mean_ic_t1:+.4f} (std {res.std_ic_t1:.4f})")
    print(f"{sp}    t-stat: {res.t_stat_t1:+.2f}, p-val: {res.pval_t1:.4f}")
    print(f"{sp}")
    print(f"{sp}  t+21 (one month drift):")
    print(f"{sp}    IC mean: {res.mean_ic_t21:+.4f} (std {res.std_ic_t21:.4f})")
    print(f"{sp}    t-stat: {res.t_stat_t21:+.2f}, p-val: {res.pval_t21:.4f}")

    if res.mean_ic_t63 is not None:
        print(f"{sp}")
        print(f"{sp}  t+63 (two months drift):")
        print(f"{sp}    IC mean: {res.mean_ic_t63:+.4f} (std {res.std_ic_t63:.4f})")
        print(f"{sp}    t-stat: {res.t_stat_t63:+.2f}, p-val: {res.pval_t63:.4f}")

    if res.long_short_annual_return is not None:
        print(f"{sp}")
        print(f"{sp}  Long-short (SUE > median vs <=):")
        print(f"{sp}    Quarterly return: {res.long_short_annual_return / 4:+.2f}%")
        print(f"{sp}    Annualized return: {res.long_short_annual_return:+.2f}%")
        print(f"{sp}    p-value: {res.long_short_pvalue:.4f}")


if __name__ == "__main__":
    main()
