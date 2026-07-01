# Dataset manifest

Auto-generated snapshot of every bundled CSV/CSV.GZ file under `datasets/` — answers "is this result on stale data?" without opening each file by hand. For each dataset: file size, row/column count, the date range covered (parsed from the first column, falling back to a `date`/`Date`/`timestamp` column), which `experiments/build_*.py` (or other) script produces it, and a content hash (sha256, first 12 hex chars) to detect silent changes between runs.

Regenerate with: `python experiments/build_dataset_manifest.py`

Generated: 2026-07-01 23:04 UTC

| file | size | rows × cols | date range | builder | sha256 |
|---|---|---|---|---|---|
| `alt_zoo.csv` | 1.4MB | 4,889 × 28 | 2007-01-03 → 2026-06-09 | unknown | `d29173b62a76` |
| `btc_usd_daily.csv` | 32.0KB | 1,636 × 2 | 2022-01-01 → 2026-06-24 | unknown | `4b827126492c` |
| `dealer_positions.csv.gz` | 4.4KB | 687 × 3 | 2013-04-03 → 2026-05-27 | unknown | `70aabbb56198` |
| `epu_daily.csv` | 256.6KB | 15,147 × 2 | 1985-01-01 → 2026-06-21 | unknown | `bfca3a31a56d` |
| `equity_universe_info.csv` | 824.4KB | 500 × 23 | — | experiments/build_equity_universe.py | `53377f6045a5` |
| `equity_universe_prices.csv.gz` | 14.0MB | 4,132 × 501 | 2010-01-04 → 2026-06-08 | experiments/build_equity_universe.py | `fe298b54f4b4` |
| `etf_universe_info.csv` | 171.2KB | 250 × 11 | — | experiments/build_etf_universe.py | `e7d35efa1894` |
| `etf_universe_prices.csv.gz` | 7.0MB | 4,132 × 251 | 2010-01-04 → 2026-06-08 | experiments/build_etf_universe.py | `bfb2d647a7a5` |
| `ff_europe_monthly.csv` | 36.5KB | 430 × 8 | 1990-07-31 → 2026-04-30 | experiments/build_factor_data.py | `d11f1d314c9c` |
| `ff_factors_daily.csv.gz` | 212.4KB | 15,813 × 8 | 1963-07-01 → 2026-04-30 | experiments/build_factor_data.py | `67b82db3eb94` |
| `ff_factors_monthly.csv` | 64.2KB | 754 × 8 | 1963-07-31 → 2026-04-30 | experiments/build_factor_data.py | `515c0603679f` |
| `ff_industry12_monthly.csv` | 166.5KB | 1,198 × 13 | 1926-07-31 → 2026-04-30 | experiments/build_factor_data.py | `2a4559a4aafd` |
| `fred_macro.csv` | 1010.0KB | 9,507 × 21 | 1990-01-01 → 2026-06-09 | experiments/build_fred_macro.py, experiments/build_fred_macro_plus.py | `569892eebd80` |
| `fred_macro_plus.csv` | 1.3MB | 9,507 × 28 | 1990-01-01 → 2026-06-09 | experiments/build_fred_macro_plus.py | `ec109e5034ed` |
| `fresh_GLD.csv` | 35.6KB | 1,289 × 2 | 2004-11-18 → 2009-12-31 | experiments/run_barbell_gfc.py, experiments/run_timing_mining.py | `e3ebf097a238` |
| `fresh_QQQ.csv` | 77.9KB | 2,722 × 2 | 1999-03-10 → 2009-12-31 | experiments/run_barbell_gfc.py, experiments/run_timing_mining.py | `9c31aad02e5a` |
| `fresh_SPY.csv` | 120.8KB | 4,264 × 2 | 1993-01-29 → 2009-12-31 | experiments/run_barbell_gfc.py, experiments/run_timing_mining.py | `a732ba3a0dcb` |
| `funding_rates.csv.gz` | 27.3KB | 8,529 × 5 | 2000-07-03 → 2026-06-10 | experiments/run_funding_stress.py | `4cc6d8dc616d` |
| `gex_snapshots.csv` | 151B | 1 × 8 | 2026-06-09 → 2026-06-09 | experiments/run_gamma_exposure.py | `0aee3dbdad6d` |
| `gpr_daily.csv.gz` | 354.5KB | 15,148 × 4 | 1985-01-01 → 2026-06-22 | unknown | `68e29f4a2e72` |
| `jkp_cop_portfolios.csv.gz` | 224.7KB | 16,458 × 8 | 1951-11-30 → 2025-12-31 | experiments/run_quality_longonly.py | `4a44917a9940` |
| `jkp_factors.csv.gz` | 385.2KB | 33,088 × 4 | 1926-01-31 → 2025-12-31 | experiments/build_factor_data.py | `fa2d01b18e53` |
| `jkp_factors_individual.csv.gz` | 3.9MB | 339,720 × 4 | 1926-01-31 → 2025-12-31 | posterioralpha/data/factors.py | `aed2c70558b1` |
| `levered_etfs.csv.gz` | 93.6KB | 5,023 × 3 | 2006-06-21 → 2026-06-09 | unknown | `69d5a20eb878` |
| `levered_stacked.csv` | 839.3KB | 5,140 × 15 | 2006-01-03 → 2026-06-09 | unknown | `6d83cfd54d94` |
| `liquid_alts_wide.csv` | 922.8KB | 4,385 × 20 | 2009-01-02 → 2026-06-09 | unknown | `6377c8e83c14` |
| `net_liquidity.csv` | 265.5KB | 5,594 × 5 | 2005-01-03 → 2026-06-11 | experiments/build_fred_macro.py | `2447e5cc13ed` |
| `openap_ls_returns.csv.gz` | 1.4MB | 1,188 × 213 | 1926-01-31 → 2024-12-31 | experiments/build_factor_data.py | `7199392d2856` |
| `openap_signal_doc.csv` | 177.3KB | 331 × 28 | — | experiments/build_factor_data.py | `590a308e4048` |
| `overnight_panel.csv.gz` | 259.7KB | 8,397 × 5 | 1993-01-29 → 2026-06-09 | experiments/run_overnight_split.py | `a616f841d1ce` |
| `portfolio_data.csv` | 500.3KB | 5,032 × 6 | 2005-01-03 → 2024-12-30 | unknown | `adddad96ec18` |
| `quality_etfs.csv` | 770.4KB | 5,394 × 12 | 2005-01-03 → 2026-06-11 | unknown | `764b1deb37d3` |
| `retail_alts.csv` | 713.9KB | 4,385 × 12 | 2009-01-02 → 2026-06-09 | unknown | `a34bd6461c01` |
| `sp500_top100_adj_close.csv` | 4.2MB | 2,511 × 99 | 2016-04-18 → 2026-04-13 | unknown | `54878e800deb` |
| `stress_panel.csv.gz` | 421.2KB | 4,890 × 14 | 2007-01-03 → 2026-06-09 | experiments/run_liquidity_stress.py | `8b1d93af52e2` |
| `treasury_auctions.csv.gz` | 119.1KB | 7,255 × 8 | 2003-01-06 → 2026-06-11 | unknown | `b077b0771689` |
| `tug_conditioners.csv.gz` | 109.2KB | 16,655 × 3 | 1960-01-04 → 2026-06-09 | unknown | `e7e4d1e8fdd3` |
| `vix_term_structure.csv` | 190.9KB | 4,134 × 3 | 2010-01-04 → 2026-06-09 | experiments/run_liquidity_predictive.py | `4cda204b4169` |
| `vix_termstructure.csv` | 225.9KB | 4,892 × 3 | 2007-01-03 → 2026-06-11 | unknown | `6fc38d2cdb98` |
| `withheld_taxes.csv.gz` | 26.1KB | 5,194 × 2 | 2005-10-03 → 2026-06-08 | unknown | `199029985642` |

## Notes

**(a) Near-duplicate VIX term-structure files**

- **`vix_term_structure.csv`**: 4,134 rows × 2 cols (`['VIX', 'VIX3M']`), 2010-01-04 → 2026-06-09. Builder: experiments/run_liquidity_predictive.py. Referenced by 11 script(s): experiments/run_barbell_governors.py, experiments/run_exploration_lab.py, experiments/run_liquidity_4layer.py, experiments/run_liquidity_calendar.py, experiments/run_liquidity_continuous.py, experiments/run_liquidity_disagreement.py, experiments/run_liquidity_levered.py, experiments/run_liquidity_predictive.py, experiments/run_liquidity_regime_weighted.py, experiments/run_model_disagreement.py, experiments/run_overnight_split.py.
- **`vix_termstructure.csv`**: 4,892 rows × 2 cols (`['VIX', 'VIX3M']`), 2007-01-03 → 2026-06-11. Builder: unknown. Referenced by 5 script(s): experiments/plot_daily_leading.py, experiments/run_carry_vrp.py, experiments/run_daily_leading_signal.py, experiments/run_regime_exogenous.py, experiments/run_regime_paper_update.py.
- Same schema (`Date, VIX, VIX3M`). 4,134 overlapping dates; max abs difference on overlapping VIX/VIX3M values is 0.180 (negligible — different download vintages of the same series, not different data).
- **Recommendation**: treat **`vix_termstructure.csv`** as canonical — it has the wider date coverage (back to 2007-01-03, a strict superset of the other file's range) and history should never be lost by consolidating onto it. `vix_term_structure.csv` is currently the more heavily-referenced file (11 scripts vs. 5), so consolidating means repointing those scripts to `vix_termstructure.csv` and then deleting `vix_term_structure.csv`. Note `vix_term_structure.csv` has an active cache-on-miss builder (experiments/run_liquidity_predictive.py) that re-downloads from a hardcoded 2010-01-01 start if the file is ever deleted — that start date would need bumping to match `vix_termstructure.csv`'s 2007-01-03 history before the consolidation, or the download would silently truncate history again.

**(b) Files no script references at all** (candidates for deletion)

- None — every dataset is referenced by at least one script.

**(c) Stale candidates** (date range ends more than 180 days before generation date 2026-07-01)

- `fresh_GLD.csv` — last date 2009-12-31 (6,026 days ago) — **intentional**: pre-2010 fresh-sample cache for the timing miner's PROMOTED gate (`fresh_<ticker>.csv`), deliberately frozen history, not stale data
- `fresh_QQQ.csv` — last date 2009-12-31 (6,026 days ago) — **intentional**: pre-2010 fresh-sample cache for the timing miner's PROMOTED gate (`fresh_<ticker>.csv`), deliberately frozen history, not stale data
- `fresh_SPY.csv` — last date 2009-12-31 (6,026 days ago) — **intentional**: pre-2010 fresh-sample cache for the timing miner's PROMOTED gate (`fresh_<ticker>.csv`), deliberately frozen history, not stale data
- `portfolio_data.csv` — last date 2024-12-30 (548 days ago)
- `openap_ls_returns.csv.gz` — last date 2024-12-31 (547 days ago)
- `jkp_cop_portfolios.csv.gz` — last date 2025-12-31 (182 days ago)
- `jkp_factors.csv.gz` — last date 2025-12-31 (182 days ago)
- `jkp_factors_individual.csv.gz` — last date 2025-12-31 (182 days ago)
