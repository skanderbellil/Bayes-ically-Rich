# Dip-buy study — data inventory and pre-registered design

Written **before** any out-of-sample (2022+) result was computed. Parameters below are frozen;
every variant that is run is listed here and reported in REPORT.md whether it helps or not.

## 1. Data inventory (what this environment can actually reach)

| Need | Available? | Source used | Notes |
|---|---|---|---|
| US universe incl. delisted | **Yes, partly** | CTA EOD summaries (`ftp.nyse.com/cts_summary_files`), 3,230 days 2014-01-02 → 2026-10-30 | Every security on the CTA tape each day, so delisted names are present until their last session. **NYSE- and NYSE American-listed only**: Nasdaq-listed names are on the UTP tape, which has no equivalent free file. |
| Official open/close (auction) vs first trade | **Yes** | Primary-venue open/last in the same files | Verified against NYSE XDP trade conditions (auction prints flagged `O`/`6`) on 2023-10-02/03: primary open = opening-auction print for 100% of NYSE and 99% of NYSE American listings. |
| Auction volumes | Sample days only | XDP trade samples 2023-10-02/03 | The daily file has only full-day venue volume. |
| Pre-open indicative auction prices | Sample days only | XDP imbalance feed 2020-09-09 (NYSE), 2026-07-01/02 (NYSE + NYSE American) | Used to test the tradable (pre-open) signal. |
| Intraday bars / pre-market | Recent window only | Yahoo chart API: 5m bars (last ~60 days) with pre/post-market | Live tickers only. |
| Bid/ask quotes | **No** (only 2 TAQ NBBO days of ~10 GB each) | — | Spread costs use auction-to-auction (no spread) plus haircut scenarios. |
| Corporate actions | Partial | Legacy files flag ex-dividend days (`XD`); splits detected from ratio + sustained volume jump; Yahoo dividends for 2019+ | |
| Market cap, analyst coverage (historical) | **No** | — | Proxies: $ volume, price level, Yahoo earnings dates with an EPS estimate (covered or not, live tickers only). |
| Earnings / news dates | Partial | Yahoo earnings calendar (live tickers) | Delisted names cannot be classified. |
| Europe | Daily only, survivorship-biased | Yahoo daily OHLC for European listings | Xetra/Euronext/LSE open and close are call-auction prices. |

## 2. Frozen definitions (taken from the first pass, not re-tuned)
* open = primary-venue official opening-auction print; close = primary-venue last print (closing auction when one prints).
* agap = (open / prev close − 1) − (SPY open / SPY prev close − 1).
* **Signal (gap-down):** agap ≤ −max(3%, 3 × sd60), sd60 = std of the prior 60 daily abnormal returns (min 50).
  Previous close ≥ $5, |raw gap| < 50%. Split days and known ex-dividend days excluded.
* Q = quintile of the prior calendar year's median daily $ volume (prior-year median price ≥ $5, ≥ 200 sessions).
* Trade: buy at the open, sell at the close, same day. Base cost 0.10% round trip, as in the first pass.
* Inference: means with **standard errors clustered by date**.

## 3. Sample split
* Usable sample 2015-01 → 2026-10 (2014 is burn-in for the prior-year buckets).
* **In-sample 2015-01-01 → 2021-12-31 (≈ 59%)**. **Out-of-sample 2022-01-01 → 2026-10-30.**
* No parameter is chosen on any data. The in-sample period is used only to check that the pipeline reproduces the first-pass pattern.

## 4. Primary hypothesis (one test)
H1: the mean auction-to-auction open→close return, net of 0.10%, of gap-down signals in Q1–Q2 is > 0 out-of-sample (date-clustered t > 2).

## 5. Pre-listed variants (all reported)
Execution models (same signals):
1. E1 auction→auction (official open → official close)
2. E2 strict limit: fill only if consolidated low < official open
3. E3 independent entry print: entry at the median first print of the *other* venues (signal still uses the auction)
4. E4 cross-measurement: signal from other venues' first prints, entry at the official open
5. E5 slippage +0.5% and +1.0% on entry
6. E6 (recent window, Yahoo 5m) entry at 5/15/30-min VWAP, 09:35 and 09:45 prices; pre-market gap signal
7. E7 (sample days, XDP) indicative-price signal at 09:25/09:28/09:29:30

Kill tests: gap-up mirror under E1/E2/E3; open = consolidated high/low frequency by Q on ordinary days.

Cross-sections (E1 and E2): Q; NYSE vs NYSE American; gap size (3–5, 5–10, 10–20, >20%); SPY gap < −0.5% vs not;
earnings day vs not (live tickers); EPS-estimate coverage (yes/no); price (<$10, $10–20, ≥$20); year.

Portfolios (E1, E2, E3, E5 +1%): SPY + overlay (≤20% gross, ≤2% per name, rebuy SPY at close); static 80/20;
compounding and fixed-$ sizing ($2k, $5k, $20k per ticket), with a participation cap on full-day primary-venue volume.

Total pre-listed variants: 7 execution models × 2 directions + 8 cross-section families + 4 × 2 portfolio constructions.
Any further variant run later is added below with a timestamp.

## 6. Addenda
(none yet)
