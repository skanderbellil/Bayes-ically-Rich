# Ricardo with a Bloomberg Terminal — L/S Thesis Stress Test

A **diagnostic** (not an optimizer) for the thematic long/short book:

> In the AI buildout, rents migrate to **inelastic supply** (power, grid, fuel,
> silicon/packaging chokepoints) while **leveraged compute middlemen** (neoclouds
> renting depreciating GPUs on debt) and **AI-displaced labor arbitrage** get
> repriced down. → Long inelastic supply, short fragile middlemen + labor arb.

The whole point is to ask *is the alpha thesis-driven or just 2–3 lucky names?*
So every weight is **equal-weight**, nothing is fit. Guardrail-respecting: no
lookahead, all assumptions in comments.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # yfinance pandas numpy matplotlib requests
python run.py                          # uses cached data if present
python run.py --force                  # re-download everything
```

Outputs land in `outputs/`: 7 PNG charts, `summary.csv`, `name_contributions.csv`,
and `report.txt` (the full printed report).

## What each file does

| File | Role |
|------|------|
| `config.py` | The universe (`ticker → {book, tier, note}`), window, borrow/beta params. |
| `data.py` | Downloads **adjusted closes**, FX-normalizes to USD, cleans the panel. |
| `portfolio.py` | Equal-weight books, the 3 variants, borrow haircut, rolling beta. |
| `analysis.py` | Metrics, per-name contribution, fragility index, dispersion, pairs, tiers. |
| `charts.py` | The 7 diagnostic charts. |
| `run.py` | Orchestrates everything and writes the report. |

## Methodology & assumptions (the important part)

**Data source.** The brief asks for yfinance adjusted closes. In this sandbox all
egress is forced through a TLS-re-terminating proxy that resets `curl_cffi`'s
browser-impersonated handshake, so yfinance cannot connect. We therefore call the
**same Yahoo endpoint yfinance wraps** (`/v8/finance/chart/...`) over a plain
`requests` session and read the `adjclose` series (splits + dividends applied).
Economically identical to `yf.download(auto_adjust=True)`. Everything is cached to
CSV so reruns are offline and deterministic.

**FX.** The universe contains currencies beyond the brief's list (adds CHF, HKD,
GBp/pence). Rather than leave them unconverted, we **auto-detect each ticker's
quote currency** from Yahoo metadata and fetch the matching `<CUR>USD=X` pair.
London pence (`GBp`, e.g. RR.L) is handled as `GBP / 100`. FX is aligned per-date
and forward-filled (same/prior-day rate — no lookahead).

**Calendar & cleaning.** Master calendar = SPY's US trading days. Each USD series
is reindexed to it, forward-filled **≤ 3 days** (bridges foreign holidays / the
date line without inventing data), then any ticker missing **> 10%** of the
calendar is reported and **dropped**. Final inner-join aligns the survivors.
*In this run: `KAP.L` dropped (99% missing — thin GDR); `KLA` fetched under its
Yahoo symbol `KLAC`. Nothing else lost.*

**No lookahead.** Equal weights reset at each month-end and drift through the
month. The beta-neutral short-scale `k` for month *M* uses the trailing-60d betas
observed **through the last day of month M−1**, applied with a one-day lag.

**Borrow haircut.** Short book charged `borrow_apr/252` daily, weighted by the
drifting short weights. Default **4%**; **15%** for hard-to-borrow
`CRWV / NBIS / IREN`. Shown **with and without**.

**Contribution.** Additive daily decomposition (long name `+w·r`, short name
`−w·r − borrow`), so per-name pieces **sum to the arithmetic L/S PnL**. The ~1pt
gap vs the compounded variant return is pure arithmetic-vs-geometric and is
reconciled in the report.

## The three variants

- **(a) long_only** — the long book alone.
- **(b) dollar_neutral** — +100% long / −100% short, equal-weight, monthly rebalance.
- **(c) beta_neutral** — long − `k`·short, `k = β_long/β_short` (trailing 60d vs
  SPY, set at prior month-end), so the combined book is ≈ beta-neutral.

## Headline findings (this run, 2026-01-05 → 2026-07-10)

- **Long-only +40%**, dollar-neutral **+27%** net of borrow, beta-neutral **+33%** —
  all crushing SPY (+10%) and the QQQ/XLI blend (+16%). Sharpe 1.3–2.0.
- **The Nebius test bites.** Dropping the single worst short (**NBIS**, which ran
  +130% against the short) *adds +10.2%*; dropping the best (**UPWK**) costs
  −6.8%. The spread across the three fragility runs is **~17% on a 27% return** —
  the *net* L/S is materially **name-fragile**, even though no single name is >8%
  of gross contribution.
- **Borrow drag ≈ 4.1%/yr** on the dollar-neutral book — real, not thesis-breaking.
- **Tier that carried it: Tier 4 silicon chokepoints (+86%)**; Tier 1 gen/fuel
  actually *lagged* (−2%). The long thesis is very tier-differentiated.
- **Short-basket dispersion is HIGH** (~14%/month cross-sectional std) — the shorts
  are a bag of idiosyncratic outcomes, so **balance-sheet-aware selection**
  (short the debt-funded neoclouds, not the whole basket) would have dominated the
  naive equal-weight short. See the CRWV/NBIS pair: long-CRWV/short-NBIS was −59%.

See `outputs/report.txt` for the full numbers and the five-bullet conclusion.
