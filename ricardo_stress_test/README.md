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
| `fundamentals.py` | **(iter 2)** Point-in-time balance-sheet data → a rules-based financing-**fragility** score → tilted short weights. |
| `portfolio.py` | Equal-weight books, the variants (incl. the balance-sheet-aware short), borrow haircut, rolling beta. |
| `analysis.py` | Metrics, per-name contribution, fragility index, dispersion, pairs, tiers. |
| `charts.py` | The 8 diagnostic charts. |
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

## The variants

- **(a) long_only** — the long book alone.
- **(b) dollar_neutral** — +100% long / −100% short, equal-weight, monthly rebalance.
- **(c) beta_neutral** — long − `k`·short, `k = β_long/β_short` (trailing 60d vs
  SPY, set at prior month-end), so the combined book is ≈ beta-neutral.
- **(d) dollar_neutral_bsaware** *(iter 2)* — same 100/100 structure, but the short
  leg is weighted by **ex-ante financing fragility** instead of equal weight.
- **(e) dollar_neutral_riskparity** *(iter 2)* — both legs weighted by
  **inverse trailing volatility** (equal *risk* contribution, not equal dollars);
  60d vol, set at prior month-end, no lookahead.

## Iteration 2 — "fit the idea, not the data"

Iteration 1's conclusion #5 was that *balance-sheet-aware* short selection should
beat the naive equal-weight short (you were short the wrong neocloud — NBIS ran
+130% but sits on net cash; CRWV is the debt-funded burner). Iteration 2 tests
that **idea** without fitting the **data**:

- **Fragility score** = simple average of three **financing/solvency** percentile
  ranks — *net-debt/revenue*, *cash-burn ÷ cash pile*, *total-debt ÷ cash*.
  Deliberately **no profitability** term (that's business quality, not financing
  fragility — NBIS is the reason: deeply lossmaking yet net-cash, so *not* fragile).
- **No return-fitting, no free parameters.** Ranks → simple average → short weight
  **proportional** to fragility. No temperature, no optimizer, no metric weights.
- **No lookahead.** Fundamentals are point-in-time as of the quarter public by the
  backtest start (a 45-day filing-lag rule picks the 2025-09-30 quarter), pulled
  from Yahoo's `fundamentals-timeseries` endpoint, and the tilt is held **static**
  (a structural characteristic, not a timing signal).

**The honest result:** the tilt correctly down-weighted NBIS (net cash → shorted
*less*) and loaded the debt-funded burners (CRWV, APLD, WULF → shorted *more*) — but
it **did not help this window** (dollar-neutral +27.2% → +24.9%). Rank-correlation
between fragility and realized short success was ≈ **0**. In H1 2026 the financially
*fragile* neoclouds **rallied** on the AI-capex bid while the financially *sturdy*
labor-arb names (UPWK, FVRR) were the shorts that actually worked. So the short
leg's H1 problem was **direction, not weighting** — a genuine finding, not a
curve-fit. The infrastructure is generalizable: point it at any short book and it
produces the same ex-ante fragility tilt.

### So what *does* work? Weight by risk, not dollars.

If balance sheet didn't help, what's a better weighting that stays simple and
general? Go back to the *actual* failure mode: iteration 1 found the book was
"hostage to 1–2 names," and those names (NBIS, WULF, CRWV) are the
**highest-volatility** names in the universe. Equal *dollar* weight hands a
60%-vol neocloud the same notional as a 25%-vol staffing name — i.e. equal
dollars = wildly *unequal risk*. The fix isn't a better return forecast (that's
curve-fitting); it's **equalizing risk contribution → inverse-volatility
weighting**: weight ∝ 1/(trailing 60d vol), set at each month-end, no lookahead,
no fundamentals, no tuning.

It wins on every axis this window (dollar-neutral, with borrow):

| Short/both-leg weighting | YTD | Ann vol | Sharpe | Max DD |
|---|---|---|---|---|
| equal weight | +27.2% | 40.7% | 1.27 | −20.5% |
| balance-sheet-aware | +24.9% | 41.7% | 1.16 | −22.1% |
| **inverse-vol (risk parity)** | **+33.0%** | **39.4%** | **1.52** | **−16.8%** |

Why it works *without forecasting*: inverse-vol shorts the low-vol labor-arb names
*more* (RAND, ADEN, RHI) and the hyper-vol neoclouds *less* (NBIS/IREN/CIFR ~4%
vs 6.7% equal). The low-vol staffers were exactly the shorts that worked — the
weighting leaned into them purely to balance risk, not because it predicted them.
And by capping any single name's risk share it *directly cures* the one-name
fragility from iteration 1's bullet 1. This is the recommended default: **size by
risk, express the thesis in what you include, not in dollar concentration.**

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
  are a bag of idiosyncratic outcomes. See the CRWV/NBIS pair: long-CRWV/short-NBIS
  was −59%.
- **Balance-sheet-aware short (iter 2):** correctly identified the fragile vs sturdy
  names ex-ante, but the tilt *cost* 2.4% this window because fragile ≠ underperformer
  in H1 2026 — the neoclouds rallied on AI-capex euphoria. Fragility as a *short
  timing* signal was neutral-to-negative here; as a *risk-management* signal (avoid
  shorting cash-rich names to size like debt-funded ones) it is still the right frame.

See `outputs/report.txt` for the full numbers and the seven-bullet conclusion.
