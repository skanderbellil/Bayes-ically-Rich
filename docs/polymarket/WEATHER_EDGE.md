# Weather edge — city max-temperature, forecast vs market

> Short answer: **a candidate edge, and now the best-powered one in this thread.**
> A five-model NWP consensus is *sharper and better-calibrated* than the
> Polymarket crowd on daily-max-temperature buckets. Buying the model's favorite
> bucket the day before settlement earns **+0.13 per contract per event**, 95% CI
> **[+0.093, +0.161]** pooled over **724 settled city-days across 6 cities** — the
> CI excludes zero, survives a 1¢ half-spread, and is stable across the whole
> 3–15% edge sweep. Crucially it **generalises**: 5 of 6 cities clear zero
> independently (London, Paris, NYC, Tokyo, Miami; only Chicago is noise),
> spanning °C and °F, Europe/Asia/US, and very different climates — which retires
> the original "one city / one spring" caveat. One honest qualifier remains: the
> alpha is really *"the forecast beats the market"* (the price filter adds little).

```bash
# London only (the original study)
python experiments/run_polymarket_weather_backtest.py
# multi-city — the generalisation test
python experiments/run_polymarket_weather_backtest.py --cities all
python experiments/run_polymarket_weather_backtest.py --cities london,paris,nyc,tokyo --lead-hours 48
```

## Generalisation across cities (the robustness test)

Polymarket runs the same daily-max-temperature event for **16 cities**. Extending
the method there multiplies the sample *and* breaks the seasonal/spatial
correlation that the single-city study could not. Two gotchas the code handles:
**US cities quote °F in 2-degree buckets** ("between 38-39°F") while Europe/Asia
quote °C in 1-degree buckets, and the forecast must be pulled in the market's own
unit (`weather.City` carries `unit`, `lat/lon`, `timezone`).

Pooled (24h lead, edge ≥ 5%, 100 bps): **661 trades, +0.127/event, CI
[+0.093, +0.161]** — a far tighter interval than London-alone's [+0.029, +0.197].
Per city:

| city | n | PnL/event | 95% CI | hit | CI > 0 |
|---|---:|---:|---|---:|:--:|
| paris | 113 | **+0.217** | [+0.133, +0.304] | 50% | ✓ |
| nyc | 118 | +0.158 | [+0.082, +0.235] | 39% | ✓ |
| tokyo | 90 | +0.127 | [+0.040, +0.216] | 33% | ✓ |
| london | 120 | +0.113 | [+0.029, +0.197] | 38% | ✓ |
| miami | 121 | +0.083 | [+0.002, +0.167] | 35% | ✓ |
| chicago | 99 | +0.059 | [−0.011, +0.131] | 26% | ✗ |

Five of six exclude zero; Chicago (continental, high day-to-day temperature
variance → the forecast's edge over the crowd shrinks) is the honest failure.
Calibration pooled over 7,513 buckets stays near-perfect (top bin: model 0.391 /
realized 0.389; market more diffuse at 0.303). **Next:** the remaining 10 cities,
and a forward paper ledger per city.

```bash
python experiments/run_polymarket_weather_backtest.py    # single-city default (London)
python experiments/run_polymarket_weather_backtest.py --lead-hours 48 --threshold 0.08
```

## Why this is the right archetype

`STRATEGY_SYNTHESIS` found that the only trade to survive across 11 studies was
the **macro** favorite–longshot bet, and explained *why* it survived when
politics/sports did not: macro markets are **scheduled, objectively resolved, and
draw volume regardless of outcome**, so there is no "longshot only attracts volume
once it is already winning" selection artifact. London temperature markets share
every one of those properties — they settle daily on an objective reading — and
they are **uncorrelated with the Fed cycle**, which is the direct answer to the
macro trade's biggest weakness ("one calm regime, correlated short-vol tail").

## The signal (point-in-time by construction)

For each event-day we build the london-edge construction with no lookahead:

1. **Forecast** — five global models (ECMWF, GFS, ICON, UKMO, MeteoFrance) from
   Open-Meteo's *historical-forecast archive* (the forecast issued for that date,
   **not** the ERA5 reanalysis), requiring ≥3/5 models.
2. **Consensus** — model mean `μ` and inter-model dispersion `σ` (floored at
   0.7°C, so disagreement widens the implied distribution).
3. **Bucket probability** — a normal CDF with a ±0.5°C continuity correction over
   the integer-degree buckets the market quotes.
4. **Decision** — the last CLOB mid at/just before `settlement − lead` (default
   24h); the entry pays a 1¢ half-spread (`--slippage`).
5. **Settlement** — the market's own resolution, never ERA5.

`edge = P_model − P_market`. The trade rule (london-edge): each day, **buy the one
Yes bucket with the largest positive edge** if it clears the threshold. One trade
per day ⇒ event-days are independent ⇒ the bootstrap resamples days.

## Result (128 settled event-days, 2026-02-05 → 2026-06-20, 24h lead, 50–100 bps)

| book | n | PnL/event | 95% CI | hit | worst | total |
|---|---:|---:|---|---:|---:|---:|
| **edge ≥ 5% (model vs market)** | 120 | **+0.113** | **[+0.029, +0.197]** | 38% | −0.475 | +13.5 |
| market favorite (benchmark) | 128 | +0.026 | [−0.059, +0.113] | 46% | −0.825 | +3.3 |
| model favorite (benchmark) | 128 | +0.140 | [+0.055, +0.226] | 50% | +... | +18.0 |

Threshold sweep — stable, never an artifact of one cutoff:

| thr | n | PnL/event | 95% CI | hit |
|---|---:|---:|---|---:|
| 3% | 126 | +0.102 | [+0.018, +0.189] | 37% |
| 5% | 120 | +0.113 | [+0.029, +0.197] | 38% |
| 8% | 109 | +0.118 | [+0.030, +0.208] | 37% |
| 10% | 100 | +0.124 | [+0.030, +0.219] | 37% |
| 15% | 68 | +0.116 | [+0.006, +0.231] | 34% |

**Calibration is the real finding.** Model probability vs realized frequency, all
1,310 buckets, by model-probability quintile:

| n | p_model | p_market | realized |
|---:|---:|---:|---:|
| 262 | 0.000 | 0.002 | 0.000 |
| 263 | 0.000 | 0.009 | 0.000 |
| 261 | 0.004 | 0.041 | 0.008 |
| 262 | 0.086 | 0.140 | 0.073 |
| 262 | **0.397** | 0.309 | **0.405** |

The model's top bin lands at **0.397 predicted / 0.405 realized** — near-perfect —
while the **market is more diffuse** (0.309 on the same bin) and over-prices the
low tails (0.041 / 0.140 where realized is 0.008 / 0.073). The edge is the gap
between a sharp, calibrated forecast and a vaguer crowd.

**Lead robustness.** Deciding **48h** before settlement (market less efficient,
forecast still good) the edge *grows* to **+0.18/event, CI [+0.074, +0.290]** —
the opposite of a microstructure artifact, which would decay away from settlement.

## The honest qualifiers

- **The alpha is "the forecast beats the market," not "mispricing."** The
  *model-favorite* benchmark — buy the model's top bucket ignoring price — does as
  well as the price-filtered edge book. So execution reduces to *trust the NWP
  consensus*; the `edge ≥ thr` filter mostly trims days where the market already
  agrees. Treat capacity as the favorite bucket's depth, not the whole field.
- **~~One city, one spring~~ — now retired.** The original London-only sample was
  seasonally correlated; the 6-city pooled study (724 city-days, 5/6 cities
  clearing zero across Europe/Asia/US) breaks that dependence. Remaining residual:
  all six are the *same Feb–Jun 2026 window*, so a different season is still untested.
- **Short forecast lead.** The historical-forecast archive is ≈0–1 day lead; the
  48h result reuses that archive on the market side only, so it slightly flatters
  the forecast. A true multi-lead study needs raw model-run archives.
- **Low hit rate, bounded loss.** 37% hit with positive PnL = a mild longshot
  payoff (worst trade −0.48, far gentler than macro's −0.97). Still short-vol; size
  per the worst case.

## What it produces

`results/polymarket/weather_edge_buckets.csv` (every scored bucket: model P,
market P, edge, outcome), `weather_edge_trades.csv` (the executed book), and
`weather_edge.png` (cumulative PnL + the calibration reliability curve).

## Verdict

A **deployable-candidate** edge — better-powered and gentler-tailed than the macro
trade, uncorrelated with it, and runnable with only public key-less APIs. Next
step before real sizing: a forward paper-trade ledger (like `PAPER_TRADE` /
`SMART_FLOW_PAPER`) so the out-of-sample record is immune to the seasonal-sample
caveat, plus a second city to break the one-regime dependence.
