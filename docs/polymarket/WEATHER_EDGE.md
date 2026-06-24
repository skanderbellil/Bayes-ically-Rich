# Weather edge — city max-temperature, forecast vs market

> **Short answer (corrected 2026-06-24): not a deployable edge.** The original
> "+0.13 per contract, generalises across 5/6 cities" headline was an artifact of
> a **forecast-lead leak**: the backtest paired a market price from 24–48h before
> settlement with a *same-day* model forecast it could not have had at decision
> time. Once the forecast is lead-matched to the decision, the edge collapses from
> **+0.127/event to +0.035/event** (95% CI [+0.004, +0.066]), only **2 of 6**
> cities clear zero, and it is fragile across the threshold sweep. It is now
> "not distinguishable from a marginal, regime-specific residual."

## The bug: a forecast-lead leak (not ERA5 lookahead)

The strategy decides `--lead-hours` (24–48h) before settlement and takes the
market price as-of that moment — correctly, no price lookahead. But the forecast
came from `point_in_time_forecast(target)`, which pulled Open-Meteo's
*historical-forecast archive* with `start_date=end_date=target` and no lead
parameter. That archive returns the **most-recent (≈same-day) run** for a past
date — a forecast sharper than anything a trader had 24–48h earlier.

The author avoided ERA5 reanalysis (so this is *not* the gross "archive = the
answer" hindsight — measured forecast MAE is 0.67 °C, not ~0, with real station
biases). The subtler trap: the archive's default series is the **short-lead run**,
not the run available at the decision time.

### Lead-skill curve (5-model mean daily Tmax vs ERA5, pooled MAE, °C)

| forecast used | pooled MAE | London | NYC | Paris | Tokyo | Miami | Chicago |
|---|---:|---:|---:|---:|---:|---:|---:|
| **lead-0** (what the buggy code used) | **0.67** | 0.49 | 0.89 | 0.78 | 0.69 | 0.51 | 0.68 |
| prev-day-1 (honest 24h-ahead) | 0.87 | 0.73 | 1.16 | 0.80 | 0.87 | 0.61 | 1.06 |
| prev-day-2 (honest 48h-ahead) | 1.04 | 0.95 | 1.47 | 0.78 | 1.01 | 0.71 | 1.31 |

The honest forecast a trader had 24–48h out is 30–55% less accurate. With 1 °C
buckets, that gap is the whole edge.

## Control vs honest (6 cities, 24h decision, edge ≥ 5%, 100 bps)

| | lead-0 control (leaks ~1d) | lead-matched (honest) |
|---|---:|---:|
| PnL/event | **+0.127** | **+0.035** |
| 95% CI | [+0.093, +0.161] | [+0.004, +0.066] |
| hit rate | 37% | 26% |
| cities clearing zero | 5 of 6 | **2 of 6** (Paris, Tokyo) |
| threshold sweep | all clear zero | only 5% & 15% clear zero |

~72% of the measured edge was the leak. The residual is threshold-fragile, fails
in London (the original study city), and the calibration gap nearly closes (honest
top-bin: model 0.365 / realized 0.334 — the model is now slightly *over*confident).

## The fix (in code)

`point_in_time_forecast(target, city, lead_days)` now takes a `lead_days`:

* `lead_days=0` — the lead-0 archive run (sharp, **leaks** forward info). Kept
  only to reproduce the original (wrong) numbers.
* `lead_days>=1` — the run issued ~`lead_days` days earlier, reconstructed from
  the hourly `temperature_2m_previous_day{N}` series (max over the local day).
  This is the honest, no-lookahead forecast.

`run_polymarket_weather_backtest.py --forecast-lead-days` defaults to **auto** =
`ceil(lead_hours/24)`, so the forecast lead always matches the decision lead.

```bash
# honest, lead-matched (default):
python experiments/run_polymarket_weather_backtest.py --cities london,paris,nyc,tokyo,miami,chicago
# reproduce the original leaked result for comparison:
python experiments/run_polymarket_weather_backtest.py --cities london,paris,nyc,tokyo,miami,chicago --forecast-lead-days 0
```

## Why the archetype was still worth testing

`STRATEGY_SYNTHESIS` found the only surviving trade was the macro favorite–longshot
bet, because macro markets are scheduled, objectively resolved, and draw volume
regardless of outcome — no volume-selection artifact. Daily city-temperature
markets share those properties and are uncorrelated with the Fed cycle, so the
*idea* was sound. The execution leaked, and once corrected the edge is marginal.

## Verdict

**Not deployable.** The honest edge (+0.035/event) is too small, too
threshold-fragile, and too concentrated (2/6 cities) to size, and it fails in the
original study city. The lesson generalises: with Open-Meteo's historical-forecast
archive you **must** pin the forecast lead to the decision lead, or every
weather-market backtest will quietly read ~1 day into the future. The lead-matched
machinery is now in place if a future study wants to test a genuinely longer-lead
or different-season hypothesis.
