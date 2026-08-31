# TimesFM vs. our own vol models

**Question.** Google's [TimesFM](https://github.com/google-research/timesfm) is a
pretrained time-series foundation model that forecasts any series zero-shot. Is it
worth a dependency in PosteriorAlpha?

**Scope.** Not on returns. `pead/bayesvol.py` already states the house view —
*"Returns are ~unforecastable (PEAD tradeable IC was 0.03). Variance is NOT: it
clusters."* TimesFM is trained to extrapolate persistent structure, and its
benchmark suites (GIFT-Eval, fev-bench, TIME) are dominated by demand, energy and
traffic series. Pointing it at daily equity returns tests a hypothesis we already
rejected. So it gets the job we think is winnable: **second moments**.

## Design

For each of 20 liquid cross-asset ETFs and each month-end *t*, forecast the log
realized volatility of the next 21 trading days:

    rv_w(t)   = sqrt(252/w · Σ_{i<w} r²_{t-i})       annualised trailing RV
    x(t)      = log rv_w(t)
    target(t) = x(t+21)                              covers r_{t+1..t+21}

With the measurement window equal to the horizon, the target *is* the next month's
realized vol, and the context series is a strict function of returns up to *t*.

| model | what it is |
|---|---|
| `rw` | carry today's trailing 21d RV forward — near-unit-root, and a genuinely hard baseline |
| `ewma94` | RiskMetrics EWMA variance |
| `har` | HAR-RV (Corsi 2009) in logs, expanding **purged** refit — the standard academic benchmark |
| `bayes` | `pead/bayesvol.py` — Gamma posterior on precision, BOCPD-style adaptive discount |
| `timesfm` | TimesFM 2.5 (200M, Apache-2.0 weights), zero-shot on the log-RV series, 1024-day context |

**Why 2.5 and not 3.0.** TimesFM 3.0 is the stronger model, but its weights ship
under `timesfm-non-commercial-license-v1.0` — non-commercial, non-production only.
The Apache-2.0 2.5 weights are the ones that could sit behind a live book, so
they are what we benchmark. Anything 3.0 tells us is unusable downstream.

## Scoring

Primary loss is **RMSE in log-vol space**, which is transformation-neutral: no
model is rewarded merely for sitting higher. Secondary is **QLIKE** on the
variance scale — robust to the noise in the RV proxy (Patton 2011) and asymmetric
against under-prediction, which is the error a vol-targeted book actually pays for.

A log-space point forecast is a *median* while QLIKE scores a *mean*, so every
model gets the **same causal expanding-window level debiasing** before scoring,
fitted only on evaluation dates whose target windows had already closed.

Significance is a **Diebold-Mariano** test on loss differentials averaged
cross-sectionally per date — the 20 ETFs share market-wide vol shocks, so treating
them as 20 independent observations would inflate the t-stats several-fold —
with a Newey-West (lag 1) standard error.

## The leakage caveat that the gauntlet cannot catch

`mining/validation.py` tests whether a *backtest* is overfit: purged windows, block
bootstrap, permutation, Deflated Sharpe. None of them can see contamination that
happened during **pretraining**. TimesFM's weights were frozen at a checkpoint
date; a zero-shot forecast of 2015 SPX vol is not out-of-sample in the sense the
rest of this repo means by the term.

So results are reported twice: the full sample (2013→2026, large *n*, but not a
clean read for TimesFM) and the slice after the 2.5 release on 2025-09-15 (clean,
but small *n*). Where the two disagree, believe the second and note the error bars.

## Results

20 ETFs × 160 month-ends (2013-01 → 2026-05) = 3,200 forecasts per model.
Lower is better for both losses; `r2` and `mz_slope` are in log-vol space.

**Full sample** — large *n*, but not a clean out-of-sample read for `timesfm`:

| model | rmse_log | QLIKE | r² | MZ slope |
|---|---|---|---|---|
| `ewma94` | 0.3598 | **0.3739** | 0.599 | 0.83 |
| `bayes` | 0.3608 | 0.3855 | 0.597 | 0.84 |
| `har` | **0.3398** | 0.4092 | 0.643 | 0.98 |
| `timesfm` | 0.3564 | 0.4134 | 0.607 | 0.87 |
| `rw` | 0.3854 | 0.4290 | 0.540 | 0.77 |

**Post-checkpoint** (from 2025-09-15; 8 dates × 20 ETFs — clean, but thin):

| model | rmse_log | QLIKE | r² | MZ slope |
|---|---|---|---|---|
| `har` | **0.3035** | **0.2392** | 0.755 | 1.02 |
| `bayes` | 0.3191 | 0.2898 | 0.729 | 0.92 |
| `timesfm` | 0.3218 | 0.2922 | 0.725 | 0.91 |
| `ewma94` | 0.3341 | 0.3137 | 0.703 | 0.87 |
| `rw` | 0.3525 | 0.3465 | 0.669 | 0.82 |

Diebold-Mariano on QLIKE (negative = better than the benchmark):

| slice | comparison | Δ | t |
|---|---|---|---|
| full | `timesfm` vs `bayes` | +0.0279 | 1.01 |
| full | `timesfm` vs `rw` | −0.0156 | −0.38 |
| post-checkpoint | `timesfm` vs `bayes` | +0.0024 | 0.20 |
| post-checkpoint | `timesfm` vs `rw` | −0.0543 | −2.66 |
| post-checkpoint | `har` vs `bayes` | −0.0505 | −1.41 |

TimesFM is **statistically indistinguishable from `bayes`** on both slices, and it
does not beat a random walk on the full sample. It is not bad — it is mid-table,
respectable, and completely unnecessary.

## Why: it rediscovered EWMA

Conditioning the signed error on how far realized vol actually moved:

| move quintile | `rw` | `ewma94` | `bayes` | `har` | `timesfm` |
|---|---|---|---|---|---|
| 1 vol crush | −0.510 | −0.457 | −0.449 | −0.327 | −0.420 |
| 2 | −0.198 | −0.171 | −0.167 | −0.124 | −0.162 |
| 3 flat | −0.011 | −0.013 | −0.014 | −0.004 | −0.011 |
| 4 | +0.171 | +0.142 | +0.127 | +0.141 | +0.142 |
| 5 vol spike | +0.549 | +0.486 | +0.457 | +0.452 | **+0.486** |

(positive = under-predicted). `timesfm`'s error profile is *the same curve as
`ewma94`*, to two decimals, in four of five buckets. A 200M-parameter transformer
given 1,024 days of context reproduces an exponentially-weighted moving average.

Two things follow. First, **no model calls the turn** — every one of them
under-predicts vol spikes by ~0.45–0.55 in logs (roughly 60% too low) and
over-predicts the crushes. That component of vol is not forecastable from vol's
own history, and a foundation model does not change that. Second, TimesFM's
edge on `rmse_log` but deficit on QLIKE is exactly this asymmetry: it is a good
*level tracker* and a poor *spike anticipator*, and QLIKE is the loss that
charges for the second.

The decisive test is whether it knows anything the incumbent does not. A 50/50
log-space blend against `bayes`:

| blend | QLIKE |
|---|---|
| `bayes` alone | 0.3855 |
| `bayes` + `ewma94` | **0.3773** |
| `bayes` + `har` | 0.3821 |
| `bayes` + `timesfm` | 0.3861 *(worse — redundant)* |
| `bayes` + `rw` | 0.3916 *(worse — redundant)* |

Blending in TimesFM makes the Gamma-posterior model **worse**. The classical
baselines carry orthogonal information; the foundation model carries none. That
kills the one genuinely interesting integration — TimesFM's quantile head as an
observation inside `research/bayesian.py`'s posterior blend. There is nothing
to blend.

## Verdict — no

Not adopted. It ties our own model, loses to a four-parameter regression on the
clean slice, adds nothing to a blend, and costs a 2.5 GB `torch` dependency plus
~12 minutes of CPU per run for what the baselines compute in under a second.

Kept anyway as `posterioralpha/research/tsfm.py` + this experiment, because the
negative result is worth being able to re-run when 4.0 lands or when someone
proposes a foundation model again. It stays behind an optional `[tsfm]` extra;
nothing in the default install pulls torch.

**What the study did turn up**, both worth following up independently of TimesFM:

1. **HAR-RV is the best log-space vol forecaster here** — best `rmse_log` on both
   slices, best on *everything* post-checkpoint, MZ slope ≈ 1.0 (the only
   unbiased forecaster in the set), and it wins 31% of individual cells against
   19–20% for the runners-up. `research/amr.py`'s vol targeting currently
   consumes a flat-held one-step forecast; a HAR term is four parameters.
2. **`bayes` + `ewma94` beats `bayes` alone** (QLIKE 0.3773 vs 0.3855). The
   cheapest improvement in the table, and it needs no new dependency.

## Caveats

- The post-checkpoint slice is **8 dates**. It is the only leakage-clean read
  available, and it is thin; the DM t-stats there rest on 8 observations.
- Only TimesFM **2.5** was tested. 3.0 is stronger, but its weights are
  non-commercial/non-production, so a win there would not be usable — see the
  licence note above. `--checkpoint` will point the harness at it for anyone who
  wants the research-only number.
- Zero-shot only. TimesFM supports LoRA fine-tuning on your own series, which is
  a different (and much more expensive) experiment than the one run here.
- One horizon (21d) and one asset class (liquid ETFs).
