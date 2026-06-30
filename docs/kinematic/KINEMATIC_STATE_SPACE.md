# Kinematic State-Space Signals from Price Alone

**The generative layer** — model price as a noisy observation of a latent
kinematic state `[level, velocity, acceleration]` and ask, honestly, which of
the states it produces are real signal and which are noise the differencing
operator amplified. Benchmarked against **the integral layer** (EMA/MACD), which
*is* the first integral of price.

> Reproduce: `python experiments/run_kinematic_signals.py`
> Code: `posterioralpha/kinematic/` (filter · signals · regime · backtest · diagnostics)
> Artifacts: `data/kinematic/` (CSVs + `kinematic_diagnostics.png`)

---

## TL;DR — what's real, what's garbage

| Component | Verdict | Evidence |
|-----------|---------|----------|
| **Kalman level / velocity** | Real, clean causal estimate | velocity SNR-vs-smoother 0.47 (naive Δ: 0.12); variance ÷3 vs naive |
| **Kalman acceleration** | **Noise on real prices** | corr to smoother **−0.18**, SNR 0.26; rescued on synthetic (corr 0.98) but not on SPY — real prices aren't constant-acceleration |
| **Velocity → daily momentum** | **Loses, robustly** | net Sharpe −0.77; **worse** than EMA/MACD; negative across the *entire* Q/R sweep |
| **Local-level innovation → reversion** | Weak-real, cost-fragile | OOS IC +0.059; net Sharpe 0.68 @1bp → **0.26 @5bp** |
| **Dynamic Kalman hedge (pair)** | Best result, still doesn't clear the bar | net Sharpe 1.20 @1bp, survives 2008; but 0.20 @5bp, **DSR 0.68 < 0.95** |
| **2-state velocity regime gate** | **Hurts** | gates 78% of OOS into the losing momentum leg; gated 0.06 vs reversion-only 0.68 |
| **Path signatures** | Overfit corner (flagged) | tiny IC +0.024; DSR 0.01 |

**One-sentence read:** the filter is a genuinely better *estimator* of the
latent drift than finite differencing, but at daily frequency on a liquid index
the only thing that drift estimate is good for is **fading** it, not following
it — and the acceleration state is differencing noise that the filter suppresses
but cannot resurrect. The single robustly-positive use of the state-space
machinery is the **adaptive hedge ratio**, and even that fails a deflated-Sharpe
test once you count how many variants were tried.

---

## Data

| Item | Value |
|------|-------|
| Instruments | SPY (1993–2009), QQQ (1999–2009), GLD (2004–2009) — bundled `fresh_*` closes |
| Single-name study | SPY, 4,264 days; train/test split 50/50 at 2001-07-11 |
| Pair study | SPY~QQQ overlap, 2,722 days (1999–2009) |
| Frequency | Daily; everything operates on **log price** (scale-invariant, velocity ≈ drift) |

**Honest box.** History ends 2009-12-31, so the pair test is dominated by — and
usefully stress-tested on — the 2008 crisis. A single liquid index means tiny
effective breadth; read *shapes and signs*, not headline Sharpes. Costs are
charged on turnover; index ETFs are cheap, so the default column is 1 bp with a
5 bp stress column shown throughout.

---

## Non-negotiable rigor (how the traps were closed)

- **Strict causality.** Every tradeable quantity is a *filtered* (`a_{t|t}`) or
  *one-step-predicted* (`a_{t|t-1}`) state, computed from data ≤ t. The RTS
  **smoother** (`a_{t|T}`, conditions on the future) is physically segregated —
  it is a *separate method* returning a *separate array*, never stored on the
  result object, and is used only for plots and the noise-amplification
  reference. The causality contract is documented at the top of `filter.py`.
  - We also hand-rolled the regime HMM's forward pass, because `hmmlearn`'s
    `score_samples` returns the forward–**backward** posterior (it peeks at the
    future); our `trend_prob` uses a forward-only `α̂_t`.
  - The pair trade uses the **innovation** spread, not the contemporaneous
    filtered residual (which the filter shrinks to ≈0 by construction and would
    fabricate a Sharpe of ~7.8). P&L is the *next-day* hedged return with beta
    formed at t.
- **Walk-forward.** `(q, r)`, EMA spans, regime params, and OLS beta are all fit
  on train only. The velocity signal additionally gets a 5-fold expanding
  walk-forward where Q/R is refit every fold (`walk_forward`).
- **Net of costs.** Gross *and* net (1 bp and 5 bp) reported everywhere, with
  annualised turnover.
- **Honest diagnostics.** PnL concentration, effective-bet counts, a Q/R
  sensitivity sweep, and a deflated Sharpe over every variant tried.

---

## 1 — Kinematic filter

Constant-acceleration (white-noise-jerk) transition, scalar observation noise:

```
state x = [level, velocity, acceleration]      F = [[1,1,½],[0,1,1],[0,0,1]]
y_t = level_t + noise(r)                        Q = q · Qjerk   (single scalar q)
```

The whole process noise is one scalar `q` (jerk spectral density). With one `r`
the model has **two parameters**, MLE-fit on train via the prediction-error
decomposition — which also makes the Q/R sweep a clean 1-D object.

```
TRAIN MLE → q=1.43e-06  r=7.71e-05  q/r=0.0185
OOS standardised innovation: mean +0.00, std 1.24, lag-1 AC +0.28
```

The innovation std ≈ 1.24 (not 1.0) and lag-1 autocorrelation +0.28 say the
constant-acceleration model is **mildly misspecified** on real SPY — a first
honest flag that the latent kinematics are an approximation, not the truth.

---

## 2 — Noise amplification: does the filter rescue the derivatives?

The point of the exercise. Differentiating amplifies high-frequency noise; each
order multiplies it. The question is whether the filter's optimal smoothing buys
back enough to make velocity — and especially **acceleration** — usable.

### (a) Synthetic ground truth (the model is *exactly* right)

| order | corr to truth | SNR | 
|-------|--------------:|----:|
| velocity · Kalman | 1.00 | 3.4e7 |
| velocity · naive Δ | 1.00 | 1.9e6 |
| **accel · Kalman** | **0.98** | **29.6** |
| **accel · naive ΔΔ** | **0.23** | **0.058** |

When the data really are constant-acceleration, the filter **rescues
acceleration** decisively (corr 0.98 vs 0.23; SNR ×500). So the machinery works.

### (b) Real SPY (smoother = low-noise reference, diagnostic only)

| order | var of estimate | corr to smoother | SNR vs smoother | lag-1 AC |
|-------|----------------:|-----------------:|----------------:|---------:|
| velocity · Kalman | 1.1e-4 | **0.65** | **0.47** | +0.80 |
| velocity · naive Δ | 2.0e-4 | 0.39 | 0.12 | −0.08 |
| **accel · Kalman** | 1.0e-7 | **−0.18** | 0.26 | +0.73 |
| **accel · naive ΔΔ** | 4.3e-4 | 0.11 | 0.004 | −0.50 |

Variance shrink (filter vs naive): **velocity ×3, acceleration ×104.**

**The honest negative result.** On real prices the filter clearly improves
*velocity* (correlation to the smoother 0.65 vs 0.39; SNR ~4× the naive). But
the filtered **acceleration does not track the smoother at all** (corr −0.18) —
the filter suppresses the differencing noise by 100× yet what survives is not a
coherent latent acceleration, because **real SPY is not a constant-acceleration
process** (§1's misspecification, now cashed out). The strong negative lag-1 AC
of naive ΔΔ (−0.50) is the textbook signature of pure differencing noise; the
filter removes that but replaces it with an over-smoothed series that is its own
artefact, not signal. Acceleration is garbage here, and no amount of filtering
rescues a derivative the underlying process doesn't actually have.

---

## 3 — Local-level innovation as a mean-reversion signal

Model `price = level + noise`, level a random walk. The one-step innovation
`print − E[level | past]`, standardised and negated, is the reversion tilt.

```
MLE q/r = 25  → on daily closes the filter trusts the print (little measurement
                noise), so the innovation ≈ the 1-day return: this signal is, in
                substance, short-horizon return REVERSAL.
OOS IC (vs next-day return) = +0.059
```

| horizon (d) | 1 | 2 | 3 | 5 | 10 | 20 |
|-------------|---|---|---|---|----|----|
| IC | 0.059 | 0.054 | 0.065 | 0.051 | 0.068 | 0.042 |
| hit-rate | 0.514 | 0.505 | 0.517 | 0.508 | 0.512 | 0.505 |

Reversion backtest (OOS): **gross 0.79 · net@1bp 0.68 · net@5bp 0.26**, ann
turnover 192×.

**Verdict: weak-but-real, cost-fragile.** A genuine OOS IC that decays slowly,
but it *is* daily mean-reversion in disguise — hit-rate barely over 51% and a
192× turnover that surrenders two-thirds of the Sharpe by 5 bp. Tradable only
where costs are genuinely ~1 bp.

---

## 4 — Dynamic Kalman hedge ratio (SPY~QQQ) vs static OLS

State `[alpha, beta]` random walk, `logSPY = alpha + beta·logQQQ + noise`.
Static benchmark: OLS beta fit on train and frozen.

```
static OLS beta = 0.32 (frozen)
adaptive beta_t : 0.51 → 0.81  (mean OOS 0.62)
```

The relationship genuinely drifts (the figure's bottom-left panel) — a frozen
0.32 is simply *wrong* for most of the OOS period.

**Spread stationarity (OOS).** A hedge that still holds has a small,
non-drifting spread with a short half-life:

| spread | OOS std | OOS \|mean\| | half-life (d) |
|--------|--------:|-------------:|--------------:|
| static OLS | 0.120 | 0.157 | 174 |
| adaptive Kalman (innov) | 0.0068 | 0.0001 | — (mean ≈ 0) |

The static spread drifts with a mean of 0.16 and a 174-day half-life — it has
**broken**: it is no longer a mean-reverting residual, just a wandering mispriced
leg. The adaptive spread stays centred at zero.

**Causal reversion backtest** (signal known at close t; P&L from next-day hedged
return with beta formed at t):

| spread | gross | net@1bp | net@5bp | ann turnover |
|--------|------:|--------:|--------:|-------------:|
| static OLS | 0.17 | 0.14 | 0.05 | 36× |
| **adaptive Kalman** | **1.45** | **1.20** | 0.20 | 208× |

Net PnL by year — **the adaptive hedge survives 2008, the static one stalls:**

| | 2004 | 2005 | 2006 | 2007 | **2008** | 2009 |
|--|-----:|-----:|-----:|-----:|---------:|-----:|
| static OLS | −0.00 | +0.04 | −0.04 | +0.12 | +0.05 | −0.05 |
| adaptive Kalman | −0.02 | +0.13 | +0.01 | +0.10 | **+0.19** | +0.13 |

**Verdict: the one place the state-space machinery clearly pays.** The adaptive
beta keeps the pair hedged as the relationship moves and earns through the
crisis where the static hedge is left holding a mispriced, drifting spread. The
honest asterisk: 208× turnover means the edge is real at 1 bp but mostly gone by
5 bp, and the sample is one pair over eleven years.

---

## 5 — Regime gate (2-state HMM on filtered velocity)

A causal 2-state Gaussian HMM on the filtered velocity, blending momentum (trend
state) and reversion (revert state) by `P(trend | velocity ≤ t)`.

```
emission mean velocity:  trend +8.7e-4   revert −7.9e-4
self-persistence:        trend 0.98      revert 0.92
mean OOS P(trend) = 0.78
```

| strategy | gross | net@1bp |
|----------|------:|--------:|
| momentum only | −0.73 | −0.77 |
| reversion only | +0.79 | +0.68 |
| **regime-gated** | +0.13 | **+0.06** |

**Verdict: the gate hurts.** It labels 78% of the OOS window "trend" (by
velocity magnitude) and so routes most days into the losing momentum leg,
dragging the good reversion signal down to ~zero. The lesson is specific and
honest: **velocity-magnitude regimes are not momentum-vs-reversion-profitability
regimes** at this frequency. A regime layer built on the wrong axis is worse than
no layer.

---

## 6 — Integral baseline (EMA/MACD) and the signature corner

EMA and MACD *are* the first integral of price — the natural layer-1 competitor
to the Kalman velocity.

| signal | IC | gross | net@1bp | ann turnover |
|--------|---:|------:|--------:|-------------:|
| Kalman velocity | −0.053 | −0.73 | −0.77 | 82× |
| EMA(50) | −0.030 | −0.14 | −0.17 | 42× |
| MACD(12,26,9) | −0.040 | −0.29 | −0.32 | 36× |
| path-signature ⚠ | +0.024 | +0.24 | +0.22 | 38× |

**Verdict: the Kalman velocity does *not* beat a plain EMA/MACD — it is the
worst of the trend trio.** All three lose (daily index trend-following is the
wrong sign), but the filter's extra machinery buys nothing over a 50-day EMA. The
path-signature is the only positive trend number, which is exactly why it is
flagged: depth-2 signatures are a rich feature space on a short single-asset
sample, and a +0.02 IC there is a data-mining warning, not a result.

---

## Rigor diagnostics

**Walk-forward velocity (Q/R refit each of 5 folds):** gross −0.63, net@1bp
−0.68, net@5bp −0.85. Losing OOS, refit honestly, concentrated in the 2008 fold.

**Concentration / breadth (adaptive pair, the headline positive):** top-10 days
= 31% of net PnL, Gini\|pnl\| 0.61; participation N_eff 381 of 1,361 days;
autocorr-adjusted independent bets ≈ 1,666. The result is spread across years
but a third of it lives in ten days — fragile, as expected for one pair.

**Q/R sensitivity sweep (velocity net Sharpe vs q/r):**

| q/r | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 1e-1 | 1e0 |
|-----|-----:|-----:|-----:|-----:|-----:|----:|
| net Sharpe | −0.14 | −0.35 | −0.53 | −0.73 | −0.91 | −1.11 |

Monotonically negative across **five orders of magnitude** — the velocity-
momentum failure is not a curve-fit to one noise ratio, it is structural. (A
real edge would show a plateau; this shows a slope.)

**Deflated Sharpe (Bailey–López de Prado), n_trials = 7 variants tried:**

| variant | Sharpe (ann) | DSR |
|---------|-------------:|----:|
| adaptive Kalman pair | 1.20 | **0.68** |
| innovation reversion | 0.68 | 0.17 |
| path-signature ⚠ | 0.22 | 0.01 |
| EMA(50) | −0.17 | 0.00 |
| MACD | −0.32 | 0.00 |
| Kalman velocity | −0.77 | 0.00 |
| walk-forward velocity | −0.68 | 0.00 |

**Nothing clears DSR > 0.95.** The best result (the adaptive pair) reaches only
0.68 once the seven variants are counted as the multiple-testing family. This is
the most important line in the document: read honestly, the kinematic layer
produced **one promising mechanism (adaptive hedging) and a pile of
noise-amplified trend signals**, and even the promising one is not, on this
sample, statistically distinguishable from luck after deflation.

---

## What to take forward

1. **Use the filter as an estimator, not a trend oracle.** Filtered *velocity* is
   a real, low-noise read on drift; the value is in *conditioning* (regime,
   sizing, hedging), not in trading its sign at daily frequency.
2. **Drop acceleration.** On real prices it is differencing noise the filter
   suppresses but cannot make coherent. Keep it only on processes that genuinely
   have a constant-acceleration component.
3. **The adaptive hedge ratio is the keeper** — it is the one place where letting
   a latent parameter drift demonstrably beats the static benchmark, including
   through the 2008 stress. Next step: a basket of pairs (kills the
   single-sample / breadth problem) and an explicit turnover penalty in the q
   calibration (208× is the binding constraint, not the alpha).
4. **Don't gate on the wrong axis.** A regime layer has to separate states by the
   thing you're switching between (momentum vs reversion *profitability*), not by
   a convenient observable (velocity magnitude).
