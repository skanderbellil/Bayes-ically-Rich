# Pod hiring policies — can the winner's curse be traded against?

`experiments/run_hiring_policies.py` · QQQ, seed 7, 3 seats, 2017-06-30 → 2026-06-08,
every decision walk-forward, Alpaca ETF costs. Reproduces deterministically
(miner seed = pod seed + review index).

## The question

The first pod run (`run_quant_pod.py`) measured the winner's curse directly:
across the pod's own hires, research Sharpe at hire is *negatively* correlated
with realized live Sharpe,

```
live = 2.07 − 0.73 × research   (n = 21 sleeves with ≥ 60 live days,
                                 r = −0.52, p = 0.016, slope SE = 0.27)
  first  half (2017→2020): slope −0.54  (r = −0.49, p = 0.15, n = 10)
  second half (2020→2026): slope −1.00  (r = −0.57, p = 0.07, n = 11)

mean live Sharpe, top-third by raw research Sharpe:   0.58
mean live Sharpe, bottom-third (predicted best):      1.72
```

So: among candidates that already passed the gate, the *highest* backtest
Sharpe is the *worst* hire. Can the pod exploit this by changing only the
order in which gate-passers fill free seats?

## The policies

| policy | hire order among gate-passers |
|---|---|
| `raw` | highest holdout Sharpe first (status quo; identical to `run_quant_pod.py`) |
| `dsr` | highest Deflated Sharpe Ratio first (theory-driven shrinkage, no meta-fitting) |
| `shrunk` | regress live-vs-research on the pod's own completed sleeves so far; order by predicted live Sharpe (the negative slope is *learned*, not assumed) |
| `median` | start at the median-ranked passer and expand outward; the top pick is hired last |

Gate, seats, retirement rules, and costs are unchanged.

## Results

| policy | sharpe | cagr | max_dd | hires | mean_live_sharpe | haircut_corr |
|---|---|---|---|---|---|---|
| raw | 0.87 | 0.110 | −0.242 | 22 | 1.03 | −0.52 |
| dsr | 0.87 | 0.110 | −0.242 | 22 | 1.03 | −0.52 |
| shrunk | 0.89 | 0.109 | −0.235 | 22 | 1.05 | −0.51 |
| median | **0.98** | 0.105 | **−0.165** | 21 | 1.06 | −0.45 |
| QQQ half-exposure (benchmark) | 0.94 | 0.108 | −0.186 | — | — | — |

`mean_live_sharpe` / `haircut_corr` are computed across hired sleeves with
≥ 60 live days; `haircut_corr` = Pearson corr(research Sharpe at hire,
realized live Sharpe) — the per-policy winner's-curse measure.

### Why raw == dsr (bitwise)

Ordering can only matter when a review produces more gate-passers than free
seats. Of 26 mined reviews, only 10 had ≥ 2 passers — and in those, DSR
changed the top pick **0** times and the full order **0** times. Within a
single review every candidate shares the same holdout length and the same
trial count/variance, so DSR is a monotone transform of holdout Sharpe there:
it can shrink *levels* (and it does, hard — that's its job at the gate), but
it cannot *reorder* a single review's survivors. The hired (date, sleeve)
sets are identical, hence identical books.

### Why shrunk barely improves while median jumps

The hypothesis from the handoff was that shrunk "still hires ~22 and just
reorders". Confirmed, with the sharper mechanism visible in the hire overlap:

- `shrunk` shares **17 of 22** hires with raw. It can only act at the few
  multi-passer reviews, only after ≥ 6 sleeves have completed (≈ 2019), and
  even then it reorders a list whose members mostly all get hired anyway as
  seats free up over subsequent reviews.
- `median` shares only **11 of 21** hires with raw — it picks genuinely
  different sleeves, systematically skipping the top-ranked passer (the
  sleeve the regression says is most luck-loaded), and ends up hiring one
  fewer. Its book also shows the mildest internal winner's curse
  (haircut_corr −0.45) and by far the shallowest drawdown (−0.165, vs
  −0.242 raw and −0.186 for the benchmark — most of the gap is 2022).

## Robustness — is median's win real?

Stationary block bootstrap of the daily books (2000 draws, mean block 63
trading days, the *same* blocks applied to every policy so draws are
comparable):

| | p05 | p50 | p95 |
|---|---|---|---|
| raw / dsr | 0.46 | 0.89 | 1.37 |
| shrunk | 0.48 | 0.92 | 1.39 |
| median | 0.51 | 1.00 | 1.46 |
| QQQ half | 0.50 | 0.96 | 1.45 |

- **P(median > raw) = 0.70** — directionally favorable, far from decisive.
- **P(median > QQQ half-exposure) = 0.58** — a coin flip.

Leave-one-sleeve-out jackknife: median's full-book Sharpe 0.98 stays in
[0.90, 1.07] dropping any single hire (raw: 0.87 in [0.78, 0.97]) — the edge
over raw does not hinge on one lucky sleeve, but it never escapes the
benchmark's bootstrap band either.

## Takeaway

**The winner's curse is real; the "hire the median" fix is not (yet) a
policy.** Defensible conclusions, in order of confidence:

1. **Do not hire the top backtest Sharpe.** The sleeve-level evidence is
   significant (p = 0.016, and the slope steepened in the second half):
   top-third research Sharpe delivered 0.58 live, bottom-third 1.72. This is
   a statement about *avoiding* the extreme right tail of gate-passers, and
   it is the robust part.
2. **DSR is a gate tool, not a selection tool.** It cannot discriminate
   among survivors of the same review by construction here; expecting it to
   fix hiring order was a category error.
3. **Meta-learning the haircut (`shrunk`) is structurally throttled** in
   this pod: with 3 seats, quarterly reviews, and mostly ≤ seat-count
   passers, reordering changes almost nothing. To matter it would need to
   also *veto* hires (shrink the book), not just reorder.
4. **`median` beating everything (0.98, −0.165 DD) is suggestive, not
   established.** One ordering choice on n = 21 hires, P = 0.58 vs the dumb
   benchmark; the honest reading is that its outperformance is concentrated
   in avoiding two or three high-research-Sharpe 2021–22 hires. A real test
   needs other underlyings (SPY), other seeds, and ideally "median" vs the
   simpler rule it proxies for: *cap the research Sharpe you're willing to
   hire* — which is what the negative slope actually recommends.

The pod's larger verdict stands: even the best re-selection policy tested is
statistically indistinguishable from half-exposure QQQ. Selection effort at
the hiring stage cannot rescue a candidate pool whose top end is mostly luck.
