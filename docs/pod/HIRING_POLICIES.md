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
| `inverse` | *lowest* research Sharpe first — the pure exploit of the negative slope (⚠️ hard-codes a fact learned from this sample; the walk-forward-honest versions are `shrunk`/`veto`) |
| `veto` | `shrunk` with teeth: same learned predicted-live ranking, but a passer with predicted live Sharpe ≤ 0 is not hired at all (an idle seat earns cash); unlike reordering, the veto can act at single-passer reviews |

Gate, seats, retirement rules, and costs are unchanged.

## Results

| policy | sharpe | cagr | max_dd | hires | mean_live_sharpe | haircut_corr |
|---|---|---|---|---|---|---|
| raw | 0.87 | 0.110 | −0.242 | 22 | 1.03 | −0.52 |
| dsr | 0.87 | 0.110 | −0.242 | 22 | 1.03 | −0.52 |
| shrunk | 0.89 | 0.109 | −0.235 | 22 | 1.05 | −0.51 |
| median | **0.98** | 0.105 | −0.165 | 21 | 1.06 | −0.45 |
| inverse | 0.96 | 0.097 | **−0.159** | 20 | 1.03 | −0.39 |
| veto | 0.89 | 0.107 | −0.235 | 21 | 1.07 | −0.52 |
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

| | p05 | p50 | p95 | P(> raw) | P(> QQQ half) |
|---|---|---|---|---|---|
| raw / dsr | 0.46 | 0.89 | 1.37 | — | 0.34 |
| shrunk | 0.48 | 0.92 | 1.39 | 0.86 | 0.39 |
| median | 0.51 | 1.00 | 1.46 | 0.70 | 0.58 |
| inverse | 0.49 | 0.97 | 1.43 | 0.67 | 0.51 |
| veto | 0.48 | 0.92 | 1.40 | 0.79 | 0.40 |
| QQQ half | 0.50 | 0.96 | 1.45 | — | — |

- Every anti-curse policy is *directionally* better than raw, none decisively.
- **No policy clears the dumb benchmark: the best, median, is at P = 0.58 — a
  coin flip — and `inverse`, which exploits the slope with full hindsight of
  its sign, sits at P = 0.51, i.e. exactly on the benchmark.**

Leave-one-sleeve-out jackknife: median's full-book Sharpe 0.98 stays in
[0.90, 1.07] dropping any single hire (raw: 0.87 in [0.78, 0.97]) — the edge
over raw does not hinge on one lucky sleeve, but it never escapes the
benchmark's bootstrap band either.

## Is the inverse correlation usable, then?

This is the question the `inverse` and `veto` rows answer, and the answer has
a low ceiling built into the pod's structure:

1. **Selection only binds at 10 of 26 reviews** (those with ≥ 2 passers), and
   runners-up usually get hired a quarter later anyway as seats free up.
   Any ordering policy — however clever — can only shuffle ~half the hires.
2. **The slope's usable content is "avoid the extreme top," and that's it.**
   r = −0.52 means R² ≈ 0.27 on n ≤ 21: the predicted difference between the
   median-ranked and lowest-ranked passer is well inside the noise, which is
   why `inverse` (0.96) does not even beat `median` (0.98) and gives up CAGR
   (0.097 — the lowest-research-Sharpe passers skew toward low-exposure dials).
3. **The learned veto almost never fires.** Predicted live ≤ 0 requires
   research Sharpe > 2.84; that happened once in nine years (21 vs 22 hires).
   A veto strong enough to matter (e.g. "predicted must beat half-QQQ," i.e.
   refuse research Sharpe > 1.55) would cut ~8 of 22 hires — but choosing
   that threshold *now*, on the same 21 sleeves the regression was fit to, is
   curve-fitting; it's the variant to take to SPY / other seeds, not to tune here.
4. **The pool, not the picker, is the constraint.** Gate-passers deliver a
   mean live Sharpe ≈ 1.03 at well under full exposure; no reordering of that
   pool reliably clears half-QQQ (0.94). The correlation is a *guardrail*
   (cap what you'll pay for a backtest), not a source of book alpha.

## Out-of-sample: SPY, and a wider search space

The doc above flagged the next test: other underlyings, and the
capital-sizing variant. Both ran (`--underlying SPY`, `--extended` adds
three short-horizon families — `macro_lag_dial`: 10–60d macro z with a
1–21d reaction lag; `rev_dial`: 2–15d price reversal/continuation;
`lag_vote2`: two macro layers with independent lags — i.e. more short-term
relationships, combinations and lags for the miner to discover).

Book Sharpe by configuration (bench = half-exposure underlying):

| policy | QQQ | SPY | QQQ-ext | SPY-ext |
|---|---|---|---|---|
| raw | 0.87 | 0.75 | 0.79 | 0.68 |
| shrunk | 0.89 | 0.75 | 0.79 | 0.70 |
| median | 0.98 | **0.58** | 0.67 | 0.79 |
| inverse | 0.96 | 0.57 | 0.75 | 0.79 |
| veto | 0.89 | 0.75 | 0.70 | 0.59 |
| sized (raw hires, pred-live weights) | 0.87 | 0.72 | 0.63 | 0.51 |
| benchmark | 0.94 | 0.85 | 0.94 | 0.85 |

P(policy > benchmark), joint block bootstrap: best anywhere is median's
0.58 on the original QQQ run; on SPY everything is ≤ 0.25.

1. **`median`'s QQQ win did not replicate — it was curve-fit.** On SPY it
   is the *worst* policy (0.58 vs raw 0.75, P(median > raw) = 0.09, i.e.
   raw beats it in 91% of bootstrap draws); on QQQ-extended it loses again
   (P = 0.02); on SPY-extended it wins again (P = 0.93). Two up, two down:
   exactly what a noise variable looks like. The same applies to `inverse`.
   `shrunk`/`veto` hug raw everywhere (±0.02) — never harmful, never material.
2. **`sized` is consistently harmful** (QQQ 0.87, SPY 0.72, ext 0.63/0.51;
   P(> raw) ≤ 0.51 in all four configurations). The haircut regression has
   predictive *sign* but not predictive *precision*: using its point
   estimates as capital weights adds estimation error on top of selection
   noise. If sizing can work at all it needs heavy shrinkage toward equal
   weight, not raw regression output.
3. **The wider search space made the pod worse on both underlyings**
   (raw: QQQ 0.87 → 0.79, SPY 0.75 → 0.68; QQQ pool mean live Sharpe
   1.03 → 0.77). More short-term families, combinations and lags = more
   trials = the fixed gate (perm/boot p ≤ 0.25) passes more luck, even
   though the DSR is charged for the extra trials. "Maybe the miner needs
   more to choose from" is precisely the intuition the winner's curse
   punishes.
4. **The sleeve-level curse replicates where the pool has edge, and is
   unstable where it doesn't:** SPY slope −0.63 (p = 0.048), SPY-extended
   −0.48 (p = 0.09) — but QQQ-extended flips to +0.32 (p = 0.57, n = 15).
   The *negative slope* is a property of this pod's gate + pool, not a law;
   any policy hard-coded to it inherits that fragility.

## Takeaway (revised after out-of-sample)

**The winner's curse is real; no hiring policy tested converts it into
book alpha, and "hire the median" failed its out-of-sample test.** Defensible conclusions, in order of confidence:

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
4. **`median` beating everything on QQQ (0.98, −0.165 DD) was sample
   luck.** The out-of-sample section above settles what the bootstrap
   already hinted at (P = 0.58 vs benchmark): on SPY median is the worst
   policy, and its sign flips across the four configurations. The simpler
   rule it proxied for — cap the research Sharpe you'll pay for — survives
   as a guardrail at the sleeve level (top-third research Sharpe delivered
   0.35–0.58 live across QQQ/SPY; bottom-third 1.46–1.72), but no seat-level
   implementation of it beat the benchmark anywhere.
5. **Exploiting the inverse correlation head-on doesn't clear the bar
   either:** `inverse` — the most aggressive use of the slope, with
   hindsight of its sign — lands at P = 0.51 vs the benchmark on QQQ and
   collapses on SPY. Moving the haircut into capital allocation (`sized`)
   made things strictly worse in all four configurations. The remaining
   untested lever is the gate itself (raise the evidence bar / deflate
   holdout Sharpe before the B&H comparison), which shrinks the book
   rather than re-sorting it.
6. **Richer short-term search spaces are not the answer.** Adding lagged
   macro responses, fast reversal, and lagged combination families cut
   raw's book by ~0.08 Sharpe on both QQQ and SPY and degraded the hired
   pool's live quality. The binding constraint is real edge in the
   candidate pool, and widening the funnel dilutes it. Every configuration
   tested — 7 policies × 2 underlyings × 2 search spaces — lost to
   half-exposure buy-and-hold.

The pod's larger verdict stands: even the best re-selection policy tested is
statistically indistinguishable from half-exposure QQQ. Selection effort at
the hiring stage cannot rescue a candidate pool whose top end is mostly luck.

## Epilogue 2: the gate, the panel, and the levered vehicle

Three more levers ran after the out-of-sample verdict (`--gate strict`,
`--with-liquidity` / `--rich-macro`, `--underlying QLD`).

**Tightening the gate is the worst idea tested.** Strict evidence
(perm/boot p ≤ 0.10 + DSR ≥ 0.5) cut QQQ raw from 0.87 to 0.63 and SPY
from 0.75 to 0.34 — fewer hires, more cash, and the surviving hires were
*not* better (QQQ mean live Sharpe 0.86 vs 1.03). Under a negative
research→live relation this is expected with hindsight: demanding stronger
in-sample evidence selects *harder* winners, which is exactly what the
curse punishes. The full selection axis — loose gate ↔ strict gate, every
hire ordering, sleeve sizing — is now mapped, and no point on it beats
half-exposure buy & hold on a 1× vehicle.

**Richer FRED inputs change which dials win, not how well.** Two
enrichments: the Fed net-liquidity complex (WALCL/TGA/RRP/net — the
strongest council seat, previously invisible to the miner) and
`fred_macro_plus.csv` (7 more series: 3m bill, JPY, continued claims,
mortgage rate, CCC OAS, sentiment, permits). Effects flip sign by
underlying — QQQ 0.87 → 0.78 (though liquidity-only cut the drawdown to
−18% at flat Sharpe), SPY 0.75 → 0.82, QLD 0.89 → 0.93 — and nothing
clears its benchmark (best: SPY median 0.85 vs benchmark 0.85, P = 0.52).
More searchable series is the same multiple-testing tax as more families,
paid in a different currency.

**The levered vehicle is where the pod finally beats its own benchmark —
and still loses to plain QQQ.** On QLD (2× QQQ, live data from 2006) the
pod's book clears half-QLD for the first time in any configuration:
raw 0.89 vs 0.86 (P = 0.60), rich-panel 0.93 vs 0.86 (P = 0.70), with
CAGR 15.8–17.6% at −34/−37% DD, and the `sized` overlay finally helps
(0.90, −31% DD). But the honest yardstick over the same 2014–2026 span is
**QQQ buy & hold: Sharpe 0.94, CAGR 19.4%, DD −35%** — more return AND
more Sharpe than every pod variant at the same drawdown. The absolute-
return frontier on this data runs from QQQ B&H (19.4%, −35%) to QLD B&H
(31.5%, −64%); the mining pod sits below that line everywhere.

### Where this leaves the absolute-return question

Maximizing CAGR inside an e ∈ [0, 1] dial on a 1× vehicle is impossible
by construction — the ceiling is buy & hold minus timing slippage. The
levers that actually move absolute returns, in order of defensibility on
this repo's own evidence:

1. **Hold the underlying.** QQQ B&H beat every mined configuration. The
   burden of proof is on any machinery that goes below 100% exposure.
2. **Take the validated fixed dial to a levered vehicle.** The
   council/champion-stack dial (Sharpe 1.05, DD −16% on QQQ — a *fixed*,
   already-validated construction, not a re-selecting pod) applied to
   QLD/SSO is the CAGR route the earlier threads pointed at: leverage
   harvested through a dial that has survived fresh-sample tests.
3. **Stop re-mining.** Every variation of "search more / select smarter /
   gate harder" tested across 13 configurations made things worse or
   changed nothing. The pod's value is the haircut measurement itself —
   it prices the winner's curse at roughly −0.7 live Sharpe per unit of
   research Sharpe — not the book it trades.
