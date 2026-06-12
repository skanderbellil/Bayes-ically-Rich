# Alpha Mining in This Repo — the Research Program from Commit History

> Durable record of the research arcs on `claude/automated-alpha-mining-*`
> (126 commits as of 2026-06), so future sessions don't re-derive what was
> tried, kept, or killed. The per-experiment verdicts live in
> `experiments/README.md` (the canonical log — one row per study, with the
> honest outcome in the description); this file is the arc-level map.
> Read alongside `systematic-equity-strategies.md` (external corpus) and
> `research-roadmap.md` (what to do next).

## Arc 1 — Bayesian regime engine (the founding line)

Initial PosteriorAlpha system (`c69de85`) → AMR strategy + 14-asset universe
(`3a9a3bc`) → Viterbi replaced with forward-filtered HMM states to kill a
look-ahead bias (`6964561`) → BOCPD + 3-state HMM hybrids (`7b51575`) →
BOCPD-AMR v2 (CVaR, multi-asset BOCPD, low-vol tilt, `83bda78`) → v3
(data-driven λ via EWMA Omega ratio, `7611418`) → v4 (ERL-adaptive EWMA
halflife; the credibility discount and Kelly guard were *removed* because
they fought the main signal, `2d4e26e`) → SPY-only timing ablation
(`9fab14e`). Lessons: decoders leak (only forward-filtered states are valid);
data-driven calibration beat hand-tuned constants; deleting components was
the v4 improvement.

## Arc 2 — PEAD program

`d1f1674` → `b8a4394`: full post-earnings-announcement-drift pipeline
(signal extraction, cap-spectrum confirmation of the information-diffusion
hypothesis, walk-forward validation, cost/capacity/AUM realism, Bayesian
vol-scaled sizing, dynamic exits). Final honest verdicts in `docs/pead/`:
the signal is real and not overfit, but after costs and hustle it does not
robustly justify active management vs. indexing for retail capital.

## Arc 3 — Framework consolidation + explorations

Explorations #1–8 (low-turnover factors fail vs. survivorship-matched
benchmarks; cross-asset ETF portfolios were "the one real, robust result";
multi-pod ETF strategies; dynamic vol management) → everything consolidated
into the `posterioralpha` package with the explicit 4-stage pipeline
(`105d7b6`): data → research → backtest → validation.

## Arc 4 — Net liquidity & the layered-vote champion thread

The longest arc. Net liquidity (Fed assets − TGA − RRP) vs. equities →
layer-by-layer construction of a continuous "vote": liquidity + trend →
forward-looking layers (VIX term structure, credit appetite) → dollar layer
(patches the 2022 grinding bear) → tug-of-war layer (overnight − intraday
spread) → debt-ceiling calendar windows. Validated at each step with rolling
OOS win rates, GFC stress tests, fresh-sample checks. Retail surprise: the
levered-ETF route (weight = vote in QLD + cash) beats the margin fiction.
Champion stack frozen at `2087b1d`. Key diagnostic findings: liquidity loads
on the overnight leg; layer *disagreement* is a real risk gauge; model
disagreement inverts (indecision, not conflict, marks transitions).

## Arc 5 — Plumbing nowcasts (the null graveyard)

Auction-demand surprises, RRP-floor forensics, funding-spread scarcity,
withheld-tax nowcast, dealer-inventory fragility: all clean, recorded nulls
— excellent macro data, no equity signal, or the mechanism-true gauge loses
to its VIX shadow. Kept deliberately as negative knowledge.

## Arc 6 — Council & pod governance

Blinded period-council (LLM domain specialists with no-dates/no-tickers
contexts, mirror leakage audits, an oracle falsification test), multi-asset
retail version with Alpaca costs. Then the quant-pod simulator
(research → hire → monitor → retire) and hiring policies vs. the winner's
curse: only median-picking moves the book, and not decisively; the inverse
research→live correlation means the ceiling is the benchmark.

## Arc 7 — Automated alpha mining

`1b0063e` →: evolutionary search over parameterized signal families
(momentum, reversal, vol-scaled, channel, idio-reversal, skew, plus
BOCPD/macro-gated) with a randomized holdout gauntlet — random purged
windows, block bootstrap, permutation tests, Deflated Sharpe
(`posterioralpha/mining/validation.py`). Extended with a FRED macro panel,
family-capped finalists, and a timing-dial miner whose fresh-sample gate is
the automatic final verdict (easing dials died there; the real-yield dial
was the sole cross-asset promotion).

## Arc 8 — Levered barbell & governors (current frontier)

Static QLD/GLD barbell expands the frontier; the gold leg "cannot be
selected, only held" (de-hindsighting); four novel governors tested → vol
management on levered barbells is real with a leverage dose-response,
rotation beats cash-scaling, three clean negatives; surviving pieces are
substitutes, not complements (stack test); return-objective nowcast descent
dies on its foundation while the vol objective delivers; the rotation
barbell passed a GFC test it never saw (`bebeb4f`, branch head).

## Arc 9 — Cross-sectional IC layer (added 2026-06-12)

Knowledge-base application session: `posterioralpha/mining/ic.py`
(Grinold & Kahn monthly rank-IC lab as the pre-gauntlet gate),
`posterioralpha/validation/robustness.py` (rebalance-offset dispersion +
family-level DSR wrapper), per-stock BOCPD families
(`stock_regime_momentum`, `regime_age`) added to the miner's search space,
and `experiments/run_signal_lab.py`. First results in
`research-roadmap.md` §5–6.

## House research style (observed, worth preserving)

1. Every experiment gets a one-row verdict in `experiments/README.md`,
   including — especially — the failures.
2. Fresh-sample / GFC / OOS gates decide promotions, not in-sample Sharpe.
3. Nulls are recorded, not deleted (Arc 5 exists on purpose).
4. De-hindsighting: ask whether a winning component could have been
   selected ex ante before crediting it.
5. Champion configs are frozen in dedicated commits so later work has a
   fixed reference to beat.
