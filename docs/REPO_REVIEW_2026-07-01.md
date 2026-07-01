# Repo Review — 2026-07-01

A full-repo audit: what's wrong, what's missing, what could be improved, and which
discovery routes remain unexplored. Companion artifact: `graphify-out/graph.html`
(interactive knowledge graph of the whole codebase — 2,746 nodes / 5,845 edges /
187 communities, built with [Graphify](https://github.com/safishamsi/graphify)).

Overall verdict up front: the research methodology here is unusually honest —
DSR/bootstrap/permutation gauntlets, pre-registered OOS tests, self-correcting
write-ups, forward paper-trade ledgers. Most of what follows is engineering
hygiene and coverage gaps, not methodological rot. The items are ranked roughly
by how much they could silently corrupt results.

---

## 1. Things that look actually wrong

### 1.1 The monthly Bayesian engine reports GROSS returns
`posterioralpha/backtest/bayesian.py` uses `tc` only as an optimizer penalty
(inside `max_sharpe_weights`) — realized portfolio returns are never debited
for turnover (`pf_rets = period.values @ weights`, no cost line). The AMR
engine does it right (`backtest/amr.py:361-376` deducts `tc * turnover`).
The README's methodology note — "Costs. Transaction costs are charged on
effective (post-leverage) turnover." — therefore over-claims for the whole
Bayesian family (`bayesian`, `bayesian_ewma`, `bayesian_hmm`, `bayesian_full`,
plus the baselines). Monthly rebalance keeps the bias small, but any
Bayesian-vs-AMR comparison in the docs is gross-vs-net. Fix: charge
`tc * Σ|Δw|` on the first day of each holding period, same as AMR.

### 1.2 Paper-trade resolution detection is a price heuristic, and the real
### resolution checker is dead code
`posterioralpha/polymarket/papertrade.py::_resolve_status` marks a position
`won`/`lost` when the CLOB **last-trade price** crosses 0.99/0.01. Two failure
modes: (a) a 0.99 print before actual resolution marks `won` prematurely and
freezes the ledger row even if the market later reverses (rare but real in
politics/macro tails — exactly this book's markets); (b) a market that stops
trading at e.g. 0.97 and resolves without a final print near 1.0 stays `open`
forever. Meanwhile `scan_closed_macro_events` — which queries Gamma for the
actual `outcome` field — exists in the same file and is **never called**.
The Gamma `closed`/`outcome`/UMA-resolution fields are the ground truth;
use them, keep the price heuristic only as a fallback. The same pattern
should be audited in `smartflow_papertrade.py` and the midprice/dip trackers.

### 1.3 `close_position.py` books stale, slip-free exits
Manual closes price the exit at the ledger's `current_price` — up to an hour
stale (and after a failed cron run, arbitrarily stale) — and apply no slippage,
while entries were charged `SLIP`/ask. Manual closes are therefore
systematically flattered vs held-to-resolution positions. Fetch a live price
at close time and debit the same slippage model used at entry.

### 1.4 Workflow failures are swallowed
`paper_trade.yml` runs three steps with `|| true` (regime ledgers, domain
validation). If the Gamma/CLOB API changes shape, those ledgers silently stop
updating and the forward-OOS record develops holes — the one thing a forward
test can't tolerate. At minimum: let the step fail and add a
`if: failure()` notification step (or a badge/issue-creating action), so
staleness is visible. Same for the un-guarded steps: one crash aborts every
later strategy's update *and* the commit of the earlier ones' changes.

### 1.5 Concurrent-push race between the two workflows
Both `paper_trade.yml` (hourly) and `close_position.yml` (manual) commit and
`git push` without pulling/rebasing first. A manual close that overlaps the
hourly run will fail its push (or vice versa). Add
`git pull --rebase origin <branch>` before push, or use `concurrency:` groups
to serialize the two workflows.

### 1.6 Shell-injection surface in `close_position.yml`
`${{ github.event.inputs.question }}` is interpolated raw into `run:` shell —
a question string containing `$(...)`/backticks executes in the runner.
Only collaborators can dispatch, so severity is low, but the fix is free:
pass inputs via `env:` and reference `"$QUESTION"`.

### 1.7 Packaging drift
- `paper_trade.yml` does `pip install -e ".[dev]"` — **no `[dev]` extra
  exists** in `pyproject.toml`, so the fallback hard-coded package list always
  runs and can drift from `pyproject.toml` silently.
- `requirements.txt` lists `financedatabase` twice and includes `anthropic`,
  which `pyproject.toml` doesn't declare anywhere (the council's `--llm` mode
  imports it). Add an `llm = ["anthropic>=0.40"]` extra and a real `dev` extra.
- No lockfile anywhere: the hourly cron installs whatever pandas/numpy shipped
  that hour. A pandas major release can silently change ledger semantics.
  Pin the CI environment (a `requirements-ci.lock` generated with `pip freeze`
  is enough).
- No `LICENSE` file — the repo is public; right now nobody (including future
  you on another account) has a formal right to reuse it.

---

## 2. Things missed / gaps

### 2.1 Zero tests
There is no test suite at all (the only `*test*` files are experiments). The
program's epistemics rest on `mining/validation.py`, `validation/metrics.py`,
the loaders, and the blinding/oracle machinery — a silent regression in any of
those corrupts every subsequent study, and the PEAD history (the +23.6%/yr
winners-only bug) shows exactly this class of error happens. A small, fast,
offline `pytest` suite would cover the highest-value invariants cheaply:
- metrics: Sharpe/CAGR/MaxDD on hand-computable toy series;
- loaders: shapes, date monotonicity, no-NaN guarantees on bundled datasets;
- backtest engines: a flat-price universe yields ~zero PnL net of costs; a
  known two-asset case reproduces frozen numbers (regression pin);
- lookahead tripwire: perturb returns *after* date t and assert weights at t
  are bit-identical (the strongest automated no-lookahead guarantee);
- ledger update: `update_ledger` twice in a row is a no-op (idempotency);
- blinding: `mirror` contexts satisfy anti-symmetry by construction.
Wire it into a tiny `ci.yml` on push/PR. This is the single highest-leverage
engineering improvement available.

### 2.2 The flagship equity strategy has no forward ledger
Every Polymarket book has a forward, out-of-sample paper-trade ledger — but the
**champion stack** (QLD/UUP, the project's headline scorecard) does not. Its
entire record is in-sample-era backtest. The infrastructure already exists:
a daily job that logs the vote, the exposure `e`, the implied w_QLD/w_UUP and
marks the two-ticker book would give the frozen spec the same survivorship-free
record the macro tracker has. Given `run_champion_stack.py` freezes the spec in
a docstring, a cron that *executes the frozen spec forward* is the natural next
step and closes the loop the docs themselves call for ("trust requires the
forward paper-trade").

### 2.3 Hardcoded debt-ceiling event list is a time bomb
The ceiling ×½ overlay's event list ends Jun-2023 (flagged in the docstring,
but nothing enforces it). The champion stack silently degrades as new deals
happen. Options: derive the events from the TGA series itself (a rebuild
after a flat-lined TGA is detectable causally from `fred_macro`), or add a CI
check that fails when `today - last_event > N months` to force a review.

### 2.4 experiments/README.md indexes ~120 of 181 scripts
62 scripts are missing from the index — including the *entire* recent arcs
(regime/GPR thread, midprice-YES family, insider event study, weather edge,
wallet-sentiment, bankroll sim, carry/VRP, quality long-only, dashboard/paper
infra). The README is the project's map; the newest work is the least
discoverable. Also stale: docs reference `.github/workflows/smart_flow_paper.yml`,
which was consolidated into `paper_trade.yml`; `papertrade.py` docstrings say
"daily cron" while the workflow is hourly; the top-level README's repository
layout omits the `polymarket/` package from the tree diagram's first block and
`results/` is described as gitignored while `data/paper_trade/` artifacts are
tracked. One pass of doc-sync would fix all of it. (The new
`graphify-out/graph.html` helps here too — it's a queryable map that doesn't
rot the way a hand-written index does; re-run `graphify update .` after
changes.)

### 2.5 No dataset manifest
`datasets/` holds ~43 MB of bundled CSVs built at different times by different
scripts, with no record of *when* each was built, from what source, over what
date range, or by which script version. Studies silently mix vintages (e.g.
`vix_term_structure.csv` vs `vix_termstructure.csv` — two files, near-duplicate
names). Add `datasets/MANIFEST.md` (or a JSON emitted by each build script):
filename → builder script, build date, start/end dates, row count, sha256.
Cheap, and it makes "is this result on stale data?" answerable.

### 2.6 Ledger history is append-only in git but not in the data model
Hourly `paper-trade: update` commits (~8,700/yr at the current cadence) are
already 88% of recent history. That's a deliberate choice (the git history IS
the timestamping/notarization of the forward test), but consider: (a) pushing
ledger commits to a dedicated `paper-trade-data` branch so the main branch
history stays readable, or (b) appending to a history CSV (one row per
mark-to-market) instead of overwriting `current_price` in place — right now
intraday price paths are only recoverable by walking git history.

### 2.7 Smaller items
- `papertrade.py` imports `numpy` unused; `scan_closed_macro_events` dead (§1.2).
- 26 broad `except Exception` in the package — most are justified around
  network parsing, but a few wrap pure-pandas logic where they'd mask bugs.
- `data/paper_trade/dashboard.html` is regenerated hourly but only viewable
  via raw-file download; GitHub Pages (serve `data/paper_trade/`) would give
  a live public dashboard for free.
- The `.gitignore` un-ignore list for `data/paper_trade/*.csv` is 25 lines of
  enumerated files; `!data/paper_trade/*.csv` + explicit ignores for caches
  would be less error-prone (new ledgers currently need a gitignore edit or
  they silently don't commit — a new strategy's ledger could be lost).

---

## 3. Methodological notes (mostly already handled, stated for the record)

1. **Sequential selection on one sample.** The champion-stack arc (vote layers,
   ceiling, UUP slack, slow-down) was assembled by sequential testing on the
   same 2011→ QQQ sample. `run_retroactive_dsr.py` addresses this properly
   (family B passes at 80 assumed trials) and `run_adaptive_vote.py` +
   GFC checks add fresh-sample evidence — good. But the *stacking order* itself
   was chosen by results; the honest number for the full stack is the forward
   ledger (§2.2), which doesn't exist yet.
2. **Survivorship** is flagged everywhere it applies (good), but remains the
   biggest data limitation — see route R1 below for the free fix.
3. **Polymarket sample is one regime.** Nearly every study spans 2024→2026 —
   one election cycle, one calm-macro regime; the docs say so. The forward
   ledgers are the right answer; just don't let §1.2/§1.4 corrupt them.
4. **Costs on Polymarket books** are modeled as ¢/share half-spreads; the paper
   trackers entering at the live ask are the honest instrument — good pattern,
   keep extending it (the payup book deserves a live-ask tracker too).

---

## 4. Unexplored discovery routes

Ranked by (evidence it could matter) × (cheapness given existing infrastructure).

**R1 — Point-in-time equity universe via EDGAR + index-membership history.**
The roadmap calls a PIT universe "the single biggest data upgrade available"
and it's still open. Two free routes: (a) historical S&P 500/400/600
constituent lists are reconstructible from Wikipedia revision history +
published index-change announcements — that alone converts
`run_equity_cross_section` / the retail labs from "survivorship-biased,
judge spreads only" to honest levels; (b) SEC EDGAR (free, PIT by
construction) supplies fundamentals and earnings dates for the PEAD pipeline,
removing the yfinance live-names bias that currently caps PEAD as "research
artefact". This unblocks value/quality cross-sections too (Ilmanen
completeness: value ✗ carry ✓(VRP) — value is the remaining style hole, and it
failed on JKP only as a *factor-level* construct; a PIT single-name value test
was never run).

**R2 — Order-book depth accumulation cron (the GEX trick, applied to Polymarket).**
`run_gamma_exposure.py --log` accumulates a GEX history precisely because free
historical options OI doesn't exist. Polymarket has the identical situation:
`fetch.order_book_features` (spread/microprice/depth imbalance) is reachable
live but has **no history anywhere**. Add one step to the hourly cron that
snapshots books for open ledger tokens into a compressed CSV. Six months from
now you own the only depth history you'll ever get — it's the missing input
for the payup book's execution modeling (its break-even is ≈1¢ half-spread,
currently *assumed*, not measured per token per day).

**R3 — Cross-venue prediction-market arbitrage / calibration transfer
(Kalshi vs Polymarket).** Never mentioned in the whole thread. Kalshi has a
free API with historical data, overlapping macro/weather/politics markets, and
a *different* (KYC'd US retail) clientele. Three cheap studies: price gaps on
equivalent contracts (true arb, execution-bound but measurable); whether the
favorite-longshot miscalibration replicates on a different venue (the cleanest
possible OOS test of the macro-tails candidate — different market maker,
different crowd, same events); and Kalshi's longer macro history extends the
"one calm regime" sample backward.

**R4 — LLM council as a live probability forecaster on Polymarket.**
The two strongest sub-systems never meet: the blinded-council + mirror-audit
machinery (built to let LLMs vote without lookahead-through-memory) and the
prediction-market stack. A council seat that reads the *blinded* description of
an open market (no dates/names — the blinding code exists) and outputs a
probability gives you a calibrated LLM-vs-market residual; trade paper-size
where the residual is large. All infrastructure — structured-output calls,
Batches API, leakage audit, forward ledgers — already exists in-repo. This is
the highest-novelty cheap experiment available.

**R5 — Champion-stack forward ledger** (§2.2) — listed here because it's also
the cheapest new *evidence*, not just engineering.

**R6 — Portfolio-of-everything.** `run_orthogonal_ensemble.py` risk-parities
equity core + zoo + ETF momentum, but the Polymarket books (macro leader,
payup/flow) were never added as sleeves despite being the most orthogonal
return streams in the repo (event-resolution timing, no equity beta). Even a
toy monthly risk-parity across {champion stack, ETF-mom L/S, liquid-alts,
macro-leader ledger} would answer whether the prediction-market work earns a
capital allocation or stays a curiosity.

**R7 — Turn-of-month / FOMC-cycle overlays on the champion.** The calendar
thread covered tax dates, QRA, quarter-end RRP and auctions, but the two most
robust published calendar effects — turn-of-month equity returns and the
even/odd FOMC-week cycle (Cieslak-Morse-Vissing-Jørgensen) — were never tested
against the vote stack. Both are one-afternoon studies on data already bundled
(`fred_macro` has the FOMC-relevant series; OHLC panels exist).

**R8 — Regime-conditional sizing of the macro-leader book.** The regime thread
(GPR/geo-calm validation) gates *entry*; Kelly sizing (new dashboard code) is
regime-blind. The macro favorite-longshot candidate is explicitly a
short-vol trade that "worked because no tail fired" — sizing it down when the
funding-stress gauge (`run_funding_stress.py`, kept as a monitoring trigger)
or GPR spikes is the mechanism-aligned refinement, and both signals are
already computed in-repo.

---

## 5. Suggested immediate punch list

1. Charge TC in `backtest/bayesian.py` returns (or amend the README claim). §1.1
2. Resolution via Gamma `outcome` (revive `scan_closed_macro_events`); price
   heuristic as fallback only. §1.2
3. Live price + slippage in `close_position.py`. §1.3
4. Remove `|| true`; add failure notification; rebase-before-push;
   env-indirect the workflow inputs. §1.4–1.6
5. Add `pytest` suite + `ci.yml` (metrics, loaders, lookahead tripwire,
   ledger idempotency). §2.1
6. Champion-stack forward paper-trade cron. §2.2 / R5
7. Doc-sync pass: experiments README index, workflow names, cron cadence;
   add LICENSE; fix packaging extras; dataset MANIFEST. §1.7, §2.4, §2.5
8. Start the order-book snapshot cron (R2) — the sooner it starts, the sooner
   the history exists.
