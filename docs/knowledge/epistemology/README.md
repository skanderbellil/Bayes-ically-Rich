# Epistemology reference library

Three uploaded reference documents (2026-07-08) on the philosophy of knowledge,
kept here because every strategy in this repo is an epistemological bet: a claim
about what can be known from data, how confident that knowledge can be, and how
long it stays true once acted on. Read the PDFs for the full treatment; this
index summarizes each and maps the ideas onto the repo's modules so future
sessions can grep them.

## The documents

### `01-how-do-we-know-anything.pdf` — Phase 1: the classical canon

A primer on the core problem of induction and its 20th-century responses.

- **Hume**: "it always worked before" can never justify "it will work again" —
  the justification is circular. Every backtest lives inside this problem.
- **Popper / falsification**: don't collect confirmations, make bold claims and
  try to break them. The one-line test: *"what would have to happen for me to
  admit this is wrong?"* Decide it in advance — after the fact everyone is a
  champion at explaining failures away.
- **Duhem–Quine**: a failed test never tells you *which* assumption broke (the
  edge, the data, the execution assumption, the regime). You always test a
  bundle.
- **Lakatos**: protecting a core belief with auxiliary patches is legitimate
  *while the patches predict new things that come true* (Neptune), and
  degenerate the moment they only explain failures away (Vulcan). Judge a
  strategy's research program by whether its fixes forecast, not excuse.
- **Prediction ≠ explanation**: Newton predicted perfectly for 200 years with a
  wrong "why". A profitable ledger does not validate the story attached to it.
- **The consolation**: we never get proof, only correction — and a belief that
  survived heavy correction (a forward ledger, not a backtest) is the most
  reliable thing available.

### `02-phase2-after-the-classical-canon.pdf` — Phase 2: the reading roadmap

The syllabus for what comes after Popper/Kuhn/Lakatos/Quine/Jaynes/Mayo/Pearl.
Five frontiers, each relaxing a hidden Phase-1 assumption, with key readings:

1. **Decision under deep uncertainty** — Knight/Keynes risk vs. uncertainty,
   Ellsberg, imprecise probability, Gilboa–Schmeidler maxmin, Savage's
   small-world warning ("your Polymarket models are small worlds; markets are
   large ones — the gap is where blowups live").
2. **Social epistemology** — testimony, expert vetting, peer disagreement
   (every trade is applied disagreement theory), Condorcet, Hayek, when
   prediction markets aggregate information vs. amplify herding.
3. **Reflexivity** — Soros, Merton's self-fulfilling prophecies, MacKenzie's
   performativity (Black-Scholes), Lo's Adaptive Markets, alpha decay as an
   epistemological law, Ole Peters' ergodicity economics.
4. **The physics of inference** — MaxEnt, MDL/compression, bounded rationality
   (Gigerenzer vs. Kahneman), NP-hardness of exact Bayesian updating.
5. **Ethics of belief** — Clifford vs. James, Rudner/Douglas inductive risk
   (significance thresholds are value judgments), Fricker, Peirce's pragmatism.

Flagged original-contribution frontier: **epistemology of reflexive systems**
(Popperian severity + ergodicity + performativity applied to markets).

### `03-phase2-the-embedded-knower.pdf` — Phase 2: the five lessons, taught

The worked version of the roadmap, with the trading implications spelled out:

- **Lesson 1 (deep uncertainty)**: a backtest's sharp "edge = 3.7 bps" is an
  artifact of one historical path; the honest object is an *interval* that
  widens as conditions drift from sample. Regime change is exactly when a sharp
  credence should dissolve back into an interval. Sizing under
  maxmin-over-a-prior-set behaves very differently from Kelly on a point
  estimate. *"How wide is my interval?" beats "what's my p?"*
- **Lesson 2 (social knowing)**: Condorcet's theorem runs on **independence**;
  break it and a crowd of a million has the wisdom of the three people who
  thought first (information cascade). "Whale wallet moved" is testimony — is
  the actor informed, or early in a cascade? *Separating information from
  imitation in order flow is social epistemology as a signal-processing
  problem.* If you can't name why you'd know something the counterparty
  doesn't, don't trade.
- **Lesson 3 (reflexivity)**: discovered edges expire *because* they're
  discovered — induction in markets has an expiration date by construction.
  Backtests are photographs of a reflexive system taken before it noticed the
  camera. Non-ergodicity: you live on one path, not in the ensemble average
  (the +50%/−40% coin ruins everyone despite positive expectation);
  Kelly-style sizing is the practical answer.
- **Lesson 4 (bounded inference)**: simple rules beat optimization when data is
  thin relative to the world's complexity (bias–variance); regularization,
  shrinkage, and empirical-Bayes priors are MDL/MaxEnt in working clothes; a
  fast crude signal that trades today can dominate a perfect posterior that
  arrives tomorrow.
- **Lesson 5 (responsible belief)**: the significance threshold is an
  inductive-risk decision — false-positive edge costs capital, false-negative
  costs opportunity; where alpha is set *is* the answer to "which error do I
  fear more". Kill-criteria should be pre-registered (Clifford's shipowner:
  decide the hull inspection before sailing).

## Mapping to this repo

| Idea | Where it already lives | Where it's missing |
|---|---|---|
| Forward ledgers as survival-testing (Popper) | `data/paper_trade/*` hourly ledgers, `docs/knowledge/CHAMPION_PAPER_TRADE.md` | Pre-registered kill thresholds per strategy |
| Independence vs. cascade in crowd signals (Lesson 2) | `smart_flow` counts co-buying wallets (`run_smart_flow_paper_update.py`) | No test of *whether the buyers are independent* — consensus counting rewards cascades |
| Interval credences / maxmin sizing (Lesson 1) | — | All sizing is point-estimate `--fraction 0.10` |
| Non-ergodicity / time-average growth (Lesson 3) | `run_bankroll_sim.py` (capital-constrained path simulation) | Sizing not derived from time-average growth |
| Alpha decay as law (Lesson 3) | Regime re-validation step refreshes `validated_domains.json` | No decay monitoring on live edges |
| Simple-beats-optimal (Lesson 4) | Champion stack's frozen spec, equal-weight bankroll | — |
