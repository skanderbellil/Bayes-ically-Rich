"""
Council backtest: period-isolated, blinded, domain-specialist decisions.

The research loop is decomposed in two directions at once:

  BY TIME    — the backtest is a sequence of independent PERIOD decisions;
               the context handed to the council at period t contains only
               information knowable at t (publication-lagged macro, trailing
               prices), so the engine is walk-forward by construction.
  BY SUBJECT — each period is assessed by a COUNCIL of specialists (rates,
               credit, liquidity, volatility, trend, dollar), each seeing
               only its own domain's slice of the context.

Anti-lookahead design (the LLM problem)
---------------------------------------
A language model knows market history: shown "March 2020" it can recall the
recovery instead of analyzing the data. Three defenses:

  1. BLINDING   — contexts carry no dates, no tickers, no raw levels: only
                  relative time, z-scores and normalized cumulative paths
                  (`blinding.py`).
  2. RANDOMIZED — each period context is independently jittered (random
                  rounding/scale presentation) so byte patterns can't be
                  memorized across runs.
  3. MIRROR AUDIT — every specialist also assesses a sign-flipped copy of
                  each context. An honest analyst's stance must flip with
                  the data (anti-symmetry ≈ 1); a model recognizing the
                  episode and recalling what actually happened won't flip
                  (`council.py: mirror_audit`). This measures residual
                  leakage instead of assuming the blinding worked.

Backends: deterministic statistical specialists (offline, the baseline) and
Claude-backed specialists via structured outputs (`llm.py`, needs
ANTHROPIC_API_KEY; uses the Batches API for large period counts).
"""
from posterioralpha.council.council import CouncilBacktest, CouncilConfig
from posterioralpha.council.specialists import DOMAINS, StatisticalSpecialist

__all__ = ["CouncilBacktest", "CouncilConfig", "DOMAINS", "StatisticalSpecialist"]
