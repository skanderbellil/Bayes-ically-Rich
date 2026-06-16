"""Lightweight topic classification of Polymarket markets.

Keyword rules over the question text plus the parent-event ticker/title. Coarse by
design — its job is to split the universe (especially "politics / election" vs the
rest) so a finding can be checked for robustness beyond the 2024-election complex
that dominates top-volume resolved markets.
"""
from __future__ import annotations

import re

import pandas as pd

# order matters: first match wins
_RULES: list[tuple[str, str]] = [
    ("politics", r"trump|harris|biden|elect|president|senate|govern|electoral|"
                 r"republican|democrat|\bgop\b|vance|desantis|congress|primary|"
                 r"nomine|swing state|popular vote|speaker|impeach|cabinet|"
                 r"shutdown|parliament|prime minister|chancellor|mayor"),
    ("geopolitics", r"\bwar\b|ceasefire|iran|russia|ukraine|israel|gaza|hamas|"
                    r"nuclear|invade|military|nato|zelensky|putin|hostage|strike"),
    ("crypto",   r"bitcoin|\bbtc\b|ethereum|\beth\b|crypto|solana|\bsol\b|dogecoin|"
                 r"\bxrp\b|microstrategy|coinbase|stablecoin|\bnft\b"),
    ("macro",    r"\bfed\b|interest rate|rate cut|rate hike|inflation|\bcpi\b|"
                 r"recession|\bgdp\b|jobs report|unemployment|powell|fomc"),
    ("sports",   r"nba|\bnfl\b|super bowl|world cup|premier league|champions league|"
                 r"finals|playoff|\bufc\b|world series|stanley cup|olympic|"
                 r"\bf1\b|grand prix|wimbledon|masters|win the"),
    ("culture",  r"album|movie|box office|oscar|grammy|spotify|youtube|twitter|"
                 r"\bgta\b|rotten tomatoes|time person|nobel"),
]
_COMPILED = [(name, re.compile(pat, re.I)) for name, pat in _RULES]


def market_category(question: str, event_ticker: str = "", event_title: str = "") -> str:
    """Classify one market into a coarse topic bucket ('other' if no rule matches)."""
    text = " ".join(str(x or "") for x in (question, event_ticker, event_title))
    for name, rx in _COMPILED:
        if rx.search(text):
            return name
    return "other"


def classify_markets(meta: pd.DataFrame) -> pd.Series:
    """Topic label per market, indexed like ``meta`` (uses event fields if present)."""
    et = meta["event_ticker"] if "event_ticker" in meta.columns else ""
    tt = meta["event_title"] if "event_title" in meta.columns else ""
    et = et if isinstance(et, pd.Series) else pd.Series("", index=meta.index)
    tt = tt if isinstance(tt, pd.Series) else pd.Series("", index=meta.index)
    return pd.Series(
        [market_category(q, a, b) for q, a, b in zip(meta["question"], et, tt)],
        index=meta.index, name="category",
    )
