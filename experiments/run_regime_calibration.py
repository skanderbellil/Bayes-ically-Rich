#!/usr/bin/env python3
"""
Regime-dependent calibration error in Polymarket
================================================

Tests the behavioural hypothesis (ChatGPT/▢ session): traders **underestimate
disruptive events during calm regimes and overestimate them right after shocks**,
so prices lag the true hazard rate. Two falsifiable predictions:

* **P1 — calm underpricing.** For *disruptive* domains (geopolitics, crypto/
  financial, politics, AI), realized YES frequency should exceed the price,
  i.e. calibration error ``CE = outcome - price > 0`` — strongest at low prices
  (the tail-adjacent events normalcy bias most affects).
* **P2 — regime dependence.** Underpricing should be **larger after calm** and
  **smaller/negative after recent shocks**: ``CE`` falls as a strictly-lagged,
  per-domain "surprise intensity" rises (mean-reversion in calibration).

Method (no lookahead): the decision price is the last daily mid **7 days before
resolution** (reusing the token-history cache the mid-price study already built),
the outcome is the market's own settlement, and surprise intensity for a market
counts disruptive longshot-YES resolutions in the **same domain** that settled in
the 90 days **before this market's decision date** (leave-one-out, strictly past).

Sports/team moneylines are kept as a *control* domain: an underdog winning is not
a "disruptive event" in the behavioural sense, so the regime effect should be
absent there even if a favorite-longshot mispricing is present.

Usage
-----
  python experiments/run_regime_calibration.py
  python experiments/run_regime_calibration.py --n-markets 2500 --lookback-days 7
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from posterioralpha.polymarket.fetch import GAMMA_URL, _get, fetch_token_history
from posterioralpha.polymarket.paths import RESULTS_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-5s  %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DISRUPTIVE = {"geopolitics", "crypto/financial", "politics", "ai/tech"}

_KW = {
    "geopolitics": ["strike", "war", "invade", "invasion", "attack", "nuclear", "missile",
                    "ceasefire", "troops", "military", "iran", "russia", "ukraine", "israel",
                    "gaza", "taiwan", "north korea", "coup", "annex", "sanction", "hostage"],
    "crypto/financial": ["bitcoin", "ethereum", "solana", "crypto", "btc", "eth", "price of",
                         "reach $", "hit $", "all-time high", "ath", "recession", "s&p",
                         "nasdaq", "market cap", "fed", "rate cut", "rate hike", "cpi",
                         "inflation", "gdp", "default"],
    "ai/tech": ["openai", "gpt", "agi", "anthropic", "claude", "gemini", "llm", "ai model",
                "artificial intelligence", "deepmind", "grok", "nvidia"],
    "politics": ["election", "president", "senate", "governor", "vote", "poll", "trump",
                 "biden", "kamala", "nominee", "impeach", "resign", "primary", "democrat",
                 "republican", "parliament", "prime minister", "approval"],
}


def classify(question: str, leg: str) -> str:
    if leg not in ("yes", "no"):
        return "sports/other"
    q = question.lower()
    for dom, kws in _KW.items():
        if any(k in q for k in kws):
            return dom
    return "other-binary"


def fetch_universe(n_markets: int, min_volume: float, page_size: int = 100) -> pd.DataFrame:
    rows, offset, seen = [], 0, 0
    while seen < n_markets:
        batch = _get(GAMMA_URL, {"limit": page_size, "offset": offset, "closed": "true",
                                 "order": "volumeNum", "ascending": "false"})
        if not batch:
            break
        for m in batch:
            seen += 1
            try:
                vol = float(m.get("volumeNum") or 0.0)
                if vol < min_volume:
                    continue
                toks = json.loads(m.get("clobTokenIds") or "[]")
                prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
                labels = json.loads(m.get("outcomes") or "[]")
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if len(toks) != len(prices) or not toks:
                continue
            for i, (tok, settled) in enumerate(zip(toks, prices)):
                oc = 1.0 if settled >= 0.98 else (0.0 if settled <= 0.02 else None)
                if oc is None:
                    continue
                leg = (str(labels[i]) if i < len(labels) else ("Yes" if i == 0 else "No")).strip().lower()
                q = m.get("question") or ""
                rows.append({"token": str(tok), "outcome": oc, "leg": leg,
                             "domain": classify(q, leg), "end_date": m.get("endDate"),
                             "volume": vol, "question": q})
            if seen >= n_markets:
                break
        if len(batch) < page_size:
            break
        offset += page_size
    df = pd.DataFrame(rows)
    logger.info("universe: %d markets scanned → %d settled legs", seen, len(df))
    return df


def decision_price(token: str, lookback_days: int):
    """Last daily mid <= (resolution - lookback). Returns (decision_date, price) or None."""
    s = fetch_token_history(token, fidelity_minutes=1440, use_cache=True)
    if s is None or len(s) < 3:
        return None
    s = s[~s.index.duplicated(keep="last")].sort_index()
    t_res = s.index.max()
    sub = s[s.index <= t_res - pd.Timedelta(days=lookback_days)]
    if sub.empty:
        return None
    p = float(sub.iloc[-1])
    if not (0.02 <= p <= 0.98):
        return None
    return sub.index[-1].tz_localize(None) if sub.index[-1].tz else sub.index[-1], p


def tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 and x.std() > 0 else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-markets", type=int, default=2000)
    ap.add_argument("--min-volume", type=float, default=5_000.0)
    ap.add_argument("--lookback-days", type=int, default=7)
    ap.add_argument("--surprise-window", type=int, default=90, help="trailing days for surprise intensity")
    ap.add_argument("--surprise-thresh", type=float, default=0.20, help="price below which a YES is a 'surprise'")
    args = ap.parse_args()
    ensure_dirs()

    uni = fetch_universe(args.n_markets, args.min_volume)
    if uni.empty:
        raise SystemExit("empty universe")

    rows = []
    for k, r in enumerate(uni.itertuples(index=False), 1):
        dp = decision_price(r.token, args.lookback_days)
        if k % 500 == 0:
            logger.info("  priced %d/%d (%d usable)", k, len(uni), len(rows))
        if dp is None:
            continue
        ddate, price = dp
        rows.append({"token": r.token, "domain": r.domain, "leg": r.leg,
                     "decision_date": pd.Timestamp(ddate), "price": price, "outcome": r.outcome,
                     "res_date": pd.Timestamp(r.end_date).tz_localize(None) if r.end_date else pd.Timestamp(ddate),
                     "volume": r.volume, "question": r.question})
    P = pd.DataFrame(rows).dropna(subset=["decision_date", "res_date"])
    P["CE"] = P["outcome"] - P["price"]                       # calibration error
    P = P.sort_values("res_date").reset_index(drop=True)
    logger.info("panel: %d priced legs  (%s → %s)", len(P), P.res_date.min().date(), P.res_date.max().date())

    # ---- surprise intensity: trailing, strictly-lagged, per domain ----
    surprises = P[(P.domain.isin(DISRUPTIVE)) & (P.price < args.surprise_thresh) & (P.outcome == 1)]
    sd = surprises.groupby("domain")["res_date"].apply(lambda s: np.sort(s.values))
    win = pd.Timedelta(days=args.surprise_window)
    def s_intensity(row):
        arr = sd.get(row["domain"])
        if arr is None:
            return 0
        cut_hi = np.datetime64(row["decision_date"]); cut_lo = np.datetime64(row["decision_date"] - win)
        return int(((arr >= cut_lo) & (arr < cut_hi)).sum())
    P["S"] = P.apply(s_intensity, axis=1)

    # ================= REPORT =================
    print("\n" + "="*92)
    print("REGIME-DEPENDENT CALIBRATION ERROR — Polymarket   (n=%d legs, decision = res-%dd)"
          % (len(P), args.lookback_days))
    print("="*92)

    print("\n[P1] CALIBRATION ERROR (outcome - price) BY DOMAIN  [CE>0 = market underprices]")
    g = P.groupby("domain").agg(n=("CE","size"), mean_price=("price","mean"),
            realized=("outcome","mean"), CE=("CE","mean"), CE_t=("CE", tstat))
    g["disruptive"] = ["✓" if d in DISRUPTIVE else "" for d in g.index]
    print(g.round(4).sort_values("CE", ascending=False).to_string())

    print("\n[P1b] CE BY PRICE BUCKET, disruptive domains only  (tail = low price)")
    D = P[P.domain.isin(DISRUPTIVE)].copy()
    D["pb"] = pd.cut(D.price, [0,.1,.2,.35,.5,.7,1.0])
    gb = D.groupby("pb", observed=True).agg(n=("CE","size"), realized=("outcome","mean"),
            mean_price=("price","mean"), CE=("CE","mean"), CE_t=("CE", tstat))
    print(gb.round(4).to_string())

    print("\n[P2] REGIME DEPENDENCE — CE vs trailing surprise intensity S (disruptive domains)")
    # calm vs turbulent split (per-domain demeaned S): low-S vs high-S
    D = D.assign(Sd=D.groupby("domain")["S"].transform(lambda s: s - s.mean()))
    calm = D[D.Sd <= 0]; turb = D[D.Sd > 0]
    print("  CALM     (S<=domain mean): n=%4d  CE=%+.4f (t=%+.2f)  realized=%.3f  price=%.3f"
          % (len(calm), calm.CE.mean(), tstat(calm.CE), calm.outcome.mean(), calm.price.mean()))
    print("  TURBULENT(S> domain mean): n=%4d  CE=%+.4f (t=%+.2f)  realized=%.3f  price=%.3f"
          % (len(turb), turb.CE.mean(), tstat(turb.CE), turb.outcome.mean(), turb.price.mean()))
    print("  >> hypothesis P2 predicts CALM CE > TURBULENT CE.  diff = %+.4f" % (calm.CE.mean()-turb.CE.mean()))
    # OLS: CE ~ price + S (domain fixed effects via demeaning CE & S within domain)
    D2 = D.assign(CEd=D.groupby("domain")["CE"].transform(lambda s: s-s.mean()))
    x = D2["Sd"].to_numpy(); y = D2["CEd"].to_numpy()
    if x.std() > 0:
        beta = np.cov(x, y, ddof=1)[0,1]/x.var(ddof=1)
        se = np.sqrt(((y-beta*x)**2).sum()/(len(x)-1)) / (np.sqrt(((x-x.mean())**2).sum()))
        print("  OLS within-domain: dCE/dS = %+.5f  (t=%+.2f)  [neg ⇒ supports P2]" % (beta, beta/se))

    print("\n[CONTROL] sports/other (underdog wins are NOT 'disruptive events'):")
    sp = P[P.domain == "sports/other"]
    print("  n=%d  CE=%+.4f (t=%+.2f)  — favorite-longshot mispricing can exist, but no regime story"
          % (len(sp), sp.CE.mean(), tstat(sp.CE)))

    out = RESULTS_DIR / "regime_calibration_panel.csv"
    P.to_csv(out, index=False)
    g.to_csv(RESULTS_DIR / "regime_calibration_by_domain.csv")
    print("\nSaved: %s , regime_calibration_by_domain.csv" % out.name)
    print("✓ done.")


if __name__ == "__main__":
    main()
