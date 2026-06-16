"""Cross-outcome structure of multi-candidate Polymarket events.

A single event (e.g. an election) is several mutually-exclusive outcome markets
whose Yes-prices are mechanically coupled: exactly one resolves Yes, so the prices
sum to ~1 and changes sum to ~0. This module characterises that joint behaviour —
how tightly the field is arbitrage-free, who trades against whom (substitution),
and what happens after one candidate is shocked (the Biden→Harris archetype).

All dynamics are in log-odds (``signals.to_logodds``); a "well-covered" event is one
where we hold ≥ ``min_k`` constituents and their prices sum to ~1 (so the field isn't
missing big candidates).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import to_logodds


def event_panels(panel: pd.DataFrame, meta: pd.DataFrame, min_k: int = 4,
                 sum_lo: float = 0.85, sum_hi: float = 1.20, min_days: int = 20) -> dict:
    """Map each well-covered multi-outcome event → its constituent Yes-price sub-panel."""
    out = {}
    for ev, idx in meta.groupby("event_ticker").groups.items():
        cols = [m for m in idx if m in panel.columns]
        if len(cols) < min_k:
            continue
        sub = panel[cols].dropna(how="all")
        live = sub.notna().sum(axis=1)
        sub = sub[live >= max(min_k, int(0.7 * len(cols)))]
        if len(sub) < min_days:
            continue
        if sum_lo < sub.sum(axis=1, min_count=1).median() < sum_hi:
            out[str(ev)] = sub
    return out


def _label(meta: pd.DataFrame, market: str, n: int = 22) -> str:
    q = str(meta.loc[market, "question"]) if market in meta.index else market
    return q[:n]


def coupling_summary(evp: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-event coupling: mean pairwise log-odds-change corr + the dominant substitute pair."""
    rows = []
    for ev, sub in evp.items():
        dz = to_logodds(sub).clip(-6, 6).diff()
        C = dz.corr()
        cols = list(C.columns)
        iu = np.triu_indices(len(cols), 1)
        vals = C.values[iu]
        mean_corr = float(np.nanmean(vals))
        # most-negative pair = the dominant substitution
        k = int(np.nanargmin(np.where(np.isfinite(vals), vals, np.inf)))
        a, b = cols[iu[0][k]], cols[iu[1][k]]
        rows.append({
            "event": ev, "k": sub.shape[1],
            "sum_tightness": float((sub.sum(axis=1, min_count=max(3, int(0.8 * sub.shape[1]))) - 1).abs().median()),
            "mean_corr": mean_corr,
            "dominant_pair": f"{_label(meta, a, 16)} ↔ {_label(meta, b, 16)}",
            "pair_corr": float(C.loc[a, b]),
        })
    return pd.DataFrame(rows).sort_values("k", ascending=False)


def redistribution_drift(evp: dict, drop_thresh: float = 0.5, horizon: int = 5) -> dict:
    """After a sharp single-day drop in one candidate, the biggest beneficiary's forward drift.

    Negative forward drift ⇒ the beneficiary overshoots and reverts (overreaction);
    positive ⇒ it under-reacts and keeps climbing (laggy redistribution / momentum).
    """
    drifts, benef_conc = [], []
    for sub in evp.values():
        z = to_logodds(sub).clip(-6, 6)
        dz = z.diff()
        for t in range(1, len(sub) - horizon):
            day = dz.iloc[t]
            if day.min() < -drop_thresh:
                dropper = day.idxmin()
                gains = day.drop(dropper)
                if gains.max() <= 0:
                    continue
                ben = gains.idxmax()
                pos = gains.clip(lower=0)
                benef_conc.append(float(gains.max() / (pos.sum() + 1e-9)))  # Herfindahl-ish
                d = float(z[ben].iloc[t + horizon] - z[ben].iloc[t])
                if np.isfinite(d):
                    drifts.append(d)
    drifts = np.array(drifts)
    return {
        "n_shocks": int(len(drifts)),
        "mean_drift": float(drifts.mean()) if len(drifts) else float("nan"),
        "frac_momentum": float((drifts > 0).mean()) if len(drifts) else float("nan"),
        "mean_beneficiary_share": float(np.mean(benef_conc)) if benef_conc else float("nan"),
    }


def rank_calibration(evp: dict, outcomes: pd.Series) -> pd.DataFrame:
    """Calibration of candidates by within-field price rank (favorite / 2nd / trailing).

    Restricted to genuine winner-take-all fields (outcomes sum to 1). Reveals whether
    the field over/under-prices the leader vs the challenger vs the also-rans.
    """
    rows = []
    for ev, sub in evp.items():
        ids = list(sub.columns)
        o = outcomes.reindex(ids)
        if not np.isclose(o.sum(), 1.0):          # exactly one winner ⇒ real field
            continue
        order = sub.median().sort_values(ascending=False)
        for r, m in enumerate(order.index):
            grp = "favorite" if r == 0 else ("2nd" if r == 1 else "trailing")
            rows.append({"event": ev, "rank": r, "group": grp,
                         "price": float(sub[m].median()), "outcome": float(o[m])})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby("group", observed=True)
    return pd.DataFrame({
        "n": g.size(), "mean_price": g["price"].mean(), "yes_rate": g["outcome"].mean(),
        "resid": g["outcome"].mean() - g["price"].mean(),
    }).reindex(["favorite", "2nd", "trailing"]).reset_index()


def relative_value_reversion(evp: dict) -> dict:
    """Lag-1 autocorr of the top-2 candidates' log-odds spread changes (negative ⇒ reverts)."""
    acs = []
    for sub in evp.values():
        top2 = sub.mean().sort_values(ascending=False).index[:2]
        z = to_logodds(sub[top2]).clip(-6, 6)
        d = (z.iloc[:, 0] - z.iloc[:, 1]).diff().dropna()
        if len(d) > 15 and d.std() > 1e-6:
            acs.append(d.autocorr(1))
    acs = np.array([a for a in acs if np.isfinite(a)])
    return {"n_events": int(len(acs)), "mean_autocorr": float(acs.mean()) if len(acs) else float("nan"),
            "frac_reverting": float((acs < 0).mean()) if len(acs) else float("nan")}
