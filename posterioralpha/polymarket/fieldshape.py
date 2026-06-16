"""Does a field's *shape* condition how predictive its prices are?

A multi-outcome event can be **peaked** — one candidate towering over the field
(e.g. an incumbent at 0.85) — or **flat** — several comparably-priced outcomes
splitting the mass (a genuine toss-up). Both are valid probability distributions,
but they encode very different states of knowledge. This module asks whether that
shape is *itself* informative beyond the prices: conditional on the leader's price,
does the leader of a peaked field win more (or less) often than the leader of a flat
field?

Shape is summarised by the **normalised Shannon entropy** of the renormalised price
vector ``p`` (``p`` clipped > 0, divided by its sum so it integrates to 1)::

    H_norm = -Σ pᵢ·log pᵢ / log k        ∈ [0, 1]

``H_norm → 0`` is maximally peaked (one outcome ≈ 1), ``H_norm → 1`` is a flat field
(all k outcomes ≈ 1/k). The companion concentration measure is the Herfindahl
``Σ pᵢ²`` (1 = peaked, 1/k = flat).

For each well-covered event we reduce to **one row** — the field leader over a fixed
days-to-resolution window — so the binary "did the leader win" fact is counted once
per event (snapshots within a market are not independent). Calibration is then the
leader's realised win-rate minus its mean implied price, bucketed by field shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def field_shape_events(evp: dict, outcomes: pd.Series, days_lo: int = 7,
                       days_hi: int = 90, min_k: int = 3, min_obs: int = 5) -> pd.DataFrame:
    """One row per multi-outcome event: field shape + the leader's price and fate.

    The window ``[days_lo, days_hi]`` days before the event's last observation avoids
    the trivially-converged final days (price → 0/1) and the thin opening. The
    *leader* is the constituent with the highest mean price over that window; its
    ``fav_win`` is its eventual Yes-resolution, counted once.

    Columns
    -------
    entropy, hhi   : mean normalised-entropy / Herfindahl of the field over the window
    p_fav, fav_win : leader's mean implied price and its realised 0/1 outcome
    resid          : ``fav_win - p_fav`` (>0 ⇒ leader under-priced)
    brier_price    : mean (outcome - price)² over *all* constituents (information content)
    brier_unif     : same vs a uniform 1/k forecaster (the no-shape baseline)
    wta            : True if the field is winner-take-all (outcomes sum to 1)
    """
    rows = []
    for ev, sub in evp.items():
        ids = list(sub.columns)
        if len(ids) < min_k:
            continue
        o = outcomes.reindex(ids)
        if o.isna().any():
            continue
        clean = sub.dropna(how="all")
        if clean.empty:
            continue
        last = clean.index[-1]
        dl = np.array([(last - d).days for d in sub.index])
        win = sub[(dl >= days_lo) & (dl <= days_hi)]
        if len(win) < min_obs:
            continue

        ents, hhis = [], []
        for _, r in win.iterrows():
            p = r.dropna()
            if len(p) < min_k:
                continue
            pn = p.clip(lower=1e-6)
            pn = pn / pn.sum()
            ents.append(float(-(pn * np.log(pn)).sum() / np.log(len(pn))))
            hhis.append(float((pn ** 2).sum()))
        if not ents:
            continue

        leader = win.mean().idxmax()
        bp, bu = [], []
        for mid in ids:
            ser = win[mid].dropna()
            if ser.empty:
                continue
            y = float(o[mid])
            bp.append(float(((y - ser) ** 2).mean()))
            bu.append((y - 1.0 / len(ids)) ** 2)

        rows.append({
            "event": ev, "k": len(ids), "n_obs": int(len(win)),
            "entropy": float(np.mean(ents)), "hhi": float(np.mean(hhis)),
            "p_fav": float(win[leader].mean()), "fav_win": float(o[leader]),
            "brier_price": float(np.mean(bp)), "brier_unif": float(np.mean(bu)),
            "wta": bool(np.isclose(o.sum(), 1.0)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["resid"] = df["fav_win"] - df["p_fav"]
    return df


def constituent_rows(evp: dict, outcomes: pd.Series, days_lo: int = 7,
                     days_hi: int = 90, min_k: int = 3, min_obs: int = 5) -> pd.DataFrame:
    """One row per *constituent* of each field: its window-mean price, outcome, field shape.

    Where ``field_shape_events`` keeps only the leader (n = #events), this keeps every
    candidate (n = Σ k) so calibration can be measured *price-matched* — comparing
    like-priced outcomes in peaked vs flat fields, which removes the mechanical fact
    that peaked fields simply have a higher-priced favorite. Still event-clustered:
    constituents of one field share a single winner, so bootstrap by ``event``.
    """
    rows = []
    for ev, sub in evp.items():
        ids = list(sub.columns)
        if len(ids) < min_k:
            continue
        o = outcomes.reindex(ids)
        if o.isna().any():
            continue
        clean = sub.dropna(how="all")
        if clean.empty:
            continue
        last = clean.index[-1]
        dl = np.array([(last - d).days for d in sub.index])
        win = sub[(dl >= days_lo) & (dl <= days_hi)]
        if len(win) < min_obs:
            continue
        ents = []
        for _, r in win.iterrows():
            p = r.dropna()
            if len(p) >= min_k:
                pn = p.clip(lower=1e-6); pn = pn / pn.sum()
                ents.append(float(-(pn * np.log(pn)).sum() / np.log(len(pn))))
        if not ents:
            continue
        field_entropy = float(np.mean(ents))
        for mid in ids:
            ser = win[mid].dropna()
            if ser.empty:
                continue
            rows.append({"event": ev, "market": mid, "k": len(ids),
                         "price": float(ser.mean()), "outcome": float(o[mid]),
                         "entropy": field_entropy})
    return pd.DataFrame(rows)


def bucket_by_shape(df: pd.DataFrame, col: str = "entropy", n_bins: int = 2) -> pd.DataFrame:
    """Tag each event peaked/flat (or n-tile) by ``col`` and summarise leader calibration."""
    if df.empty:
        return df
    if n_bins == 2:
        labels = ["peaked", "flat"]
    elif n_bins == 3:
        labels = ["peaked", "mid", "flat"]
    else:
        labels = None
    s = df.copy()
    s["shape"] = pd.qcut(s[col], n_bins, labels=labels, duplicates="drop")
    g = s.groupby("shape", observed=True)
    out = pd.DataFrame({
        "n_events": g.size(),
        "mean_entropy": g["entropy"].mean(),
        "mean_p_fav": g["p_fav"].mean(),
        "fav_win_rate": g["fav_win"].mean(),
        "resid": g["resid"].mean(),
        "brier_skill": 1.0 - g["brier_price"].sum() / g["brier_unif"].sum(),
    }).reset_index()
    return out
