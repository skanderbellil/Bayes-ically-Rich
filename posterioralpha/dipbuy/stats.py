"""Inference helpers: date-clustered standard errors and portfolio metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clustered_mean(x: pd.Series, clusters: pd.Series) -> dict:
    """Mean of x with a standard error clustered by ``clusters`` (e.g. date).

    For a pure mean, the CR0 cluster-robust variance is sum_g (sum_{i in g} e_i)^2 / N^2 (with G/(G-1) correction).
    """
    x = pd.Series(np.asarray(x, float)); c = pd.Series(np.asarray(clusters))
    ok = x.notna()
    x, c = x[ok], c[ok]
    n = len(x)
    if n < 3:
        return {"N": n, "mean": x.mean() if n else np.nan, "se": np.nan, "t": np.nan, "days": c.nunique()}
    e = x - x.mean()
    sg = e.groupby(c.values).sum()
    G = len(sg)
    var = (sg ** 2).sum() / n ** 2 * (G / (G - 1) if G > 1 else 1)
    se = np.sqrt(var)
    return {"N": n, "days": G, "mean": x.mean(), "se": se, "t": x.mean() / se if se > 0 else np.nan,
            "median": x.median(), "win": (x > 0).mean()}


def table(df: pd.DataFrame, val: str, by, cluster="date") -> pd.DataFrame:
    rows = {}
    for k, g in df.groupby(by, observed=True):
        rows[k] = clustered_mean(g[val], g[cluster])
    out = pd.DataFrame(rows).T
    if out.empty:
        return pd.DataFrame(columns=["N", "days", "mean", "se", "t", "median", "win"])
    if isinstance(by, (list, tuple)) and len(by) > 1:
        out.index = pd.MultiIndex.from_tuples(out.index, names=by)
    else:
        out.index.name = by if isinstance(by, str) else by[0]
    return out


def perf(r: pd.Series, bench: pd.Series | None = None) -> dict:
    r = r.dropna()
    c = (1 + r).cumprod(); yrs = len(r) / 252
    d = {"CAGR": c.iloc[-1] ** (1 / yrs) - 1, "vol": r.std() * np.sqrt(252),
         "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
         "maxDD": (c / c.cummax() - 1).min(), "x$1": c.iloc[-1]}
    if bench is not None:
        d["corr_idx"] = r.corr(bench.reindex(r.index))
    return d
