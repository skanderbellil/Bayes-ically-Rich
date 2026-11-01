"""Security-type map (common stock vs ETF / notes / preferred / units / warrants ...).

Sources: symbol-mapping messages (type 3) in the NYSE XDP sample files (2020-09-09, 2023-10-02/03,
2026-07-01/02) and NYSE's current NYSE / NYSE American symbol-mapping files. XDP security types:
C common, A ADR, H ADS, O ordinary shares, S shares of beneficial interest  -> kept as equity;
E ETF, M misc/notes/trust, P preferred, U (closed-end funds in these files, e.g. ADX, AVK), W warrant, R rights, I index-linked/other, D debt,
L, T test -> excluded. Symbols seen in none of these (mostly gone before 2020) stay 'unknown'.
"""
from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd

EQUITY_TYPES = {"C", "A", "H", "O", "S"}


def _norm(sym: str) -> str:
    return sym.strip().replace(" ", "/").replace(".", "/")


def from_xdp(path: Path) -> pd.DataFrame:
    op = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with op(path, "rt") as f:
        for line in f:
            if line.startswith("3,"):
                p = line.split(",")
                rows.append((_norm(p[2]), p[6]))
    return pd.DataFrame(rows, columns=["sym", "stype"]).drop_duplicates()


def from_mapping(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="|", header=None, dtype=str, keep_default_na=False)
    d = d[d[0].str.len() > 0]
    return pd.DataFrame({"sym": d[0].map(_norm), "stype": d[9]}).dropna().drop_duplicates()


def build(xdp_dir: Path) -> pd.DataFrame:
    parts = []
    for f in sorted(xdp_dir.iterdir()):
        n = f.name
        if n.startswith(("EQY_US_TAQ_NYSE_TRADES", "EQY_US_TAQ_AMEX_TRADES", "EQY_US_NYSE_REF_IMB", "EQY_US_AMEX_REF_IMB", "NYSE_IMBALANCES")):
            parts.append(from_xdp(f))
        elif "SymbolMapping" in n and n.endswith(".txt"):
            parts.append(from_mapping(f))
    t = pd.concat(parts).drop_duplicates()
    g = t.groupby("sym").stype.agg(lambda s: "".join(sorted(set(s))))
    out = pd.DataFrame({"stypes": g})
    out["equity"] = out.stypes.map(lambda s: any(c in EQUITY_TYPES for c in s))
    return out
