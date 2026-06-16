"""
Free factor datasets cited in ``docs/knowledge/systematic-equity-strategies.md``.

Three sources, three builders (network; run once, cached in ``datasets/``):

build_ff_factors()   — Kenneth French Data Library (§4.1 of the KB):
                       US FF5 + momentum (monthly since 1926/63 AND daily —
                       daily is what strategy-return regressions need),
                       Europe 5 factors + momentum (monthly), and the 12
                       value-weighted industry portfolios (monthly).
                       The standard attribution benchmark: regress any
                       strategy on FF5+UMD before crediting skill
                       ("Buffett's Alpha" template).

build_jkp_factors()  — Jensen-Kelly-Pedersen global factor data (§4.3):
                       13 theme-level long-short factor returns, monthly,
                       value-weighted (capped), for usa / developed /
                       emerging / world_ex_us. The cross-country
                       out-of-sample validation panel.

build_openap()       — Chen-Zimmermann Open Source Asset Pricing (§4.2):
                       monthly long-short returns for 200+ published
                       predictors + the signal documentation table. The
                       ready-made input for factor-momentum and
                       meta-strategy research. Requires ``gdown``
                       (``pip install -e .[data]``) — files live on
                       Google Drive.

All return series are stored as decimals (Ken French publishes percent;
converted here), indexed by month-end (monthly) or trade date (daily).
"""
import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from posterioralpha.data.loaders import DATASETS_DIR

logger = logging.getLogger(__name__)

# ── Kenneth French ───────────────────────────────────────────────────────────

_FF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

FF_FACTORS_MONTHLY_CSV = DATASETS_DIR / "ff_factors_monthly.csv"
FF_FACTORS_DAILY_CSV   = DATASETS_DIR / "ff_factors_daily.csv.gz"
FF_EUROPE_MONTHLY_CSV  = DATASETS_DIR / "ff_europe_monthly.csv"
FF_INDUSTRY12_CSV      = DATASETS_DIR / "ff_industry12_monthly.csv"


def _fetch_ff_table(zip_name: str, daily: bool = False) -> pd.DataFrame:
    """
    Download one French-library zip and parse the FIRST data block.

    The library's CSVs have free-text preambles and multiple tables
    (monthly, then annual); the first contiguous block of YYYYMM (or
    YYYYMMDD) rows is the one we want. -99.99 / -999 are missing markers.
    """
    import requests

    r = requests.get(_FF_BASE + zip_name, timeout=120)
    r.raise_for_status()
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    raw = zf.read(zf.namelist()[0]).decode("latin-1")

    pat = re.compile(r"^\s*\d{8}\s*," if daily else r"^\s*\d{6}\s*,")
    lines = raw.splitlines()
    start = next(i for i, ln in enumerate(lines) if pat.match(ln))
    end = start
    while end < len(lines) and pat.match(lines[end]):
        end += 1

    cols = [c.strip() for c in lines[start - 1].split(",")]
    cols[0] = "Date"
    body = "\n".join([",".join(cols)] + lines[start:end])
    df = pd.read_csv(io.StringIO(body))

    fmt = "%Y%m%d" if daily else "%Y%m"
    df["Date"] = pd.to_datetime(df["Date"].astype(str).str.strip(), format=fmt)
    if not daily:
        df["Date"] = df["Date"] + pd.offsets.MonthEnd(0)
    df = df.set_index("Date").apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999.0], np.nan) / 100.0
    df.columns = [c.replace(" ", "") for c in df.columns]
    return df.sort_index()


def build_ff_factors() -> Dict[str, pd.DataFrame]:
    """Fetch and cache the four French-library tables. Returns them keyed by file."""
    out: Dict[str, pd.DataFrame] = {}

    logger.info("FF: US 5 factors + momentum (monthly) …")
    us_m = _fetch_ff_table("F-F_Research_Data_5_Factors_2x3_CSV.zip").join(
        _fetch_ff_table("F-F_Momentum_Factor_CSV.zip"), how="left")
    us_m.to_csv(FF_FACTORS_MONTHLY_CSV)
    out["ff_factors_monthly"] = us_m

    logger.info("FF: US 5 factors + momentum (daily) …")
    us_d = _fetch_ff_table("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                           daily=True).join(
        _fetch_ff_table("F-F_Momentum_Factor_daily_CSV.zip", daily=True),
        how="left")
    us_d.to_csv(FF_FACTORS_DAILY_CSV, compression="gzip")
    out["ff_factors_daily"] = us_d

    logger.info("FF: Europe 5 factors + momentum (monthly) …")
    eu_m = _fetch_ff_table("Europe_5_Factors_CSV.zip").join(
        _fetch_ff_table("Europe_Mom_Factor_CSV.zip"), how="left")
    eu_m.to_csv(FF_EUROPE_MONTHLY_CSV)
    out["ff_europe_monthly"] = eu_m

    logger.info("FF: 12 industry portfolios (monthly, value-weighted) …")
    ind = _fetch_ff_table("12_Industry_Portfolios_CSV.zip")
    ind.to_csv(FF_INDUSTRY12_CSV)
    out["ff_industry12_monthly"] = ind
    return out


# ── Jensen-Kelly-Pedersen ────────────────────────────────────────────────────

_JKP_URL = ("https://jkpfactors-data.s3.amazonaws.com/public/"
            "%5B{region}%5D_%5B{theme}%5D_%5B{freq}%5D_%5B{weight}%5D.zip")

JKP_FACTORS_CSV = DATASETS_DIR / "jkp_factors.csv.gz"
JKP_INDIV_CSV   = DATASETS_DIR / "jkp_factors_individual.csv.gz"
JKP_REGIONS = ("usa", "developed", "emerging", "world_ex_us")


def build_jkp_factors(
    regions=JKP_REGIONS,
    theme: str = "all_themes",
    weight: str = "vw_cap",
) -> pd.DataFrame:
    """
    Fetch theme-level monthly factor returns per region from the JKP public
    S3 bucket and cache them as one long-format table
    (location, name, date, ret). ``vw_cap`` is the paper's headline
    weighting (value-weighted, capped).
    """
    import requests

    frames = []
    for region in regions:
        url = _JKP_URL.format(region=region, theme=theme,
                              freq="monthly", weight=weight)
        logger.info(f"JKP: {region} / {theme} (monthly, {weight}) …")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(zf.open(zf.namelist()[0]), parse_dates=["date"])
        frames.append(df[["location", "name", "date", "ret"]])

    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(JKP_FACTORS_CSV, index=False, compression="gzip")
    return panel


def build_jkp_individual(
    regions=JKP_REGIONS,
    weight: str = "vw_cap",
) -> pd.DataFrame:
    """
    Fetch the 153 INDIVIDUAL JKP characteristic factors per region (the
    ``all_factors`` download, not the 13-theme aggregate) and cache them as
    one long table (location, name, date, ret) with ``ret`` oriented to
    positive expected return (raw ret × the published ``direction`` sign).

    This is the matched-breadth panel: ~153 factors per region rivals the
    OpenAP US panel's 212, which is what the seasonality OOS test needs
    (the 13-theme set was too thin — 4 factors per leg — to carry it).
    """
    import requests

    frames = []
    for region in regions:
        url = _JKP_URL.format(region=region, theme="all_factors",
                              freq="monthly", weight=weight)
        logger.info(f"JKP indiv: {region} / all_factors (monthly, {weight}) …")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        df = pd.read_csv(zf.open(zf.namelist()[0]), parse_dates=["date"])
        df["ret"] = df["ret"] * df["direction"]   # orient to positive premium
        frames.append(df[["location", "name", "date", "ret"]])

    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(JKP_INDIV_CSV, index=False, compression="gzip")
    return panel


# ── Chen-Zimmermann Open Source Asset Pricing ────────────────────────────────

_OPENAP_LS_GDRIVE_ID  = "10sOryk_ddjkXagaajTKUk1nwJs2ZLRiI"   # wide monthly L/S returns
_OPENAP_DOC_GDRIVE_ID = "1Sev9s6cPFUGgxp1pFiej0lGzpsMqJCI2"   # signal documentation

OPENAP_LS_CSV  = DATASETS_DIR / "openap_ls_returns.csv.gz"
OPENAP_DOC_CSV = DATASETS_DIR / "openap_signal_doc.csv"


def build_openap() -> pd.DataFrame:
    """
    Fetch the predictor long-short return matrix (wide: date × predictor,
    percent → stored as decimals) and the signal documentation table.
    Google Drive hosting → needs ``gdown`` (``pip install -e .[data]``).
    """
    try:
        import gdown
    except ImportError as exc:                      # pragma: no cover
        raise ImportError(
            "build_openap needs gdown:  pip install -e .[data]") from exc

    logger.info("OpenAP: signal documentation …")
    gdown.download(id=_OPENAP_DOC_GDRIVE_ID, output=str(OPENAP_DOC_CSV),
                   quiet=True)

    logger.info("OpenAP: monthly long-short predictor returns …")
    tmp = OPENAP_DOC_CSV.with_suffix(".tmp.csv")
    gdown.download(id=_OPENAP_LS_GDRIVE_ID, output=str(tmp), quiet=True)
    ls = pd.read_csv(tmp, parse_dates=["date"], index_col="date")
    tmp.unlink()
    ls = (ls / 100.0).sort_index()                  # published in percent
    ls.index = ls.index + pd.offsets.MonthEnd(0)
    ls.to_csv(OPENAP_LS_CSV, compression="gzip")
    return ls
