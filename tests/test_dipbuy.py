"""Unit tests for the dip-buy data layer and inference helpers (no network)."""
import numpy as np
import pandas as pd

from posterioralpha.dipbuy import cts, stats

LEGACY = """Mon Jun  1 20:04:01 EDT 2018 prod
MsgCat,MsgType,Network,SeqNum,OutputTime,Symbol,Suffix,PartId,PrClsDate,HighPrice,LowPrice,LastPrice,TotVolume,OpenPrice,Tick,FinStat,ShortSale
Equity,Con_EOD,A,1,_16:15:05.298,GS,,NYSE,,_230.25,_228.09,_228.35,3323297, , ,Not_Bkrpt,
Equity,Con_EOD,A,2,_20:04:01.177,GS,,FINRA,,_230.25,_228.09,_228.35,3330852, , ,Not_Bkrpt,
Equity,Part_EOD,A,3,_20:04:01.177,GS,,NYSE,,_230.24,_228.14,_228.35,657874,_228.40,0+
Equity,Part_EOD,A,4,_20:04:01.177,GS,,FINRA,,_230.25,_228.10,_228.30,1142842,_228.50,0+
Equity,Part_EOD,A,5,_20:04:01.177,GS,,BATS,,_230.22,_228.11,_228.34,354640,_228.45,0+
Equity,Part_EOD,A,6,_20:04:01.177,GS,,CHX,05/30/2018,_229.00,_0,_0,0,_0,0+
Equity,Con_EOD,B,7,_20:04:01.280,NG,,NYSE_MKT,,_4.85,_4.66,_4.68,1207067, , ,Not_Bkrpt,
Equity,Part_EOD,B,8,_20:04:01.280,NG,,NYSE_MKT,,_4.85,_4.66,_4.68,91898,_4.82,0+
"""
PILLAR = """Mon Oct 02 21:33:45 EDT 2023 prod
Msg Cat/Type,TransTime,SeqNum,Symbol,instrumentType,partiId,prevClosePriceDate,lastClosePrice,highPrice,lowPrice,openPrice,totalVolume,shortSaleRestrictionInd,primaryListingMarketPartiId,financialStausInd,numberOfParticipants
ConsolidatedEODSummary,_16:15:00.,_1,NG     ,Equity,FINRA,_,3.540000,3.820000,3.455000,,2100000, ,NYSE_MKT,0,16,
ConsolidatedEODSummary,_20:04:00.,_2,NG     ,Equity,FINRA,_,3.550000,3.820000,3.455000,,2169538, ,NYSE_MKT,0,16,
ParticipantEODSummary,_20:04:00.,_3,NG     ,Equity,NYSE_MKT,_,3.550000,3.820000,3.460000,3.820000,163902,,,,,0-
ParticipantEODSummary,_20:04:00.,_4,NG     ,Equity,NSX,_09/29/2023,3.700000,0.000000,0.000000,0.000000,1,,,,,-
ParticipantEODSummary,_20:04:00.,_5,NG     ,Equity,ARCA,_,3.560000,3.800000,3.460000,3.750000,50000,,,,,-
ConsolidatedEODSummary,_20:04:00.,_6,QQQX   ,Equity,ARCA,_,1,1,1,,10, ,NASDAQ,0,1,
"""


def test_parse_legacy_primary_from_network_and_last_snapshot():
    d = cts.parse_eod(LEGACY).set_index("sym")
    assert d.loc["GS", "prim"] == "NYSE"            # network A, even though last trade was FINRA
    assert d.loc["GS", "cons_vol"] == 3330852       # latest snapshot wins
    assert abs(d.loc["GS", "p_open"] - 228.40) < 1e-4
    assert d.loc["GS", "nven"] == 1                 # CHX echoes a prior close -> ignored; FINRA not an 'alt open'
    assert d.loc["NG", "prim"] == "AMEX?"           # network B resolved later
    assert abs(d.loc["NG", "p_open"] - 4.82) < 1e-4


def test_parse_pillar_and_filters():
    d = cts.parse_eod(PILLAR).set_index("sym")
    assert "QQQX" not in d.index                    # not NYSE/AMEX-listed and not a kept ETF
    assert d.loc["NG", "prim"] == "AMEX"
    assert d.loc["NG", "cons_vol"] == 2169538
    assert abs(d.loc["NG", "p_open"] - 3.82) < 1e-4
    assert abs(d.loc["NG", "alt_open_med"] - 3.75) < 1e-4   # NSX row has a prior-close date -> excluded
    assert d.loc["NG", "arca_vol"] == 50000


def test_clustered_se_inflates_with_clustering():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(200), 10)
    shock = rng.normal(0, 1, 200)[dates]             # common within-date component
    x = shock + rng.normal(0, 1, len(dates))
    c = stats.clustered_mean(pd.Series(x), pd.Series(dates))
    iid_se = x.std(ddof=1) / np.sqrt(len(x))
    assert c["se"] > 1.8 * iid_se                    # ~sqrt(1 + 9*0.5) = 2.3x
    assert c["days"] == 200


def test_parse_fractional_labels_2026():
    txt = PILLAR.replace("ConsolidatedEODSummary", "FractionalConsolidatedEODSummary").replace("ParticipantEODSummary", "FractionalParticipantEODSummary")
    d = cts.parse_eod(txt).set_index("sym")
    assert d.loc["NG", "prim"] == "AMEX" and abs(d.loc["NG", "p_open"] - 3.82) < 1e-4


def test_no_silent_empty_parse():
    """Every real trading-day file must yield NYSE rows; guards against future label changes."""
    import pytest
    from posterioralpha.dipbuy.paths import CTS_DAY_DIR
    files = sorted(CTS_DAY_DIR.glob("*.parquet"))
    if not files:
        pytest.skip("no parsed days on disk")
    import pandas as pd
    empty = [f.stem for f in files if len(pd.read_parquet(f, columns=["sym"])) == 0]
    assert not empty, f"{len(empty)} empty parsed days, first {empty[:3]}"
