"""
Configuration for the "Ricardo with a Bloomberg terminal" stress test.

THESIS (stated for the record, so every diagnostic below can be read against it):
    In the AI buildout, economic rents migrate toward INELASTIC SUPPLY -- power
    generation, fuel, the grid, and physical build-out chokepoints (transformers,
    switchgear, cable, cooling, lithography). Meanwhile LEVERAGED COMPUTE MIDDLEMEN
    (neoclouds renting depreciating GPUs on debt) and AI-DISPLACED LABOR-ARBITRAGE
    businesses get repriced downward. So: long inelastic supply, short fragile
    middlemen + labor arbitrage.

This module is pure data (no logic) so the universe is auditable at a glance.

ASSUMPTIONS / DESIGN NOTES
--------------------------
* Universe currencies exceed the FX list given in the brief. The brief lists
  EUR/JPY/KRW/GBP/SEK/DKK, but the universe also contains CHF (ADEN.SW),
  HKD (0522.HK) and GBp/pence (RR.L). Rather than leave those unconverted, the
  data layer AUTO-DETECTS each ticker's quote currency from Yahoo metadata and
  fetches the matching `<CUR>USD=X` pair. GBp (London pence) is handled as
  GBP / 100. See data.py.
* Tiers/books are the analytic groupings the thesis is built on; they drive the
  tier-attribution and book-construction steps.
"""

# ---------------------------------------------------------------------------
# Backtest window and economic parameters
# ---------------------------------------------------------------------------
START_DATE = "2026-01-01"
END_DATE = "2026-07-12"          # "today" for this run (see brief)

RISK_FREE_ANNUAL = 0.04          # rf = 4% for Sharpe
TRADING_DAYS = 252               # annualization factor

# Borrow-cost haircut on the SHORT book (annualized, charged daily apr/252).
BORROW_APR_DEFAULT = 0.04        # 4% general collateral
BORROW_APR_HTB = 0.15            # 15% hard-to-borrow
HARD_TO_BORROW = {"CRWV", "NBIS", "IREN"}

BETA_WINDOW = 60                 # rolling window (days) for beta-neutral scaling
ROLLING_BETA_PLOT_WINDOW = 20    # rolling 20d beta chart

# Data-cleaning thresholds
FFILL_LIMIT = 3                  # forward-fill at most 3 days across holidays
MAX_MISSING_FRAC = 0.10          # drop any ticker missing >10% of the calendar

# ---------------------------------------------------------------------------
# Universe:  ticker -> {book, tier, note}
# `book` is the coarse grouping (long tiers vs short baskets); `tier` is the
# finer thesis bucket used for attribution.
# ---------------------------------------------------------------------------
UNIVERSE = {
    # ================= LONGS: inelastic supply =================
    # ---- Tier 1: generation + fuel (the scarce electron / fuel) ----
    "GEV":     dict(book="LONG", tier="Tier1_gen_fuel", note="GE Vernova - gas+grid turbines"),
    "ENR.DE":  dict(book="LONG", tier="Tier1_gen_fuel", note="Siemens Energy - turbines/grid (EUR)"),
    "7011.T":  dict(book="LONG", tier="Tier1_gen_fuel", note="Mitsubishi Heavy - power/nuclear (JPY)"),
    "CCJ":     dict(book="LONG", tier="Tier1_gen_fuel", note="Cameco - uranium"),
    "KAP.L":   dict(book="LONG", tier="Tier1_gen_fuel", note="Kazatomprom GDR - uranium (USD/thin)"),
    "LEU":     dict(book="LONG", tier="Tier1_gen_fuel", note="Centrus - enrichment"),
    "BWXT":    dict(book="LONG", tier="Tier1_gen_fuel", note="BWX Tech - nuclear components"),
    "CEG":     dict(book="LONG", tier="Tier1_gen_fuel", note="Constellation - nuclear IPP"),
    "VST":     dict(book="LONG", tier="Tier1_gen_fuel", note="Vistra - IPP"),
    "TLN":     dict(book="LONG", tier="Tier1_gen_fuel", note="Talen - IPP"),

    # ---- Tier 2: grid / transmission / conductor & copper ----
    "ETN":      dict(book="LONG", tier="Tier2_grid", note="Eaton - electrical"),
    "6501.T":   dict(book="LONG", tier="Tier2_grid", note="Hitachi - grid/HVDC (JPY)"),
    "267260.KS":dict(book="LONG", tier="Tier2_grid", note="HD Hyundai Electric - transformers (KRW)"),
    "298040.KS":dict(book="LONG", tier="Tier2_grid", note="Hyosung Heavy - transformers (KRW)"),
    "010120.KS":dict(book="LONG", tier="Tier2_grid", note="LS Electric - switchgear (KRW)"),
    "POWL":     dict(book="LONG", tier="Tier2_grid", note="Powell - electrical systems"),
    "CLF":      dict(book="LONG", tier="Tier2_grid", note="Cleveland-Cliffs - GOES steel"),
    "PRY.MI":   dict(book="LONG", tier="Tier2_grid", note="Prysmian - cable (EUR)"),
    "NEX.PA":   dict(book="LONG", tier="Tier2_grid", note="Nexans - cable (EUR)"),
    "NKT.CO":   dict(book="LONG", tier="Tier2_grid", note="NKT - HV cable (DKK)"),
    "PWR":      dict(book="LONG", tier="Tier2_grid", note="Quanta - T&D construction"),
    "MYRG":     dict(book="LONG", tier="Tier2_grid", note="MYR Group - T&D construction"),
    "MTZ":      dict(book="LONG", tier="Tier2_grid", note="MasTec - infrastructure"),
    "FCX":      dict(book="LONG", tier="Tier2_grid", note="Freeport - copper"),
    "SCCO":     dict(book="LONG", tier="Tier2_grid", note="Southern Copper - copper"),

    # ---- Tier 3: datacenter building / thermal / power gear ----
    "VRT":     dict(book="LONG", tier="Tier3_building", note="Vertiv - cooling/power"),
    "MOD":     dict(book="LONG", tier="Tier3_building", note="Modine - thermal"),
    "NVT":     dict(book="LONG", tier="Tier3_building", note="nVent - electrical connect"),
    "SPXC":    dict(book="LONG", tier="Tier3_building", note="SPX Tech - HVAC/cooling"),
    "MTRS.ST": dict(book="LONG", tier="Tier3_building", note="Munters - datacenter cooling (SEK)"),
    "6367.T":  dict(book="LONG", tier="Tier3_building", note="Daikin - cooling (JPY)"),
    "CAT":     dict(book="LONG", tier="Tier3_building", note="Caterpillar - gensets"),
    "CMI":     dict(book="LONG", tier="Tier3_building", note="Cummins - gensets"),
    "GNRC":    dict(book="LONG", tier="Tier3_building", note="Generac - power gen"),
    "RR.L":    dict(book="LONG", tier="Tier3_building", note="Rolls-Royce - SMR/power (GBp/pence)"),
    "BE":      dict(book="LONG", tier="Tier3_building", note="Bloom Energy - fuel cells"),
    "XYL":     dict(book="LONG", tier="Tier3_building", note="Xylem - water/cooling"),

    # ---- Tier 4: silicon chokepoints (litho / packaging / materials) ----
    "ASML":    dict(book="LONG", tier="Tier4_silicon", note="ASML - EUV litho"),
    "TSM":     dict(book="LONG", tier="Tier4_silicon", note="TSMC ADR - foundry"),
    "BESI.AS": dict(book="LONG", tier="Tier4_silicon", note="BE Semiconductor - hybrid bonding (EUR)"),
    "0522.HK": dict(book="LONG", tier="Tier4_silicon", note="ASMPT - advanced packaging (HKD)"),
    "6146.T":  dict(book="LONG", tier="Tier4_silicon", note="Disco - dicing/grinding (JPY)"),
    "6315.T":  dict(book="LONG", tier="Tier4_silicon", note="TOWA - molding (JPY)"),
    "4062.T":  dict(book="LONG", tier="Tier4_silicon", note="Ibiden - ABF substrate (JPY)"),
    "2802.T":  dict(book="LONG", tier="Tier4_silicon", note="Ajinomoto - ABF film (JPY)"),
    "AMKR":    dict(book="LONG", tier="Tier4_silicon", note="Amkor - OSAT packaging"),
    "KLA":     dict(book="LONG", tier="Tier4_silicon", note="KLA - process control"),
    "MU":      dict(book="LONG", tier="Tier4_silicon", note="Micron - HBM memory"),

    # ================= SHORTS =================
    # ---- Fragile middle: leveraged compute middlemen (neoclouds) ----
    "ORCL":    dict(book="SHORT", tier="fragile_middle", note="Oracle - debt-funded DC capex"),
    "CRWV":    dict(book="SHORT", tier="fragile_middle", note="CoreWeave - neocloud (HTB)"),
    "NBIS":    dict(book="SHORT", tier="fragile_middle", note="Nebius - neocloud (HTB)"),
    "IREN":    dict(book="SHORT", tier="fragile_middle", note="IREN - miner->AI (HTB)"),
    "APLD":    dict(book="SHORT", tier="fragile_middle", note="Applied Digital - DC host"),
    "WULF":    dict(book="SHORT", tier="fragile_middle", note="TeraWulf - miner->AI"),
    "CIFR":    dict(book="SHORT", tier="fragile_middle", note="Cipher - miner->AI"),

    # ---- Labor arbitrage: AI-displaceable services ----
    "CNXC":    dict(book="SHORT", tier="labor_arbitrage", note="Concentrix - BPO/call centers"),
    "TEP.PA":  dict(book="SHORT", tier="labor_arbitrage", note="Teleperformance - BPO (EUR)"),
    "RHI":     dict(book="SHORT", tier="labor_arbitrage", note="Robert Half - staffing"),
    "ADEN.SW": dict(book="SHORT", tier="labor_arbitrage", note="Adecco - staffing (CHF)"),
    "RAND.AS": dict(book="SHORT", tier="labor_arbitrage", note="Randstad - staffing (EUR)"),
    "FVRR":    dict(book="SHORT", tier="labor_arbitrage", note="Fiverr - gig marketplace"),
    "UPWK":    dict(book="SHORT", tier="labor_arbitrage", note="Upwork - gig marketplace"),
    "CHGG":    dict(book="SHORT", tier="labor_arbitrage", note="Chegg - edtech (AI-displaced)"),
}

# Benchmarks. SPY is the primary; "BLEND" is an equal-weight of QQQ and XLI.
BENCH_TICKERS = ["SPY", "QQQ", "XLI"]

# Convenience views -----------------------------------------------------------
def tickers_by_book(book):
    return [t for t, m in UNIVERSE.items() if m["book"] == book]

def tickers_by_tier(tier):
    return [t for t, m in UNIVERSE.items() if m["tier"] == tier]

LONG_TICKERS = tickers_by_book("LONG")
SHORT_TICKERS = tickers_by_book("SHORT")
ALL_EQUITY_TICKERS = list(UNIVERSE.keys())

LONG_TIERS = ["Tier1_gen_fuel", "Tier2_grid", "Tier3_building", "Tier4_silicon"]
SHORT_TIERS = ["fragile_middle", "labor_arbitrage"]
