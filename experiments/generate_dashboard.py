#!/usr/bin/env python3
"""Generate a self-contained HTML dashboard from the three paper-trade ledgers.

Output: data/paper_trade/dashboard.html
Run directly or called by the cron workflows after each ledger update.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "paper_trade"
OUT  = DATA / "dashboard.html"

REPO        = "skanderbellil/Bayes-ically-Rich"
ACTIONS_URL = f"https://github.com/{REPO}/actions/workflows/close_position.yml"


# ---------------------------------------------------------------------------
# Load & compute MTM for one ledger
# ---------------------------------------------------------------------------

def load_strategy(path: Path, entry_col: str, q_col: str,
                  label: str, strategy_id: str) -> dict:
    if not path.exists():
        return {"label": label, "strategy_id": strategy_id,
                "open": [], "resolved": [], "unrealized": 0,
                "realized": 0, "nW": 0, "nL": 0, "nF": 0}
    df = pd.read_csv(path)
    for c in [entry_col, "current_price", "pnl", "bet_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bet_fraction" not in df.columns or df["bet_fraction"].isna().all():
        df["bet_fraction"] = 0.10

    opened   = df[df["status"] == "open"].copy()
    resolved = df[df["status"].notna() & (df["status"] != "open")].copy()
    opened["mtm"] = (opened["current_price"] / opened[entry_col] - 1) * opened["bet_fraction"]

    def safe_str(val, fallback=""):
        return str(val) if pd.notna(val) else fallback

    def row_to_dict(r, include_mtm=False):
        d = {
            "question":   str(r[q_col]),
            "entry":      round(float(r[entry_col]), 4) if pd.notna(r[entry_col]) else None,
            "current":    round(float(r["current_price"]), 4) if "current_price" in r and pd.notna(r["current_price"]) else None,
            "status":     str(r["status"]),
            "pnl":        round(float(r["pnl"]), 4) if pd.notna(r.get("pnl")) else None,
            "exit_date":  safe_str(r.get("exit_date")),
            "entry_date": safe_str(r.get("entry_date")),
            "token":      str(r["token"]) if "token" in r.index and pd.notna(r.get("token")) else "",
            "fraction":   round(float(r["bet_fraction"]), 4) if "bet_fraction" in r.index and pd.notna(r.get("bet_fraction")) else 0.1,
        }
        if include_mtm:
            d["mtm"] = round(float(r["mtm"]), 4) if pd.notna(r.get("mtm")) else None
        return d

    open_rows     = [row_to_dict(r, include_mtm=True)
                     for _, r in opened.sort_values("mtm", ascending=False).iterrows()]
    resolved_rows = [row_to_dict(r)
                     for _, r in resolved.iterrows()]

    return {
        "label":       label,
        "strategy_id": strategy_id,
        "open":        open_rows,
        "resolved":    resolved_rows,
        "unrealized":  round(float(opened["mtm"].sum()), 4),
        "realized":    round(float(resolved["pnl"].dropna().sum()), 4),
        "nW": int((resolved["status"] == "won").sum()),
        "nL": int((resolved["status"] == "lost").sum()),
        "nF": int((resolved["status"] == "flipped").sum()),
    }


def load_dip_confirm(path: Path, label: str = "Dip-Confirm YES",
                     strategy_id: str = "dip_confirm") -> dict:
    """Load the dip-confirm ledger, which has an extra 'watching' status."""
    base = {"label": label, "strategy_id": strategy_id,
            "watching": [], "open": [], "resolved": [],
            "unrealized": 0, "realized": 0, "nW": 0, "nL": 0, "nExpired": 0,
            "primary": False}
    if not path.exists():
        return base
    df = pd.read_csv(path)
    for c in ["entry_ask", "watch_price", "min_price", "current_price", "pnl", "bet_fraction"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "bet_fraction" not in df.columns or df["bet_fraction"].isna().all():
        df["bet_fraction"] = 0.10

    def safe(val, fallback=""):
        return str(val) if pd.notna(val) else fallback

    watching_rows = []
    for _, r in df[df["status"] == "watching"].iterrows():
        wp  = r["watch_price"] if pd.notna(r.get("watch_price")) else None
        mp  = r["min_price"]   if pd.notna(r.get("min_price"))   else None
        cp  = r["current_price"] if pd.notna(r.get("current_price")) else None
        dip = round((wp - mp) / wp, 4) if (wp and mp and wp > 0) else 0
        watching_rows.append({
            "question":    str(r["question"]),
            "watch_date":  safe(r.get("watch_date")),
            "watch_price": round(float(wp), 4) if wp else None,
            "min_price":   round(float(mp), 4) if mp else None,
            "current":     round(float(cp), 4) if cp else None,
            "dip_pct":     dip,
            "status":      "watching",
        })

    opened = df[df["status"] == "open"].copy()
    if not opened.empty and "entry_ask" in opened.columns:
        opened["mtm"] = (opened["current_price"] / opened["entry_ask"] - 1) * opened["bet_fraction"]
    else:
        opened["mtm"] = 0.0

    open_rows = []
    for _, r in opened.sort_values("mtm", ascending=False).iterrows():
        open_rows.append({
            "question":    str(r["question"]),
            "entry_date":  safe(r.get("entry_date")),
            "watch_price": round(float(r["watch_price"]), 4) if pd.notna(r.get("watch_price")) else None,
            "entry":       round(float(r["entry_ask"]), 4)   if pd.notna(r.get("entry_ask"))   else None,
            "current":     round(float(r["current_price"]), 4) if pd.notna(r.get("current_price")) else None,
            "mtm":         round(float(r["mtm"]), 4) if pd.notna(r.get("mtm")) else None,
            "token":       str(r["token"]) if "token" in r.index and pd.notna(r.get("token")) else "",
            "fraction":    round(float(r["bet_fraction"]), 4) if pd.notna(r.get("bet_fraction")) else 0.1,
            "status":      "open",
        })

    resolved = df[df["status"].isin(["won", "lost", "expired"])].copy()
    resolved_rows = []
    for _, r in resolved.iterrows():
        resolved_rows.append({
            "question":   str(r["question"]),
            "entry_date": safe(r.get("entry_date")),
            "exit_date":  safe(r.get("exit_date")),
            "watch_price": round(float(r["watch_price"]), 4) if pd.notna(r.get("watch_price")) else None,
            "entry":      round(float(r["entry_ask"]), 4) if pd.notna(r.get("entry_ask")) else None,
            "status":     str(r["status"]),
            "pnl":        round(float(r["pnl"]), 4) if pd.notna(r.get("pnl")) else None,
        })

    return {
        **base,
        "watching":    watching_rows,
        "open":        open_rows,
        "resolved":    resolved_rows,
        "unrealized":  round(float(opened["mtm"].sum()), 4),
        "realized":    round(float(resolved[resolved["status"].isin(["won","lost"])]["pnl"].dropna().sum()), 4),
        "nW":          int((resolved["status"] == "won").sum()),
        "nL":          int((resolved["status"] == "lost").sum()),
        "nExpired":    int((resolved["status"] == "expired").sum()),
    }


# ---------------------------------------------------------------------------
# Load backtest results
# ---------------------------------------------------------------------------

def load_band_sweep() -> list:
    """Load band sweep results for backtest comparison (5d horizon, hold to resolution)."""
    path = DATA / "band_sweep.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path)
    bands_of_interest = [
        (0.10, 0.20), (0.10, 0.30), (0.10, 0.50),
        (0.20, 0.40), (0.30, 0.50), (0.10, 0.90), (0.50, 0.90),
    ]
    rows = []
    for lo, hi in bands_of_interest:
        sub = df[(df["lo"] == lo) & (df["hi"] == hi) & (df["horizon_d"] == 5)]
        if sub.empty or int(sub.iloc[0]["n"]) < 5:
            continue
        r = sub.iloc[0]
        rows.append({
            "band":     f"[{lo:.2f},{hi:.2f})",
            "n":        int(r["n"]),
            "win_rate": round(float(r["win_rate"]), 4),
            "sharpe":   round(float(r["sharpe"]), 3),
            "mean_ret": round(float(r["mean_ret"]), 4),
            "live":     (lo == 0.10 and hi == 0.50),
        })
    return rows


def load_midprice_backtest() -> dict:
    """Load mid-priced YES historical backtest (5d horizon, [0.10, 0.50) band)."""
    raw_path     = DATA / "midprice_backtest.csv"
    summary_path = DATA / "midprice_backtest_summary.csv"
    bench_path   = DATA / "midprice_yes_benchmark.csv"
    BET = 0.10

    if not all(p.exists() for p in [raw_path, summary_path, bench_path]):
        return {"available": False}

    bench = pd.read_csv(bench_path).iloc[0]

    # Build flat-bet cumulative equity curve: 5d horizon, [0.10, 0.50) band
    raw  = pd.read_csv(raw_path)
    mask = (raw["horizon_d"] == 5) & (raw["price"] >= 0.10) & (raw["price"] < 0.50)
    tr   = raw[mask].copy().sort_values("t_res")
    tr["pnl"] = np.where(tr["outcome"] == 1,
                         BET * (1.0 / tr["price"] - 1.0), -BET)

    # Aggregate same-day resolutions, then cumulate
    by_date = (tr.groupby("t_res")["pnl"]
                 .sum()
                 .cumsum()
                 .reset_index()
                 .rename(columns={"pnl": "cum", "t_res": "date"}))
    equity = [{"date": r["date"], "cum": round(r["cum"], 4)}
              for _, r in by_date.iterrows()]

    # Per-horizon stats for [0.10, 0.50) band
    summ = pd.read_csv(summary_path)
    band = summ[(summ["lo"] == 0.10) & (summ["hi"] == 0.50)].sort_values("horizon_d")
    by_horizon = [{"h": int(r["horizon_d"]), "n": int(r["n"]),
                   "wr": round(float(r["win_rate"]), 4),
                   "sharpe": round(float(r["sharpe"]), 3)}
                  for _, r in band.iterrows()]

    return {
        "available":  True,
        "n":          int(bench["n"]),
        "win_rate":   round(float(bench["win_rate"]), 4),
        "mean_ret":   round(float(bench["mean_ret"]) * BET, 4),
        "sharpe":     round(float(bench["sharpe"]), 3),
        "period":     f"{equity[0]['date'][:7]} – {equity[-1]['date'][:7]}",
        "equity":     equity,
        "by_horizon": by_horizon,
    }


def load_regime_backtest() -> dict:
    """Regime-calm disruptive-tail study: per-domain matched-proxy validation +
    the geopolitics audit headline (from validated_domains.json)."""
    vpath = ROOT / "data" / "polymarket_calibration_snapshot" / "validated_domains.json"
    if not vpath.exists():
        return {"available": False}
    info = json.loads(vpath.read_text())
    per = info.get("per_domain", {})
    rows = []
    for dom, r in per.items():
        rows.append({
            "domain":    dom,
            "proxy":     r.get("proxy", "—"),
            "verdict":   r.get("verdict", "—"),
            "n_calm":    r.get("n_calm"),
            "base_edge": r.get("base_edge"),
            "base_t":    r.get("base_t"),
            "gap":       r.get("gap"),
            "oos_edge":  r.get("oos_edge"),
            "oos_t":     r.get("oos_t"),
            "net_5c":    r.get("net_5c"),
            "boot_p":    r.get("boot_p"),
            "top1":      r.get("top1"),
            "reason":    r.get("reason", ""),
        })
    # verdict order: validated first, then by base_t desc
    rows.sort(key=lambda x: (x["verdict"] != "VALIDATED", -(x["base_t"] or -99)))
    return {
        "available":  True,
        "generated":  info.get("generated", ""),
        "method":     info.get("method", ""),
        "max_price":  info.get("max_price"),
        "validated":  info.get("validated_domains", []),
        "per_domain": rows,
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<title>Paper-Trade Dashboard</title>
<style>
:root {
  --bg:#0f1117; --card:#1a1d27; --border:#2a2d3e;
  --green:#22c55e; --red:#ef4444; --yellow:#f59e0b;
  --blue:#3b82f6; --text:#e2e8f0; --muted:#94a3b8; --flip:#a78bfa;
}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;font-size:14px;padding-bottom:72px;}

/* ── PAGES ── */
.page{display:none;padding:16px 16px 28px;}
.page.active{display:block;}
.page-hdr{margin-bottom:14px;}
.page-hdr h1{font-size:20px;font-weight:700;}
.page-hdr h2{font-size:17px;font-weight:700;}
.updated{font-size:11px;color:var(--muted);margin-top:2px;}

/* ── BOTTOM NAV ── */
.bottom-nav{
  position:fixed;bottom:0;left:0;right:0;
  background:var(--card);border-top:1px solid var(--border);
  display:flex;z-index:50;
  padding-bottom:env(safe-area-inset-bottom,0px);
}
.nav-btn{
  flex:1;background:none;border:none;padding:10px 0 8px;
  display:flex;flex-direction:column;align-items:center;gap:2px;
  cursor:pointer;color:var(--muted);transition:color .15s;
  -webkit-tap-highlight-color:transparent;
}
.nav-btn.active{color:var(--blue);}
.nav-icon{font-size:18px;line-height:1;}
.nav-lbl{font-size:10px;font-weight:500;}

/* ── DESKTOP TABS (sticky header, wide screens only) ── */
.dtabs-wrap{
  display:none;background:var(--card);border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:40;
}
.dtabs{display:flex;max-width:1200px;margin:0 auto;padding:0 20px;}
.dtab{
  background:none;border:none;padding:14px 16px;font-size:13px;font-weight:500;
  color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;
  margin-bottom:-1px;transition:color .15s;white-space:nowrap;
}
.dtab.active{color:var(--blue);border-bottom-color:var(--blue);}

/* ── CARDS ── */
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:12px;}
.card-title{font-size:14px;font-weight:600;margin-bottom:10px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;}

/* ── KPIs ── */
.port-kpis{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:14px;}
.port-kpi{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:12px 14px;}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:10px;}
.kpi{background:rgba(255,255,255,.04);border-radius:8px;padding:10px 12px;}
.kpi-lbl{font-size:10px;color:var(--muted);margin-bottom:2px;text-transform:uppercase;letter-spacing:.4px;}
.kpi-val{font-size:18px;font-weight:700;}
.port-kpi .kpi-lbl{font-size:10px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.4px;}
.port-kpi .kpi-val{font-size:22px;font-weight:700;}

/* ── BADGES & PILLS ── */
.badge{font-size:11px;padding:2px 8px;border-radius:99px;font-weight:500;}
.badge-green{background:rgba(34,197,94,.18);color:var(--green);}
.badge-red{background:rgba(239,68,68,.18);color:var(--red);}
.badge-grey{background:rgba(148,163,184,.12);color:var(--muted);}
.pill{display:inline-block;font-size:10px;padding:1px 6px;border-radius:4px;font-weight:600;}
.pill-won{background:rgba(34,197,94,.18);color:var(--green);}
.pill-lost{background:rgba(239,68,68,.18);color:var(--red);}
.pill-flipped{background:rgba(167,139,250,.18);color:var(--flip);}
.pill-open{background:rgba(59,130,246,.12);color:var(--blue);}

/* ── COLOR HELPERS ── */
.pos{color:var(--green);}.neg{color:var(--red);}.neu{color:var(--text);}

/* ── W/L/F BAR ── */
.wlf-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;}
.wlf-bar{flex:1;height:5px;border-radius:3px;overflow:hidden;display:flex;gap:1px;}
.wlf-seg{height:100%;border-radius:2px;}
.wlf-lbl{font-size:11px;color:var(--muted);white-space:nowrap;}

/* ── TABLES ── */
.tbl-wrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin-top:8px;}
table{width:100%;border-collapse:collapse;font-size:12px;}
th{text-align:left;padding:5px 8px;color:var(--muted);font-weight:500;border-bottom:1px solid var(--border);white-space:nowrap;}
td{padding:5px 8px;border-bottom:1px solid rgba(42,45,62,.6);vertical-align:middle;}
tr:last-child td{border-bottom:none;}
.q{max-width:180px;word-break:break-word;}

/* ── STRATEGY FILTER PILLS ── */
.seg{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px;}
.seg-btn{
  background:var(--card);border:1px solid var(--border);border-radius:6px;
  padding:5px 11px;font-size:12px;font-weight:500;color:var(--muted);
  cursor:pointer;transition:all .15s;-webkit-tap-highlight-color:transparent;
}
.seg-btn.active{background:var(--blue);border-color:var(--blue);color:#fff;}

/* ── CUT BUTTON ── */
.cut-btn{
  background:rgba(239,68,68,.1);color:var(--red);
  border:1px solid rgba(239,68,68,.25);border-radius:4px;
  padding:3px 8px;font-size:10px;font-weight:600;
  cursor:pointer;white-space:nowrap;transition:background .15s;
}
.cut-btn:hover{background:rgba(239,68,68,.25);}
.expand-btn{
  background:rgba(59,130,246,.1);color:#60a5fa;
  border:1px solid rgba(59,130,246,.25);border-radius:4px;
  padding:5px 8px;font-size:11px;cursor:pointer;
  transition:background .15s;margin-right:4px;
  min-width:28px;min-height:28px;line-height:1;
  touch-action:manipulation;-webkit-tap-highlight-color:transparent;
}
.expand-btn:hover,.expand-btn:active{background:rgba(59,130,246,.25);}

/* ── MODAL (slides up from bottom on mobile) ── */
.modal-backdrop{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
  z-index:100;align-items:flex-end;justify-content:center;
}
.modal-backdrop.open{display:flex;}
.modal{
  background:#1a1d27;border:1px solid #2a2d3e;
  border-radius:16px 16px 0 0;padding:24px;
  width:100%;max-height:90vh;overflow-y:auto;
}
.modal h3{font-size:15px;margin-bottom:6px;}
.modal-q{font-size:12px;color:var(--muted);margin-bottom:16px;line-height:1.5;}
.modal-section{margin-bottom:14px;}
.modal-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin-bottom:5px;}
.code-box{
  background:#0f1117;border:1px solid #2a2d3e;border-radius:6px;
  padding:10px 12px;font-family:'Menlo','Consolas',monospace;
  font-size:11px;color:#e2e8f0;word-break:break-all;line-height:1.6;white-space:pre-wrap;
}
.modal-actions{display:flex;gap:8px;margin-top:16px;flex-wrap:wrap;}
.btn{
  padding:10px 14px;border-radius:8px;font-size:13px;font-weight:600;
  cursor:pointer;border:none;transition:background .15s;
  -webkit-tap-highlight-color:transparent;
}
.btn-copy{background:rgba(59,130,246,.18);color:var(--blue);border:1px solid rgba(59,130,246,.35);}
.btn-copy:hover{background:rgba(59,130,246,.32);}
.btn-gh{
  background:rgba(34,197,94,.12);color:var(--green);
  border:1px solid rgba(34,197,94,.28);text-decoration:none;
  display:inline-flex;align-items:center;gap:5px;
}
.btn-gh:hover{background:rgba(34,197,94,.25);}
.btn-cancel{background:rgba(255,255,255,.05);color:var(--muted);border:1px solid #2a2d3e;}
.btn-cancel:hover{background:rgba(255,255,255,.1);}

/* ── EQUITY CHART ── */
.eq-sub{font-size:12px;color:var(--muted);margin-bottom:10px;}

/* ── META ── */
.meta{font-size:11px;color:var(--muted);margin-top:4px;line-height:1.7;padding:14px 16px;border-top:1px solid var(--border);}

/* ── DESKTOP OVERRIDES ── */
@media(min-width:640px){
  body{padding-bottom:0;}
  .bottom-nav{display:none;}
  .dtabs-wrap{display:block;}
  .page{max-width:1200px;margin:0 auto;padding:24px;}
  .port-kpis{grid-template-columns:repeat(4,1fr);}
  .modal-backdrop{align-items:center;padding:20px;}
  .modal{border-radius:14px;max-width:520px;max-height:none;}
  .strategy-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:16px;}
  .strategy-grid .card{margin-bottom:0;}
  .kpi-val{font-size:20px;}
}
@media(max-width:380px){
  .hide-xs{display:none;}
}
/* ── BACKTEST PAGE ── */
.bt-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;}
.bt-kpi{background:rgba(255,255,255,.04);border-radius:8px;padding:10px 12px;text-align:center;}
.bt-kpi-lbl{font-size:10px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.4px;}
.bt-kpi-val{font-size:17px;font-weight:700;}
.horizon-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid rgba(42,45,62,.5);}
.horizon-row:last-child{border-bottom:none;}
.horizon-d{width:26px;font-size:11px;color:var(--muted);flex-shrink:0;text-align:right;}
.horizon-bar-wrap{flex:1;height:6px;background:#2a2d3e;border-radius:3px;overflow:hidden;}
.horizon-bar{height:100%;border-radius:3px;background:var(--blue);transition:width .4s;}
.horizon-bar.live{background:var(--green);}
.horizon-stats{font-size:11px;color:var(--muted);white-space:nowrap;min-width:120px;text-align:right;}
.horizon-live-tag{font-size:9px;font-weight:700;color:var(--green);margin-left:4px;
                   background:rgba(34,197,94,.15);padding:1px 5px;border-radius:3px;}
.bt-section-lbl{font-size:10px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);
                  margin:14px 0 8px;font-weight:600;}
.bt-note{font-size:11px;color:var(--muted);line-height:1.6;padding:10px 12px;
          background:rgba(255,255,255,.03);border-radius:8px;border-left:3px solid var(--border);}
/* ── SCHEDULE CARD ── */
.sched-row{display:flex;align-items:center;padding:9px 0;border-bottom:1px solid rgba(42,45,62,.6);}
.sched-row:last-child{border-bottom:none;}
.sched-dot{width:8px;height:8px;border-radius:50%;margin-right:10px;flex-shrink:0;}
.sched-name{flex:1;font-size:13px;}
.sched-time{font-size:12px;color:var(--muted);margin-right:12px;white-space:nowrap;}
.sched-cd{font-size:12px;font-weight:600;min-width:58px;text-align:right;font-variant-numeric:tabular-nums;}
.sched-cd.soon{color:var(--yellow);}
.sched-cd.now{color:var(--green);}
.sched-run{font-size:11px;color:var(--blue);text-decoration:none;margin-left:10px;
           padding:3px 8px;border:1px solid rgba(59,130,246,.3);border-radius:6px;
           white-space:nowrap;flex-shrink:0;transition:background .15s;
           -webkit-tap-highlight-color:transparent;}
.sched-run:hover{background:rgba(59,130,246,.15);}
/* ── HEADER + RELOAD ── */
.hdr-row{display:flex;align-items:flex-start;justify-content:space-between;gap:12px;}
.reload-btn{background:rgba(59,130,246,.12);color:var(--blue);
            border:1px solid rgba(59,130,246,.3);border-radius:8px;
            padding:7px 12px;font-size:12px;font-weight:600;cursor:pointer;
            display:inline-flex;align-items:center;gap:6px;white-space:nowrap;flex-shrink:0;
            -webkit-tap-highlight-color:transparent;transition:background .15s;}
.reload-btn:hover{background:rgba(59,130,246,.22);}
.reload-btn:active{background:rgba(59,130,246,.3);}
.reload-btn .spin{display:inline-block;}
.reload-btn.loading{pointer-events:none;opacity:.7;}
.reload-btn.loading .spin{animation:spin .7s linear infinite;}
@keyframes spin{to{transform:rotate(360deg);}}
.sched-note{font-size:10px;color:var(--muted);margin-top:9px;line-height:1.55;}
</style>
</head>
<body>

<!-- Desktop sticky tab bar -->
<div class="dtabs-wrap">
  <div class="dtabs">
    <button class="dtab active" onclick="showPage('overview')" id="dtab-overview">📊 Overview</button>
    <button class="dtab" onclick="showPage('open')" id="dtab-open">📋 Open</button>
    <button class="dtab" onclick="showPage('equity')" id="dtab-equity">📈 Equity</button>
    <button class="dtab" onclick="showPage('history')" id="dtab-history">🏁 History</button>
    <button class="dtab" onclick="showPage('bt')" id="dtab-bt">🔬 Backtests</button>
  </div>
</div>

<!-- ── OVERVIEW ── -->
<div id="page-overview" class="page active">
  <div class="page-hdr">
    <div class="hdr-row">
      <div>
        <h1>Polymarket Paper-Trade</h1>
        <div class="updated">Last updated: <strong id="ts">—</strong></div>
      </div>
      <button class="reload-btn" id="reload-btn" onclick="reloadData()" title="Re-fetch the latest deployed data (bypasses cache)">
        <span class="spin">🔄</span> Reload
      </button>
    </div>
  </div>
  <div id="port-kpis" class="port-kpis"></div>
  <div class="card" style="margin-bottom:12px">
    <div class="card-title" style="margin-bottom:6px">⏱ Next Updates <span style="font-size:11px;font-weight:400;color:var(--muted)">(Paris time)</span></div>
    <div id="schedule-rows"></div>
    <div class="sched-note">
      Tap <strong>Run ↗</strong> to fetch fresh data now on GitHub Actions (press “Run workflow” there).
      New data deploys ~2–5 min later — then tap <strong>🔄 Reload</strong> to pull it in.
    </div>
  </div>
  <div id="strategy-grid" class="strategy-grid"></div>
  <div class="meta">
    10 % of bankroll per bet · PnL shown as % of that allocation<br>
    <strong>YES variants</strong> [10–20%] [10–30%] [10–50%] [20–40%]: buy YES in band ~5 d before resolution, hold to settlement. Portfolio equity (Overview/Equity) uses <strong>YES [10–50%]</strong> only to avoid double-counting overlapping trades.<br>
    <strong>Smart Flow</strong>: follow ≥ 3 smart-wallet consensus buys, exit on reversal signal.<br>
    <strong>Macro</strong>: hold dominant leg of macro resolution markets until year-end.
  </div>
</div>

<!-- ── OPEN POSITIONS ── -->
<div id="page-open" class="page">
  <div class="page-hdr"><h2>Open Positions</h2></div>
  <div id="open-filter" class="seg"></div>
  <div id="open-list"></div>
</div>

<!-- ── EQUITY CURVES ── -->
<div id="page-equity" class="page">
  <div class="page-hdr"><h2>Equity Curves</h2></div>
  <div id="equity-charts"></div>
</div>

<!-- ── HISTORY ── -->
<div id="page-history" class="page">
  <div class="page-hdr"><h2>History</h2></div>
  <div id="history-list"></div>
</div>

<!-- ── BACKTESTS ── -->
<div id="page-bt" class="page">
  <div class="page-hdr"><h2>Backtests</h2></div>
  <div id="bt-content"></div>
</div>

<!-- Bottom nav (mobile) -->
<nav class="bottom-nav">
  <button class="nav-btn active" onclick="showPage('overview')" id="nav-overview">
    <span class="nav-icon">📊</span><span class="nav-lbl">Overview</span>
  </button>
  <button class="nav-btn" onclick="showPage('open')" id="nav-open">
    <span class="nav-icon">📋</span><span class="nav-lbl">Open</span>
  </button>
  <button class="nav-btn" onclick="showPage('equity')" id="nav-equity">
    <span class="nav-icon">📈</span><span class="nav-lbl">Equity</span>
  </button>
  <button class="nav-btn" onclick="showPage('history')" id="nav-history">
    <span class="nav-icon">🏁</span><span class="nav-lbl">History</span>
  </button>
  <button class="nav-btn" onclick="showPage('bt')" id="nav-bt">
    <span class="nav-icon">🔬</span><span class="nav-lbl">Tests</span>
  </button>
</nav>

<!-- Cut-position modal -->
<div id="cut-modal" class="modal-backdrop" onclick="hideCut()">
  <div class="modal" onclick="event.stopPropagation()">
    <h3>✂ Close position</h3>
    <div class="modal-q" id="modal-q"></div>
    <div class="modal-section">
      <div class="modal-lbl">Run locally</div>
      <div class="code-box" id="modal-cmd"></div>
    </div>
    <div class="modal-section">
      <div class="modal-lbl">GitHub Actions — paste these when prompted</div>
      <div class="code-box" id="modal-inputs"></div>
    </div>
    <div class="modal-actions">
      <button class="btn btn-copy" id="copy-btn" onclick="copyCmd()">📋 Copy command</button>
      <a class="btn btn-gh" id="modal-gh" href="" target="_blank" rel="noopener">⚡ Open Actions</a>
      <button class="btn btn-cancel" onclick="hideCut()">Cancel</button>
    </div>
  </div>
</div>

<script>
const DATA        = %%DATA%%;
const ACTIONS_URL = %%ACTIONS_URL%%;
const S_COLORS    = ['#06b6d4','#3b82f6','#f59e0b','#818cf8','#22c55e','#a78bfa'];

// ── NAVIGATION ───────────────────────────────────────────────────────────────
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn, .dtab').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  const nb = document.getElementById('nav-' + id);   if (nb) nb.classList.add('active');
  const dt = document.getElementById('dtab-' + id);  if (dt) dt.classList.add('active');
  window.scrollTo(0, 0);
}

// ── FORMATTERS ───────────────────────────────────────────────────────────────
function fmtPct(v) {
  if (v === null || v === undefined) return '—';
  return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%';
}
function cls(v)        { return v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : 'neu'; }
function pill(s)       { return `<span class="pill pill-${s}">${s}</span>`; }
const esc = s => String(s)
  .replace(/&/g,'&amp;').replace(/"/g,'&quot;')
  .replace(/</g,'&lt;').replace(/>/g,'&gt;');

// ── SPARKLINE ────────────────────────────────────────────────────────────────
function miniTrack(entry, current) {
  if (entry == null || current == null) return '<span style="color:var(--border)">—</span>';
  const W = 80, H = 14, mid = 7;
  const clamp = v => Math.min(Math.max(v, 0), 1);
  const ex = Math.round(clamp(entry)   * W);
  const cx = Math.round(clamp(current) * W);
  const col = current >= entry ? '#22c55e' : '#ef4444';
  const lx = Math.min(ex, cx), rx = Math.max(ex, cx);
  return [
    `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">`,
    `<rect x="0" y="${mid-1}" width="${W}" height="2" fill="#2a2d3e" rx="1"/>`,
    `<rect x="${lx}" y="${mid-3}" width="${Math.max(rx-lx,1)}" height="6" fill="${col}" opacity="0.3" rx="2"/>`,
    `<line x1="${ex}" y1="2" x2="${ex}" y2="${H-2}" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round"/>`,
    `<circle cx="${cx}" cy="${mid}" r="3.5" fill="${col}"/>`,
    '</svg>',
  ].join('');
}

// ── W/L/F BAR ────────────────────────────────────────────────────────────────
function wlfRow(nW, nL, nF) {
  const tot = nW + nL + nF;
  if (!tot) return '';
  const segs = [
    nW ? `<div class="wlf-seg" style="width:${(nW/tot*100).toFixed(1)}%;background:var(--green)"></div>` : '',
    nL ? `<div class="wlf-seg" style="width:${(nL/tot*100).toFixed(1)}%;background:var(--red)"></div>`   : '',
    nF ? `<div class="wlf-seg" style="width:${(nF/tot*100).toFixed(1)}%;background:var(--flip)"></div>`  : '',
  ].join('');
  return `<div class="wlf-row">
    <div class="wlf-bar">${segs}</div>
    <span class="wlf-lbl">${nW}W&nbsp;${nL}L${nF ? '&nbsp;' + nF + 'F' : ''}</span>
  </div>`;
}

// ── EQUITY CURVE DATA ─────────────────────────────────────────────────────────
function strategyEquity(s) {
  const today = DATA.generated.split(' ')[0];

  // Find earliest entry date to anchor the curve at 0
  const allDates = [...s.open, ...s.resolved].map(r => r.entry_date).filter(Boolean).sort();
  const pts = allDates.length ? [{date: allDates[0], cum: 0}] : [];

  // Walk resolved positions in exit-date order
  const closed = s.resolved
    .filter(r => r.exit_date && r.pnl != null)
    .sort((a, b) => a.exit_date.localeCompare(b.exit_date));

  let cum = 0;
  for (const r of closed) {
    cum += r.pnl;
    if (pts.length && pts[pts.length - 1].date === r.exit_date) {
      pts[pts.length - 1].cum = +cum.toFixed(4);
    } else {
      pts.push({date: r.exit_date, cum: +cum.toFixed(4)});
    }
  }

  // Today's trailing point = realized + unrealized
  const todayCum = +(cum + s.unrealized).toFixed(4);
  if (pts.length && pts[pts.length - 1].date === today) {
    pts[pts.length - 1].cum = todayCum;
  } else {
    pts.push({date: today, cum: todayCum});
  }

  return pts;
}

function portfolioEquity() {
  const today   = DATA.generated.split(' ')[0];
  const primary = DATA.strategies.filter(s => s.primary);

  const allStartDates = primary
    .flatMap(s => [...s.open, ...s.resolved].map(r => r.entry_date).filter(Boolean))
    .sort();
  const pts = allStartDates.length ? [{date: allStartDates[0], cum: 0}] : [];

  const allClosed = primary
    .flatMap(s => s.resolved.filter(r => r.exit_date && r.pnl != null))
    .sort((a, b) => a.exit_date.localeCompare(b.exit_date));

  let cum = 0;
  for (const r of allClosed) {
    cum += r.pnl;
    if (pts.length && pts[pts.length - 1].date === r.exit_date) {
      pts[pts.length - 1].cum = +cum.toFixed(4);
    } else {
      pts.push({date: r.exit_date, cum: +cum.toFixed(4)});
    }
  }

  const totalUnrealized = primary.reduce((s, d) => s + d.unrealized, 0);
  const todayCum = +(cum + totalUnrealized).toFixed(4);
  if (pts.length && pts[pts.length - 1].date === today) {
    pts[pts.length - 1].cum = todayCum;
  } else {
    pts.push({date: today, cum: todayCum});
  }

  return pts;
}

// ── EQUITY SVG ────────────────────────────────────────────────────────────────
function equitySVG(pts, color) {
  const W = 320, H = 155;
  const pad = {t: 18, r: 14, b: 28, l: 46};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  if (!pts || pts.length < 2) {
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">
      <text x="${W/2}" y="${H/2}" fill="#94a3b8" font-size="11"
            text-anchor="middle" dominant-baseline="middle">No resolved trades yet</text>
    </svg>`;
  }

  const cums  = pts.map(p => p.cum);
  const minC  = Math.min(Math.min(...cums), 0);
  const maxC  = Math.max(Math.max(...cums), 0);
  const range = maxC - minC || 0.001;
  const n     = pts.length;

  const xOf = i => pad.l + (i / (n - 1)) * cW;
  const yOf = c => pad.t + cH - ((c - minC) / range) * cH;
  const y0  = yOf(0);

  // Polyline + area
  const pstr  = pts.map((p, i) => `${xOf(i).toFixed(1)},${yOf(p.cum).toFixed(1)}`).join(' ');
  const areaD = `M${xOf(0).toFixed(1)},${y0.toFixed(1)} ` +
    pts.map((p, i) => `L${xOf(i).toFixed(1)},${yOf(p.cum).toFixed(1)}`).join(' ') +
    ` L${xOf(n-1).toFixed(1)},${y0.toFixed(1)} Z`;

  const lastCum  = cums[n - 1];
  const dotColor = lastCum >= 0 ? '#22c55e' : '#ef4444';

  // Y axis ticks: zero (solid), max, min
  const ticks = [{v: 0, y: y0, lbl: '0%'}];
  if (maxC > 0.005) ticks.push({v: maxC, y: pad.t,      lbl: (maxC * 100).toFixed(0) + '%'});
  if (minC < -0.005) ticks.push({v: minC, y: pad.t + cH, lbl: (minC * 100).toFixed(0) + '%'});

  const gridLines = ticks.map(t =>
    `<line x1="${pad.l}" y1="${t.y.toFixed(1)}" x2="${(pad.l+cW).toFixed(1)}" y2="${t.y.toFixed(1)}"
           stroke="#2a2d3e" stroke-width="1"${t.v === 0 ? '' : ' stroke-dasharray="3,2"'}/>`
  ).join('');
  const yLabels = ticks.map(t =>
    `<text x="${pad.l - 5}" y="${(t.y + 3.5).toFixed(1)}"
           fill="#94a3b8" font-size="9" text-anchor="end">${t.lbl}</text>`
  ).join('');

  // X axis labels: first, mid (if enough points), last
  const xLbl = (i, anchor) =>
    `<text x="${xOf(i).toFixed(1)}" y="${(H - 6).toFixed(1)}"
           fill="#94a3b8" font-size="9" text-anchor="${anchor}">${pts[i].date.slice(5)}</text>`;
  let xLabels = xLbl(0, 'middle') + xLbl(n - 1, 'middle');
  if (n > 4) {
    const mi = Math.round((n - 1) / 2);
    xLabels += xLbl(mi, 'middle');
  }

  // Segment step-fill (positive green / negative red bands)
  const bands = [];
  for (let i = 0; i < n - 1; i++) {
    const x1 = xOf(i), x2 = xOf(i + 1);
    const c1 = pts[i].cum, c2 = pts[i + 1].cum;
    const yTop1 = yOf(Math.max(c1, 0)), yTop2 = yOf(Math.max(c2, 0));
    const yBot1 = yOf(Math.min(c1, 0)), yBot2 = yOf(Math.min(c2, 0));
    if (Math.max(c1, c2) > 0)
      bands.push(`<polygon points="${x1.toFixed(1)},${y0.toFixed(1)} ${x1.toFixed(1)},${yTop1.toFixed(1)} ${x2.toFixed(1)},${yTop2.toFixed(1)} ${x2.toFixed(1)},${y0.toFixed(1)}" fill="#22c55e" opacity="0.10"/>`);
    if (Math.min(c1, c2) < 0)
      bands.push(`<polygon points="${x1.toFixed(1)},${y0.toFixed(1)} ${x1.toFixed(1)},${yBot1.toFixed(1)} ${x2.toFixed(1)},${yBot2.toFixed(1)} ${x2.toFixed(1)},${y0.toFixed(1)}" fill="#ef4444" opacity="0.10"/>`);
  }

  // Dot labels for every resolved point (skip first anchor)
  const dotLabels = [];
  pts.slice(1, -1).forEach((p, ii) => {
    const i = ii + 1;
    const x = xOf(i), y = yOf(p.cum);
    const dc = p.cum >= 0 ? '#22c55e' : '#ef4444';
    dotLabels.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5" fill="${dc}" opacity="0.8"/>`);
  });

  return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block">
    ${gridLines}
    ${yLabels}
    ${bands.join('')}
    <polyline points="${pstr}" fill="none" stroke="${color}" stroke-width="2"
              stroke-linejoin="round" stroke-linecap="round"/>
    ${dotLabels.join('')}
    <circle cx="${xOf(n-1).toFixed(1)}" cy="${yOf(lastCum).toFixed(1)}"
            r="4.5" fill="${dotColor}" stroke="var(--bg)" stroke-width="2"/>
    ${xLabels}
  </svg>`;
}

// ── PAGE: OVERVIEW ────────────────────────────────────────────────────────────
function buildOverview() {
  const primary  = DATA.strategies.filter(s => s.primary);
  const totalU   = primary.reduce((s, d) => s + d.unrealized, 0);
  const totalR   = primary.reduce((s, d) => s + d.realized,   0);
  const totalMTM = totalU + totalR;
  const totalO   = DATA.strategies.reduce((s, d) => s + d.open.length, 0);

  // Real equal-weighted portfolio: each strategy funded with its own $1k bankroll.
  const p = DATA.portfolio;
  if (p) {
    document.getElementById('port-kpis').innerHTML = `
      <div class="port-kpi">
        <div class="kpi-lbl">Equity ($${p.capital.toLocaleString()} → )</div>
        <div class="kpi-val ${cls(p.mtm)}">$${Math.round(p.equity).toLocaleString()}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Portfolio MTM</div>
        <div class="kpi-val ${cls(p.mtm)}">${fmtPct(p.mtm)}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Unrealized</div>
        <div class="kpi-val ${cls(p.unrealized)}">${fmtPct(p.unrealized)}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Realized</div>
        <div class="kpi-val ${cls(p.realized)}">${fmtPct(p.realized)}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Open Positions</div>
        <div class="kpi-val neu">${totalO}</div>
      </div>`;
  } else {
    document.getElementById('port-kpis').innerHTML = `
      <div class="port-kpi">
        <div class="kpi-lbl">Portfolio MTM</div>
        <div class="kpi-val ${cls(totalMTM)}">${fmtPct(totalMTM)}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Unrealized</div>
        <div class="kpi-val ${cls(totalU)}">${fmtPct(totalU)}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Realized</div>
        <div class="kpi-val ${cls(totalR)}">${fmtPct(totalR)}</div>
      </div>
      <div class="port-kpi">
        <div class="kpi-lbl">Open Positions</div>
        <div class="kpi-val neu">${totalO}</div>
      </div>`;
  }

  document.getElementById('strategy-grid').innerHTML = DATA.strategies.map(d => {
    const tot  = d.unrealized + d.realized;
    const bdge = tot > 0.001 ? '<span class="badge badge-green">▲ profitable</span>'
               : tot < -0.001 ? '<span class="badge badge-red">▼ drawdown</span>'
               : '<span class="badge badge-grey">flat</span>';
    const nR = d.nW + d.nL + d.nF;
    const nWatch = (d.watching || []).length;
    const watchLine = nWatch ? `<span style="color:var(--accent)">${nWatch} watching</span> · ` : '';
    return `<div class="card">
      <div class="card-title">${d.label} ${bdge}</div>
      <div class="kpis">
        <div class="kpi"><div class="kpi-lbl">Total MTM</div><div class="kpi-val ${cls(tot)}">${fmtPct(tot)}</div></div>
        <div class="kpi"><div class="kpi-lbl">Unrealized</div><div class="kpi-val ${cls(d.unrealized)}">${fmtPct(d.unrealized)}</div></div>
        <div class="kpi"><div class="kpi-lbl">Realized</div><div class="kpi-val ${cls(d.realized)}">${fmtPct(d.realized)}</div></div>
      </div>
      <div style="font-size:12px;color:var(--muted);margin-bottom:${nR?'6px':'0'}">
        ${watchLine}${d.open.length} open · ${d.resolved.length} resolved${nR ? ' (' + d.nW + 'W/' + d.nL + 'L' + (d.nF ? '/' + d.nF + 'F' : '') + ')' : ''}
      </div>
      ${d.bankroll ? `<div style="font-size:12px;margin:0 0 6px;padding-top:6px;border-top:1px solid var(--border)">
        <span style="color:var(--muted)">Real $${d.bankroll.capital.toFixed(0)} acct (10%/trade):</span>
        <strong class="${cls(d.bankroll.total_return)}">$${d.bankroll.final_equity.toFixed(0)} (${fmtPct(d.bankroll.total_return)})</strong>
        <span style="color:var(--muted)"> · R ${fmtPct(d.bankroll.realized/d.bankroll.capital)} · U ${fmtPct(d.bankroll.unrealized/d.bankroll.capital)}</span>${d.bankroll.trades_skipped ? ` · <span style="color:var(--muted)">${d.bankroll.trades_skipped} unfundable</span>` : ''}${d.bankroll.edge_tstat !== null && d.bankroll.edge_tstat !== undefined ? ` · edge t=${d.bankroll.edge_tstat.toFixed(2)}` : ''}
      </div>` : ''}
      ${nR ? wlfRow(d.nW, d.nL, d.nF) : ''}
    </div>`;
  }).join('');
}

// ── POSITION PRICE CHART ─────────────────────────────────────────────────────
function buildPriceChart(hist, entryPx, entryDate) {
  const W = 520, H = 130;
  const pad = {t: 12, r: 18, b: 24, l: 44};
  const cW = W - pad.l - pad.r, cH = H - pad.t - pad.b;

  const times  = hist.map(p => p.t);
  const prices = hist.map(p => +p.p);
  const tMin = times[0], tMax = times[times.length - 1];
  const rawMin = Math.min(...prices, entryPx);
  const rawMax = Math.max(...prices, entryPx);
  const margin = Math.max((rawMax - rawMin) * 0.15, 0.03);
  const pMin = Math.max(0, rawMin - margin);
  const pMax = Math.min(1, rawMax + margin);

  const tx = t => pad.l + (tMax > tMin ? (t - tMin) / (tMax - tMin) : 0.5) * cW;
  const ty = p => pad.t + (1 - (p - pMin) / Math.max(pMax - pMin, 0.001)) * cH;

  const lastPx = prices[prices.length - 1];
  const col = lastPx >= entryPx ? '#22c55e' : '#ef4444';

  const polyPts = hist.map(p => `${tx(p.t).toFixed(1)},${ty(p.p).toFixed(1)}`).join(' ');
  const fillPts = `${tx(tMin).toFixed(1)},${(pad.t + cH).toFixed(1)} ${polyPts} ${tx(tMax).toFixed(1)},${(pad.t + cH).toFixed(1)}`;

  const ey = +ty(entryPx).toFixed(1);

  // Y-axis grid ticks
  const step = (pMax - pMin) <= 0.15 ? 0.05 : 0.10;
  const yTicks = [];
  for (let v = Math.ceil(pMin / step) * step; v <= pMax + 1e-9; v = Math.round((v + step) * 1000) / 1000) {
    yTicks.push(+v.toFixed(3));
  }

  let svg = `<svg viewBox="0 0 ${W} ${H}" width="100%" style="display:block;max-width:${W}px">`;

  // Grid
  svg += yTicks.map(v => {
    const y = ty(v).toFixed(1);
    return `<line x1="${pad.l}" y1="${y}" x2="${pad.l+cW}" y2="${y}" stroke="#1e2436" stroke-width="1"/>` +
           `<text x="${pad.l-5}" y="${+y+3.5}" fill="#64748b" font-size="9" text-anchor="end">${v.toFixed(2)}</text>`;
  }).join('');

  // Fill + line
  svg += `<polygon points="${fillPts}" fill="${col}" opacity="0.1"/>`;
  svg += `<line x1="${pad.l}" y1="${ey}" x2="${pad.l+cW}" y2="${ey}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4,3"/>`;
  svg += `<polyline points="${polyPts}" fill="none" stroke="${col}" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/>`;

  // Entry label
  svg += `<text x="${pad.l-5}" y="${ey+3.5}" fill="#94a3b8" font-size="8.5" text-anchor="end">${entryPx.toFixed(3)}</text>`;

  // Current dot + label
  const lx = +tx(tMax).toFixed(1), ly = +ty(lastPx).toFixed(1);
  svg += `<circle cx="${lx}" cy="${ly}" r="3.5" fill="${col}"/>`;
  const lbX = lx + 6 + (W - pad.r - lx < 40 ? -50 : 0);
  svg += `<text x="${lbX}" y="${ly+4}" fill="${col}" font-size="10" font-weight="600">${lastPx.toFixed(3)}</text>`;

  // X axis dates
  const fmt = ts => new Date(ts * 1000).toISOString().slice(5, 10);
  svg += `<text x="${pad.l}" y="${H-4}" fill="#64748b" font-size="9">${entryDate || fmt(tMin)}</text>`;
  svg += `<text x="${pad.l+cW}" y="${H-4}" fill="#64748b" font-size="9" text-anchor="end">${fmt(tMax)}</text>`;

  svg += '</svg>';
  return svg;
}

async function expandPos(btn) {
  const token     = btn.dataset.token;
  const entryDate = btn.dataset.entryDate;
  const entryPx   = parseFloat(btn.dataset.entry) || 0;
  const rowId     = btn.dataset.chart;
  const tr = document.getElementById(rowId);
  if (!tr) return;
  if (tr.style.display !== 'none') {
    tr.style.display = 'none';
    btn.textContent = '▶';
    return;
  }
  const td = tr.querySelector('td');
  td.innerHTML = '<span style="color:var(--muted);font-size:12px;padding:4px 0;display:block">Loading…</span>';
  tr.style.display = '';
  btn.textContent = '▼';
  try {
    const url = `https://clob.polymarket.com/prices-history?market=${encodeURIComponent(token)}&interval=max&fidelity=60`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const json = await resp.json();
    let hist = json.history || [];
    if (entryDate) {
      const since = new Date(entryDate).getTime() / 1000 - 86400;
      hist = hist.filter(p => p.t >= since);
    }
    if (!hist.length) {
      td.innerHTML = '<span style="color:var(--muted);font-size:12px;padding:4px 0;display:block">No price data available</span>';
      return;
    }
    td.innerHTML = buildPriceChart(hist, entryPx, entryDate);

    // Refresh Now / MTM / mini-track in the main row with live price
    const livePx  = +hist[hist.length - 1].p;
    const frac    = parseFloat(btn.dataset.fraction) || 0.1;
    const mainRow = tr.previousElementSibling;
    if (mainRow) {
      const tds = mainRow.querySelectorAll('td');
      if (tds[1]) tds[1].innerHTML = miniTrack(entryPx, livePx);
      if (tds[3]) tds[3].innerHTML = `${livePx.toFixed(3)} <span style="font-size:9px;color:var(--accent);vertical-align:middle">live</span>`;
      if (tds[4] && entryPx > 0) {
        const mtm = (livePx / entryPx - 1) * frac;
        tds[4].textContent = (mtm >= 0 ? '+' : '') + (mtm * 100).toFixed(1) + '%';
        tds[4].className   = cls(mtm);
      }
    }
  } catch(e) {
    td.innerHTML = `<span style="color:var(--red);font-size:11px;padding:4px 0;display:block">Failed to load: ${e.message}</span>`;
    btn.textContent = '▶';
    tr.style.display = 'none';
  }
}

// ── PAGE: OPEN POSITIONS ──────────────────────────────────────────────────────
function buildOpen() {
  // Filter pill buttons
  document.getElementById('open-filter').innerHTML = DATA.strategies.map((d, i) => {
    const nWatch = (d.watching || []).length;
    const total = d.open.length + nWatch;
    return `<button class="seg-btn${i===0?' active':''}" onclick="filterOpen(${i})" id="ofilt-${i}">
      ${d.label} <span style="opacity:.65">${total}</span>
    </button>`;
  }).join('');

  // Strategy sections
  document.getElementById('open-list').innerHTML = DATA.strategies.map((d, i) => {
    const show = i === 0 ? 'block' : 'none';
    const nWatch = (d.watching || []).length;
    if (!d.open.length && !nWatch) {
      return `<div id="open-sec-${i}" style="display:${show}">
        <div class="card"><div style="color:var(--muted);font-size:13px;text-align:center;padding:20px">No open positions</div></div>
      </div>`;
    }
    const rows = d.open.flatMap((r, ri) => {
      const chartId = `chart-${i}-${ri}`;
      const hasToken = r.token && r.token.length > 8;
      const expandBtn = hasToken
        ? `<button class="expand-btn" title="Show price chart" data-token="${r.token}" data-entry-date="${r.entry_date||''}" data-entry="${r.entry != null ? r.entry : 0}" data-fraction="${r.fraction != null ? r.fraction : 0.1}" data-chart="${chartId}" onclick="expandPos(this)">▶</button>`
        : '';
      const mainRow = `<tr>
      <td class="q">${r.question}</td>
      <td style="padding-right:2px">${miniTrack(r.entry, r.current)}</td>
      <td class="hide-xs">${r.entry   != null ? r.entry.toFixed(3)   : '—'}</td>
      <td>${r.current != null ? r.current.toFixed(3) : '—'}</td>
      <td class="${cls(r.mtm)}">${fmtPct(r.mtm)}</td>
      <td style="white-space:nowrap">${expandBtn}<button class="cut-btn" data-sid="${d.strategy_id}" data-q="${esc(r.question)}" onclick="openCut(this)">✂</button></td>
    </tr>`;
      const chartRow = hasToken
        ? `<tr id="${chartId}" style="display:none"><td colspan="6" style="padding:6px 14px 14px"></td></tr>`
        : '';
      return [mainRow, chartRow];
    }).join('');

    const watchRows = nWatch ? (d.watching || []).map(r => {
      const dipPct = r.dip_pct ? (r.dip_pct * 100).toFixed(1) : '0.0';
      const dipped = r.dip_pct >= 0.10;
      return `<tr style="opacity:.8">
        <td class="q">${r.question}</td>
        <td style="padding-right:2px">—</td>
        <td class="hide-xs" style="color:var(--muted)">${r.watch_price != null ? r.watch_price.toFixed(3) : '—'}</td>
        <td>${r.current != null ? r.current.toFixed(3) : '—'}</td>
        <td style="color:${dipped ? 'var(--green)' : 'var(--muted)'}">−${dipPct}%${dipped ? ' ✓' : ''}</td>
        <td><span style="font-size:10px;color:var(--accent)">watching</span></td>
      </tr>`;
    }).join('') : '';

    const posCount = d.open.length ? `${d.open.length} open` : '';
    const watchCount = nWatch ? `${nWatch} watching` : '';
    const subtitle = [posCount, watchCount].filter(Boolean).join(' · ');

    return `<div id="open-sec-${i}" style="display:${show}">
      <div class="card">
        <div class="card-title">${d.label}
          <span style="font-size:12px;font-weight:400;color:var(--muted)">${subtitle}</span>
        </div>
        <div class="tbl-wrap">
          <table>
            <thead><tr>
              <th>Question</th><th>Track</th>
              <th class="hide-xs">Entry</th><th>Now</th><th>MTM / Dip</th><th></th>
            </tr></thead>
            <tbody>${rows}${watchRows}</tbody>
          </table>
        </div>
      </div>
    </div>`;
  }).join('');
}

function filterOpen(idx) {
  DATA.strategies.forEach((_, i) => {
    const sec = document.getElementById('open-sec-' + i);
    const btn = document.getElementById('ofilt-' + i);
    if (sec) sec.style.display = i === idx ? 'block' : 'none';
    if (btn) btn.classList.toggle('active', i === idx);
  });
}

// ── PAGE: EQUITY CURVES ───────────────────────────────────────────────────────
function buildEquity() {
  const portPts  = portfolioEquity();
  const portLast = portPts.length ? portPts[portPts.length - 1].cum : 0;

  let html = `<div class="card">
    <div class="card-title">📊 All Strategies Combined</div>
    <div class="eq-sub ${cls(portLast)}">${fmtPct(portLast)} combined MTM</div>
    ${equitySVG(portPts, '#3b82f6')}
  </div>`;

  DATA.strategies.forEach((d, i) => {
    const pts  = strategyEquity(d);
    const last = pts.length ? pts[pts.length - 1].cum : 0;
    const col  = S_COLORS[(i + 1) % S_COLORS.length];
    const nR   = d.nW + d.nL + d.nF;
    html += `<div class="card">
      <div class="card-title">${d.label}</div>
      <div class="eq-sub ${cls(last)}">${fmtPct(last)} · ${d.open.length} open${nR ? ' · ' + d.nW + 'W/' + d.nL + 'L' + (d.nF ? '/' + d.nF + 'F' : '') : ''}</div>
      ${equitySVG(pts, col)}
    </div>`;
  });

  document.getElementById('equity-charts').innerHTML = html;
}

// ── PAGE: HISTORY ─────────────────────────────────────────────────────────────
function buildHistory() {
  const sections = DATA.strategies.map(d => {
    if (!d.resolved.length) return '';
    const rows = [...d.resolved]
      .sort((a, b) => (b.exit_date || '').localeCompare(a.exit_date || ''))
      .map(r => `<tr>
        <td class="q">${r.question}</td>
        <td>${pill(r.status)}</td>
        <td class="${cls(r.pnl)}">${fmtPct(r.pnl)}</td>
        <td style="color:var(--muted);white-space:nowrap">${r.exit_date}</td>
      </tr>`).join('');
    return `<div class="card">
      <div class="card-title">${d.label}</div>
      <div class="tbl-wrap">
        <table>
          <thead><tr><th>Question</th><th>Status</th><th>PnL</th><th>Date</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>`;
  }).join('');

  document.getElementById('history-list').innerHTML = sections.trim() ||
    '<div class="card"><div style="color:var(--muted);font-size:13px;text-align:center;padding:20px">No resolved positions yet</div></div>';
}

// ── MODAL ─────────────────────────────────────────────────────────────────────
let _currentCmd = '';

function openCut(btn) {
  const sid = btn.dataset.sid;
  const q   = btn.dataset.q;
  const cmd = `python experiments/close_position.py --strategy ${sid} --question "${q.replace(/\\/g,'\\\\').replace(/"/g,'\\"')}"`;
  _currentCmd = cmd;
  document.getElementById('modal-q').textContent      = q;
  document.getElementById('modal-cmd').textContent    = cmd;
  document.getElementById('modal-inputs').textContent = `Strategy : ${sid}\nQuestion : ${q}`;
  document.getElementById('modal-gh').href            = ACTIONS_URL;
  document.getElementById('copy-btn').textContent     = '📋 Copy command';
  document.getElementById('cut-modal').classList.add('open');
}
function hideCut() { document.getElementById('cut-modal').classList.remove('open'); }
function copyCmd() {
  navigator.clipboard.writeText(_currentCmd).then(() => {
    const b = document.getElementById('copy-btn');
    b.textContent = '✓ Copied!';
    setTimeout(() => { b.textContent = '📋 Copy command'; }, 1800);
  });
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') hideCut(); });

// ── PAGE: BACKTESTS ───────────────────────────────────────────────────────────
function buildBacktest() {
  const bt  = DATA.backtests || {};
  const mp  = bt.midprice_yes || {};
  const sf  = DATA.strategies.find(s => s.strategy_id === 'smart_flow') || {};
  const mac = DATA.strategies.find(s => s.strategy_id === 'macro') || {};
  let html  = '';

  // ── Regime-calm disruptive tail (per-domain validation) ─────────────────────
  const rg = bt.regime || {};
  if (rg.available) {
    const vBadge = v => v === 'VALIDATED'
      ? '<span class="pill pill-won">VALIDATED</span>'
      : '<span class="pill pill-lost">killed</span>';
    const fmtE = v => (v == null ? '—' : (v >= 0 ? '+' : '') + v.toFixed(3));
    const rows = (rg.per_domain || []).map(r => `<tr>
        <td style="white-space:nowrap">${esc(r.domain)}</td>
        <td>${vBadge(r.verdict)}</td>
        <td style="color:var(--muted);font-size:11px">${esc(r.proxy)}</td>
        <td class="${cls(r.base_edge)}">${fmtE(r.base_edge)}</td>
        <td>${r.base_t == null ? '—' : r.base_t.toFixed(2)}</td>
        <td class="${cls(r.gap)}">${fmtE(r.gap)}</td>
        <td class="${cls(r.oos_edge)}">${fmtE(r.oos_edge)}</td>
        <td>${r.boot_p == null ? '—' : r.boot_p.toFixed(2)}</td>
        <td>${r.top1 == null ? '—' : (r.top1*100).toFixed(0)+'%'}</td>
      </tr>`).join('');
    html += `<div class="card">
      <div class="card-title">Regime-Calm Disruptive Tail
        <span style="font-size:11px;font-weight:400;color:var(--muted)">per-domain kill test · 1,576-leg panel · price≤${rg.max_price ?? 0.35}</span>
      </div>
      <div class="bt-note" style="margin-bottom:12px">
        Cheap YES legs of disruptive markets are underpriced <strong>while that domain is calm</strong>.
        Each domain is judged against its <strong>own exogenous proxy</strong> and put through a strict
        kill battery (base t, walk-forward OOS, cost@3¢/5¢, week-clustered bootstrap, placebo regime,
        winner concentration). Only survivors trade live. <strong>Validated: ${(rg.validated||[]).join(', ') || 'none'}</strong>.
      </div>
      <div class="tbl-wrap"><table>
        <thead><tr>
          <th>Domain</th><th>Verdict</th><th>Proxy</th><th>Calm edge/$1</th><th>t</th>
          <th>Calm−Turb</th><th>OOS edge</th><th>boot P≤0</th><th>top1</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table></div>
      <div class="bt-note" style="margin-top:10px">
        <strong>Geopolitics</strong> (GPR-calm) is the lone survivor: calm edge
        <strong>+0.41/$1 (t 3.9)</strong>, walk-forward OOS <strong>+0.36 (t 3.3)</strong>,
        bootstrap P(edge≤0)=0, top-1 only 17% of profit.
        Crypto/politics fail significance; ai/tech lacks tail data.
      </div>
      <div class="bt-note" style="margin-top:8px">
        <strong>Realistic-cost backtest</strong> (entry at ask = mid + 3¢, 2% stake, hold to
        settlement, n=15): net edge <strong>+0.40/$1 (t 3.2)</strong>, win 53%, profit factor 7.3,
        CAGR +70%, <strong>Sharpe 1.26</strong>, maxDD −6%, Calmar 11.9; bootstrap 90% CI
        [+0.19, +0.57]. Cost isn't the killer (edge holds to 5¢) — the honest limits are small n,
        lumpy payoff, and thin capacity. Forward ledger: <strong>Regime (geo-calm)</strong> on the Overview tab.
      </div>
    </div>`;
  }

  // ── Mid-priced YES ──────────────────────────────────────────────────────────
  if (mp.available) {
    const eq  = mp.equity;               // [{date, cum}]
    const MAX_EQ_PTS = 80;               // downsample for SVG clarity
    const step = Math.max(1, Math.floor(eq.length / MAX_EQ_PTS));
    const eq80 = eq.filter((_, i) => i % step === 0 || i === eq.length - 1);

    // Scale so chart Y axis shows "portfolio %" (cum * 100 = % of initial capital)
    // (each trade is 10% of capital; cum = sum of those returns)
    const chartPts = eq80.map(p => ({date: p.date, cum: +(p.cum / 10).toFixed(4)}));

    const finalPct  = fmtPct(eq[eq.length - 1].cum / 10);
    const maxDD     = (() => {
      let peak = 0, dd = 0;
      for (const p of eq) { peak = Math.max(peak, p.cum); dd = Math.max(dd, peak - p.cum); }
      return fmtPct(-(dd / 10));
    })();

    // Horizon comparison bars (max win rate for scale)
    const maxWR = Math.max(...mp.by_horizon.map(h => h.wr));
    const horizonRows = mp.by_horizon.map(h => {
      const isLive = h.h === 5;
      const barPct = (h.wr / maxWR * 100).toFixed(1);
      return `<div class="horizon-row">
        <span class="horizon-d">${h.h}d</span>
        <div class="horizon-bar-wrap">
          <div class="horizon-bar${isLive?' live':''}" style="width:${barPct}%"></div>
        </div>
        <span class="horizon-stats">
          ${(h.wr*100).toFixed(0)}% win&nbsp; Sharpe ${h.sharpe.toFixed(2)}
          ${isLive ? '<span class="horizon-live-tag">LIVE</span>' : ''}
        </span>
      </div>`;
    }).join('');

    html += `<div class="card">
      <div class="card-title">Mid-priced YES
        <span style="font-size:11px;font-weight:400;color:var(--muted)">${mp.period} · ${mp.n} trades</span>
      </div>
      <div class="bt-kpis">
        <div class="bt-kpi">
          <div class="bt-kpi-lbl">Portfolio return</div>
          <div class="bt-kpi-val pos">${finalPct}</div>
        </div>
        <div class="bt-kpi">
          <div class="bt-kpi-lbl">Win rate</div>
          <div class="bt-kpi-val">${(mp.win_rate*100).toFixed(1)}%</div>
        </div>
        <div class="bt-kpi">
          <div class="bt-kpi-lbl">Sharpe</div>
          <div class="bt-kpi-val">${mp.sharpe.toFixed(2)}</div>
        </div>
      </div>
      <div class="bt-note" style="margin-bottom:12px">
        Flat 10% sizing per trade, non-compounded · ${mp.n} resolved markets from Polymarket ·
        Buy YES at ask when price in [10%,50%] with 2–10 days to resolution, hold to settlement.
        Max drawdown: <strong>${maxDD}</strong> from peak.
      </div>
      ${equitySVG(chartPts, '#3b82f6')}
      <div class="bt-section-lbl">Win rate by entry horizon (band 10%–50%)</div>
      ${horizonRows}
    </div>`;
  } else {
    html += `<div class="card"><div class="bt-note">Mid-priced YES backtest data not found. Run <code>python experiments/run_midprice_yes_backtest.py</code> to generate it.</div></div>`;
  }

  // ── Band Comparison ───────────────────────────────────────────────────────────
  const bsw = (DATA.backtests || {}).band_sweep || [];
  if (bsw.length) {
    const maxSh = Math.max(...bsw.map(b => b.sharpe));
    const bsRows = bsw.map(b => {
      const barPct = (Math.max(b.sharpe, 0) / maxSh * 100).toFixed(1);
      const color  = b.live   ? 'var(--yellow)'
                   : b.sharpe >= 0.35 ? 'var(--green)'
                   : b.sharpe >= 0.20 ? 'var(--blue)' : '#4b5563';
      return `<div class="horizon-row">
        <span class="horizon-d" style="width:80px;font-size:10px;text-align:left;flex-shrink:0">${b.band}</span>
        <div class="horizon-bar-wrap">
          <div class="horizon-bar" style="width:${barPct}%;background:${color}"></div>
        </div>
        <span class="horizon-stats">
          ${(b.win_rate*100).toFixed(0)}% win · Sh&nbsp;${b.sharpe.toFixed(2)} · n=${b.n}
          ${b.live ? '<span class="horizon-live-tag">CURRENT</span>' : ''}
        </span>
      </div>`;
    }).join('');
    html += `<div class="card" style="margin-top:12px">
      <div class="card-title">Price Band Comparison
        <span style="font-size:11px;font-weight:400;color:var(--muted)">5d horizon · hold to resolution · 2-year backtest</span>
      </div>
      <div class="bt-note" style="margin-bottom:12px">
        All 4 live band variants tracked simultaneously in paper-trade. Cheapest bets carry
        the strongest edge — the lower the price band, the higher the implied edge when markets
        resolve YES. Portfolio equity on the Equity tab uses <strong>YES [10–50%]</strong> only to avoid overlap.
      </div>
      ${bsRows}
    </div>`;
  }

  // ── Smart Flow ───────────────────────────────────────────────────────────────
  const sfRes = sf.resolved || [];
  const sfN   = sfRes.length;
  const sfW   = sf.nW || 0, sfL = sf.nL || 0, sfF = sf.nF || 0;
  const sfWR  = sfN ? (sfW / sfN * 100).toFixed(0) : '—';
  const sfPnl = fmtPct(sf.realized);

  html += `<div class="card" style="margin-top:12px">
    <div class="card-title">Smart Flow
      <span style="font-size:11px;font-weight:400;color:var(--muted)">Live paper-trade · inception Jun 2026</span>
    </div>
    <div class="bt-kpis">
      <div class="bt-kpi">
        <div class="bt-kpi-lbl">Resolved</div>
        <div class="bt-kpi-val neu">${sfN}</div>
      </div>
      <div class="bt-kpi">
        <div class="bt-kpi-lbl">Win rate</div>
        <div class="bt-kpi-val">${sfWR}%</div>
      </div>
      <div class="bt-kpi">
        <div class="bt-kpi-lbl">Realized PnL</div>
        <div class="bt-kpi-val ${cls(sf.realized)}">${sfPnl}</div>
      </div>
    </div>
    <div class="bt-note">
      <strong>No pre-launch historical backtest</strong> — strategy depends on live leaderboard wallet data
      that doesn't exist for past dates. Live paper-trade started Jun 2026.<br><br>
      <strong>Edge logic:</strong> When ≥ 3 top Polymarket wallets (ranked by 7d/30d profit, excluding
      market-makers) all buy the same outcome within 7 days, enter long at the ask. Exit early if
      smart-wallet net flow reverses (sellers > buyers) — this is the <em>flipped</em> status you see in History.
      ${sfN > 0 ? `<br><br>Current record: <strong>${sfW}W / ${sfL}L / ${sfF} flip</strong>. Too early to draw conclusions — aim for 50+ resolved trades.` : ''}
    </div>
  </div>`;

  // ── Macro ────────────────────────────────────────────────────────────────────
  const macO = (mac.open || []).length;
  const macR = (mac.resolved || []).length;

  html += `<div class="card" style="margin-top:12px">
    <div class="card-title">Macro (Fed cuts)
      <span style="font-size:11px;font-weight:400;color:var(--muted)">Live paper-trade · inception Jun 2026</span>
    </div>
    <div class="bt-kpis">
      <div class="bt-kpi">
        <div class="bt-kpi-lbl">Open</div>
        <div class="bt-kpi-val neu">${macO}</div>
      </div>
      <div class="bt-kpi">
        <div class="bt-kpi-lbl">Resolved</div>
        <div class="bt-kpi-val neu">${macR}</div>
      </div>
      <div class="bt-kpi">
        <div class="bt-kpi-lbl">Unrealized</div>
        <div class="bt-kpi-val ${cls(mac.unrealized)}">${fmtPct(mac.unrealized)}</div>
      </div>
    </div>
    <div class="bt-note">
      <strong>No historical backtest</strong> — macro markets on Polymarket only became liquid and
      numerous from 2025 onward. Strategy holds a low-turnover position all year so sample size
      grows very slowly.<br><br>
      <strong>Edge logic:</strong> Scan multi-outcome macro events (e.g. "How many Fed rate cuts in 2026?"),
      buy the current field leader (highest-priced outcome), hold to year-end resolution.
      Conviction bet that the market's dominant macro view is also the right one.
    </div>
  </div>`;

  // ── Dip-Confirm YES ──────────────────────────────────────────────────────────
  const dc = DATA.strategies.find(s => s.strategy_id === 'dip_confirm') || {};
  const dcW   = (dc.watching || []).length;
  const dcO   = (dc.open     || []).length;
  const dcRes = (dc.resolved || []).filter(r => r.status !== 'expired');
  const dcExp = dc.nExpired || 0;

  const dcWatchRows = (dc.watching || []).map(r => {
    const dipPct = r.dip_pct ? (r.dip_pct * 100).toFixed(1) : '0.0';
    const dipped = r.dip_pct >= 0.10;
    return `<tr>
      <td style="max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${r.question}</td>
      <td>${r.watch_date || ''}</td>
      <td>${r.watch_price != null ? r.watch_price.toFixed(3) : '—'}</td>
      <td>${r.min_price   != null ? r.min_price.toFixed(3)   : '—'}</td>
      <td style="color:${dipped ? 'var(--green)' : 'var(--muted)'}">−${dipPct}%${dipped ? ' ✓' : ''}</td>
      <td>${r.current != null ? r.current.toFixed(3) : '—'}</td>
    </tr>`;
  }).join('');

  html += `<div class="card" style="margin-top:12px">
    <div class="card-title">Dip-Confirm YES [10–50%]
      <span style="font-size:11px;font-weight:400;color:var(--muted)">Live paper-trade · watch 5–9d out, enter after dip≥10%→recover</span>
    </div>
    <div class="bt-kpis">
      <div class="bt-kpi"><div class="bt-kpi-lbl">Watching</div><div class="bt-kpi-val neu">${dcW}</div></div>
      <div class="bt-kpi"><div class="bt-kpi-lbl">Open</div><div class="bt-kpi-val neu">${dcO}</div></div>
      <div class="bt-kpi"><div class="bt-kpi-lbl">Resolved</div><div class="bt-kpi-val neu">${dcRes.length}</div></div>
      <div class="bt-kpi"><div class="bt-kpi-lbl">Win rate</div>
        <div class="bt-kpi-val">${dcRes.length ? (dc.nW / dcRes.length * 100).toFixed(0) + '%' : '—'}</div></div>
      <div class="bt-kpi"><div class="bt-kpi-lbl">Realized PnL</div>
        <div class="bt-kpi-val ${cls(dc.realized)}">${fmtPct(dc.realized)}</div></div>
    </div>
    ${dcW > 0 ? `<div style="margin-top:10px;overflow-x:auto">
      <table style="width:100%;font-size:11px;border-collapse:collapse">
        <thead><tr style="color:var(--muted);border-bottom:1px solid #374151">
          <th style="text-align:left;padding:4px 6px">Market</th>
          <th style="padding:4px 6px">Watched</th>
          <th style="padding:4px 6px">Watch px</th>
          <th style="padding:4px 6px">Min px</th>
          <th style="padding:4px 6px">Max dip</th>
          <th style="padding:4px 6px">Now</th>
        </tr></thead>
        <tbody>${dcWatchRows}</tbody>
      </table>
    </div>` : ''}
    <div class="bt-note" style="margin-top:10px">
      <strong>Backtest edge (2098-market full universe):</strong>
      H=5d dip≥10%: <strong>74% win rate</strong> (n=27, 95% CI 55–87%) vs 40% base.
      H=7d: 88% win rate (n=8). Edge disappears at H≥10d.<br><br>
      <strong>How it works:</strong> Markets in [10–50%] YES at 5–9 days out are placed on watch.
      If the YES price drops ≥10% then recovers to ≥95% of the watch price before <2 days remain,
      the position is entered at the live ask. Markets that never dip expire unfilled.
      ${dcExp > 0 ? `<br><span style="color:var(--muted)">${dcExp} markets expired without triggering.</span>` : ''}
    </div>
  </div>`;

  document.getElementById('bt-content').innerHTML = html;
}

// ── SCHEDULE COUNTDOWN ───────────────────────────────────────────────────────
// Cron definitions: every N hours starting at startH UTC, at :minute past the hour
const CRONS = [
  {label:'All strategies', color:'#f59e0b', every:1,  startH:0,  minute:15, wf:'paper_trade.yml'},
];
const PARIS_TZ = 'Europe/Paris';
// Base URL for the Actions workflow pages (derive from the close-position ACTIONS_URL)
const WF_BASE = ACTIONS_URL.replace(/[^/]+$/, '');

function nextCronRun(every, startH, minute) {
  const now = new Date();
  const nowMs = now.getTime();
  const slots = [];
  for (let h = startH; h < 24; h += every) slots.push(h);
  for (const h of slots) {
    const t = new Date(Date.UTC(
      now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate(), h, minute, 0));
    if (t.getTime() > nowMs + 2000) return t;
  }
  // wrap to next day
  return new Date(Date.UTC(
    now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate() + 1, slots[0], minute, 0));
}

function fmtParisTime(d) {
  return d.toLocaleTimeString('fr-FR', {timeZone: PARIS_TZ, hour:'2-digit', minute:'2-digit'});
}

function fmtParisDay(d) {
  const opts = {timeZone: PARIS_TZ, year:'numeric', month:'2-digit', day:'2-digit'};
  const now  = new Date();
  const dStr = d.toLocaleDateString('fr-FR', opts);
  if (dStr === now.toLocaleDateString('fr-FR', opts)) return 'Today';
  const tmr = new Date(now.getTime() + 86400000);
  if (dStr === tmr.toLocaleDateString('fr-FR', opts)) return 'Tomorrow';
  return dStr;
}

function fmtCountdown(ms) {
  if (ms <= 0) return {text: 'running', cls: 'now'};
  const h = Math.floor(ms / 3600000);
  const m = Math.floor((ms % 3600000) / 60000);
  const s = Math.floor((ms % 60000) / 1000);
  const text = h > 0
    ? `${h}h ${String(m).padStart(2,'0')}m`
    : m > 0
      ? `${m}m ${String(s).padStart(2,'0')}s`
      : `${s}s`;
  const cls = ms < 300000 ? 'soon' : ''; // yellow if < 5 min
  return {text, cls};
}

let _nextRuns = [];

function buildSchedule() {
  _nextRuns = CRONS.map(c => nextCronRun(c.every, c.startH, c.minute));
  document.getElementById('schedule-rows').innerHTML = CRONS.map((c, i) => {
    const day  = fmtParisDay(_nextRuns[i]);
    const time = fmtParisTime(_nextRuns[i]);
    return `<div class="sched-row">
      <div class="sched-dot" style="background:${c.color}"></div>
      <div class="sched-name">${c.label}</div>
      <div class="sched-time">${day} ${time}</div>
      <div class="sched-cd" id="scd-${i}">—</div>
      <a class="sched-run" href="${WF_BASE}${c.wf}" target="_blank" rel="noopener"
         title="Open GitHub Actions to run ${c.label} now">Run ↗</a>
    </div>`;
  }).join('');
}

function tickSchedule() {
  const now = Date.now();
  let needRebuild = false;
  CRONS.forEach((c, i) => {
    if (_nextRuns[i] && _nextRuns[i].getTime() <= now) {
      _nextRuns[i] = nextCronRun(c.every, c.startH, c.minute);
      needRebuild = true;
    }
  });
  if (needRebuild) { buildSchedule(); return; }
  CRONS.forEach((_, i) => {
    const el = document.getElementById('scd-' + i);
    if (!el) return;
    const {text, cls} = fmtCountdown(_nextRuns[i].getTime() - now);
    el.textContent = text;
    el.className   = 'sched-cd' + (cls ? ' ' + cls : '');
  });
}

// ── MANUAL RELOAD ─────────────────────────────────────────────────────────────
// The page is static (data baked in at build time), so "reload" re-fetches the
// freshest deployed HTML. A cache-busting query param sidesteps the Pages CDN
// (max-age=600) and the browser cache so a just-finished deploy shows immediately.
function reloadData() {
  const btn = document.getElementById('reload-btn');
  if (btn) btn.classList.add('loading');
  const u = new URL(window.location.href);
  u.searchParams.set('t', Date.now().toString());
  window.location.replace(u.toString());
}

// ── INIT ──────────────────────────────────────────────────────────────────────
document.getElementById('ts').textContent = DATA.generated;
buildOverview();
buildOpen();
buildEquity();
buildHistory();
buildBacktest();
buildSchedule();
setInterval(tickSchedule, 1000);
</script>
</body>
</html>
"""


def generate() -> None:
    def strat(path, entry_col, q_col, label, sid, primary=True):
        s = load_strategy(path, entry_col, q_col, label, sid)
        s["primary"] = primary
        return s

    strategies = [
        strat(DATA / "midprice_yes_10_20_positions.csv", "entry_ask", "question",
              "YES [10–20%]", "midprice_yes_10_20", primary=False),
        strat(DATA / "midprice_yes_10_30_positions.csv", "entry_ask", "question",
              "YES [10–30%]", "midprice_yes_10_30", primary=False),
        strat(DATA / "midprice_yes_positions.csv", "entry_ask", "question",
              "YES [10–50%]", "midprice_yes", primary=True),
        strat(DATA / "midprice_yes_20_40_positions.csv", "entry_ask", "question",
              "YES [20–40%]", "midprice_yes_20_40", primary=False),
        strat(DATA / "smart_flow_positions.csv", "entry_ask", "question",
              "Smart Flow", "smart_flow", primary=True),
        strat(DATA / "smart_flow_roi_positions.csv", "entry_ask", "question",
              "Smart Flow (ROI)", "smart_flow_roi", primary=True),
        strat(DATA / "macro_positions.csv", "entry_price", "leader_question",
              "Macro (Fed cuts)", "macro", primary=True),
        strat(DATA / "validated_regime_positions.csv", "entry_ask", "question",
              "Regime (geo-calm)", "validated_regime", primary=True),
        load_dip_confirm(DATA / "dip_confirm_positions.csv"),
    ]
    # attach real bankroll-sim metrics (from run_bankroll_sim.py) by strategy id
    bankroll_by_file: dict = {}
    portfolio: dict | None = None
    bsum = DATA / "bankroll_summary.csv"
    if bsum.exists():
        bdf = pd.read_csv(bsum)
        for _, r in bdf.iterrows():
            et = r.get("edge_tstat")
            rec = {
                "capital":        float(r["capital"]),
                "final_equity":   float(r["final_equity"]),
                "total_return":   float(r["total_return"]),
                "realized":       float(r["realized"]) if pd.notna(r.get("realized")) else 0.0,
                "unrealized":     float(r["unrealized"]) if pd.notna(r.get("unrealized")) else 0.0,
            }
            if str(r["file"]) == "" or str(r["strategy"]).startswith("GLOBAL"):
                # equal-weighted portfolio of all strategies
                portfolio = {
                    "capital":      rec["capital"],
                    "equity":       rec["final_equity"],
                    "mtm":          rec["total_return"],
                    "realized":     rec["realized"] / rec["capital"] if rec["capital"] else 0.0,
                    "unrealized":   rec["unrealized"] / rec["capital"] if rec["capital"] else 0.0,
                }
                continue
            rec["trades_taken"]   = int(r["trades_taken"])
            rec["trades_skipped"] = int(r["trades_skipped"])
            rec["edge_tstat"]     = (float(et) if pd.notna(et) and str(et) != "" else None)
            bankroll_by_file[str(r["file"])] = rec
    file_by_id = {
        "midprice_yes": "midprice_yes_positions.csv",
        "smart_flow": "smart_flow_positions.csv",
        "smart_flow_roi": "smart_flow_roi_positions.csv",
        "dip_confirm": "dip_confirm_positions.csv",
        "macro": "macro_positions.csv",
    }
    for s in strategies:
        s["bankroll"] = bankroll_by_file.get(file_by_id.get(s["strategy_id"], ""))

    payload = {
        "generated":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "strategies": strategies,
        "portfolio":  portfolio,
        "backtests":  {
            "midprice_yes": load_midprice_backtest(),
            "band_sweep":   load_band_sweep(),
            "regime":       load_regime_backtest(),
        },
    }
    html = _HTML.replace("%%DATA%%",        json.dumps(payload, ensure_ascii=False))
    html = html.replace("%%ACTIONS_URL%%",  json.dumps(ACTIONS_URL))
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {OUT}")
    for s in strategies:
        total = s["unrealized"] + s["realized"]
        sign  = "+" if total >= 0 else ""
        print(f"  {s['label']:22s}  MTM {sign}{total*100:.1f}%  "
              f"(open {len(s['open'])}, {s['nW']}W/{s['nL']}L"
              + (f"/{s['nF']}flip" if s.get("nF") else "") + ")")


if __name__ == "__main__":
    generate()
