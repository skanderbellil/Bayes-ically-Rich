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


# ---------------------------------------------------------------------------
# Load & compute MTM for one ledger
# ---------------------------------------------------------------------------

def load_strategy(path: Path, entry_col: str, q_col: str, label: str) -> dict:
    if not path.exists():
        return {"label": label, "open": [], "resolved": [], "unrealized": 0,
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

    def row_to_dict(r):
        return {
            "question":  str(r[q_col]),
            "entry":     round(float(r[entry_col]), 4) if pd.notna(r[entry_col]) else None,
            "current":   round(float(r["current_price"]), 4) if pd.notna(r.get("current_price")) else None,
            "mtm":       round(float(r["mtm"]), 4) if "mtm" in r and pd.notna(r["mtm"]) else None,
            "status":    str(r["status"]),
            "pnl":       round(float(r["pnl"]), 4) if pd.notna(r.get("pnl")) else None,
            "exit_date": str(r.get("exit_date", "")) if pd.notna(r.get("exit_date", "")) else "",
        }

    open_rows     = [row_to_dict(r) for _, r in opened.sort_values("mtm", ascending=False).iterrows()]
    resolved_rows = [row_to_dict(r) for _, r in resolved.iterrows()]

    return {
        "label":      label,
        "open":       open_rows,
        "resolved":   resolved_rows,
        "unrealized": round(float(opened["mtm"].sum()), 4),
        "realized":   round(float(resolved["pnl"].dropna().sum()), 4),
        "nW": int((resolved["status"] == "won").sum()),
        "nL": int((resolved["status"] == "lost").sum()),
        "nF": int((resolved["status"] == "flipped").sum()),
    }


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Polymarket Paper-Trade Dashboard</title>
<style>
  :root {
    --bg: #0f1117; --card: #1a1d27; --border: #2a2d3e;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b;
    --blue: #3b82f6; --text: #e2e8f0; --muted: #94a3b8;
    --flip: #a78bfa;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Inter', system-ui, sans-serif; font-size: 14px; padding: 24px; }
  h1 { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  .updated { color: var(--muted); font-size: 12px; margin-bottom: 24px; }
  .strategies { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 20px; margin-bottom: 32px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 20px; }
  .card-title { font-size: 15px; font-weight: 600; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 99px; font-weight: 500; }
  .badge-green { background: rgba(34,197,94,.18); color: var(--green); }
  .badge-red   { background: rgba(239,68,68,.18);  color: var(--red); }
  .badge-grey  { background: rgba(148,163,184,.12); color: var(--muted); }
  .kpis { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 14px; }
  .kpi { background: rgba(255,255,255,.04); border-radius: 8px; padding: 10px 12px; }
  .kpi-label { font-size: 11px; color: var(--muted); margin-bottom: 3px; text-transform: uppercase; letter-spacing: .4px; }
  .kpi-value { font-size: 18px; font-weight: 700; }
  .pos { color: var(--green); } .neg { color: var(--red); } .neu { color: var(--text); }
  details { margin-top: 10px; }
  summary { cursor: pointer; font-size: 12px; color: var(--muted); user-select: none; padding: 4px 0; }
  summary:hover { color: var(--text); }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }
  th { text-align: left; padding: 5px 8px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--border); }
  td { padding: 5px 8px; border-bottom: 1px solid rgba(42,45,62,.6); vertical-align: top; }
  tr:last-child td { border-bottom: none; }
  .q { max-width: 280px; }
  .pill { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 4px; font-weight: 600; }
  .pill-won     { background: rgba(34,197,94,.18);  color: var(--green); }
  .pill-lost    { background: rgba(239,68,68,.18);  color: var(--red); }
  .pill-flipped { background: rgba(167,139,250,.18);color: var(--flip); }
  .pill-open    { background: rgba(59,130,246,.12); color: var(--blue); }
  .meta { font-size: 11px; color: var(--muted); margin-top: 20px; border-top: 1px solid var(--border); padding-top: 12px; }
</style>
</head>
<body>
<h1>📊 Polymarket Paper-Trade Dashboard</h1>
<div class="updated">Last updated: <strong id="ts">—</strong> &nbsp;·&nbsp; auto-refreshes each cron run</div>

<div class="strategies" id="strategies"></div>

<div class="meta">
  Positions sized at 10% of bankroll. MTM = mark-to-market unrealized PnL on open positions.<br>
  <strong>Mid-priced YES</strong>: buy YES tokens in [0.10,0.50] ~5d before resolution, hold to settlement.<br>
  <strong>Smart Flow</strong>: follow ≥3 smart-wallet consensus buys, consensus-exit on reversal signal.<br>
  <strong>Macro</strong>: hold the dominant leg of macro resolution markets until year-end.
</div>

<script>
const DATA = %%DATA%%;

function fmt(v, pct=false) {
  if (v === null || v === undefined) return '—';
  const s = (v >= 0 ? '+' : '') + (pct ? (v*100).toFixed(1)+'%' : v.toFixed(4));
  return s;
}
function cls(v) { return v > 0.001 ? 'pos' : v < -0.001 ? 'neg' : 'neu'; }

function statusPill(s) {
  return `<span class="pill pill-${s}">${s}</span>`;
}

function buildCard(d) {
  const total = d.unrealized + d.realized;
  const totCls = cls(total);
  const badge = total > 0 ? '<span class="badge badge-green">▲ profitable</span>'
              : total < 0 ? '<span class="badge badge-red">▼ drawdown</span>'
              : '<span class="badge badge-grey">flat</span>';
  const resolved_summary = d.nW || d.nL || d.nF
    ? `${d.nW}W / ${d.nL}L${d.nF ? ' / '+d.nF+' flip' : ''}`
    : 'none yet';

  const openRows = d.open.map(r => `
    <tr>
      <td class="q">${r.question}</td>
      <td>${r.entry?.toFixed(3) ?? '—'}</td>
      <td>${r.current?.toFixed(3) ?? '—'}</td>
      <td class="${cls(r.mtm)}">${fmt(r.mtm)}</td>
    </tr>`).join('');

  const resolvedRows = d.resolved.map(r => `
    <tr>
      <td class="q">${r.question}</td>
      <td>${statusPill(r.status)}</td>
      <td class="${cls(r.pnl)}">${fmt(r.pnl)}</td>
      <td style="color:var(--muted)">${r.exit_date}</td>
    </tr>`).join('');

  return `
  <div class="card">
    <div class="card-title">${d.label} ${badge}</div>
    <div class="kpis">
      <div class="kpi">
        <div class="kpi-label">Total MTM</div>
        <div class="kpi-value ${totCls}">${fmt(total)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Unrealized</div>
        <div class="kpi-value ${cls(d.unrealized)}">${fmt(d.unrealized)}</div>
      </div>
      <div class="kpi">
        <div class="kpi-label">Realized</div>
        <div class="kpi-value ${cls(d.realized)}">${fmt(d.realized)}</div>
      </div>
    </div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:8px">
      ${d.open.length} open &nbsp;·&nbsp; ${d.resolved.length} resolved (${resolved_summary})
    </div>

    ${d.open.length ? `
    <details open>
      <summary>Open positions (${d.open.length})</summary>
      <table>
        <thead><tr><th>Question</th><th>Entry</th><th>Now</th><th>MTM</th></tr></thead>
        <tbody>${openRows}</tbody>
      </table>
    </details>` : ''}

    ${d.resolved.length ? `
    <details>
      <summary>Resolved (${d.resolved.length})</summary>
      <table>
        <thead><tr><th>Question</th><th>Status</th><th>PnL</th><th>Date</th></tr></thead>
        <tbody>${resolvedRows}</tbody>
      </table>
    </details>` : ''}
  </div>`;
}

document.getElementById('ts').textContent = DATA.generated;
document.getElementById('strategies').innerHTML = DATA.strategies.map(buildCard).join('');
</script>
</body>
</html>
"""


def generate() -> None:
    strategies = [
        load_strategy(
            DATA / "midprice_yes_positions.csv", "entry_ask", "question",
            "Mid-priced YES"),
        load_strategy(
            DATA / "smart_flow_positions.csv", "entry_ask", "question",
            "Smart Flow"),
        load_strategy(
            DATA / "macro_positions.csv", "entry_price", "leader_question",
            "Macro (Fed cuts)"),
    ]
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "strategies": strategies,
    }
    html = _HTML.replace("%%DATA%%", json.dumps(payload, ensure_ascii=False))
    DATA.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {OUT}")
    # quick summary to stdout
    for s in strategies:
        total = s["unrealized"] + s["realized"]
        sign = "+" if total >= 0 else ""
        print(f"  {s['label']:20s}  MTM {sign}{total:.4f}  "
              f"(open {len(s['open'])}, {s['nW']}W/{s['nL']}L"
              + (f"/{s['nF']}flip" if s["nF"] else "") + ")")


if __name__ == "__main__":
    generate()
