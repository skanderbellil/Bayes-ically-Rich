# Order-Book Snapshots — owning the depth history Polymarket doesn't keep

## Why

Polymarket's CLOB exposes full **live** order-book depth (`/book?token_id=…`)
but publishes **no historical books** — once an hour passes, that depth is gone
forever. The repo already solved this exact problem once: dealer gamma had no
public history either, so the cron started appending `datasets/gex_snapshots.csv`
and the project now owns a GEX series nobody else has. This does the same for
order books.

The immediate consumer is **execution modeling for PAYUP_FOLLOW**: its
break-even table is quoted against an *assumed* ~1¢ half-spread per share
(see `PAYUP_FOLLOW.md`). With a few weeks of snapshots, the half-spread, dollar
depth, and book imbalance become **measurable per token per day** — and every
paper ledger's fill assumptions can be audited against real quoted spreads.

## Cadence & universe

Every hourly `paper_trade.yml` run (step "Order-book snapshots",
`experiments/run_book_snapshot.py`) snapshots the books of the union of token
ids across all **open** positions in the `data/paper_trade/*_positions.csv`
ledgers (`token` column; `leader_token` for the macro ledger), deduped and
capped at 150 tokens. A missed snapshot (API down, no open positions) exits 0 —
the history just has an hourly gap.

## Schema

`data/paper_trade/book_snapshots.csv.gz` — one row per token per run:

| column            | meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `timestamp`       | UTC, ISO minute resolution (`2026-07-01T14:15Z`)               |
| `token_id`        | CLOB token id — **string** (77-digit ints overflow float64)   |
| `best_bid` / `best_ask` / `mid` / `spread` | top of book; `spread/2` is the measured half-spread |
| `microprice`      | size-weighted top-of-book price                                |
| `bid_depth_10` / `ask_depth_10` | dollar depth (Σ price·size) within 10¢ of mid    |
| `depth_imbalance` | top-of-book (bid−ask)/(bid+ask) size imbalance ∈ [−1, 1]       |
| `n_bid_levels` / `n_ask_levels` | book depth in levels                             |

Appends are deduped on (`timestamp`, `token_id`) and rewritten atomically. When
the compressed file passes ~8 MB (~40 MB raw) the run logs a rotation warning;
history is never auto-deleted.

## Loading it

```python
import pandas as pd

df = pd.read_csv(
    "data/paper_trade/book_snapshots.csv.gz",
    dtype={"token_id": str},          # never let token ids become floats
    parse_dates=["timestamp"],
)
half_spread = df.groupby([df.timestamp.dt.date, "token_id"])["spread"].median() / 2
```
