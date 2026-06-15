# Level control — the sharp-move "edge" was mostly price level

> **This study corrects `EVENT_TRADE_HARVEST.md`.** The hold-to-resolution profit
> is real, but it is the favorite–longshot **calibration of the entry price band**,
> not the upside-vol detection. Controlled for price level, the vol signal sign-flips
> — and is already priced.

```bash
python experiments/run_polymarket_band_control.py --refresh
python experiments/run_polymarket_event_trades.py --min-price 0.15 --max-price 0.30   # band-restricted
```

## The objection

The harvest trade entered around price 0.27, and `upside_vol` is only high for
markets that are *alive* (away from 0.01). So high-vol episodes carry a higher
price, and the apparent "calibration edge" could just be: **markets priced
0.15–0.30 underprice Yes** (a favorite–longshot effect), with upside vol acting as
a proxy for "this market has a tradable price". The control: hold price fixed
inside narrow bands and compare high- vs low-vol episodes.

## Within-band test (250 markets, 11,262 episodes)

| price band | n | Yes-rate hi vol | Yes-rate lo vol | **hi − lo** | fwd Δz hi | fwd Δz lo |
|---|---:|---:|---:|---:|---:|---:|
| [0.05, 0.15) | 1060 | 10.7% | 20.6% | **−9.9%** | −0.137 | −0.080 |
| [0.15, 0.30) | 677 | 38.9% | 42.4% | **−3.5%** | −0.209 | +0.036 |
| [0.30, 0.50) | 585 | 42.6% | 43.5% | −0.9% | +0.035 | +0.011 |
| [0.50, 0.70) | 331 | 57.0% | 56.3% | +0.7% | −0.063 | −0.072 |
| [0.70, 0.95) | 409 | 80.5% | 51.0% | **+29.4%** | +0.069 | +0.055 |

Episode-weighted mean within-band lift ≈ **−0.4%** (2/5 bands positive). Pooled, the
vol signal carries **no** level-controlled Yes-rate edge — but it is far from flat:
**the sign flips with price level.**

- **Longshots/mids that spike up → reversion.** In [0.05, 0.30) high upside vol
  resolves Yes *less* than quiet markets at the same price, with strongly negative
  forward drift (−0.137, −0.209). A spiking longshot is noise/overreaction.
- **Favorites that spike up → momentum.** In [0.70, 0.95) high vol resolves Yes
  **+29.4%** more (80.5% vs 51.0%), with positive forward drift. A spiking favorite
  is information.

The naive pooled study averaged these opposite effects to ~zero and mistook the
**band calibration** for a vol edge.

## Does any of it trade? (hold-to-resolution & drift, 1% slippage)

| trade | n | avg PnL/trade | trade-Sharpe | note |
|---|---:|---:|---:|---|
| mids [0.15,0.30] **all-vol** Yes → resolution | 66 | **+0.114** | +0.236 | the real edge: band calibration |
| mids [0.15,0.30] **high-vol** Yes → resolution | 54 | +0.090 | +0.191 | vol filter *hurts* vs all-vol |
| favs [0.70,0.95] high-vol Yes → resolution | 32 | −0.073 | −0.162 | +29% lift already priced in |
| mids [0.15,0.30] high-vol **No**, 10d horizon | 95 | −0.008 | −0.043 | reversion ≈ breakeven net of slip |
| favs [0.70,0.95] high-vol Yes, 10d horizon | 52 | −0.039 | −0.172 | momentum doesn't beat slip |

Three conclusions, in order of importance:

1. **The only robust positive PnL is the favorite–longshot calibration of the
   [0.15, 0.30] band** — buy Yes there and hold to resolution earns ~+11¢ per $1
   contract (trade-Sharpe 0.24), **regardless of vol**. Adding the upside-vol filter
   *reduces* it (+0.090 vs +0.114): the vol detector had merely been *riding* this
   structural mispricing, not creating edge.
2. **The level-controlled vol signal is real but already priced.** The favorites'
   +29% Yes-lift does not pay (buying Yes at ~0.78 that resolves 72% *loses* −7¢),
   and the longshot reversion barely covers slippage. The market prices the signal.
3. So the genuine, harvestable effect in this universe is **structural
   (favorite–longshot mispricing of mid-low-probability markets)**, not a
   **timing/volatility** signal. That is the honest headline.

### Caveats (unchanged + reinforced)

Thin per-cell counts (32–95 trades), trades cluster in the 2024-election complex,
winners lock up capital ~80 days (PnL not time-annualised), survivorship-tilted
universe, and the favorite–longshot calibration itself is the kind of effect that
is notoriously sensitive to the universe and to settlement assumptions. Read the
signs, not the t-stats. This is a research artifact — and a cautionary tale about
confounds — not a deployable edge.

Next steps that *could* still find timing alpha: join live order-book imbalance
(`fetch.order_book_features`) at the spike to separate informed favorite-moves from
noise longshot-moves *before* the price adjusts; and test the [0.15,0.30] calibration
on a non-election universe to see whether the structural mispricing is real or a
2024 artifact.
