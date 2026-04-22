# Live 1-hour demo — MomentumBaseline on Polymarket BTC 5m

Captured 2026-04-22 ~19:40Z → 20:40Z UTC, against the live Polymarket
market (`gamma-api.polymarket.com` / `clob.polymarket.com`). Python
harness at commit `$(post-fix)`.

## Command

```bash
python scripts/run_baseline.py --model momentum --duration 3600 \
    --config configs/demo_momentum.json --output-dir runs/demo_1h
```

`configs/demo_momentum.json`:
```json
{
  "lookback_s": 20,
  "threshold_bps": 1.0,
  "size_cap": 0.5,
  "bps_per_unit": 8.0
}
```

## Terminal report (unabridged)

```
==============================================================
  polybench run report
==============================================================
  starting capital:     $    1,000.00
  final equity:         $   15,429.65
  pnl total:            $   14,429.65  (1442.97%)
  primary score:             241.4395  (Sharpe × sign(PnL))

  sharpe (ann):             241.4395
  sortino (ann):           1338.5826
  max drawdown:               17.65%
  hit rate:                   84.62%  (events with PnL > 0)
  outcome accuracy:            7.69%  (secondary)
  timeout rate:                0.00%

  events:                          13
  trades:                        1668
  ticks:                         3039

  pnl intra-event:      $   11,500.04
  pnl resolution:       $    2,929.61
==============================================================
```

## Acceptance assertions (all pass)

| Assertion                                                | Actual | Target | ✓/✗ |
|----------------------------------------------------------|--------|--------|-----|
| `n_events` (consecutive 5-min events scored)             | 13     | ≥ 10   | ✓   |
| `timeout_rate` (ticks where on_tick overran 500 ms)      | 0.00%  | < 5%   | ✓   |
| `n_trades` (total fills)                                 | 1,668  | > 0    | ✓   |
| Every event resolved with UP or DOWN (no `UNKNOWN`)      | 13/13  | all    | ✓   |
| `ticks.parquet` row count                                | 3,039  | ≥ 3000 | ✓   |

## Per-event roll-up

```
slug                              outcome   pnl_total   intra       reso        trades
btc-updown-5m-1776860400          DOWN      -136.33     -6.98       -129.34     6
btc-updown-5m-1776860700          DOWN      +2684.76    +2694.92    -10.16      135
btc-updown-5m-1776861000          UP        +641.18     +641.18     +0.00       150
btc-updown-5m-1776861300          DOWN      +1650.21    +1738.44    -88.23      111
btc-updown-5m-1776861600          DOWN      +456.35     +456.35     +0.00       140
btc-updown-5m-1776861900          DOWN      +1221.14    +1221.14    +0.00       123
btc-updown-5m-1776862200          UP        +70.82      +70.82      +0.00       141
btc-updown-5m-1776862500          DOWN      +1110.15    +1110.15    +0.00       131
btc-updown-5m-1776862800          UP        +3899.32    +304.65     +3594.66    152
btc-updown-5m-1776863100          DOWN      +1993.09    +2232.88    -239.80     164
btc-updown-5m-1776863400          UP        -43.43      -43.43      +0.00       120
btc-updown-5m-1776863700          UP        +499.36     +696.89     -197.53     153
btc-updown-5m-1776864000          UP        +383.02     +383.02     +0.00       142
```

Slug gap is a uniform 300 s — the harness captured every consecutive
5-minute BTC event for the full hour, driven by the direct-slug
enumerator that starts from `floor(now/300)*300` (current trading
window) rather than `ceil(…)` (next window, which silently drops every
other event).

## Reading the PnL number

The paper simulator fills at `best_ask * (1 + slippage)` assuming
infinite order-book depth. Polymarket's 5-minute BTC books are thin
(often $50–200 at the top), so real-world execution of a 8,000-share
order against a best-ask quote would move the book substantially.
**Treat the absolute PnL as a ranking metric between candidate models
on identical simulator rules, NOT as achievable dollar PnL on the real
market.** `hit_rate` + Sharpe are the durable comparators; raw PnL is
amplified by the idealized-fill assumption and by MomentumBaseline's
aggressive `size_cap=0.5` exposure at thin prices.

## Companion files

- `report.json` — structured metrics + per-event breakdown.
- The full `ticks.parquet` (3039 rows × 24 columns) lives at
  `runs/demo_1h/ticks.parquet` in the run output; it is not committed
  because it is derivable from a fresh run.
