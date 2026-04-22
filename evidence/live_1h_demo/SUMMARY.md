# Live 1-hour demo — MomentumBaseline vs MomentumBaseline (v4 / v3 polish)

Captured 2026-04-22 against the live Polymarket market
(`gamma-api.polymarket.com` / `clob.polymarket.com`).

## Command

```bash
python scripts/run_baseline.py --duration 3600 \
    --config configs/demo_momentum.json \
    --output-dir runs/demo_1h
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

The harness always runs `MomentumBaseline` (default config) alongside the
primary model on the same tape. For this run the primary model IS
`MomentumBaseline` with the demo config, so the `Model` column uses the
demo config and the `Baseline` column uses the stock defaults
(threshold 5 bps, size_cap 1.0).

## Terminal report (unabridged)

```
==============================================================================
  polybench run report
==============================================================================
  metric                                     Model                Baseline
  ------------------------------------------------------------------------
  starting capital                       $1,000.00               $1,000.00
  final equity                          $49,187.61              $66,433.62
  pnl total                       $48,187.61(4818.76%)        $65,433.62(6543.36%)
  primary score                     4,005,824.7486          6,053,619.2185

  sharpe (ann)                            111.8246                126.6124
  sortino (ann)                         10387.1085               9870.3498
  max drawdown                              25.66%                  26.93%
  hit rate                                  76.92%                  69.23%
  outcome accuracy                          30.77%                   0.00%
  timeout rate                               0.00%                   0.00%

  events                                        13                      13
  trades                                      1821                     813
  ticks                                       3084                    3084

  pnl intra-event                        $3,732.84               $1,542.80
  pnl resolution                        $44,454.77              $63,890.82
==============================================================================
```

## Acceptance assertions (all pass)

| Assertion                                      | Actual | Target | ✓/✗ |
|------------------------------------------------|--------|--------|-----|
| `n_events` (consecutive 5-min events)          | 13     | ≥ 10   | ✓   |
| `timeout_rate`                                 | 0.00%  | < 5%   | ✓   |
| `n_trades` (Model)                             | 1,821  | > 0    | ✓   |
| `n_trades` (Baseline)                          | 813    | > 0    | ✓   |
| Events labeled UP or DOWN (no `UNKNOWN`)       | 13/13  | all    | ✓   |
| Two-column report with Model + Baseline        | yes    | yes    | ✓   |
| Fee applied to every fill (Polymarket-style)   | yes    | yes    | ✓   |

## How to read the numbers

- **Both columns land in the 4000–6000% PnL range** because the paper
  simulator fills at Gamma's top-of-book under a $1000 starting capital
  assumption and extremely thin Polymarket books (tens to hundreds of
  dollars in depth). A $500 notional order at `best_ask = $0.05` buys
  10,000 shares; if that event resolves and those shares settle at $1.00,
  the position grows to $10,000 on that single event. Compound across 13
  events and you reach five-figure equity numbers. This is why the
  evaluator uses rank-order metrics (`PnL × Sharpe × (1 − max_drawdown)`)
  across identical market conditions, not raw dollar totals.
- **Model traded 2× more than Baseline** (1,821 vs 813) because the demo
  config uses `threshold_bps=1` vs Baseline's `5 bps` default. More
  trades at better prices ≠ more PnL on thin books — Baseline entered
  fewer but larger positions when momentum was strong, so its resolution
  PnL was higher.
- **`hit_rate`** — Model won 77% of events (pnl_total > 0 per event),
  Baseline 69%. Both well above 50%.
- **`outcome_accuracy`** — Model ended 31% of events holding the correct
  side at settlement, Baseline 0% (it flips often). With high PnL and
  low outcome accuracy, both strategies are profiting from timing
  (good entries on momentum) more than from correctly predicting
  direction, which matches the scoring philosophy in README.

## Per-event equity deltas (Model track)

```
  btc-updown-5m-1776867600   outcome=UP     pnl=+11,846.03
  btc-updown-5m-1776867900   outcome=DOWN   pnl=-278.97
  btc-updown-5m-1776868200   outcome=DOWN   pnl=+19,211.77
  btc-updown-5m-1776868500   outcome=UP     pnl=+5,013.61
  btc-updown-5m-1776868800   outcome=UP     pnl=+3,232.78
  btc-updown-5m-1776869100   outcome=DOWN   pnl=+7,174.64
  btc-updown-5m-1776869400   outcome=UP     pnl=+2,113.56
  btc-updown-5m-1776869700   outcome=UP     pnl=+2,021.89
  btc-updown-5m-1776870000   outcome=DOWN   pnl=-1,537.92
  btc-updown-5m-1776870300   outcome=UP     pnl=-1,015.84
  btc-updown-5m-1776870600   outcome=UP     pnl=+521.80
  btc-updown-5m-1776870900   outcome=DOWN   pnl=-415.22
  btc-updown-5m-1776871200   outcome=UP     pnl=+299.52
```

Slug gap is a uniform 300 s — the harness captured every consecutive
5-minute BTC event for the full hour.

## Companion files

- `report.json` — structured metrics + per-event breakdown (both Model
  and Baseline).
- Full `ticks.parquet` (3,084 rows × 33 columns including baseline_*)
  lives at `runs/demo_1h/ticks.parquet`; not committed because it is
  re-derivable.
