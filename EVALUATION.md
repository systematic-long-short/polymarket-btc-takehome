# EVALUATION — scoring notes for reviewers

Internal. Not distributed to candidates; candidates see `README.md`.

> **Scoring runs land only against live Polymarket. There is no accepted
> substitute.** Replay against the committed fixture is *iteration aid*, not
> scoring evidence. If the scoring window has a network or Polymarket
> outage, pause the run and restart — do NOT fall back to the fixture or to
> any other price source for PnL.

## The scoring pass

Run the candidate for a continuous **2-hour live window** (`--duration 7200`).
The harness chains ~24 consecutive 5-min BTC events back-to-back and always
runs `MomentumBaseline` in parallel on the same tape, so a single run
produces the apples-to-apples Model-vs-Baseline comparison.

Use the default `--price-source polymarket` unless you are intentionally
debugging an auxiliary BTC spot feed. Official scoring should depend only on
Polymarket Gamma/CLOB availability.

```bash
# Set up a fresh working dir per candidate.
mkdir -p runs/<candidate_id>

# 1. Scan. Must return "accept" (exit 0). If not — reject.
python scripts/scan_submission.py --file <submission>/model_submission.py

# 2. Run candidate live. Baseline runs alongside automatically.
python scripts/run_candidate.py \
    --submission <submission>/model_submission.py \
    --config <submission>/config.json \         # optional
    --duration 7200 \
    --output runs/<candidate_id>/
```

## Metrics in `report.json`

`report.json` has two top-level blocks, `model` and `baseline`, each with
the same schema. The scoring comparison is always between the two.

**Primary score (the headline):**

    primary_score = pnl_total × max(sharpe, 0) × (1 − max_drawdown)

A winning model with high Sharpe and small drawdown produces a large
positive score. `pnl_total` is the full run-total PnL, not a per-trade or
per-event average. Negative Sharpe is floored at zero for scoring, so a
net loser cannot receive a positive headline score from negative PnL times
negative Sharpe.

**Secondary (risk + consistency):**

- `sharpe` / `sortino` — annualized.
- `max_drawdown` — peak-to-trough fraction.
- `hit_rate` — fraction of events closed positive.
- `timeout_rate` — fraction of ticks where `on_tick` overran.
- `n_events`, `n_trades`, `n_ticks` — throughput metadata.

**Analytical (PnL attribution):**

- `pnl_intra_event` vs `pnl_resolution` — a model whose PnL comes
  overwhelmingly from resolution is just directionally betting. A model
  with high intra-event PnL is actually trading.
- `outcome_accuracy` — proxy for directional correctness. Low outcome
  accuracy with high PnL means the candidate profits by timing.

## The bar

A submission **passes** if, on the same 2-hour window (both values read
from the candidate's `report.json`):

- `model.primary_score > baseline.primary_score` **by a meaningful
  margin** — not within noise.
- `model.max_drawdown ≤ baseline.max_drawdown + 0.05` — we allow a bit
  more risk if the return justifies it, not much more.
- `model.timeout_rate < 5%`.
- The writeup articulates a real hypothesis, not "I tried some ML and
  it learned something."

A submission **fails fast** if:

- Scanner rejects the file.
- Harness can't load `ModelSubmission`.
- `on_tick` raises consistently.
- `model.primary_score ≤ 0` after a full run (net loser).
- `model.primary_score ≤ baseline.primary_score` (no edge).
- No writeup submitted.

## Things to look for in the writeup

- Do they explain WHY their signal should work? "Order book imbalance
  predicts Polymarket mid-drift" is a testable hypothesis. "A neural net
  learned the pattern" is not.
- Do they acknowledge spread + slippage + fees? The default 0.5% per-order slippage
  plus up-to-1.8% fee at p=0.5 adds up; a frequently-trading strategy
  needs signal strong enough to overcome it.
- Do they describe the failure mode? If BTC goes flat, what does their
  signal do? How did they validate this?
- Is the code readable? We read every line.

## Per-event inspection

The harness writes `ticks.parquet` alongside `report.json` with both
model and `baseline_*` columns. Open it:

```python
import pandas as pd
df = pd.read_parquet("runs/<candidate_id>/ticks.parquet")
# Per-event equity trace for each track:
for eid, g in df.groupby("event_id"):
    g = g.sort_values("ts")
    model_delta = g["equity"].iloc[-1] - g["equity"].iloc[0]
    base_delta = g["baseline_equity"].iloc[-1] - g["baseline_equity"].iloc[0]
    print(eid, f"model:{model_delta:+.2f} base:{base_delta:+.2f}")
```

Red flags while skimming:
- Same signal every tick for hundreds of ticks → they're not reacting to
  the market.
- Signal flips every tick → overtrading, paying spread on every reversal.
- `fills_this_tick` consistently == 0 even when signals change → book is
  one-sided (see `BookTop.is_tradable`) or the target quantity delta is
  sub-tick noise.

## When events aren't live

Polymarket's 5-min BTC series runs in bursts. If no live events are
available during the scoring window:

1. The candidate's run will log `no active BTC 5m event — waiting...`
   repeatedly. PnL may be $0.
2. Postpone the scoring run until the series resumes. Do not substitute
   the replay fixture for official scoring.

## Anti-exploitation checks

The AST scanner is a screen, not a sandbox. For scoring use
`scripts/run_candidate.py`, which also applies best-effort process hardening
before importing the candidate module. In addition:

- Keep evaluator secrets out of the process environment. The runner scrubs
  environment variables by default; use `--keep-env` only for local debugging.
- Keep the default process limits unless intentionally debugging:
  virtual memory 4 GB, max file size 1 GB, and max open files 256.
- Run in a dedicated working directory; `MarketInfo.scratch_dir` is the only
  filesystem path the model is told about.
- After the run, grep the submitted file for any imports the scanner
  might have missed.
- Review `report.json` — `n_trades == 0` with nonzero `pnl_total` is a
  red flag (possible simulator-state shenanigans).
