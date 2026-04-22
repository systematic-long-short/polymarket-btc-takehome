# EVALUATION — scoring notes for reviewers

Internal. Not distributed to candidates; candidates see `README.md`.

## The scoring pass

Run the candidate for a continuous **2-hour live window** (`--duration 7200`).
Over that window, the harness will chain ~24 consecutive 5-min events when
live events are available. Run the three baselines in **the same window**
so ambient price conditions are identical.

```bash
# Set up a fresh working dir per candidate.
mkdir -p runs/<candidate_id>

# 1. Scan. Must return "accept" (exit 0). If not — reject.
python scripts/scan_submission.py --file <submission>/model_submission.py

# 2. Run candidate live.
python scripts/run_candidate.py \
    --submission <submission>/model_submission.py \
    --config <submission>/config.json \         # optional
    --duration 7200 \
    --output runs/<candidate_id>/candidate

# 3. Run each baseline in a parallel window (separate processes).
python scripts/run_baseline.py --model momentum --duration 7200 \
    --output-dir runs/<candidate_id>/baseline_momentum
python scripts/run_baseline.py --model alwaysup --duration 7200 \
    --output-dir runs/<candidate_id>/baseline_alwaysup
python scripts/run_baseline.py --model random --duration 7200 \
    --output-dir runs/<candidate_id>/baseline_random
```

## Metrics to read from `report.json`

Primary:
- `pnl_total` — dollars.
- `pnl_pct` — vs. `starting_capital`.
- `primary_score` — `|Sharpe| × sign(PnL)`. This is the headline.

Risk:
- `sharpe` — annualized on tick-level returns.
- `sortino` — downside-only.
- `max_drawdown` — peak-to-trough.

Consistency:
- `n_events` — typically ~24 over a 2h run.
- `hit_rate` — fraction of events closed positive.
- PnL stdev across events (compute from the `events[]` list).

Analytical:
- `pnl_intra_event` vs `pnl_resolution` — a model whose PnL comes
  overwhelmingly from resolution is essentially just directionally
  betting. A model with high intra-event PnL is actually trading.
- `outcome_accuracy` — a proxy for directional correctness. Low outcome
  accuracy with high PnL means the candidate profits by timing, not by
  predicting.
- `timeout_rate` — if >5% the model is too slow.

## The bar

A submission **passes** if, on the same 2-hour window:
- `primary_score(candidate) > primary_score(momentum)` **by a meaningful
  margin** — not within noise. Check Sharpe magnitude + PnL sign align.
- `max_drawdown(candidate) ≤ max_drawdown(momentum) + 0.05` — we're fine
  with more drawdown if the returns justify it, not if it's just
  leverage dressed up as alpha.
- `timeout_rate < 5%`.
- The writeup articulates a real hypothesis — not "I tried some ML and
  it learned something."

A submission **fails fast** if:
- The scanner rejects the file.
- The harness can't load the `ModelSubmission` class.
- The run errors out repeatedly (`on_tick` raising).
- `primary_score(candidate) ≤ primary_score(random)`.
- No writeup submitted.

## Things to look for in the writeup

- Do they explain WHY their signal should work? "Order book imbalance
  predicts Polymarket mid-drift" is a testable hypothesis. "A neural net
  learned the pattern" is not.
- Do they acknowledge the spread + slippage? The default 2% is
  punishing; a strategy that trades frequently needs signal strong
  enough to overcome that.
- Do they describe the failure mode? If the BTC market goes flat, what
  does their signal do? How did they validate this?
- Is the code readable? We read every line of every submission.

## Per-event inspection

The harness writes `ticks.parquet` alongside `report.json`. Open it:

```python
import pandas as pd
df = pd.read_parquet("runs/<candidate_id>/candidate/ticks.parquet")
# Per-event PnL curve:
for eid, g in df.groupby("event_id"):
    g = g.sort_values("ts")
    print(eid, "pnl delta:", g["equity"].iloc[-1] - g["equity"].iloc[0])
```

Red flags while skimming:
- Same signal every tick for hundreds of ticks → they're not reacting to
  the market.
- Signal side flips every tick → they're overtrading and paying spread.
- `fills_this_tick` consistently == 0 even when signals change → the
  book is one-sided or their target quantities are sub-tick noise.

## When events aren't live

Polymarket's 5-min BTC series runs in bursts. If no live events are
available during the 2h window:

1. The candidate's scoring run will log `no active BTC 5m event —
   waiting...` repeatedly. Their PnL may be $0 (no ticks processed).
2. Postpone the scoring run until the series resumes; do not substitute
   replay against the committed fixture for official scoring.
3. For development feedback on candidates already submitted, use replay
   against the committed fixture — it's deterministic but does not
   reflect live market microstructure.

## Anti-exploitation checks

The AST scanner is a screen, not a sandbox. For the scoring run we
also:

- Cap process memory via `ulimit -v 4194304` (~4 GB).
- Run in a dedicated working directory (`runs/<candidate_id>/candidate/`);
  the harness only passes `market_info.scratch_dir` to the model.
- After the run, grep the model file for any novel imports we missed.
- Review `report.json` — `n_trades == 0` with nonzero PnL is a red flag
  (possible fake PnL by side effects on simulator state).
