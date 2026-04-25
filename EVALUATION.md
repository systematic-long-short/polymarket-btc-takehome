# EVALUATION — reviewer workflow

Internal. Do not distribute this file to candidates; candidates only need
`README.md` and a single `model_submission.py`.

## Non-negotiable market data policy

- Official scoring is live-only against Polymarket BTC 5-minute up/down events.
- Polymarket CLOB order books are the only source for executable fills and
  marks.
- Gamma is allowed for event discovery and final resolution metadata only.
- There is no synthetic book fill path and no Gamma executable-price fallback.
- Do not use Coinbase or Binance-US.
- The optional BTC spot feed is either disabled (`--price-source polymarket`,
  the default) or Binance global (`--price-source binance`).
- Candidate signals execute on the next recorded tick, never the same tick.
- Depth-aware execution is intentionally out of scope for this version.
- Fees remain Polymarket-style per share:
  `fee = shares * fee_rate * p * (1 - p)`.

## Pre-window operational soak

Run a soak shortly before a scoring window. This loads only the reference
MomentumBaseline and validates the live feed path:

```bash
python scripts/soak_validate.py \
    --duration 3600 \
    --price-source polymarket \
    --postmortem-timeout 120 \
    --min-events 10 \
    --max-clob-stale-count 0 \
    --max-missing-book-fraction 0.20
```

If you intend to expose the optional BTC spot fields during candidate scoring,
also run:

```bash
python scripts/soak_validate.py \
    --duration 3600 \
    --price-source binance \
    --postmortem-timeout 120 \
    --min-events 10 \
    --max-clob-stale-count 0 \
    --max-missing-book-fraction 0.20
```

The validation output checks active tick coverage, CLOB book freshness,
WebSocket reconnect/stale counts, missing and one-sided book rows, Gamma
discovery/resolution lag, unknown event count, and BTC source mode. If the
soak fails, postpone scoring rather than substituting replay data or another
price source.

## Official candidate run

Use the host-side runner. It builds or reuses the evaluator image, mounts only
the candidate file read-only, mounts one writable output directory at
`/output`, and runs the candidate inside a locked-down container:

```bash
mkdir -p runs/<candidate_id>

python scripts/run_official_evaluation.py \
    --submission <submission_dir>/model_submission.py \
    --output runs/<candidate_id> \
    --duration 7200 \
    --price-source polymarket \
    --postmortem-timeout 600
```

The runner applies:

- non-root container user
- read-only image filesystem plus tmpfs `/tmp` and `/run`
- no mounted evaluator secrets
- no Docker socket mount
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- PID, memory, CPU, file-size, and file-descriptor limits
- one controlled writable output mount

Inside the container, `scripts/run_candidate.py --official` performs its own
fail-closed checks before importing candidate code. It rejects debug bypass
flags, root execution, writable evaluator filesystem, writable submission
mount, mounted Docker socket, missing official-run marker, and sensitive
environment variables.

Trusted local debugging can still use:

```bash
python scripts/run_candidate.py --submission <path>/model_submission.py --duration 300
```

Do not use the local path for untrusted submissions.

## Pending resolutions are not final

Short/demo runs should stop near the requested `--duration`; they do not wait
for late Gamma settlement unless `--postmortem-timeout` is set. Official runs
use a postmortem wait, but Gamma can still lag.

If an official run exits with code `3` or `report.json` contains:

```json
"scoring_status": {"state": "pending_resolution"}
```

the score is not final and must not be accepted. Reconcile later:

```bash
python scripts/reconcile_resolutions.py \
    --report runs/<candidate_id>/report.json \
    --ticks runs/<candidate_id>/ticks.parquet
```

Repeat until `scoring_status.state` is `final`. The reconciliation command
updates the settlement rows in `ticks.parquet`, event PnL in `report.json`,
top-level final equity/PnL, metrics, and feed-health unknown counts. Use
`--allow-pending` only for monitoring jobs that are expected to retry later.
Use `--allow-unresolved-final` on the official runner only with an explicit
reviewer decision to accept unresolved events.

## Final validation

After the run is final:

```bash
python scripts/validate_live_run.py \
    --report runs/<candidate_id>/report.json \
    --ticks runs/<candidate_id>/ticks.parquet \
    --min-duration 7200 \
    --min-events 20 \
    --expect-price-source polymarket \
    --max-clob-stale-count 0 \
    --max-missing-book-fraction 0.20
```

For optional Binance mode, add `--require-binance --expect-price-source binance`.
Do not pass `--allow-unknown` for accepted official scores.

## Reproducibility metadata

Every report includes a `metadata` block with:

- candidate file path and SHA256
- git commit SHA
- Python and key package versions
- exact command/config
- container image and digest when provided by the runtime
- price source, slippage, fee rate, starting capital, and duration
- feed-health summary

Keep the report and ticks together. The report is not enough to inspect
per-tick behavior; the parquet is not enough to prove the exact command/config.

## Scoring policy

`report.json` has `model` and `baseline` blocks with the same schema. The
primary score is:

```text
primary_score = pnl_total * max(sharpe, 0) * (1 - max_drawdown)
```

A submission passes only if, on the same live window:

- `model.primary_score` beats `baseline.primary_score` by a meaningful margin
- `model.max_drawdown <= baseline.max_drawdown + 0.05`
- `model.timeout_rate < 5%`
- the code is readable and the signal has a concrete hypothesis

Fail fast if the scanner rejects the file, `ModelSubmission` cannot load,
`on_tick` raises consistently, `model.primary_score <= 0`, or the model does
not beat the baseline.

Resolution PnL is reported separately from intra-event PnL. Strong trading
models should make money by entering and exiting at good prices during the
event, not only by carrying directional exposure to settlement. Review
`resolution_pnl_fraction_abs` and `resolution_pnl_dominant_warning`; a high
resolution fraction is a review warning even if the primary score is positive.

## Per-event inspection

Open `ticks.parquet` alongside `report.json`:

```python
import pandas as pd

df = pd.read_parquet("runs/<candidate_id>/ticks.parquet")
for eid, g in df.groupby("event_id"):
    g = g.sort_values("ts")
    model_delta = g["equity"].iloc[-1] - g["equity"].iloc[0]
    base_delta = g["baseline_equity"].iloc[-1] - g["baseline_equity"].iloc[0]
    print(eid, f"model:{model_delta:+.2f} base:{base_delta:+.2f}")
```

Red flags:

- same signal every tick for hundreds of ticks
- signal flips every tick and pays spread repeatedly
- `fills_this_tick == 0` despite changing signals
- nonzero PnL with `n_trades == 0`
- report state is not `final`

## When events are unavailable

If the run logs repeated `no active BTC 5m event` messages and produces too few
events or ticks, postpone scoring. Do not use replay fixtures, synthetic data,
Gamma prices, Coinbase, Binance-US, or any other substitute for official PnL.
