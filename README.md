# polymarket-btc-takehome

A take-home project for technical candidates.

You write a Python model that emits buy / sell / flat signals every second
against Polymarket's 5-minute BTC up/down event. We paper-trade your signal
against the real live market and score you on raw PnL, risk-adjusted return,
and consistency across many events.

Keep it real: you're competing against a non-trivial momentum baseline that
trades the same tape. Beating it is the bar.

---

## The task

Subclass `polybench.Model`, implement `on_tick(tick) -> Signal`, submit.
That's it.

```python
from polybench import FLAT, Model, Side, Signal, Tick

class ModelSubmission(Model):
    def on_tick(self, tick: Tick) -> Signal | None:
        # Look at `tick`, decide what to do, return a Signal.
        if tick.btc_last > some_threshold:
            return Signal(side=Side.UP, size=0.5)
        return FLAT
```

The harness runs your class for a continuous 1–2 hour window, chaining
consecutive 5-min events back-to-back. You can keep any state you like on
`self`. The harness calls you at 1 Hz during events, skips you between them.

---

## Setup (under 5 minutes, assuming Python 3.11+)

```bash
git clone <repo-url> polymarket-btc-takehome
cd polymarket-btc-takehome
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # ≈ 1–2 GB, takes 5–10 min (torch + tensorflow)
pip install -e .
```

Then:

```bash
# Replay against the committed fixture (offline, fast, deterministic)
python replay.py --model polybench.baselines:MomentumBaseline \
    --data tests/fixtures/recorded_event.parquet

# Run live against Polymarket (requires active BTC 5m events)
python scripts/run_baseline.py --model momentum --duration 300
```

> **Live scoring is Polymarket-only.** The harness auto-discovers the
> currently-trading 5-min BTC event every ~3 seconds by enumerating the
> canonical slug `btc-updown-5m-<end_ts>` directly from Gamma. When one
> event resolves it rolls straight to the next, chaining ~12 events per hour
> during the scoring run. If discovery genuinely fails, the run fails —
> there is no synthetic or cross-exchange fallback for PnL.
>
> Demo runs stop near the requested `--duration`. If the final event has not
> resolved yet, it is reported as `UNKNOWN` rather than waiting minutes after
> a short smoke test.

---

## What the harness gives you each tick

```python
@dataclass(frozen=True)
class Tick:
    ts: float                  # unix seconds
    time_to_resolve: float     # seconds until event endDate

    btc_last: float            # optional BTC spot feed; 0 in default Polymarket-only live runs
    btc_bid: float             # optional BTC bid; 0 in default Polymarket-only live runs
    btc_ask: float             # optional BTC ask; 0 in default Polymarket-only live runs
    btc_source: str            # "polymarket" or "binance"

    up_bid: float              # Polymarket UP token best bid (in $0–$1)
    up_ask: float
    up_mid: float
    down_bid: float
    down_ask: float
    down_mid: float

    btc_recent: tuple[float, ...]      # rolling BTC window; empty in default Polymarket-only runs
    up_mid_recent: tuple[float, ...]   # rolling window of UP mids
    event_id: str
```

You return a `Signal`:

```python
Signal(side=Side.UP|DOWN|FLAT, size=0.0..1.0, confidence=0.0..1.0)
```

`size` is the fraction of your **starting capital** ($1,000 by default)
to allocate. `confidence` is analytics-only; it doesn't affect fills.

Return `FLAT`, `None`, or a zero-size signal to close your position.

---

## What you submit

Submit **exactly one file named `model_submission.py`**. It must define a
class `ModelSubmission` that subclasses `polybench.Model`. File name and
class name are fixed because the evaluator runs automation against those
exact strings.

Put your full model in that one file: state, feature logic, constants, and
hyperparameters. Do not submit extra files, custom dependencies, data files,
or harness changes.

---

## Libraries you can use (the full list)

Your submission may import only:

* **Python standard library** — the usual safe subset (`math`, `statistics`,
  `collections`, `itertools`, `functools`, `datetime`, `json`, `re`,
  `random`, `threading`, `queue`, `asyncio`, `pathlib`, `logging`, etc.).
  Direct filesystem, environment, subprocess, dynamic import, and
  introspection APIs are rejected by the scanner even if their modules are
  otherwise part of Python.
* **Data & numerical:** `numpy`, `pandas`, `polars`, `scipy`, `statsmodels`
* **Classical ML:** `scikit-learn`, `lightgbm`, `xgboost`, `optuna`
* **Deep learning (CPU):** `torch`, `tensorflow`
* **Technical indicators:** `ta`
* **Harness APIs you can import:** `polybench`, `httpx`, `websockets`

Data libraries are available for in-memory computation. Their direct file
readers/writers, such as `pandas.read_*`, `numpy.fromfile`, and
`polars.scan_*`, are rejected. Unknown imports are also rejected.

---

## The rules

- **Latency budget:** `on_tick` must return within **500 ms** wall-clock.
  Overruns drop that tick's signal; repeated timeouts tank your score.
- **Max 1 signal per second.** We only sample your return once per tick.
- **Model state lives on `self`.** No globals, no monkey-patching the
  harness, no mutating the `Tick` / `Signal` / `MarketInfo` dataclasses
  (they're frozen).
- **Network access from `on_tick` is allowed.** It counts against the
  500 ms budget. For slow external feeds, start a background thread in
  `on_start` and read cached state from `on_tick` — see the template for
  the pattern.
- **No direct filesystem or environment access.** Do not read files or
  environment variables from the submission. No subprocesses. No reading
  forward-looking data (the resolution price is not exposed — don't go
  looking for it in Gamma either).
- **No peeking at the resolution.** Obvious, but stated. The parquet
  recorder logs every tick's inputs, so we can verify post-hoc that your
  signal at time T didn't depend on data from T+1 or the settled outcome.

---

## How scoring works

**Primary score:** `PnL × max(Sharpe, 0) × (1 − max_drawdown)`. Bigger is better.

- `PnL` is total dollars made over the full run (final equity minus starting capital).
- `Sharpe` is annualized on tick-level returns and floored at zero for
  scoring. Consistency matters, but negative consistency does not help.
- `max_drawdown` is the worst peak-to-trough drawdown seen during the run,
  as a fraction in `[0, 1]`. The `(1 − max_drawdown)` multiplier penalizes
  volatile wins: a strategy that earns $100 cleanly beats one that earns
  the same $100 after a 40% intra-run drawdown.

**How PnL is computed.** PnL is liquidation mark-to-market, tick by tick. When you
send a signal, the simulator executes the delta against the current
Polymarket order book:

- Buys fill at `best_ask × (1 + slippage)` (default slippage 0.5% per order).
- Sells fill at `best_bid × (1 − slippage)`.
- Every fill also pays a Polymarket-style fee of
  `|notional| × fee_rate × p × (1 − p)` where `p` is the fill price in
  `[0, 1]` and `fee_rate = 0.072` by default. Fee is maximal at
  `p = 0.5` (~1.8% of notional) and collapses toward zero at the
  extremes — buying a near-certain outcome costs very little, taking a
  coin-flip position costs more.

Your equity each tick is a liquidation mark: cash plus held shares valued
at the current executable exit bid after slippage and exit fee. A one-sided
book with a valid bid can still mark or sell an existing long; a side with
no valid bid marks at zero because it cannot be exited immediately. At event
end, any held shares settle at the resolved `outcomePrices` (typically
`[1, 0]` or `[0, 1]`).

The important consequence: **you earn PnL from entering at good prices
and riding the token to a better price — not from being "correct" about
UP vs DOWN.** A candidate who buys UP at $0.40 and closes at $0.55
captures $0.15/share of edge regardless of whether the event eventually
resolves UP or DOWN.

**Secondary metrics (recorded, not the gate):**

- `sharpe` / `sortino` — annualized on tick-level returns.
- `hit_rate` — fraction of events closed with positive PnL.
- `outcome_accuracy` — fraction of events where your resolution-PnL was
  positive.
- `timeout_rate` — fraction of ticks where `on_tick` overran 500 ms.
- PnL attribution — intra-event vs resolution split.

---

## The one baseline you're compared against

Every run prints a two-column report: **Model** (your submission) vs
**Baseline** (`MomentumBaseline`), each paper-traded on the exact same
tick stream. You can't cherry-pick a favorable window; the comparison is
apples-to-apples per second.

- `MomentumBaseline` — 30-second BTC momentum → UP / DOWN / FLAT with
  magnitude-scaled size, trades dynamically within each event. This is
  the shipped example to beat. Strong submissions should materially
  outperform it on the primary score.

Run the baseline solo to see what "the bar" is producing under current
market conditions:

```bash
python scripts/run_baseline.py --duration 300
# both Model and Baseline columns in the report are identical for this
# call (you ran the baseline as your "model") — that's by design.
```

The official live path is **Polymarket-only by default**. It discovers the
active BTC 5-minute event from Gamma and polls Polymarket order books; it does
not require Binance WebSockets. Evaluators can opt into an auxiliary
BTC feed with `--price-source binance`. When this is enabled, recorded
`ticks.parquet` rows contain both Polymarket UP/DOWN order-book fields and
Binance BTC fields, with `btc_source` naming the active BTC feed. The live
validator reports `polymarket_valid_rows`
(active rows with usable UP/DOWN mids), `polymarket_two_sided_rows` (both
bid and ask populated for both tokens), and `polymarket_one_sided_rows`
(valid rows where one bid/ask is zero, which can happen near resolution).

To run the dual-feed example live for a full scoring hour:

```bash
python scripts/run_candidate.py \
    --submission model_submissions/dual_feed_momentum/model_submission.py \
    --duration 3600 \
    --price-source binance \
    --output runs/live_1h_dual_feed

python scripts/validate_live_run.py \
    --report runs/live_1h_dual_feed/report.json \
    --ticks runs/live_1h_dual_feed/ticks.parquet \
    --min-duration 3600 \
    --min-events 10 \
    --require-binance
```

For a scanner-safe reference strategy matching the built-in benchmark, use
`model_submissions/reference_momentum/model_submission.py` with the same command.

---

## Working offline (replay mode)

Waiting for live 5-min events is slow. Iterate against the committed
fixture instead:

```bash
python replay.py \
    --model-file examples/model_submission.py \
    --class ModelSubmission \
    --data tests/fixtures/recorded_event.parquet
```

The fixture is schema-compatible with live recordings — the same
`on_tick` gets called with the same `Tick` shape, so anything that works
in replay will work live.

The committed fixture is an offline regression aid with realistic synthesized
Polymarket token prices. It exercises both UP-winning and DOWN-winning
resolutions, but official scoring always happens live.

To record your own fresh fixture from live data:

```bash
# Record a live Polymarket event (scoring-equivalent data):
python scripts/run_baseline.py --model momentum --duration 600 \
    --output-dir runs/my_record
cp runs/my_record/ticks.parquet tests/fixtures/my_event.parquet
```

`scripts/synthesize_fixture.py` also exists for purely-offline iteration
when you have no internet. It is a **developer convenience only** — never
used for scoring. See the banner in that file.

---

## Time expectation

**3 days max.** If you're grinding into day 4, something's off with your
approach.

---

## FAQ

**Q: Can my model take >500 ms on the first tick of an event to do
some expensive setup?**
A: Yes — put expensive setup in `on_start`. It's not time-budgeted.

**Q: Can I use GPU libraries?**
A: `torch` and `tensorflow` are installed as CPU builds. Don't assume a
GPU at scoring time.

**Q: How do I see what my model is doing during a run?**
A: The harness writes `runs/<id>/ticks.parquet` with every tick's inputs,
your signal, the fill, and your position. Open it with pandas/polars and
investigate.

**Q: Do I have to run live for every iteration?**
A: For **scoring** yes — scoring happens only against the live market.
For **fast iteration** use the committed replay fixture; everything that
replays correctly will run correctly live. The fixture is not accepted as
an official score.

**Q: Can I call external APIs (exchange book depth, sentiment feeds)?**
A: Yes, but from a background thread if they're slow. The 500 ms budget
applies to `on_tick` return.

**Q: What happens if my signal is outside `[0, 1]`?**
A: Clamped. Negative sizes are treated as zero.

**Q: Does my position carry between events?**
A: No. Event tokens are not fungible across consecutive 5-minute markets.
Any shares still held at event end are settled against that event's final
`outcomePrices`, then the simulator starts the next event flat. If you
want to avoid resolution exposure, emit `FLAT` when `tick.time_to_resolve`
drops near zero.

**Q: Can I modify the harness code?**
A: No. Your submission is a single file. Don't touch `polybench/`.

---

## Repo layout

```
polymarket-btc-takehome/
├── README.md                     ← you are here
├── EVALUATION.md                 ← evaluator-only scoring notes
├── pyproject.toml                ← hatchling, Python ≥ 3.11
├── requirements.txt              ← installed runtime dependencies
├── Dockerfile                    ← optional reproducibility
├── src/polybench/
│   ├── __init__.py               ← re-exports Model, Tick, Signal, ...
│   ├── model.py                  ← Model ABC + dataclasses
│   ├── harness.py                ← async event loop, event rollover
│   ├── market.py                 ← Gamma + CLOB client
│   ├── pricefeed.py              ← optional BTC WS feed; off by default
│   ├── pnl.py                    ← paper-trade simulator
│   ├── metrics.py                ← Sharpe / Sortino / DD / hit rate
│   ├── recorder.py               ← parquet tick log
│   ├── replay.py                 ← offline replay engine
│   ├── submission_scan.py        ← AST security screener
│   ├── baselines.py              ← reference MomentumBaseline
│   └── cli.py                    ← polybench run|replay|scan
├── models/baseline_models.py     ← re-export of polybench.baselines
├── examples/model_submission.py  ← candidate template
├── scripts/
│   ├── run_baseline.py
│   ├── run_candidate.py          ← scan → load → run (live)
│   ├── scan_submission.py
│   ├── record_fixture.py         ← record a live event
│   └── synthesize_fixture.py     ← build fixture from BTC + synthetic PM
├── replay.py                     ← top-level alias for `polybench replay`
└── tests/
    ├── test_pnl.py
    ├── test_metrics.py
    ├── test_market_parse.py
    ├── test_submission_scan.py
    ├── test_harness_smoke.py
    └── fixtures/
        ├── recorded_event.parquet
        ├── safe_submission.py
        └── unsafe_submission.py
```

---

## License

MIT.
