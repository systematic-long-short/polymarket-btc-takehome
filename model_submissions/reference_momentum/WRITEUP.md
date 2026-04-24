# Reference Momentum Example

This example mirrors the benchmark momentum model in a normal submission file.
When run with `--price-source binance`, it uses the Binance BTC rolling window
that the harness places on each `Tick`. When the run is Polymarket-only, it
falls back to Polymarket UP-mid momentum from the same tick stream.

The signal is intentionally simple: compare the latest value to the value from
`lookback_s` seconds ago. Positive BTC or UP-mid momentum buys UP, negative
momentum buys DOWN, and small moves go flat. Size scales with move magnitude and
is capped by `size_cap`.

This is a reference scoring example, not an optimized strategy. It exists so
developers can run a known scanner-safe submission through the exact same live
candidate path used for take-home evaluation.
