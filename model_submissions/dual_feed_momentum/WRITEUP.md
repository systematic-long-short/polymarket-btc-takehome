# Dual-Feed Momentum Example

The model combines two time-aligned signals the live harness provides each
second: Polymarket UP/DOWN token momentum and Binance BTC spot momentum. The
Polymarket leg captures what traders are doing in the exact market being
scored, while the Binance leg checks that the underlying BTC move agrees with
the token move.

The model uses only the current `Tick`: `up_bid`, `up_ask`, `down_bid`,
`down_ask`, `up_mid_recent`, `btc_last`, `btc_recent`, and `btc_source`. It does
not open network connections or read files. It requires nonzero Polymarket
books and nonzero BTC spot data, so run it with `--price-source binance`.

The signal is a weighted score: recent Polymarket UP-mid drift plus recent and
event-anchor Binance BTC returns. Positive scores buy UP, negative scores buy
DOWN, and small or reversing scores go flat. The default profile is deliberately
small and exits before the final resolution minute, because live top-of-book
fills can make held-to-settlement losses dominate a one-hour score. Size scales
with confidence but is capped by notional and share count to avoid constant
oversized rebalancing.

With more time I would tune the thresholds by replaying many live recordings
from different volatility regimes and add explicit spread/depth penalties once
the harness records full order-book depth rather than top-of-book only.
