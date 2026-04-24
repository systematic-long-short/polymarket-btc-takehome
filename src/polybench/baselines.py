"""Reference baseline — the single model every candidate is scored against.

The harness always runs ``MomentumBaseline`` in parallel with the candidate's
submission on the same tape, and the report prints both side-by-side. To
pass, the candidate must materially beat this baseline on the primary score
(``PnL × Sharpe × (1 − max_drawdown)``).
"""

from __future__ import annotations

import logging
from typing import Any

from polybench.model import FLAT, MarketInfo, Model, Side, Signal, Tick

log = logging.getLogger("polybench.baselines")


class MomentumBaseline(Model):
    """30-second momentum strategy — the bar candidates must beat.

    Default live runs are Polymarket-only, so the baseline uses UP-mid token
    momentum when BTC spot fields are unavailable. If a caller explicitly
    enables an external BTC feed, it falls back to the original BTC momentum
    rule.

    This strategy trades DYNAMICALLY within each event: it re-enters, flips,
    and exits as BTC moves. Its PnL comes primarily from intra-event price
    moves on Polymarket tokens, not from holding through resolution.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._lookback_s = int(self.config.get("lookback_s", 30))
        self._threshold_bps = float(self.config.get("threshold_bps", 5.0))
        self._size_cap = float(self.config.get("size_cap", 1.0))
        # Sensitivity: 1 basis point move maps to this fraction of size.
        self._bps_per_unit = max(1e-9, float(self.config.get("bps_per_unit", 20.0)))
        self._pm_threshold = float(self.config.get("pm_threshold", 0.01))
        self._pm_per_unit = max(1e-9, float(self.config.get("pm_per_unit", 0.08)))

    def on_start(self, market_info: MarketInfo) -> None:
        log.debug("MomentumBaseline: event start %s", market_info.slug)

    def on_tick(self, tick: Tick) -> Signal | None:
        if len(tick.btc_recent) >= 2 and tick.btc_recent[-1] > 0.0:
            return self._btc_momentum(tick)
        return self._polymarket_momentum(tick)

    def _btc_momentum(self, tick: Tick) -> Signal | None:
        window = tick.btc_recent
        if len(window) < 2:
            return FLAT
        # Use the sample from ``_lookback_s`` seconds ago — approximated by
        # index offset assuming 1 Hz samples (which matches the harness's
        # 1 Hz btc_recent window).
        offset = min(len(window) - 1, self._lookback_s)
        past = window[-1 - offset]
        now_price = window[-1]
        if past <= 0.0 or now_price <= 0.0:
            return FLAT

        move_bps = ((now_price - past) / past) * 10_000.0
        if abs(move_bps) < self._threshold_bps:
            return FLAT

        magnitude = min(abs(move_bps) / self._bps_per_unit, self._size_cap)
        side = Side.UP if move_bps > 0 else Side.DOWN
        confidence = min(abs(move_bps) / (5.0 * self._threshold_bps), 1.0)
        return Signal(side=side, size=magnitude, confidence=confidence)

    def _polymarket_momentum(self, tick: Tick) -> Signal | None:
        window = tick.up_mid_recent
        if len(window) < 2:
            return FLAT
        offset = min(len(window) - 1, self._lookback_s)
        past = window[-1 - offset]
        now_price = window[-1]
        if past <= 0.0 or now_price <= 0.0:
            return FLAT

        move = now_price - past
        if abs(move) < self._pm_threshold:
            return FLAT

        magnitude = min(abs(move) / self._pm_per_unit, self._size_cap)
        side = Side.UP if move > 0 else Side.DOWN
        confidence = min(abs(move) / (5.0 * self._pm_threshold), 1.0)
        return Signal(side=side, size=magnitude, confidence=confidence)
