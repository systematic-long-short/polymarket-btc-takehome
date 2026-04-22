"""Reference baseline models.

Candidates should measurably beat ``MomentumBaseline`` on risk-adjusted return
(Sharpe × sign(PnL)) to pass. ``RandomModel`` is the floor; ``AlwaysUpModel``
calibrates how much of any reported PnL is ambient market drift.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from polybench.model import FLAT, MarketInfo, Model, Side, Signal, Tick

log = logging.getLogger("polybench.baselines")


class RandomModel(Model):
    """Pick UP, DOWN, or FLAT uniformly at random every tick.

    Expected PnL ≈ 0 across a long run; high variance. Useful as a sanity
    floor — any real signal should dominate this.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        seed = self.config.get("seed", 0)
        self._rng = random.Random(seed)
        self._size = float(self.config.get("size", 0.5))

    def on_tick(self, tick: Tick) -> Signal | None:
        choice = self._rng.randint(0, 2)
        if choice == 0:
            return Signal(side=Side.UP, size=self._size, confidence=0.33)
        if choice == 1:
            return Signal(side=Side.DOWN, size=self._size, confidence=0.33)
        return FLAT


class AlwaysUpModel(Model):
    """Buy UP at the first tick of every event, hold to resolution.

    This calibrates the "ambient up-drift + hold-to-settle" PnL a trader
    inherits without any signal. A model that only narrowly beats this is
    probably not capturing useful structure.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._size = float(self.config.get("size", 1.0))

    def on_tick(self, tick: Tick) -> Signal | None:
        return Signal(side=Side.UP, size=self._size, confidence=1.0)


class MomentumBaseline(Model):
    """30-second BTC momentum strategy — the bar candidates must beat.

    Idea: if BTC moved up by >= ``threshold_bps`` over the last ``lookback_s``
    seconds, position UP proportional to move magnitude. Symmetric for down.
    Below threshold → FLAT (no exposure).

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
        self._bps_per_unit = float(self.config.get("bps_per_unit", 20.0))

    def on_start(self, market_info: MarketInfo) -> None:
        log.debug("MomentumBaseline: event start %s", market_info.slug)

    def on_tick(self, tick: Tick) -> Signal | None:
        window = tick.btc_recent
        if len(window) < 2:
            return FLAT
        # Use the sample from ``_lookback_s`` seconds ago — approximated by
        # index offset assuming 1 Hz samples (which matches the price feed's
        # default window rate on Binance/Coinbase tickers).
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
