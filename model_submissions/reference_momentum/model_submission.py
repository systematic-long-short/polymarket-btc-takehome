"""Reference momentum model submission.

This mirrors the harness MomentumBaseline in a self-contained candidate file.
With ``--price-source binance`` it trades Binance BTC momentum; if BTC spot is
not available it falls back to Polymarket UP-mid momentum.
"""

from __future__ import annotations

from polybench import FLAT, MarketInfo, Model, Side, Signal, Tick


class ModelSubmission(Model):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._lookback_s = int(self.config.get("lookback_s", 30))
        self._threshold_bps = float(self.config.get("threshold_bps", 5.0))
        self._size_cap = float(self.config.get("size_cap", 1.0))
        self._bps_per_unit = max(1e-9, float(self.config.get("bps_per_unit", 20.0)))
        self._pm_threshold = float(self.config.get("pm_threshold", 0.01))
        self._pm_per_unit = max(1e-9, float(self.config.get("pm_per_unit", 0.08)))

    def on_start(self, market_info: MarketInfo) -> None:
        return None

    def on_tick(self, tick: Tick) -> Signal | None:
        if len(tick.btc_recent) >= 2 and tick.btc_recent[-1] > 0.0:
            return self._btc_momentum(tick)
        return self._polymarket_momentum(tick)

    def _btc_momentum(self, tick: Tick) -> Signal | None:
        window = tick.btc_recent
        if len(window) < 2:
            return FLAT
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
