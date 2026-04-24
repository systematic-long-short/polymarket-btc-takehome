"""Example candidate submission produced during wet testing.

The model is intentionally self-contained and scanner-safe. It trades a slow
Polymarket-mid trend with a small confirmation from the event's early anchor:
when UP mid has been persistently drifting higher, hold UP; when it has been
persistently drifting lower, hold DOWN. Position size is capped by both signal
confidence and a share cap so the model avoids constant resizing.
"""

from __future__ import annotations

from polybench import FLAT, MarketInfo, Model, Side, Signal, Tick


class ModelSubmission(Model):
    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._capital = float(self.config.get("starting_capital", 1000.0))
        self._warmup_s = int(self.config.get("warmup_s", 80))
        self._fast_s = int(self.config.get("fast_s", 12))
        self._slow_s = int(self.config.get("slow_s", 60))
        self._entry_threshold = float(self.config.get("entry_threshold", 0.012))
        self._exit_threshold = float(self.config.get("exit_threshold", -0.004))
        self._max_size = float(self.config.get("max_size", 0.80))
        self._base_size = float(self.config.get("base_size", 0.45))
        self._size_range = float(self.config.get("size_range", 0.35))
        self._max_shares = float(self.config.get("max_shares", 1200.0))
        self._max_spread = float(self.config.get("max_spread", 0.08))
        self._min_entry_ttr = float(self.config.get("min_entry_time_to_resolve", 20.0))
        self._force_flat_ttr = float(self.config.get("force_flat_time_to_resolve", 0.0))

        self._side = Side.FLAT
        self._shares = 0.0

    def on_start(self, market_info: MarketInfo) -> None:
        self._side = Side.FLAT
        self._shares = 0.0

    @staticmethod
    def _avg(values) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    def _trend_score(self, up_mid_recent) -> float:
        if len(up_mid_recent) < self._warmup_s:
            return 0.0

        fast_n = min(self._fast_s, len(up_mid_recent))
        slow_n = min(self._slow_s, len(up_mid_recent) - 1)
        fast_avg = self._avg(up_mid_recent[-fast_n:])

        prev_end = -fast_n
        prev_start = max(0, len(up_mid_recent) - slow_n - fast_n)
        prev_window = up_mid_recent[prev_start:prev_end]
        prev_avg = self._avg(prev_window) if prev_window else up_mid_recent[0]

        slow_move = up_mid_recent[-1] - up_mid_recent[-1 - slow_n]
        anchor_n = min(30, len(up_mid_recent))
        anchor_move = up_mid_recent[-1] - self._avg(up_mid_recent[:anchor_n])

        return 0.50 * (fast_avg - prev_avg) + 0.35 * slow_move + 0.15 * anchor_move

    def _flat(self) -> Signal:
        self._side = Side.FLAT
        self._shares = 0.0
        return FLAT

    def _hold_existing(self, tick: Tick) -> Signal:
        ask = tick.up_ask if self._side == Side.UP else tick.down_ask
        if ask <= 0.0 or self._shares <= 0.0:
            return self._flat()
        size = self._clamp((self._shares * ask) / self._capital, 0.0, 1.0)
        return Signal(side=self._side, size=size, confidence=0.75)

    def _enter(self, side: Side, ask: float, bid: float, confidence: float) -> Signal:
        if ask <= 0.0 or bid <= 0.0:
            return FLAT
        spread = ask - bid
        if spread <= 0.0 or spread > self._max_spread:
            return FLAT

        target_size = self._clamp(
            self._base_size + self._size_range * confidence,
            0.0,
            self._max_size,
        )
        target_notional = min(target_size * self._capital, self._max_shares * ask)
        if target_notional <= 0.0:
            return FLAT

        self._side = side
        self._shares = target_notional / ask
        return Signal(side=side, size=target_notional / self._capital, confidence=confidence)

    def on_tick(self, tick: Tick) -> Signal | None:
        if tick.time_to_resolve <= self._force_flat_ttr:
            return self._flat()

        score = self._trend_score(tick.up_mid_recent)
        if self._side != Side.FLAT:
            direction = 1.0 if self._side == Side.UP else -1.0
            if direction * score < self._exit_threshold:
                return self._flat()
            return self._hold_existing(tick)

        if tick.time_to_resolve < self._min_entry_ttr:
            return FLAT

        confidence = self._clamp(abs(score) / (4.0 * self._entry_threshold), 0.0, 1.0)
        if score > self._entry_threshold:
            return self._enter(Side.UP, tick.up_ask, tick.up_bid, confidence)
        if score < -self._entry_threshold:
            return self._enter(Side.DOWN, tick.down_ask, tick.down_bid, confidence)
        return FLAT
