"""Dual-feed example submission.

Uses the live Polymarket token book and the optional Binance BTC feed supplied
by the harness. Run it with ``--price-source binance`` so ``tick.btc_recent``
and ``tick.btc_source`` contain live Binance data.
"""

from __future__ import annotations

from polybench import FLAT, MarketInfo, Model, Side, Signal, Tick


class ModelSubmission(Model):
    """Momentum model that requires both Polymarket and BTC spot inputs."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config=config)
        self._capital = float(self.config.get("starting_capital", 1000.0))
        self._warmup_s = int(self.config.get("warmup_s", 45))
        self._fast_s = int(self.config.get("fast_s", 12))
        self._slow_s = int(self.config.get("slow_s", 45))
        self._entry_threshold = float(self.config.get("entry_threshold", 0.004))
        self._exit_threshold = float(self.config.get("exit_threshold", 0.001))
        self._max_size = float(self.config.get("max_size", 0.08))
        self._base_size = float(self.config.get("base_size", 0.04))
        self._size_range = float(self.config.get("size_range", 0.04))
        self._max_spread = float(self.config.get("max_spread", 0.08))
        self._min_entry_ttr = float(self.config.get("min_entry_time_to_resolve", 18.0))
        self._force_flat_ttr = float(self.config.get("force_flat_time_to_resolve", 70.0))
        self._pm_weight = float(self.config.get("polymarket_weight", 0.80))
        self._btc_weight = float(self.config.get("btc_weight", 0.05))
        self._max_shares = float(self.config.get("max_shares", 600.0))

        self._event_anchor_btc = 0.0
        self._side = Side.FLAT
        self._shares = 0.0

    def on_start(self, market_info: MarketInfo) -> None:
        self._event_anchor_btc = 0.0
        self._side = Side.FLAT
        self._shares = 0.0

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        if value < lo:
            return lo
        if value > hi:
            return hi
        return value

    @staticmethod
    def _avg(values) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _has_polymarket(self, tick: Tick) -> bool:
        return (
            tick.up_bid > 0.0
            and tick.up_ask > 0.0
            and tick.down_bid > 0.0
            and tick.down_ask > 0.0
            and tick.up_mid > 0.0
            and tick.down_mid > 0.0
        )

    def _has_btc(self, tick: Tick) -> bool:
        return tick.btc_last > 0.0 and len(tick.btc_recent) >= 2

    def _polymarket_score(self, recent) -> float:
        if len(recent) < self._warmup_s:
            return 0.0
        fast_n = min(self._fast_s, len(recent))
        slow_n = min(self._slow_s, len(recent) - 1)
        fast = self._avg(recent[-fast_n:])
        prior_start = max(0, len(recent) - slow_n - fast_n)
        prior = recent[prior_start:-fast_n]
        prior_avg = self._avg(prior) if prior else recent[0]
        slow_move = recent[-1] - recent[-1 - slow_n]
        return 0.60 * (fast - prior_avg) + 0.40 * slow_move

    def _btc_score(self, tick: Tick) -> float:
        if not self._has_btc(tick):
            return 0.0
        if self._event_anchor_btc <= 0.0:
            self._event_anchor_btc = tick.btc_last

        recent = tick.btc_recent
        offset = min(len(recent) - 1, self._slow_s)
        past = recent[-1 - offset]
        if past <= 0.0 or self._event_anchor_btc <= 0.0:
            return 0.0

        short_bps = ((tick.btc_last - past) / past) * 10_000.0
        anchor_bps = ((tick.btc_last - self._event_anchor_btc) / self._event_anchor_btc) * 10_000.0
        bounded = self._clamp(0.65 * short_bps + 0.35 * anchor_bps, -35.0, 35.0)
        return bounded / 1800.0

    def _flat(self) -> Signal:
        self._side = Side.FLAT
        self._shares = 0.0
        return FLAT

    def _hold_existing(self, tick: Tick) -> Signal:
        ask = tick.up_ask if self._side == Side.UP else tick.down_ask
        if ask <= 0.0 or self._shares <= 0.0:
            return self._flat()
        size = self._clamp((self._shares * ask) / self._capital, 0.0, 1.0)
        return Signal(side=self._side, size=size, confidence=0.70)

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
        if not self._has_polymarket(tick) or not self._has_btc(tick):
            return self._flat()

        if tick.time_to_resolve <= self._force_flat_ttr:
            return self._flat()

        score = (
            self._pm_weight * self._polymarket_score(tick.up_mid_recent)
            + self._btc_weight * self._btc_score(tick)
        )
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
