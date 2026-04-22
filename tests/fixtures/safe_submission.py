"""Safe submission fixture — passes the security scanner."""

from __future__ import annotations

from collections import deque

import numpy as np

from polybench import FLAT, Model, Side, Signal, Tick


class ModelSubmission(Model):
    def __init__(self, config=None):
        super().__init__(config=config)
        self._window: deque[float] = deque(maxlen=60)

    def on_tick(self, tick: Tick) -> Signal | None:
        self._window.append(tick.btc_last)
        if len(self._window) < 10:
            return FLAT
        arr = np.array(self._window)
        change = arr[-1] - arr[0]
        if change > 0:
            return Signal(side=Side.UP, size=0.5)
        if change < 0:
            return Signal(side=Side.DOWN, size=0.5)
        return FLAT
