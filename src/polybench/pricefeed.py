"""Optional BTC price feed — disabled by default, or Binance.

Subscribes to WebSocket channels that give us:
  - last trade price  (Binance: ``btcusdt@trade``)
  - top-of-book bid/ask (Binance: ``btcusdt@bookTicker``)

The live harness's official trading data comes from Polymarket. BTC spot is an
optional auxiliary feature; in the default ``polymarket`` mode this feed is
disabled and snapshots return zero BTC fields.

If Binance fails to connect or goes silent for >30s, the feed reconnects
with exponential backoff.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import websockets

log = logging.getLogger("polybench.pricefeed")


BINANCE_GLOBAL_URL = "wss://stream.binance.com:9443/ws"

POLYMARKET_ONLY_SOURCE = "polymarket"
VALID_SOURCES = (POLYMARKET_ONLY_SOURCE, "binance")
SILENCE_FAILOVER_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class PriceSnapshot:
    ts: float
    last: float
    bid: float
    ask: float
    source: str
    window: tuple[float, ...] = field(default_factory=tuple)


class PriceFeed:
    """Optional asynchronous BTC/USD price feed with auto-reconnect."""

    def __init__(
        self,
        source: str = "binance",
        window_size: int = 300,
        connect_timeout: float = 10.0,
    ) -> None:
        if source not in VALID_SOURCES:
            raise ValueError(
                f"source must be one of {VALID_SOURCES!r}, got {source!r}"
            )
        self._preferred = source
        self._active_source = source
        self._window: deque[float] = deque(maxlen=max(1, window_size))
        self._last = 0.0
        self._bid = 0.0
        self._ask = 0.0
        self._last_msg_ts = 0.0
        self._connect_timeout = connect_timeout
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()

    @property
    def active_source(self) -> str:
        return self._active_source

    async def start(self) -> None:
        if self._preferred == POLYMARKET_ONLY_SOURCE:
            self._active_source = POLYMARKET_ONLY_SOURCE
            self._ready.set()
            return
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._ready.clear()
        self._task = asyncio.create_task(self._run(), name="polybench-pricefeed")

    async def stop(self) -> None:
        if self._preferred == POLYMARKET_ONLY_SOURCE:
            self._stop.set()
            self._task = None
            return
        self._stop.set()
        if self._task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def wait_ready(self, timeout: float = 15.0) -> bool:
        """Block until we have at least one price update, or the timeout lapses."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def snapshot(self) -> PriceSnapshot:
        return PriceSnapshot(
            ts=self._last_msg_ts,
            last=self._last,
            bid=self._bid,
            ask=self._ask,
            source=self._active_source,
            window=tuple(self._window),
        )

    # ----- internal -----

    def _sources_to_try(self) -> Iterable[str]:
        if self._preferred == POLYMARKET_ONLY_SOURCE:
            return
        yield self._preferred

    async def _run(self) -> None:
        """Outer loop: reconnect with backoff on persistent failure."""
        attempt_counts: dict[str, int] = {s: 0 for s in VALID_SOURCES}
        source_cycle = list(self._sources_to_try())
        cycle_idx = 0

        while not self._stop.is_set():
            source = source_cycle[cycle_idx % len(source_cycle)]
            self._active_source = source
            attempt = attempt_counts[source]
            try:
                log.info("pricefeed: connecting to %s (attempt %d)", source, attempt + 1)
                await self._stream_from(source)
                # Normal return means server closed; reset attempt counter.
                attempt_counts[source] = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                attempt_counts[source] += 1
                log.warning(
                    "pricefeed: %s error (%s): %s", source, type(exc).__name__, exc
                )

            if self._stop.is_set():
                break

            # After 3 consecutive failures on one source, rotate.
            if attempt_counts[source] >= 3:
                attempt_counts[source] = 0
                cycle_idx += 1
                log.warning("pricefeed: rotating away from %s", source)
                continue

            delay = min(0.5 * (2 ** attempt_counts[source]), 8.0)
            delay *= 1 + 0.3 * (2 * random.random() - 1)
            await asyncio.sleep(max(0.0, delay))

    async def _stream_from(self, source: str) -> None:
        if source == "binance":
            await self._stream_binance(BINANCE_GLOBAL_URL)
        else:
            raise ValueError(f"unknown source {source!r}")

    async def _stream_binance(self, base_url: str) -> None:
        url = f"{base_url}/btcusdt@trade/btcusdt@bookTicker"
        async with websockets.connect(
            url,
            open_timeout=self._connect_timeout,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
            max_size=2**20,
        ) as ws:
            silence_task = asyncio.create_task(self._silence_watchdog(ws))
            try:
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self._handle_binance_msg(msg)
            finally:
                silence_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await silence_task

    async def _silence_watchdog(self, ws) -> None:
        """Close the socket if no messages arrive for >SILENCE_FAILOVER_SECONDS."""
        while not self._stop.is_set():
            await asyncio.sleep(5.0)
            if self._last_msg_ts == 0.0:
                continue
            silence = time.time() - self._last_msg_ts
            if silence > SILENCE_FAILOVER_SECONDS:
                log.warning(
                    "pricefeed: silence %.1fs on %s, forcing reconnect",
                    silence,
                    self._active_source,
                )
                with contextlib.suppress(Exception):
                    await ws.close(code=1011, reason=f"silence {silence:.1f}s")
                return

    def _handle_binance_msg(self, msg: dict) -> None:
        # btcusdt@trade events
        event = msg.get("e")
        if event == "trade" and "p" in msg:
            try:
                last = float(msg["p"])
            except (TypeError, ValueError):
                return
            self._record_last(last)
            return
        # btcusdt@bookTicker events (no "e" field)
        if "b" in msg and "a" in msg and "s" in msg:
            try:
                bid = float(msg["b"])
                ask = float(msg["a"])
            except (TypeError, ValueError):
                return
            self._bid = bid
            self._ask = ask
            self._last_msg_ts = time.time()
            self._ready.set()

    def _record_last(self, last: float) -> None:
        self._last = last
        self._window.append(last)
        self._last_msg_ts = time.time()
        self._ready.set()
