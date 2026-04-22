"""Small utilities: retry/backoff, clock, structured logging helpers."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Awaitable, Callable, TypeVar

log = logging.getLogger("polybench")

T = TypeVar("T")


def now() -> float:
    """Unix timestamp with sub-second precision."""
    return time.time()


async def with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 5,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.3,
    on_retry: Callable[[int, Exception], None] | None = None,
) -> T:
    """Call ``func()`` with exponential backoff on any exception.

    ``base_delay * 2**attempt`` capped at ``max_delay``, with +/- ``jitter`` fraction.
    Re-raises the last exception on exhaustion.
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001 — callers log specifics
            last_exc = exc
            if on_retry:
                on_retry(attempt, exc)
            if attempt == max_attempts - 1:
                break
            delay = min(base_delay * (2**attempt), max_delay)
            delay *= 1 + jitter * (2 * random.random() - 1)
            await asyncio.sleep(max(0.0, delay))
    assert last_exc is not None
    raise last_exc


def clip01(value: float) -> float:
    """Clamp to [0, 1]."""
    if value != value:  # NaN
        return 0.0
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value
