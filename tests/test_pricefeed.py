"""Tests for optional BTC price feed behavior."""

from __future__ import annotations

import pytest

from polybench.pricefeed import PriceFeed


@pytest.mark.asyncio
async def test_polymarket_price_source_disables_external_feed() -> None:
    feed = PriceFeed(source="polymarket")
    await feed.start()
    try:
        assert await feed.wait_ready(timeout=0.01)
        snap = feed.snapshot()
        assert snap.source == "polymarket"
        assert snap.last == 0.0
        assert snap.bid == 0.0
        assert snap.ask == 0.0
        assert snap.window == ()
    finally:
        await feed.stop()
