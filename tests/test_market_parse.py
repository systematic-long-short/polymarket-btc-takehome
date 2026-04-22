"""Tests for Polymarket response parsing.

These tests do NOT hit the network. They use fixed sample payloads that
match the real Gamma / CLOB response shapes we verified against live data.
"""

from __future__ import annotations

import json

import httpx
import pytest

from polybench.market import (
    BTC_5M_SERIES_SLUG,
    PolymarketClient,
    _descriptor_from_event,
    _parse_book,
    _parse_stringified_array,
)


# ---- parse helpers ----

def test_parse_stringified_array_handles_string_json() -> None:
    value = '["token_a", "token_b"]'
    assert _parse_stringified_array(value) == ["token_a", "token_b"]


def test_parse_stringified_array_handles_native_list() -> None:
    assert _parse_stringified_array(["a", "b"]) == ["a", "b"]


def test_parse_stringified_array_handles_none_and_bogus() -> None:
    assert _parse_stringified_array(None) == []
    assert _parse_stringified_array("{not json") == []
    assert _parse_stringified_array(42) == []


def test_descriptor_from_event_extracts_token_ids() -> None:
    event = {
        "id": "123",
        "slug": "btc-updown-5m-1776837600",
        "title": "Bitcoin Up or Down — 5 Minutes?",
        "endDate": "2026-04-22T20:45:00Z",
        "closed": False,
        "markets": [
            {
                "id": "9999",
                "clobTokenIds": '["UP_TOKEN_ID", "DOWN_TOKEN_ID"]',
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0.52", "0.48"]',
            }
        ],
    }
    desc = _descriptor_from_event(event)
    assert desc is not None
    assert desc.event_id == "123"
    assert desc.slug == "btc-updown-5m-1776837600"
    assert desc.up_token_id == "UP_TOKEN_ID"
    assert desc.down_token_id == "DOWN_TOKEN_ID"
    assert desc.end_date_ts > 1_700_000_000.0


def test_descriptor_from_event_rejects_missing_markets() -> None:
    assert _descriptor_from_event({"slug": "x", "endDate": "2026-04-22T20:45:00Z"}) is None
    assert _descriptor_from_event({"slug": "x", "markets": []}) is None


# ---- book parsing ----

def test_parse_book_sorts_bids_desc_asks_asc() -> None:
    payload = {
        "bids": [
            {"price": "0.49", "size": "100"},
            {"price": "0.50", "size": "200"},
        ],
        "asks": [
            {"price": "0.52", "size": "150"},
            {"price": "0.51", "size": "250"},
        ],
    }
    book = _parse_book("tok", payload, ts=0.0)
    assert book.best_bid == 0.50
    assert book.best_ask == 0.51
    assert [b.price for b in book.bids] == [0.50, 0.49]
    assert [a.price for a in book.asks] == [0.51, 0.52]
    assert book.mid == pytest.approx(0.505)


def test_parse_book_empty_sides() -> None:
    book = _parse_book("tok", {"bids": [], "asks": []}, ts=0.0)
    assert book.best_bid == 0.0
    assert book.best_ask == 0.0


# ---- async client with mock transport ----

GAMMA_EVENT_RESPONSE = [
    {
        "id": "E-1",
        "slug": "btc-updown-5m-1776837600",
        "title": "Bitcoin Up or Down — 5 Minutes?",
        "endDate": "2099-12-31T23:59:00Z",
        "active": True,
        "closed": False,
        "markets": [
            {
                "id": "M-1",
                "clobTokenIds": '["UP", "DOWN"]',
                "outcomes": '["Up", "Down"]',
                "outcomePrices": '["0.5", "0.5"]',
            }
        ],
    }
]


CLOB_BOOK_RESPONSE = {
    "bids": [{"price": "0.50", "size": "100"}],
    "asks": [{"price": "0.52", "size": "200"}],
    "bestBid": 0.50,
    "bestAsk": 0.52,
    "lastTradePrice": 0.51,
}


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def _make_live_event(slug: str, *, closed: bool = False, end_iso: str = "2099-12-31T23:59:00Z") -> list:
    return [
        {
            "id": f"E-{slug}",
            "slug": slug,
            "title": "Bitcoin Up or Down — 5 Minutes?",
            "endDate": end_iso,
            "active": True,
            "closed": closed,
            "markets": [
                {
                    "id": f"M-{slug}",
                    "clobTokenIds": '["UP", "DOWN"]',
                    "outcomes": '["Up", "Down"]',
                    "outcomePrices": '["0.505", "0.495"]',
                    "bestBid": "0.50",
                    "bestAsk": "0.51",
                    "lastTradePrice": "0.505",
                }
            ],
        }
    ]


@pytest.mark.asyncio
async def test_find_active_event_enumerates_slugs_exact_hit() -> None:
    """Discovery probes the current 5-min window (floor) first and falls
    forward to the next window if that one is empty. When the hit slug is
    the ceil(now/300)*300 boundary, we should find it within 2 probes."""
    import math, time
    hit_boundary = int(math.ceil(time.time() / 300.0) * 300)
    hit_slug = f"btc-updown-5m-{hit_boundary}"
    seen_slugs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.params.get("slug", "")
        seen_slugs.append(slug)
        if slug == hit_slug:
            return httpx.Response(200, json=_make_live_event(slug))
        return httpx.Response(200, json=[])

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        desc = await client.find_active_btc_event()
        assert desc is not None
        assert desc.slug == hit_slug
        assert desc.up_token_id == "UP"
        assert desc.down_token_id == "DOWN"
        assert desc.best_bid == 0.50
        assert desc.best_ask == 0.51
        assert desc.outcome_prices == (0.505, 0.495)
        # Hit within the first 2 probes (floor-then-ceil).
        assert hit_slug in seen_slugs[:2]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_find_active_event_falls_through_to_next_boundary() -> None:
    """First slug empty → probe +300 → hit."""
    import math, time
    expected_boundary = int(math.ceil(time.time() / 300.0) * 300)
    hit_slug = f"btc-updown-5m-{expected_boundary + 300}"
    seen_slugs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.params.get("slug", "")
        seen_slugs.append(slug)
        if slug == hit_slug:
            return httpx.Response(200, json=_make_live_event(slug))
        return httpx.Response(200, json=[])

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        desc = await client.find_active_btc_event()
        assert desc is not None
        assert desc.slug == hit_slug
        assert len(seen_slugs) >= 2           # probed at least two slugs
        assert seen_slugs[0] != hit_slug      # first one missed
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_find_active_event_skips_closed_events() -> None:
    """A closed event is not returned; enumerator keeps probing."""
    import math, time
    expected_boundary = int(math.ceil(time.time() / 300.0) * 300)
    closed_slug = f"btc-updown-5m-{expected_boundary}"
    open_slug = f"btc-updown-5m-{expected_boundary + 300}"

    def handler(request: httpx.Request) -> httpx.Response:
        slug = request.url.params.get("slug", "")
        if slug == closed_slug:
            return httpx.Response(200, json=_make_live_event(slug, closed=True))
        if slug == open_slug:
            return httpx.Response(200, json=_make_live_event(slug))
        return httpx.Response(200, json=[])

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        desc = await client.find_active_btc_event()
        assert desc is not None
        assert desc.slug == open_slug
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_find_active_event_returns_none_when_no_live_slot() -> None:
    """Every probe returns empty → None (treated as transient outage)."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        desc = await client.find_active_btc_event()
        assert desc is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_refresh_event_returns_fresh_descriptor() -> None:
    """refresh_event re-fetches a slug and returns a new descriptor with updated quotes."""
    import math, time
    expected_boundary = int(math.ceil(time.time() / 300.0) * 300)
    slug = f"btc-updown-5m-{expected_boundary}"

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            payload = _make_live_event(slug)
        else:
            # Updated quotes on second fetch
            payload = _make_live_event(slug)
            payload[0]["markets"][0]["bestBid"] = "0.55"
            payload[0]["markets"][0]["bestAsk"] = "0.56"
            payload[0]["markets"][0]["outcomePrices"] = '["0.555", "0.445"]'
        return httpx.Response(200, json=payload)

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        d1 = await client.refresh_event(slug)
        d2 = await client.refresh_event(slug)
        assert d1 is not None and d2 is not None
        assert d1.best_bid == 0.50
        assert d2.best_bid == 0.55
        assert d2.outcome_prices == (0.555, 0.445)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_book_returns_none_on_404() -> None:
    """CLOB 404 must return None so the harness can fall back to Gamma summary."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "not found"})

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        book = await client.get_book("MISSING_TOKEN")
        assert book is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_book_returns_none_on_empty_payload() -> None:
    """An empty book response is treated as a miss."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        book = await client.get_book("EMPTY_TOKEN")
        assert book is None
    finally:
        await client.aclose()


def test_descriptor_synth_up_down_books_are_complementary() -> None:
    """UP bid/ask + DOWN bid/ask should sum to ~1 via 1 − p arbitrage."""
    from polybench.market import EventDescriptor

    desc = EventDescriptor(
        event_id="E", slug="btc-updown-5m-x", question="q",
        end_date_ts=9e10, up_token_id="UP", down_token_id="DOWN",
        up_outcome_label="Up", down_outcome_label="Down",
        best_bid=0.48, best_ask=0.52, last_trade=0.50,
        outcome_prices=(0.50, 0.50),
    )
    up = desc.synth_up_book()
    down = desc.synth_down_book()
    assert up.best_bid == 0.48 and up.best_ask == 0.52
    assert down.best_bid == pytest.approx(1 - 0.52, abs=1e-9)
    assert down.best_ask == pytest.approx(1 - 0.48, abs=1e-9)


@pytest.mark.asyncio
async def test_get_book_returns_parsed_book() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=CLOB_BOOK_RESPONSE)

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        book = await client.get_book("UP")
        assert book.best_bid == 0.50
        assert book.best_ask == 0.52
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_get_outcome_prices_returns_tuple() -> None:
    payload = [
        {
            "slug": "btc-updown-5m-1776837600",
            "markets": [{"outcomePrices": '["1", "0"]'}],
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = PolymarketClient()
    await client._http.aclose()
    client._http = httpx.AsyncClient(transport=_mock_transport(handler))
    try:
        prices = await client.get_outcome_prices("btc-updown-5m-1776837600")
        assert prices == (1.0, 0.0)
    finally:
        await client.aclose()
