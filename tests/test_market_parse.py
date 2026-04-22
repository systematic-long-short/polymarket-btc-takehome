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


@pytest.mark.asyncio
async def test_find_active_event_uses_series_slug() -> None:
    seen_params: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_params.update(dict(request.url.params))
        return httpx.Response(200, json=GAMMA_EVENT_RESPONSE)

    client = PolymarketClient()
    # Swap in a mock transport on the underlying AsyncClient.
    await client._http.aclose()
    client._http = httpx.AsyncClient(
        transport=_mock_transport(handler),
        base_url="",
    )
    try:
        desc = await client.find_active_btc_event()
        assert desc is not None
        assert desc.up_token_id == "UP"
        assert desc.down_token_id == "DOWN"
        assert seen_params["series_slug"] == BTC_5M_SERIES_SLUG
        assert seen_params["closed"] == "false"
    finally:
        await client.aclose()


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
