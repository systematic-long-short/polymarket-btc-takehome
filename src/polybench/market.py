"""Polymarket Gamma + CLOB clients.

Verified endpoints (2026-04-22):
  Gamma: https://gamma-api.polymarket.com
  CLOB:  https://clob.polymarket.com

Discovery strategy for the BTC 5-minute up/down event:
  GET /events?series_slug=btc-up-or-down-5m&closed=false&limit=5
     &order=endDate&ascending=true

Event payload contains ``markets[]`` where each market has ``clobTokenIds``
(stringified JSON array — index 0 is the Up/Yes token, index 1 is the Down/No
token) and ``outcomePrices`` (stringified; e.g. ``"[0.5, 0.5]"`` pre-resolution,
``"[1, 0]"`` or ``"[0, 1]"`` post-resolution).

CLOB book endpoint:
  GET /book?token_id=<CLOB_TOKEN_ID>
  -> {"bids": [{"price": "0.50", "size": "123.4"}, ...],
      "asks": [{"price": "0.52", "size": "...."}, ...],
      "bestBid": 0.50, "bestAsk": 0.52, "lastTradePrice": 0.51}
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from polybench._utils import with_backoff
from polybench.model import MarketInfo

log = logging.getLogger("polybench.market")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
BTC_5M_SERIES_SLUG = "btc-up-or-down-5m"


@dataclass(frozen=True, slots=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True, slots=True)
class Book:
    token_id: str
    bids: tuple[Level, ...]
    asks: tuple[Level, ...]
    best_bid: float
    best_ask: float
    last_trade: float
    ts: float

    @property
    def mid(self) -> float:
        if self.best_bid <= 0.0 or self.best_ask <= 0.0:
            return self.last_trade
        return (self.best_bid + self.best_ask) / 2.0


def _parse_float(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _parse_stringified_array(value: Any) -> list[Any]:
    """Gamma returns ``clobTokenIds`` / ``outcomes`` / ``outcomePrices`` as JSON
    strings most of the time but occasionally as native arrays. Handle both.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _parse_book(token_id: str, payload: dict[str, Any], ts: float) -> Book:
    bids_raw = payload.get("bids") or []
    asks_raw = payload.get("asks") or []
    bids = tuple(
        Level(price=_parse_float(l.get("price")), size=_parse_float(l.get("size")))
        for l in bids_raw
    )
    asks = tuple(
        Level(price=_parse_float(l.get("price")), size=_parse_float(l.get("size")))
        for l in asks_raw
    )
    # CLOB sorts bids descending, asks ascending. Be defensive.
    bids = tuple(sorted(bids, key=lambda l: -l.price))
    asks = tuple(sorted(asks, key=lambda l: l.price))
    best_bid = bids[0].price if bids else _parse_float(payload.get("bestBid"))
    best_ask = asks[0].price if asks else _parse_float(payload.get("bestAsk"))
    last_trade = _parse_float(payload.get("lastTradePrice"))
    return Book(
        token_id=token_id,
        bids=bids,
        asks=asks,
        best_bid=best_bid,
        best_ask=best_ask,
        last_trade=last_trade,
        ts=ts,
    )


@dataclass
class EventDescriptor:
    """Raw event data returned by Gamma, plus its per-tick refreshable quotes.

    ``best_bid``/``best_ask``/``last_trade`` on the descriptor come from
    ``markets[0]`` on the Gamma event payload and reflect the UP token.
    DOWN-token quotes are derived by arbitrage (1 - UP). These fields act as
    a durable top-of-book source when the CLOB ``/book`` endpoint 404s on a
    given token id.
    """

    event_id: str
    slug: str
    question: str
    end_date_ts: float
    up_token_id: str
    down_token_id: str
    up_outcome_label: str   # "Up" / "Yes"
    down_outcome_label: str # "Down" / "No"
    closed: bool = False
    best_bid: float = 0.0                            # UP token best bid (from Gamma)
    best_ask: float = 0.0                            # UP token best ask
    last_trade: float = 0.0
    outcome_prices: tuple[float, float] = (0.0, 0.0) # (up_mid, down_mid) from outcomePrices
    raw: dict[str, Any] = field(default_factory=dict)

    def to_market_info(self, scratch_dir: Path) -> MarketInfo:
        return MarketInfo(
            event_id=self.event_id,
            slug=self.slug,
            question=self.question,
            end_date_ts=self.end_date_ts,
            up_token_id=self.up_token_id,
            down_token_id=self.down_token_id,
            scratch_dir=scratch_dir,
        )

    def synth_up_book(self) -> "Book":
        """Construct a minimal Book from Gamma's UP-side summary quotes."""
        import time as _time
        up_mid = self.outcome_prices[0] if self.outcome_prices[0] > 0 else self.last_trade
        bid = self.best_bid if self.best_bid > 0 else max(0.0, up_mid - 0.01)
        ask = self.best_ask if self.best_ask > 0 else min(1.0, up_mid + 0.01)
        last = self.last_trade if self.last_trade > 0 else (bid + ask) / 2.0 if (bid + ask) > 0 else 0.5
        bids = (Level(price=bid, size=0.0),) if bid > 0 else ()
        asks = (Level(price=ask, size=0.0),) if ask > 0 else ()
        return Book(
            token_id=self.up_token_id,
            bids=bids,
            asks=asks,
            best_bid=bid,
            best_ask=ask,
            last_trade=last,
            ts=_time.time(),
        )

    def synth_down_book(self) -> "Book":
        """Construct a DOWN-token Book by inverting UP via ``1 − p`` arbitrage."""
        import time as _time
        up = self.synth_up_book()
        down_bid = max(0.0, 1.0 - up.best_ask) if up.best_ask > 0 else 0.0
        down_ask = min(1.0, 1.0 - up.best_bid) if up.best_bid > 0 else 0.0
        down_last = self.outcome_prices[1] if self.outcome_prices[1] > 0 else (
            (down_bid + down_ask) / 2.0 if (down_bid + down_ask) > 0 else 0.5
        )
        bids = (Level(price=down_bid, size=0.0),) if down_bid > 0 else ()
        asks = (Level(price=down_ask, size=0.0),) if down_ask > 0 else ()
        return Book(
            token_id=self.down_token_id,
            bids=bids,
            asks=asks,
            best_bid=down_bid,
            best_ask=down_ask,
            last_trade=down_last,
            ts=_time.time(),
        )


def _iso_to_ts(iso: str | None) -> float:
    if not iso:
        return 0.0
    import datetime as _dt

    try:
        # Gamma returns "2026-04-22T20:45:00Z"
        iso_norm = iso.replace("Z", "+00:00")
        return _dt.datetime.fromisoformat(iso_norm).timestamp()
    except ValueError:
        return 0.0


def _descriptor_from_event(event_obj: dict[str, Any]) -> EventDescriptor | None:
    markets = event_obj.get("markets") or []
    if not markets:
        return None
    market = markets[0]
    token_ids = _parse_stringified_array(market.get("clobTokenIds"))
    outcomes = _parse_stringified_array(market.get("outcomes"))
    if len(token_ids) < 2:
        return None
    up_id = str(token_ids[0])
    down_id = str(token_ids[1])
    up_label = str(outcomes[0]) if len(outcomes) >= 1 else "Up"
    down_label = str(outcomes[1]) if len(outcomes) >= 2 else "Down"
    end_ts = _iso_to_ts(event_obj.get("endDate") or market.get("endDate"))
    if end_ts <= 0:
        return None
    outcome_prices_arr = _parse_stringified_array(market.get("outcomePrices"))
    up_mid = _parse_float(outcome_prices_arr[0]) if len(outcome_prices_arr) >= 1 else 0.0
    down_mid = _parse_float(outcome_prices_arr[1]) if len(outcome_prices_arr) >= 2 else 0.0
    closed_flag = bool(event_obj.get("closed") or market.get("closed"))
    return EventDescriptor(
        event_id=str(event_obj.get("id") or market.get("id") or event_obj.get("slug", "")),
        slug=str(event_obj.get("slug") or market.get("slug") or ""),
        question=str(event_obj.get("title") or market.get("question") or ""),
        end_date_ts=end_ts,
        up_token_id=up_id,
        down_token_id=down_id,
        up_outcome_label=up_label,
        down_outcome_label=down_label,
        closed=closed_flag,
        best_bid=_parse_float(market.get("bestBid")),
        best_ask=_parse_float(market.get("bestAsk")),
        last_trade=_parse_float(market.get("lastTradePrice")),
        outcome_prices=(up_mid, down_mid),
        raw=event_obj,
    )


class PolymarketClient:
    """Async read-only Gamma + CLOB client. Reuse a single instance per run."""

    def __init__(
        self,
        gamma_base: str = GAMMA_BASE,
        clob_base: str = CLOB_BASE,
        timeout: float = 5.0,
    ) -> None:
        self._http = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "polybench/0.1 (take-home harness)"},
        )
        self._gamma = gamma_base.rstrip("/")
        self._clob = clob_base.rstrip("/")

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "PolymarketClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        async def _call() -> Any:
            resp = await self._http.get(url, params=params)
            resp.raise_for_status()
            return resp.json()

        return await with_backoff(_call, max_attempts=4, base_delay=0.4, max_delay=4.0)

    async def find_active_btc_event(
        self,
        *,
        series_slug: str = BTC_5M_SERIES_SLUG,   # retained for API compat; unused
        now_ts: float | None = None,
        max_steps: int = 6,
    ) -> EventDescriptor | None:
        """Discover the currently-tradable BTC 5-min event via direct-slug enumeration.

        Polymarket's event slug is ``btc-updown-5m-<end_ts>`` where ``<end_ts>`` is
        the unix-seconds endDate aligned to a 5-minute boundary. We compute the
        next boundary ``ceil(now/300)*300`` and probe consecutive slugs with a
        tight ``GET /events?slug=…`` — the ``series_slug=`` filter is unreliable,
        this is the durable path.

        Returns the first event hit whose ``closed=false`` and whose endDate is
        still in the future. ``None`` if no event is live across the next
        ``max_steps * 5`` minutes (treat as a transient outage).
        """
        import math
        import time as _time

        threshold = now_ts if now_ts is not None else _time.time()
        # Start from the CURRENT 5-min window (floor). The event with that
        # slug is the one actively trading — its endDate is still in the
        # future by up to 300s. If that one is closed/missing (just ended),
        # fall through to the next window up. Using ceil() would miss the
        # in-flight event every time the harness rolls over after a slow
        # resolution poll, silently dropping every other event.
        boundary = int(math.floor(threshold / 300.0) * 300)

        for step in range(max_steps):
            candidate_ts = boundary + step * 300
            slug = f"btc-updown-5m-{candidate_ts}"
            try:
                data = await self._get_json(f"{self._gamma}/events", params={"slug": slug})
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    continue
                log.warning("gamma: %s probing %s", exc, slug)
                continue
            except Exception as exc:  # noqa: BLE001
                log.warning("gamma: probe failed for %s: %s", slug, exc)
                continue
            if not isinstance(data, list) or not data:
                continue
            desc = _descriptor_from_event(data[0])
            if desc is None:
                continue
            if desc.closed:
                continue
            if desc.end_date_ts <= threshold:
                continue
            return desc
        return None

    async def get_book(self, token_id: str) -> Book | None:
        """Fetch a CLOB order book. Returns ``None`` on 404 / empty responses so
        the harness can fall back to Gamma's top-of-book summary."""
        import time as _time

        try:
            data = await self._get_json(f"{self._clob}/book", params={"token_id": token_id})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise
        ts = _time.time()
        if not isinstance(data, dict):
            log.warning("clob: unexpected /book response for %s: %s", token_id, type(data))
            return None
        # An empty book (both sides empty) is effectively useless — signal miss.
        if not (data.get("bids") or data.get("asks") or data.get("bestBid") or data.get("bestAsk")):
            return None
        return _parse_book(token_id, data, ts)

    async def get_event_by_slug(self, slug: str) -> EventDescriptor | None:
        """Fetch a specific event by slug (used for rehydration / testing)."""
        data = await self._get_json(f"{self._gamma}/events", params={"slug": slug})
        if not isinstance(data, list) or not data:
            return None
        return _descriptor_from_event(data[0])

    async def refresh_event(self, slug: str) -> EventDescriptor | None:
        """Re-fetch an event by slug. Used every tick to refresh the in-memory
        EventDescriptor's ``best_bid``/``best_ask``/``outcome_prices``/``closed``."""
        return await self.get_event_by_slug(slug)

    async def get_outcome_prices(self, slug: str) -> tuple[float, float] | None:
        """Read the current ``outcomePrices`` for a market (by event slug).

        Returns ``(up_price, down_price)`` as a tuple. These are the live
        mid-prices while the event is trading and snap to ``(1.0, 0.0)``
        or ``(0.0, 1.0)`` once the UMA oracle has settled.

        Returns ``None`` if the event cannot be found or the prices are
        unparseable. The harness is responsible for determining whether
        the values represent a resolution (one near 1.0, the other near 0.0)
        vs. a mid-trade snapshot.
        """
        if not slug:
            return None
        data = await self._get_json(f"{self._gamma}/events", params={"slug": slug})
        if not isinstance(data, list) or not data:
            return None
        markets = data[0].get("markets") or []
        if not markets:
            return None
        prices = _parse_stringified_array(markets[0].get("outcomePrices"))
        if len(prices) < 2:
            return None
        try:
            return (float(prices[0]), float(prices[1]))
        except (TypeError, ValueError):
            return None
