"""Harness — the live event loop that paper-trades a ``Model`` against
Polymarket 5-minute BTC up/down events.

Architecture (single asyncio event loop):

    pricefeed task    — optional BTC feed; disabled in default Polymarket-only mode
    clob_ws task      — public Polymarket CLOB market WebSocket, cached
    event_watcher     — called from main loop: find next event when idle, detect resolution
    tick_loop (main)  — 1 Hz, assembles Tick, applies the prior tick's signal
                        through simulator, calls on_tick (500 ms budget), records row

Model's ``on_tick`` is synchronous and runs on a ThreadPoolExecutor with a
500 ms ``asyncio.wait_for`` timeout. Overruns are logged as timeouts and the
signal is dropped for the next tick (no stale-signal carry-forward).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import websockets

from polybench.market import Book, EventDescriptor, Level, PolymarketClient
from polybench.metrics import summarize
from polybench.model import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick
from polybench.pnl import BookTop, PaperSimulator
from polybench.pricefeed import PriceFeed
from polybench.recorder import Recorder, TickRow
from polybench.reporting import (
    build_reproducibility_metadata,
    event_dicts,
    scoring_status,
)

log = logging.getLogger("polybench.harness")


DEFAULT_TICK_INTERVAL_S = 1.0
DEFAULT_MODEL_BUDGET_S = 0.5
DEFAULT_CLOB_POLL_INTERVAL_S = 1.0
DEFAULT_CLOB_WS_STALE_AFTER_S = 30.0
CLOB_MARKET_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
RESOLUTION_POLL_TIMEOUT_S = 45.0   # retained for CLI compatibility; rollover no longer blocks on it
RESOLUTION_POLL_INTERVAL_S = 2.0
EVENT_DISCOVERY_INTERVAL_S = 3.0   # probe Gamma every 3s while idle
RESOLVED_EPSILON = 0.02            # |outcome_price - 1| < RESOLVED_EPSILON → resolved


@dataclass
class HarnessConfig:
    duration_s: float = 3600.0
    tick_interval_s: float = DEFAULT_TICK_INTERVAL_S
    model_budget_s: float = DEFAULT_MODEL_BUDGET_S
    clob_poll_interval_s: float = DEFAULT_CLOB_POLL_INTERVAL_S
    clob_ws_stale_after_s: float = DEFAULT_CLOB_WS_STALE_AFTER_S
    clob_ws_url: str = CLOB_MARKET_WS_URL
    resolution_poll_timeout_s: float = RESOLUTION_POLL_TIMEOUT_S
    postmortem_resolution_s: float = 0.0
    postmortem_poll_interval_s: float = 10.0
    starting_capital: float = 1000.0
    slippage_bps: float = 50.0
    fee_rate: float = 0.072   # Polymarket-style per-share fee coefficient
    price_source: str = "polymarket"
    price_window_size: int = 300
    output_dir: Path = Path("runs/latest")
    series_slug: str = "btc-up-or-down-5m"
    candidate_path: Path | None = None
    command: tuple[str, ...] = ()
    official_scoring: bool = False
    allow_unresolved_final: bool = False


class Harness:
    def __init__(
        self,
        model: Model,
        config: HarnessConfig,
        client: PolymarketClient | None = None,
        pricefeed: PriceFeed | None = None,
        baseline_model: Model | None = None,
    ) -> None:
        self._model = model
        # Always pair the candidate with a MomentumBaseline on the same tape so
        # candidates immediately see how they performed against the bar. When
        # the caller passes the MomentumBaseline as `model`, both tracks end up
        # identical — still useful as a sanity readout for the reference model.
        if baseline_model is None:
            from polybench.baselines import MomentumBaseline
            baseline_model = MomentumBaseline()
        self._baseline_model = baseline_model
        self._cfg = config
        self._client = client or PolymarketClient()
        self._own_client = client is None
        self._pricefeed = pricefeed or PriceFeed(
            source=config.price_source, window_size=config.price_window_size
        )
        self._own_pricefeed = pricefeed is None
        self._simulator = PaperSimulator(
            starting_capital=config.starting_capital,
            slippage_bps=config.slippage_bps,
            fee_rate=config.fee_rate,
        )
        self._baseline_simulator = PaperSimulator(
            starting_capital=config.starting_capital,
            slippage_bps=config.slippage_bps,
            fee_rate=config.fee_rate,
        )
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._recorder = Recorder(self._output_dir / "ticks.parquet")
        self._scratch_dir = self._output_dir / "scratch"
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

        self._current_event: EventDescriptor | None = None
        self._cached_up_book: Book | None = None
        self._cached_down_book: Book | None = None
        self._up_mid_window: list[float] = []
        self._btc_1hz_window: list[float] = []   # one sample per tick — what models see
        self._equity_curve: list[float] = []
        self._baseline_equity_curve: list[float] = []
        self._stop = asyncio.Event()
        self._clob_task: asyncio.Task | None = None
        self._resolution_task: asyncio.Task | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="polybench-model"
        )
        self._pending_signal: Signal | None = None
        self._baseline_pending_signal: Signal | None = None
        self._pending_resolution_events: dict[str, EventDescriptor] = {}
        self._pending_resolution_since: dict[str, float] = {}
        self._resolution_lags_s: list[float] = []
        self._gamma_discovery_attempts = 0
        self._gamma_discovery_failures = 0
        self._gamma_discovery_empty = 0
        self._gamma_resolution_attempts = 0
        self._gamma_resolution_failures = 0
        self._clob_ws_messages = 0
        self._clob_ws_reconnects = 0
        self._clob_ws_stale_count = 0
        self._clob_last_message_ts = 0.0
        self._missing_book_rows = 0
        self._one_sided_book_rows = 0
        self._two_sided_book_rows = 0
        self._active_tick_rows = 0

    # ---- public ----

    async def run(self) -> RunResult:
        started_ts = time.time()
        await self._pricefeed.start()
        await self._pricefeed.wait_ready(timeout=15.0)
        self._clob_task = asyncio.create_task(self._clob_poll_loop(), name="polybench-clob-ws")
        self._resolution_task = asyncio.create_task(
            self._pending_resolution_loop(), name="polybench-resolution"
        )
        try:
            await self._tick_loop(started_ts)
        finally:
            if self._current_event is not None:
                self._finish_current_event(time.time(), resolved=False, outcome=None)
            await self._postmortem_resolve_unknowns(
                max_wait_s=self._cfg.postmortem_resolution_s,
                poll_interval_s=self._cfg.postmortem_poll_interval_s,
            )
            self._stop.set()
            if self._clob_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    self._clob_task.cancel()
                    await self._clob_task
            if self._resolution_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    self._resolution_task.cancel()
                    await self._resolution_task
            if self._own_pricefeed:
                await self._pricefeed.stop()
            if self._own_client:
                await self._client.aclose()
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._recorder.flush()

        run_result = self._build_run_result(started_ts, time.time())
        try:
            self._model.on_finish(run_result)
        except Exception:  # noqa: BLE001
            log.exception("model.on_finish raised")
        self._write_report(run_result)
        return run_result

    # ---- internal loops ----

    async def _tick_loop(self, started_ts: float) -> None:
        deadline = started_ts + self._cfg.duration_s
        interval = self._cfg.tick_interval_s
        next_tick = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now >= deadline:
                if self._current_event is not None and now >= self._current_event.end_date_ts:
                    log.info("harness: duration elapsed, closing final ended event")
                    await self._resolve_and_rollover(now)
                log.info("harness: duration elapsed, stopping")
                return

            if self._current_event is None:
                discovered = await self._discover_event(now)
                if discovered is None:
                    await asyncio.sleep(min(EVENT_DISCOVERY_INTERVAL_S, max(0.1, deadline - now)))
                    next_tick = time.time()
                    continue
                if not await self._prime_books_for(discovered):
                    await asyncio.sleep(min(1.0, max(0.1, deadline - now)))
                    next_tick = time.time()
                    continue
                self._current_event = discovered
                self._on_event_start(discovered, now)
                next_tick = time.time()

            # If the event has ended, settle and roll.
            if now >= self._current_event.end_date_ts:
                await self._resolve_and_rollover(now)
                next_tick = time.time()
                continue

            # We have an active event + (hopefully) cached books.
            await self._dispatch_tick(now)

            # Pacing.
            next_tick += interval
            sleep_for = next_tick - time.time()
            if sleep_for < -interval:
                # We fell way behind — skip ahead to avoid a flood.
                next_tick = time.time() + interval
                sleep_for = interval
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)

    async def _clob_poll_loop(self) -> None:
        """Keep order books fresh from the public Polymarket CLOB WebSocket."""
        backoff_s = 0.5
        while not self._stop.is_set():
            event = self._current_event
            if event is None:
                await asyncio.sleep(0.5)
                continue
            try:
                await self._stream_clob_books(event)
                backoff_s = 0.5
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._cached_up_book = None
                self._cached_down_book = None
                self._clob_ws_reconnects += 1
                if "no CLOB WebSocket message" in str(exc):
                    self._clob_ws_stale_count += 1
                log.warning(
                    "CLOB WebSocket for %s stopped (%s); reconnecting in %.1fs",
                    event.slug,
                    exc,
                    backoff_s,
                )
                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2.0, 8.0)

    async def _stream_clob_books(self, event: EventDescriptor) -> None:
        subscribe = {
            "assets_ids": [event.up_token_id, event.down_token_id],
            "type": "market",
            "custom_feature_enabled": True,
        }
        async with websockets.connect(
            self._cfg.clob_ws_url,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=5,
        ) as ws:
            await ws.send(json.dumps(subscribe))
            log.info("CLOB WebSocket subscribed for %s", event.slug)
            while not self._stop.is_set() and self._current_event is not None:
                current = self._current_event
                if (
                    current.event_id != event.event_id
                    or current.up_token_id != event.up_token_id
                    or current.down_token_id != event.down_token_id
                ):
                    return
                try:
                    raw = await asyncio.wait_for(
                        ws.recv(), timeout=self._cfg.clob_ws_stale_after_s
                    )
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"no CLOB WebSocket message for {self._cfg.clob_ws_stale_after_s:.1f}s"
                    ) from exc
                self._apply_clob_ws_payload(event, raw)

    def _apply_clob_ws_payload(self, event: EventDescriptor, raw: str | bytes) -> None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        text = raw.strip()
        if not text or text.upper() in {"PING", "PONG"}:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            log.debug("ignored non-JSON CLOB WebSocket payload: %r", text[:80])
            return

        messages = payload if isinstance(payload, list) else [payload]
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            for token_id, book in _books_from_clob_ws_message(msg):
                self._clob_ws_messages += 1
                self._clob_last_message_ts = max(self._clob_last_message_ts, book.ts)
                if token_id == event.up_token_id:
                    self._cached_up_book = book
                elif token_id == event.down_token_id:
                    self._cached_down_book = book

    async def _discover_event(self, now: float) -> EventDescriptor | None:
        self._gamma_discovery_attempts += 1
        try:
            event = await self._client.find_active_btc_event(now_ts=now)
        except Exception as exc:  # noqa: BLE001
            self._gamma_discovery_failures += 1
            log.warning("gamma discovery failed: %s", exc)
            return None
        if event is None:
            self._gamma_discovery_empty += 1
            log.info("no active BTC 5m event yet (probed next ~30min of boundaries) — waiting %.0fs",
                     EVENT_DISCOVERY_INTERVAL_S)
            return None
        log.info(
            "locked onto event %s (ends in %.1fs, up_mid=%.4f)",
            event.slug,
            event.end_date_ts - now,
            event.outcome_prices[0],
        )
        return event

    async def _prime_books_for(self, event: EventDescriptor) -> bool:
        """Seed the book cache immediately so the first tick has data.

        Returns False if Polymarket CLOB does not provide both books yet.
        """
        up_book: Book | None = None
        down_book: Book | None = None
        try:
            up_book, down_book = await asyncio.gather(
                self._client.get_book(event.up_token_id),
                self._client.get_book(event.down_token_id),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to prime CLOB books for %s: %s", event.slug, exc)
        if up_book is None or down_book is None:
            self._cached_up_book = None
            self._cached_down_book = None
            log.warning("CLOB books not ready for %s; waiting before event start", event.slug)
            return False
        self._cached_up_book = up_book
        self._cached_down_book = down_book
        return True

    def _on_event_start(self, event: EventDescriptor, now: float) -> None:
        self._up_mid_window.clear()
        self._btc_1hz_window.clear()
        self._pending_signal = None
        self._baseline_pending_signal = None
        up_mid = self._cached_up_book.mid if self._cached_up_book else 0.5
        down_mid = self._cached_down_book.mid if self._cached_down_book else 0.5
        self._simulator.start_event(
            event_id=event.event_id, slug=event.slug, ts=now, up_mid=up_mid, down_mid=down_mid
        )
        self._baseline_simulator.start_event(
            event_id=event.event_id, slug=event.slug, ts=now, up_mid=up_mid, down_mid=down_mid
        )
        market_info = event.to_market_info(self._scratch_dir)
        try:
            self._model.on_start(market_info)
        except Exception:  # noqa: BLE001
            log.exception("model.on_start raised")
        try:
            self._baseline_model.on_start(market_info)
        except Exception:  # noqa: BLE001
            log.exception("baseline_model.on_start raised")

    async def _resolve_and_rollover(self, now: float) -> None:
        event = self._current_event
        assert event is not None
        outcome = await self._read_resolved_outcome(event)
        resolved = outcome is not None
        self._finish_current_event(now, resolved=outcome is not None, outcome=outcome)

    async def _pending_resolution_loop(self) -> None:
        while not self._stop.is_set():
            await self._resolve_pending_once()
            await asyncio.sleep(RESOLUTION_POLL_INTERVAL_S)

    async def _resolve_pending_once(self) -> int:
        resolved_count = 0
        for slug, event in list(self._pending_resolution_events.items()):
            outcome = await self._read_resolved_outcome(event)
            if outcome is None:
                continue
            if self._apply_late_resolution(slug, outcome, time.time()):
                resolved_count += 1
        return resolved_count

    async def _postmortem_resolve_unknowns(
        self,
        *,
        max_wait_s: float = 600.0,
        poll_interval_s: float = 10.0,
    ) -> None:
        """Optional final wait for events still unresolved at run shutdown."""
        if max_wait_s <= 0.0:
            return

        deadline = time.time() + max_wait_s
        total_upgraded = 0
        iteration = 0
        while self._pending_resolution_events and time.time() < deadline:
            iteration += 1
            log.info(
                "postmortem iter %d: %d unresolved slugs remaining",
                iteration,
                len(self._pending_resolution_events),
            )
            iter_upgraded = await self._resolve_pending_once()
            total_upgraded += iter_upgraded
            if iter_upgraded == 0:
                remaining_wait = deadline - time.time()
                if remaining_wait <= 0.0:
                    break
                await asyncio.sleep(min(poll_interval_s, remaining_wait))
        if total_upgraded:
            log.info("postmortem: upgraded %d UNKNOWN event rows", total_upgraded)

    async def _read_resolved_outcome(
        self, event: EventDescriptor
    ) -> tuple[float, float] | None:
        """Return final [1,0]/[0,1] prices once Gamma outcomePrices saturate."""
        try:
            self._gamma_resolution_attempts += 1
            refreshed = await self._client.refresh_event(event.slug)
        except Exception as exc:  # noqa: BLE001
            self._gamma_resolution_failures += 1
            log.warning("resolution refresh failed for %s: %s", event.slug, exc)
            return None
        if refreshed is None:
            return None
        up, down = refreshed.outcome_prices
        if up > (1.0 - RESOLVED_EPSILON) and down < RESOLVED_EPSILON:
            return (1.0, 0.0)
        if down > (1.0 - RESOLVED_EPSILON) and up < RESOLVED_EPSILON:
            return (0.0, 1.0)
        return None

    def _apply_late_resolution(
        self,
        slug: str,
        outcome: tuple[float, float],
        now: float,
    ) -> bool:
        event = self._pending_resolution_events.pop(slug, None)
        if event is None:
            return False
        pending_since = self._pending_resolution_since.pop(slug, None)
        if pending_since is not None:
            self._resolution_lags_s.append(max(0.0, now - pending_since))
        model_result = self._simulator.settle_pending_event(slug, now, outcome[0], outcome[1])
        baseline_result = self._baseline_simulator.settle_pending_event(
            slug, now, outcome[0], outcome[1]
        )
        resolved_outcome = (
            model_result.resolved_outcome
            if model_result is not None
            else _outcome_label(outcome)
        )
        self._recorder.update_resolution(
            slug=slug,
            resolution_up=outcome[0],
            resolution_down=outcome[1],
            resolved_outcome=resolved_outcome,
        )
        if model_result is not None:
            log.info(
                "event %s resolved late: model=%.4f baseline=%.4f outcome=%s",
                event.slug,
                model_result.pnl_total,
                baseline_result.pnl_total if baseline_result is not None else 0.0,
                resolved_outcome,
            )
        return True

    def _finish_current_event(
        self,
        now: float,
        *,
        resolved: bool,
        outcome: tuple[float, float] | None,
    ) -> None:
        assert self._current_event is not None
        event = self._current_event
        up_price = outcome[0] if (resolved and outcome is not None) else None
        down_price = outcome[1] if (resolved and outcome is not None) else None
        event_result = self._simulator.finish_event(now, up_price, down_price)
        baseline_result = self._baseline_simulator.finish_event(now, up_price, down_price)
        # Write a settlement row — both model and baseline tracks land at the
        # same event boundary, so one row carries both snapshots.
        price_snap = self._pricefeed.snapshot()
        row = TickRow(
            ts=now,
            event_id=event.event_id,
            slug=event.slug,
            time_to_resolve=0.0,
            btc_last=price_snap.last,
            btc_bid=price_snap.bid,
            btc_ask=price_snap.ask,
            btc_source=price_snap.source,
            up_bid=self._cached_up_book.best_bid if self._cached_up_book else 0.0,
            up_ask=self._cached_up_book.best_ask if self._cached_up_book else 0.0,
            up_mid=self._cached_up_book.mid if self._cached_up_book else 0.0,
            down_bid=self._cached_down_book.best_bid if self._cached_down_book else 0.0,
            down_ask=self._cached_down_book.best_ask if self._cached_down_book else 0.0,
            down_mid=self._cached_down_book.mid if self._cached_down_book else 0.0,
            signal_side="NONE",
            signal_size=0.0,
            signal_confidence=0.0,
            position_up=self._simulator.position.up_shares,
            position_down=self._simulator.position.down_shares,
            cash=self._simulator.position.cash,
            equity=self._simulator.position.cash,
            fills_this_tick=0,
            timeout=False,
            baseline_signal_side="NONE",
            baseline_signal_size=0.0,
            baseline_signal_confidence=0.0,
            baseline_position_up=self._baseline_simulator.position.up_shares,
            baseline_position_down=self._baseline_simulator.position.down_shares,
            baseline_cash=self._baseline_simulator.position.cash,
            baseline_equity=self._baseline_simulator.position.cash,
            baseline_fills_this_tick=0,
            baseline_timeout=False,
            resolution_up=up_price if up_price is not None else float("nan"),
            resolution_down=down_price if down_price is not None else float("nan"),
            resolved_outcome=event_result.resolved_outcome,
        )
        self._recorder.record(row)
        if not resolved:
            self._pending_resolution_events[event.slug] = event
            self._pending_resolution_since[event.slug] = now
        self._current_event = None
        self._cached_up_book = None
        self._cached_down_book = None
        self._pending_signal = None
        self._baseline_pending_signal = None
        self._up_mid_window.clear()
        self._btc_1hz_window.clear()
        log.info(
            "event %s closed: model=%.4f (intra=%.4f, reso=%.4f) baseline=%.4f outcome=%s",
            event.slug,
            event_result.pnl_total,
            event_result.pnl_intra_event,
            event_result.pnl_resolution,
            baseline_result.pnl_total,
            event_result.resolved_outcome,
        )

    async def _dispatch_tick(self, now: float) -> None:
        assert self._current_event is not None
        if self._cached_up_book is None or self._cached_down_book is None:
            # No real CLOB books yet, so no model call or paper fill is recorded.
            self._missing_book_rows += 1
            return

        up_book = self._cached_up_book
        down_book = self._cached_down_book
        self._active_tick_rows += 1
        if _is_two_sided_book(up_book) and _is_two_sided_book(down_book):
            self._two_sided_book_rows += 1
        else:
            self._one_sided_book_rows += 1
        price_snap = self._pricefeed.snapshot()

        # Update 1 Hz rolling windows (one sample per tick so candidate models
        # can reason about time-aligned history without sub-second noise).
        self._up_mid_window.append(up_book.mid)
        if len(self._up_mid_window) > self._cfg.price_window_size:
            self._up_mid_window.pop(0)
        if price_snap.last > 0.0:
            self._btc_1hz_window.append(price_snap.last)
            if len(self._btc_1hz_window) > self._cfg.price_window_size:
                self._btc_1hz_window.pop(0)

        tick = Tick(
            ts=now,
            time_to_resolve=self._current_event.end_date_ts - now,
            btc_last=price_snap.last,
            btc_bid=price_snap.bid,
            btc_ask=price_snap.ask,
            btc_source=price_snap.source,
            up_bid=up_book.best_bid,
            up_ask=up_book.best_ask,
            up_mid=up_book.mid,
            down_bid=down_book.best_bid,
            down_ask=down_book.best_ask,
            down_mid=down_book.mid,
            btc_recent=tuple(self._btc_1hz_window),
            up_mid_recent=tuple(self._up_mid_window),
            event_id=self._current_event.event_id,
        )

        up_top = BookTop(
            best_bid=up_book.best_bid,
            best_ask=up_book.best_ask,
            mid=up_book.mid,
        )
        down_top = BookTop(
            best_bid=down_book.best_bid, best_ask=down_book.best_ask, mid=down_book.mid
        )
        fills = self._simulator.apply_signal(self._pending_signal, up_top, down_top)
        equity = self._simulator.mark_to_market(up_top, down_top, now)
        self._equity_curve.append(equity)
        baseline_fills = self._baseline_simulator.apply_signal(
            self._baseline_pending_signal, up_top, down_top
        )
        baseline_equity = self._baseline_simulator.mark_to_market(
            up_top, down_top, now
        )
        self._baseline_equity_curve.append(baseline_equity)

        # Run BOTH models against the same tick in parallel (same deadline).
        # Returned signals are queued for execution on the next recorded tick.
        (signal, timed_out), (baseline_signal, baseline_timed_out) = await asyncio.gather(
            self._call_model_with_budget(self._model, tick),
            self._call_model_with_budget(self._baseline_model, tick),
        )
        if timed_out:
            self._simulator.record_timeout()
        if baseline_timed_out:
            self._baseline_simulator.record_timeout()
        self._pending_signal = signal
        self._baseline_pending_signal = baseline_signal

        effective = signal or FLAT
        baseline_effective = baseline_signal or FLAT
        row = TickRow(
            ts=now,
            event_id=self._current_event.event_id,
            slug=self._current_event.slug,
            time_to_resolve=tick.time_to_resolve,
            btc_last=tick.btc_last,
            btc_bid=tick.btc_bid,
            btc_ask=tick.btc_ask,
            btc_source=tick.btc_source,
            up_bid=tick.up_bid,
            up_ask=tick.up_ask,
            up_mid=tick.up_mid,
            down_bid=tick.down_bid,
            down_ask=tick.down_ask,
            down_mid=tick.down_mid,
            signal_side=effective.side.value if effective.side else "NONE",
            signal_size=effective.size,
            signal_confidence=effective.confidence,
            position_up=self._simulator.position.up_shares,
            position_down=self._simulator.position.down_shares,
            cash=self._simulator.position.cash,
            equity=equity,
            fills_this_tick=len(fills),
            timeout=timed_out,
            baseline_signal_side=baseline_effective.side.value if baseline_effective.side else "NONE",
            baseline_signal_size=baseline_effective.size,
            baseline_signal_confidence=baseline_effective.confidence,
            baseline_position_up=self._baseline_simulator.position.up_shares,
            baseline_position_down=self._baseline_simulator.position.down_shares,
            baseline_cash=self._baseline_simulator.position.cash,
            baseline_equity=baseline_equity,
            baseline_fills_this_tick=len(baseline_fills),
            baseline_timeout=baseline_timed_out,
        )
        self._recorder.record(row)

    async def _call_model_with_budget(
        self, model: Model, tick: Tick
    ) -> tuple[Signal | None, bool]:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            self._executor, self._safe_on_tick, model, tick
        )
        try:
            signal = await asyncio.wait_for(future, timeout=self._cfg.model_budget_s)
            return (signal, False)
        except asyncio.TimeoutError:
            log.debug("model.on_tick exceeded %.3fs budget", self._cfg.model_budget_s)
            return (None, True)

    def _safe_on_tick(self, model: Model, tick: Tick) -> Signal | None:
        try:
            result = model.on_tick(tick)
        except Exception:  # noqa: BLE001
            log.exception("model.on_tick raised")
            return None
        if result is None:
            return None
        if not isinstance(result, Signal):
            log.warning("model.on_tick returned %s, expected Signal", type(result).__name__)
            return None
        # Normalize: clip size into [0,1], clamp side to enum.
        try:
            side = result.side if isinstance(result.side, Side) else Side(str(result.side))
            size_raw = float(result.size)
            conf_raw = float(result.confidence)
        except (TypeError, ValueError):
            log.warning("model.on_tick returned an invalid Signal: %r", result)
            return None
        size = max(0.0, min(1.0, size_raw)) if math.isfinite(size_raw) else 0.0
        conf = max(0.0, min(1.0, conf_raw)) if math.isfinite(conf_raw) else 0.0
        return Signal(side=side, size=size, confidence=conf)

    # ---- run result ----

    def _build_run_result(self, started_ts: float, ended_ts: float) -> RunResult:
        def _final_equity(sim: PaperSimulator, curve: list[float]) -> float:
            canonical = sim.position.cash
            if abs(canonical) > 1e-9:
                return canonical
            return curve[-1] if curve else sim.starting_capital

        events = self._simulator.completed_events
        baseline_events = self._baseline_simulator.completed_events
        starting = self._simulator.starting_capital
        final_equity = _final_equity(self._simulator, self._equity_curve)
        baseline_final_equity = _final_equity(
            self._baseline_simulator, self._baseline_equity_curve
        )

        metrics = summarize(
            starting_capital=starting,
            final_equity=final_equity,
            equity_curve=[starting, *self._equity_curve],
            events=events,
        )
        baseline_metrics = summarize(
            starting_capital=starting,
            final_equity=baseline_final_equity,
            equity_curve=[starting, *self._baseline_equity_curve],
            events=baseline_events,
        )
        return RunResult(
            started_ts=started_ts,
            ended_ts=ended_ts,
            starting_capital=starting,
            final_equity=final_equity,
            pnl_total=final_equity - starting,
            pnl_pct=(final_equity - starting) / starting if starting else 0.0,
            events=events,
            metrics=metrics,
            baseline_events=baseline_events,
            baseline_metrics=baseline_metrics,
            baseline_final_equity=baseline_final_equity,
            baseline_pnl_total=baseline_final_equity - starting,
            baseline_pnl_pct=(
                (baseline_final_equity - starting) / starting if starting else 0.0
            ),
        )

    def _write_report(self, result: RunResult) -> None:
        report_path = self._output_dir / "report.json"
        payload: dict[str, Any] = {
            "started_ts": result.started_ts,
            "ended_ts": result.ended_ts,
            "starting_capital": result.starting_capital,
            "model": {
                "final_equity": result.final_equity,
                "pnl_total": result.pnl_total,
                "pnl_pct": result.pnl_pct,
                "metrics": result.metrics,
                "events": event_dicts(result.events),
            },
            "baseline": {
                "final_equity": result.baseline_final_equity,
                "pnl_total": result.baseline_pnl_total,
                "pnl_pct": result.baseline_pnl_pct,
                "metrics": result.baseline_metrics,
                "events": event_dicts(result.baseline_events),
            },
        }
        payload["scoring_status"] = scoring_status(
            payload,
            allow_unresolved=self._cfg.allow_unresolved_final,
        )
        payload["metadata"] = build_reproducibility_metadata(
            candidate_path=self._cfg.candidate_path,
            command=self._cfg.command,
            config=self._cfg,
            feed_health=self._feed_health_summary(result),
        )
        report_path.write_text(json.dumps(payload, indent=2, default=_json_default))
        log.info("wrote %s", report_path)

    def _feed_health_summary(self, result: RunResult) -> dict[str, Any]:
        duration_s = max(0.0, result.ended_ts - result.started_ts)
        unknown_events = sum(1 for event in result.events if event.resolved_outcome == "UNKNOWN")
        active_rows = max(0, self._active_tick_rows)
        return {
            "price_source": self._cfg.price_source,
            "duration_s": duration_s,
            "active_tick_rows": active_rows,
            "active_tick_coverage": (
                active_rows / duration_s if duration_s > 0.0 else 0.0
            ),
            "clob_ws_messages": self._clob_ws_messages,
            "clob_last_message_age_s": (
                max(0.0, result.ended_ts - self._clob_last_message_ts)
                if self._clob_last_message_ts > 0.0
                else None
            ),
            "clob_ws_reconnects": self._clob_ws_reconnects,
            "clob_ws_stale_count": self._clob_ws_stale_count,
            "missing_book_rows": self._missing_book_rows,
            "one_sided_book_rows": self._one_sided_book_rows,
            "two_sided_book_rows": self._two_sided_book_rows,
            "gamma_discovery_attempts": self._gamma_discovery_attempts,
            "gamma_discovery_failures": self._gamma_discovery_failures,
            "gamma_discovery_empty": self._gamma_discovery_empty,
            "gamma_resolution_attempts": self._gamma_resolution_attempts,
            "gamma_resolution_failures": self._gamma_resolution_failures,
            "gamma_resolution_lag_max_s": max(self._resolution_lags_s)
            if self._resolution_lags_s
            else 0.0,
            "gamma_resolution_lag_avg_s": (
                sum(self._resolution_lags_s) / len(self._resolution_lags_s)
                if self._resolution_lags_s
                else 0.0
            ),
            "pending_resolution_events": len(self._pending_resolution_events),
            "unknown_event_count": unknown_events,
        }


def _books_from_clob_ws_message(msg: dict[str, Any]) -> list[tuple[str, Book]]:
    event_type = str(msg.get("event_type") or msg.get("type") or "")
    ts = _ws_timestamp_to_seconds(msg.get("timestamp"))
    if event_type == "book":
        token_id = str(msg.get("asset_id") or "")
        book = _book_from_levels(token_id, msg.get("bids"), msg.get("asks"), ts=ts)
        return [(token_id, book)] if book is not None else []
    if event_type == "best_bid_ask":
        token_id = str(msg.get("asset_id") or "")
        book = _book_from_top(
            token_id,
            best_bid=_float_or_zero(msg.get("best_bid")),
            best_ask=_float_or_zero(msg.get("best_ask")),
            ts=ts,
        )
        return [(token_id, book)] if book is not None else []
    if event_type == "price_change":
        books: list[tuple[str, Book]] = []
        for change in msg.get("price_changes") or []:
            if not isinstance(change, dict):
                continue
            token_id = str(change.get("asset_id") or "")
            book = _book_from_top(
                token_id,
                best_bid=_float_or_zero(change.get("best_bid")),
                best_ask=_float_or_zero(change.get("best_ask")),
                ts=ts,
            )
            if book is not None:
                books.append((token_id, book))
        return books
    return []


def _is_two_sided_book(book: Book) -> bool:
    return book.best_bid > 0.0 and book.best_ask > 0.0


def _book_from_levels(
    token_id: str,
    bids_raw: Any,
    asks_raw: Any,
    *,
    ts: float,
) -> Book | None:
    if not token_id:
        return None
    bids = tuple(
        sorted(
            (
                Level(
                    price=_float_or_zero(level.get("price")),
                    size=_float_or_zero(level.get("size")),
                )
                for level in (bids_raw or [])
                if isinstance(level, dict)
            ),
            key=lambda level: -level.price,
        )
    )
    asks = tuple(
        sorted(
            (
                Level(
                    price=_float_or_zero(level.get("price")),
                    size=_float_or_zero(level.get("size")),
                )
                for level in (asks_raw or [])
                if isinstance(level, dict)
            ),
            key=lambda level: level.price,
        )
    )
    best_bid = bids[0].price if bids else 0.0
    best_ask = asks[0].price if asks else 0.0
    if best_bid <= 0.0 and best_ask <= 0.0:
        return None
    last_trade = (
        (best_bid + best_ask) / 2.0
        if best_bid > 0.0 and best_ask > 0.0
        else max(best_bid, best_ask)
    )
    return Book(
        token_id=token_id,
        bids=bids,
        asks=asks,
        best_bid=best_bid,
        best_ask=best_ask,
        last_trade=last_trade,
        ts=ts,
    )


def _book_from_top(
    token_id: str, *, best_bid: float, best_ask: float, ts: float
) -> Book | None:
    if not token_id:
        return None
    if best_bid <= 0.0 and best_ask <= 0.0:
        return None
    bids = (Level(price=best_bid, size=0.0),) if best_bid > 0.0 else ()
    asks = (Level(price=best_ask, size=0.0),) if best_ask > 0.0 else ()
    last_trade = (
        (best_bid + best_ask) / 2.0
        if best_bid > 0.0 and best_ask > 0.0
        else max(best_bid, best_ask)
    )
    return Book(
        token_id=token_id,
        bids=bids,
        asks=asks,
        best_bid=best_bid,
        best_ask=best_ask,
        last_trade=last_trade,
        ts=ts,
    )


def _float_or_zero(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def _ws_timestamp_to_seconds(value: Any) -> float:
    ts = _float_or_zero(value)
    if ts <= 0.0:
        return time.time()
    if ts > 10_000_000_000.0:
        return ts / 1000.0
    return ts


def _outcome_label(outcome: tuple[float, float]) -> str:
    up, down = outcome
    if up >= 0.99 and down <= 0.01:
        return "UP"
    if down >= 0.99 and up <= 0.01:
        return "DOWN"
    return "UNKNOWN"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# ---- printable summary ----

def format_summary(result: RunResult) -> str:
    m = result.metrics
    b = result.baseline_metrics
    bar = "=" * 78
    hdr = f"  {'metric':<24}{'Model':>24}{'Baseline':>24}"
    pnl_pct_m = f"({result.pnl_pct:.2%})"
    pnl_pct_b = f"({result.baseline_pnl_pct:.2%})"

    def fmt_dollar(label: str, model_val: float, base_val: float) -> str:
        return f"  {label:<24}{'$' + format(model_val, ',.2f'):>24}{'$' + format(base_val, ',.2f'):>24}"

    def fmt_num(label: str, model_val: float, base_val: float, width: str = ".4f") -> str:
        return f"  {label:<24}{format(model_val, width):>24}{format(base_val, width):>24}"

    def fmt_pct(label: str, model_val: float, base_val: float) -> str:
        return f"  {label:<24}{format(model_val, '.2%'):>24}{format(base_val, '.2%'):>24}"

    def fmt_int(label: str, model_val: int, base_val: int) -> str:
        return f"  {label:<24}{model_val:>24d}{base_val:>24d}"

    lines = [
        "",
        bar,
        "  polybench run report",
        bar,
        hdr,
        "  " + "-" * 72,
        fmt_dollar("starting capital", result.starting_capital, result.starting_capital),
        fmt_dollar("final equity", result.final_equity, result.baseline_final_equity),
        f"  {'pnl total':<24}{'$' + format(result.pnl_total, ',.2f'):>18}{pnl_pct_m:>6}"
        f"{'$' + format(result.baseline_pnl_total, ',.2f'):>18}{pnl_pct_b:>6}",
        fmt_num("primary score", m.get("primary_score", 0.0), b.get("primary_score", 0.0),
                width=",.4f"),
        "",
        fmt_num("sharpe (ann)", m.get("sharpe", 0.0), b.get("sharpe", 0.0)),
        fmt_num("sortino (ann)", m.get("sortino", 0.0), b.get("sortino", 0.0)),
        fmt_pct("max drawdown", m.get("max_drawdown", 0.0), b.get("max_drawdown", 0.0)),
        fmt_pct("hit rate", m.get("hit_rate", 0.0), b.get("hit_rate", 0.0)),
        fmt_pct("outcome accuracy", m.get("outcome_accuracy", 0.0), b.get("outcome_accuracy", 0.0)),
        fmt_pct("timeout rate", m.get("timeout_rate", 0.0), b.get("timeout_rate", 0.0)),
        "",
        fmt_int("events", int(m.get("n_events", 0)), int(b.get("n_events", 0))),
        fmt_int("trades", int(m.get("n_trades", 0)), int(b.get("n_trades", 0))),
        fmt_int("ticks", int(m.get("n_ticks", 0)), int(b.get("n_ticks", 0))),
        "",
        fmt_dollar("pnl intra-event", m.get("pnl_intra_event", 0.0), b.get("pnl_intra_event", 0.0)),
        fmt_dollar("pnl resolution", m.get("pnl_resolution", 0.0), b.get("pnl_resolution", 0.0)),
        bar,
        "",
    ]
    return "\n".join(lines)


# ---- convenience entrypoint ----

async def run_model(
    model: Model,
    *,
    duration_s: float,
    output_dir: Path | str = "runs/latest",
    starting_capital: float = 1000.0,
    slippage_bps: float = 50.0,
    price_source: str = "polymarket",
    tick_interval_s: float = DEFAULT_TICK_INTERVAL_S,
    model_budget_s: float = DEFAULT_MODEL_BUDGET_S,
    resolution_poll_timeout_s: float = RESOLUTION_POLL_TIMEOUT_S,
    postmortem_resolution_s: float = 0.0,
    on_summary: Callable[[str], None] | None = None,
) -> RunResult:
    cfg = HarnessConfig(
        duration_s=duration_s,
        tick_interval_s=tick_interval_s,
        model_budget_s=model_budget_s,
        resolution_poll_timeout_s=resolution_poll_timeout_s,
        postmortem_resolution_s=postmortem_resolution_s,
        starting_capital=starting_capital,
        slippage_bps=slippage_bps,
        price_source=price_source,
        output_dir=Path(output_dir),
    )
    harness = Harness(model=model, config=cfg)
    result = await harness.run()
    summary = format_summary(result)
    if on_summary:
        on_summary(summary)
    else:
        print(summary)
    return result
