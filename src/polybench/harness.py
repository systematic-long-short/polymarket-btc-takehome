"""Harness — the live event loop that paper-trades a ``Model`` against
Polymarket 5-minute BTC up/down events.

Architecture (single asyncio event loop):

    pricefeed task    — optional BTC feed; disabled in default Polymarket-only mode
    clob_poller task  — 1 Hz REST poll of UP+DOWN order books, cached
    event_watcher     — called from main loop: find next event when idle, detect resolution
    tick_loop (main)  — 1 Hz, assembles Tick, calls on_tick (500 ms budget),
                        applies signal through simulator, records row

Model's ``on_tick`` is synchronous and runs on a ThreadPoolExecutor with a
500 ms ``asyncio.wait_for`` timeout. Overruns are logged as timeouts and the
signal is dropped for that tick (no stale-signal carry-forward).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from polybench.market import Book, EventDescriptor, PolymarketClient
from polybench.metrics import summarize
from polybench.model import FLAT, MarketInfo, Model, RunResult, Side, Signal, Tick
from polybench.pnl import BookTop, PaperSimulator
from polybench.pricefeed import PriceFeed
from polybench.recorder import Recorder, TickRow

log = logging.getLogger("polybench.harness")


DEFAULT_TICK_INTERVAL_S = 1.0
DEFAULT_MODEL_BUDGET_S = 0.5
DEFAULT_CLOB_POLL_INTERVAL_S = 1.0
RESOLUTION_POLL_TIMEOUT_S = 45.0   # short live poll; postmortem pass catches lagging resolutions
RESOLUTION_POLL_INTERVAL_S = 2.0
EVENT_DISCOVERY_INTERVAL_S = 3.0   # probe Gamma every 3s while idle
RESOLVED_EPSILON = 0.02            # |outcome_price - 1| < RESOLVED_EPSILON → resolved


@dataclass
class HarnessConfig:
    duration_s: float = 3600.0
    tick_interval_s: float = DEFAULT_TICK_INTERVAL_S
    model_budget_s: float = DEFAULT_MODEL_BUDGET_S
    clob_poll_interval_s: float = DEFAULT_CLOB_POLL_INTERVAL_S
    starting_capital: float = 1000.0
    slippage_bps: float = 50.0
    fee_rate: float = 0.072   # Polymarket-style per-trade fee coefficient
    price_source: str = "polymarket"
    price_window_size: int = 300
    output_dir: Path = Path("runs/latest")
    series_slug: str = "btc-up-or-down-5m"


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
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="polybench-model"
        )

    # ---- public ----

    async def run(self) -> RunResult:
        started_ts = time.time()
        await self._pricefeed.start()
        await self._pricefeed.wait_ready(timeout=15.0)
        self._clob_task = asyncio.create_task(self._clob_poll_loop(), name="polybench-clob")
        try:
            await self._tick_loop(started_ts)
        finally:
            self._stop.set()
            if self._clob_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    self._clob_task.cancel()
                    await self._clob_task
            if self._current_event is not None:
                # Harness terminated mid-event — close with unknown resolution.
                self._finish_current_event(time.time(), resolved=False, outcome=None)
            # Post-mortem: any UNKNOWN events still in the simulator ledger get
            # one last Gamma refresh — Polymarket sometimes lags the resolution
            # flip by a few minutes, and this catches them without blocking the
            # event rollover during the run.
            await self._postmortem_resolve_unknowns()
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
                # Grace: if an event is active and its endDate is within a
                # minute past, let resolution poll complete rather than closing
                # with UNKNOWN. Caps the extra wait at RESOLUTION_POLL_TIMEOUT_S.
                if self._current_event is not None and now >= self._current_event.end_date_ts:
                    log.info("harness: duration elapsed, resolving final event before stop")
                    await self._resolve_and_rollover(now)
                log.info("harness: duration elapsed, stopping")
                return

            if self._current_event is None:
                discovered = await self._discover_event(now)
                if discovered is None:
                    await asyncio.sleep(min(EVENT_DISCOVERY_INTERVAL_S, max(0.1, deadline - now)))
                    next_tick = time.time()
                    continue
                self._current_event = discovered
                await self._prime_books_for(discovered)
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
        """Keep order books fresh.

        Every interval: (a) refresh the Gamma event to pick up updated best_bid/
        best_ask/outcome_prices, (b) try CLOB /book for each token, (c) fall back
        to Gamma-synthesized top-of-book when CLOB returns None/empty. Guarantees
        the tick loop always sees a non-empty book during a live event.
        """
        while not self._stop.is_set():
            event = self._current_event
            if event is None:
                await asyncio.sleep(0.5)
                continue
            # Refresh Gamma view of the event (best_bid/best_ask/outcome_prices).
            try:
                refreshed = await self._client.refresh_event(event.slug)
                if refreshed is not None:
                    self._current_event = refreshed
                    event = refreshed
            except Exception as exc:  # noqa: BLE001
                log.debug("gamma refresh failed for %s: %s", event.slug, exc)
            # Attempt CLOB depth; on miss, use Gamma summary.
            up_book: Book | None = None
            down_book: Book | None = None
            try:
                up_book, down_book = await asyncio.gather(
                    self._client.get_book(event.up_token_id),
                    self._client.get_book(event.down_token_id),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("clob poll errored (will use Gamma summary): %s", exc)
            if up_book is None:
                up_book = event.synth_up_book()
            if down_book is None:
                down_book = event.synth_down_book()
            self._cached_up_book = up_book
            self._cached_down_book = down_book
            await asyncio.sleep(self._cfg.clob_poll_interval_s)

    async def _discover_event(self, now: float) -> EventDescriptor | None:
        try:
            event = await self._client.find_active_btc_event(now_ts=now)
        except Exception as exc:  # noqa: BLE001
            log.warning("gamma discovery failed: %s", exc)
            return None
        if event is None:
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

    async def _prime_books_for(self, event: EventDescriptor) -> None:
        """Seed the book cache immediately so the first tick has data.

        Falls back to Gamma-synthesized top-of-book if CLOB 404s.
        """
        up_book: Book | None = None
        down_book: Book | None = None
        try:
            up_book, down_book = await asyncio.gather(
                self._client.get_book(event.up_token_id),
                self._client.get_book(event.down_token_id),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("failed to prime CLOB books for %s (using Gamma): %s", event.slug, exc)
        if up_book is None:
            up_book = event.synth_up_book()
        if down_book is None:
            down_book = event.synth_down_book()
        self._cached_up_book = up_book
        self._cached_down_book = down_book

    def _on_event_start(self, event: EventDescriptor, now: float) -> None:
        self._up_mid_window.clear()
        self._btc_1hz_window.clear()
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
        outcome = await self._poll_resolution(event)
        resolved = outcome is not None
        if not resolved:
            # Gamma's bestBid/bestAsk go stale after the market closes, but
            # outcomePrices stay accurate (they're the UMA-reported market
            # mid, snapping to 0/1 post-resolution). If we have even a close
            # approximation from the last refresh, prefer those over the
            # stale top-of-book for final flattening.
            try:
                refreshed = await self._client.refresh_event(event.slug)
            except Exception:  # noqa: BLE001
                refreshed = None
            if refreshed is not None and sum(refreshed.outcome_prices) > 0.0:
                outcome = refreshed.outcome_prices
        self._finish_current_event(now, resolved=outcome is not None, outcome=outcome)

    async def _postmortem_resolve_unknowns(
        self,
        *,
        max_wait_s: float = 600.0,
        poll_interval_s: float = 10.0,
    ) -> None:
        """After the run, re-query Gamma for events still marked UNKNOWN.

        Polymarket sometimes lags its resolution flip by 60–180 s past
        ``endDate``. The live ``_poll_resolution`` budget is short (so event
        rollover stays tight), so this post-mortem is where lagging events
        get upgraded. Retries every ``poll_interval_s`` until all events
        resolve or ``max_wait_s`` elapses. Updates BOTH the model and the
        baseline event ledgers — they share the same slug stream so a
        single Gamma refresh resolves both tracks.
        """
        import dataclasses as _dc

        model_events = self._simulator._completed_events
        baseline_events = self._baseline_simulator._completed_events
        if not model_events and not baseline_events:
            return

        def _label_from(refreshed) -> str | None:
            up, down = refreshed.outcome_prices
            if up > (1.0 - RESOLVED_EPSILON) and down < RESOLVED_EPSILON:
                return "UP"
            if down > (1.0 - RESOLVED_EPSILON) and up < RESOLVED_EPSILON:
                return "DOWN"
            if refreshed.closed:
                return "UP" if up >= down else "DOWN"
            return None

        deadline = time.time() + max_wait_s
        total_upgraded = 0
        iteration = 0
        while time.time() < deadline:
            iteration += 1
            # Collect all unique UNKNOWN slugs across both ledgers.
            unknown_slugs = {
                e.slug for e in model_events if e.resolved_outcome == "UNKNOWN"
            } | {
                e.slug for e in baseline_events if e.resolved_outcome == "UNKNOWN"
            }
            if not unknown_slugs:
                break
            log.info("postmortem iter %d: %d UNKNOWN slugs remaining", iteration, len(unknown_slugs))
            iter_upgraded = 0
            for slug in unknown_slugs:
                try:
                    refreshed = await self._client.refresh_event(slug)
                except Exception as exc:  # noqa: BLE001
                    log.debug("postmortem refresh failed for %s: %s", slug, exc)
                    continue
                if refreshed is None:
                    continue
                label = _label_from(refreshed)
                if label is None:
                    continue
                for ledger in (model_events, baseline_events):
                    for i, ev in enumerate(ledger):
                        if ev.slug == slug and ev.resolved_outcome == "UNKNOWN":
                            ledger[i] = _dc.replace(ev, resolved_outcome=label)
                            iter_upgraded += 1
                            total_upgraded += 1
            if iter_upgraded == 0:
                remaining = {
                    e.slug for e in model_events if e.resolved_outcome == "UNKNOWN"
                } | {
                    e.slug for e in baseline_events if e.resolved_outcome == "UNKNOWN"
                }
                if not remaining:
                    break
                await asyncio.sleep(poll_interval_s)
        if total_upgraded:
            log.info("postmortem: upgraded %d UNKNOWN event rows", total_upgraded)

    async def _poll_resolution(
        self, event: EventDescriptor
    ) -> tuple[float, float] | None:
        """Poll the same Gamma slug until ``closed=true`` and outcomePrices
        saturate to [1,0] or [0,1]. No cross-source fallback — if Polymarket
        hasn't resolved within the timeout, return ``None`` (UNKNOWN)."""
        deadline = time.time() + RESOLUTION_POLL_TIMEOUT_S
        while time.time() < deadline and not self._stop.is_set():
            try:
                refreshed = await self._client.refresh_event(event.slug)
            except Exception as exc:  # noqa: BLE001
                log.warning("resolution poll (refresh) failed: %s", exc)
                refreshed = None
            if refreshed is not None:
                up, down = refreshed.outcome_prices
                is_resolved_up = up > (1.0 - RESOLVED_EPSILON) and down < RESOLVED_EPSILON
                is_resolved_down = down > (1.0 - RESOLVED_EPSILON) and up < RESOLVED_EPSILON
                # Accept resolution if EITHER Gamma flags closed=true OR the
                # outcomePrices saturate. Polymarket's closed flag lags the
                # price saturation by 10–60s on many events.
                if is_resolved_up or is_resolved_down:
                    return (up, down)
                if refreshed.closed:
                    # closed but not saturated → weird; treat as resolved with
                    # whatever prices are there.
                    return (up, down)
            await asyncio.sleep(RESOLUTION_POLL_INTERVAL_S)
        log.warning(
            "event %s did not resolve within %.0fs — closing with UNKNOWN outcome",
            event.slug,
            RESOLUTION_POLL_TIMEOUT_S,
        )
        return None

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
        self._current_event = None
        self._cached_up_book = None
        self._cached_down_book = None
        self._up_mid_window.clear()
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
            # No books yet — skip tick but count it.
            return

        up_book = self._cached_up_book
        down_book = self._cached_down_book
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

        # Run BOTH models against the same tick in parallel (same deadline).
        (signal, timed_out), (baseline_signal, baseline_timed_out) = await asyncio.gather(
            self._call_model_with_budget(self._model, tick),
            self._call_model_with_budget(self._baseline_model, tick),
        )
        if timed_out:
            self._simulator.record_timeout()
        if baseline_timed_out:
            self._baseline_simulator.record_timeout()

        up_top = BookTop(best_bid=up_book.best_bid, best_ask=up_book.best_ask, mid=up_book.mid)
        down_top = BookTop(
            best_bid=down_book.best_bid, best_ask=down_book.best_ask, mid=down_book.mid
        )
        fills = self._simulator.apply_signal(signal, up_top, down_top)
        equity = self._simulator.mark_to_market(up_top, down_top, now)
        self._equity_curve.append(equity)
        baseline_fills = self._baseline_simulator.apply_signal(baseline_signal, up_top, down_top)
        baseline_equity = self._baseline_simulator.mark_to_market(up_top, down_top, now)
        self._baseline_equity_curve.append(baseline_equity)

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
                "events": [
                    dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(e)
                    for e in result.events
                ],
            },
            "baseline": {
                "final_equity": result.baseline_final_equity,
                "pnl_total": result.baseline_pnl_total,
                "pnl_pct": result.baseline_pnl_pct,
                "metrics": result.baseline_metrics,
                "events": [
                    dataclasses.asdict(e) if dataclasses.is_dataclass(e) else dict(e)
                    for e in result.baseline_events
                ],
            },
        }
        report_path.write_text(json.dumps(payload, indent=2, default=_json_default))
        log.info("wrote %s", report_path)


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
    on_summary: Callable[[str], None] | None = None,
) -> RunResult:
    cfg = HarnessConfig(
        duration_s=duration_s,
        tick_interval_s=tick_interval_s,
        model_budget_s=model_budget_s,
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
