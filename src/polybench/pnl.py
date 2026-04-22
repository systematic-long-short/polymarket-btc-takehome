"""Paper-trade simulator + PnL attribution.

Pipeline for a single event:
    simulator.start_event(starting_equity_snapshot)
    while event trading:
        book_up, book_down = latest CLOB books
        fill = simulator.apply_signal(signal, book_up, book_down)   # executes diff
        equity = simulator.mark_to_market(book_up, book_down)
    simulator.finish_event(resolved_up_price, resolved_down_price)

The simulator owns position + cash state across events (they carry over),
but records per-event PnL attribution (intra-event vs resolution).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from polybench.model import EventResult, Side, Signal

log = logging.getLogger("polybench.pnl")


@dataclass(frozen=True, slots=True)
class BookTop:
    """Minimal view of a single-token book top, enough to fill paper trades."""

    best_bid: float
    best_ask: float
    mid: float

    @property
    def is_tradable(self) -> bool:
        return self.best_bid > 0.0 and self.best_ask > 0.0

    @property
    def is_bid_tradable(self) -> bool:
        return self.best_bid > 0.0

    @property
    def is_ask_tradable(self) -> bool:
        return self.best_ask > 0.0


@dataclass(frozen=True, slots=True)
class Fill:
    """Record of a single leg executed on a tick."""

    side: Side            # which leg (UP or DOWN)
    shares_delta: float   # positive = buy, negative = sell
    fill_price: float
    notional: float       # signed: positive spent (buy), negative received (sell)


@dataclass
class Position:
    cash: float = 0.0
    up_shares: float = 0.0
    down_shares: float = 0.0

    def equity(self, up_mid: float, down_mid: float) -> float:
        return self.cash + self.up_shares * up_mid + self.down_shares * down_mid


@dataclass
class _EventAcc:
    event_id: str
    slug: str
    start_ts: float
    starting_equity: float
    last_mtm_equity: float
    last_up_mid: float = 0.0
    last_down_mid: float = 0.0
    n_trades: int = 0
    n_ticks: int = 0
    n_timeouts: int = 0


class PaperSimulator:
    """Stateful paper-trading simulator spanning one full harness run.

    Position carries across events — we do NOT flatten between events unless
    the model requests FLAT. Positions in a resolved event are naturally
    liquidated by the resolution settlement step (they pay $1 or $0 and leave
    the book as dust if not explicitly closed).
    """

    def __init__(
        self,
        starting_capital: float = 1000.0,
        slippage_bps: float = 200.0,   # 2% = 200 basis points
    ) -> None:
        self._starting_capital = float(starting_capital)
        self._slippage = float(slippage_bps) / 10_000.0
        self.position = Position(cash=self._starting_capital)
        self._current_event: _EventAcc | None = None
        self._completed_events: list[EventResult] = []

    # ---- event lifecycle ----

    @property
    def starting_capital(self) -> float:
        return self._starting_capital

    @property
    def slippage_bps(self) -> float:
        return self._slippage * 10_000.0

    @property
    def completed_events(self) -> tuple[EventResult, ...]:
        return tuple(self._completed_events)

    def start_event(
        self, event_id: str, slug: str, ts: float, up_mid: float, down_mid: float
    ) -> None:
        starting_equity = self.position.equity(up_mid, down_mid)
        self._current_event = _EventAcc(
            event_id=event_id,
            slug=slug,
            start_ts=ts,
            starting_equity=starting_equity,
            last_mtm_equity=starting_equity,
            last_up_mid=up_mid,
            last_down_mid=down_mid,
        )

    def finish_event(
        self,
        ts: float,
        resolved_up_price: float | None,
        resolved_down_price: float | None,
    ) -> EventResult:
        if self._current_event is None:
            raise RuntimeError("finish_event called with no active event")
        acc = self._current_event

        # Establish final marks: use resolved prices if we have them, else
        # keep the last observed mid (position remains open, would eventually
        # settle but we finalize PnL here).
        resolved_known = (
            resolved_up_price is not None and resolved_down_price is not None
        )
        if resolved_known:
            final_up = float(resolved_up_price)
            final_down = float(resolved_down_price)
        else:
            final_up = acc.last_up_mid
            final_down = acc.last_down_mid

        intra_pnl = acc.last_mtm_equity - acc.starting_equity

        # Snap any held shares to resolution price — this is the payoff
        # from holding through settlement.
        payout = (
            self.position.up_shares * final_up
            + self.position.down_shares * final_down
        )
        still_held_mtm = (
            self.position.up_shares * acc.last_up_mid
            + self.position.down_shares * acc.last_down_mid
        )
        resolution_pnl = payout - still_held_mtm

        # Convert held shares to cash at resolution and flatten if resolved.
        if resolved_known:
            self.position.cash += payout
            self.position.up_shares = 0.0
            self.position.down_shares = 0.0

        # Determine outcome label.
        if resolved_known:
            if final_up >= 0.99 and final_down <= 0.01:
                outcome = "UP"
            elif final_down >= 0.99 and final_up <= 0.01:
                outcome = "DOWN"
            else:
                outcome = "UNKNOWN"
        else:
            outcome = "UNKNOWN"

        total_pnl = intra_pnl + resolution_pnl
        result = EventResult(
            event_id=acc.event_id,
            slug=acc.slug,
            start_ts=acc.start_ts,
            end_ts=ts,
            resolved_outcome=outcome,
            pnl_total=total_pnl,
            pnl_intra_event=intra_pnl,
            pnl_resolution=resolution_pnl,
            n_trades=acc.n_trades,
            n_ticks=acc.n_ticks,
            n_timeouts=acc.n_timeouts,
        )
        self._completed_events.append(result)
        self._current_event = None
        return result

    # ---- per-tick ----

    def record_timeout(self) -> None:
        if self._current_event is not None:
            self._current_event.n_timeouts += 1
            self._current_event.n_ticks += 1

    def apply_signal(
        self,
        signal: Signal | None,
        up_book: BookTop,
        down_book: BookTop,
    ) -> list[Fill]:
        """Reconcile the current position toward ``signal`` using the current
        CLOB top-of-book. Returns any fills that executed."""
        if self._current_event is None:
            raise RuntimeError("apply_signal called with no active event")

        if signal is None:
            signal = Signal(side=Side.FLAT, size=0.0, confidence=0.0)

        target_up, target_down = self._target_shares(signal, up_book, down_book)
        fills = self._reconcile(target_up, target_down, up_book, down_book)
        acc = self._current_event
        acc.n_trades += len(fills)
        return fills

    def mark_to_market(
        self,
        up_book: BookTop,
        down_book: BookTop,
        ts: float | None = None,
    ) -> float:
        """Record MTM for the tick and return current equity."""
        if self._current_event is None:
            raise RuntimeError("mark_to_market called with no active event")
        up_mid = up_book.mid if up_book.mid > 0.0 else self._current_event.last_up_mid
        down_mid = (
            down_book.mid if down_book.mid > 0.0 else self._current_event.last_down_mid
        )
        equity = self.position.equity(up_mid, down_mid)
        acc = self._current_event
        acc.last_up_mid = up_mid
        acc.last_down_mid = down_mid
        acc.last_mtm_equity = equity
        acc.n_ticks += 1
        return equity

    # ---- internal ----

    def _target_shares(
        self, signal: Signal, up_book: BookTop, down_book: BookTop
    ) -> tuple[float, float]:
        size = max(0.0, min(1.0, float(signal.size)))
        notional = size * self._starting_capital
        if signal.side == Side.UP and size > 0.0 and up_book.is_ask_tradable:
            return (notional / up_book.best_ask, 0.0)
        if signal.side == Side.DOWN and size > 0.0 and down_book.is_ask_tradable:
            return (0.0, notional / down_book.best_ask)
        # FLAT or untradable target side → go to zero on both sides.
        return (0.0, 0.0)

    def _reconcile(
        self,
        target_up: float,
        target_down: float,
        up_book: BookTop,
        down_book: BookTop,
    ) -> list[Fill]:
        fills: list[Fill] = []
        fill_up = self._execute_leg(
            Side.UP,
            delta=target_up - self.position.up_shares,
            book=up_book,
        )
        if fill_up is not None:
            fills.append(fill_up)
            self.position.up_shares += fill_up.shares_delta
            self.position.cash -= fill_up.notional
        fill_down = self._execute_leg(
            Side.DOWN,
            delta=target_down - self.position.down_shares,
            book=down_book,
        )
        if fill_down is not None:
            fills.append(fill_down)
            self.position.down_shares += fill_down.shares_delta
            self.position.cash -= fill_down.notional
        return fills

    def _execute_leg(self, side: Side, *, delta: float, book: BookTop) -> Fill | None:
        if abs(delta) < 1e-9:
            return None
        if not math.isfinite(delta):
            return None
        if delta > 0.0:
            if not book.is_ask_tradable:
                return None
            fill_price = book.best_ask * (1.0 + self._slippage)
            notional = delta * fill_price                 # cash out
        else:
            if not book.is_bid_tradable:
                return None
            fill_price = book.best_bid * (1.0 - self._slippage)
            notional = delta * fill_price                 # cash in (negative notional)
        return Fill(side=side, shares_delta=delta, fill_price=fill_price, notional=notional)
