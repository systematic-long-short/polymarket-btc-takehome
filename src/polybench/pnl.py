"""Paper-trade simulator + PnL attribution.

Pipeline for a single event:
    simulator.start_event(starting_equity_snapshot)
    while event trading:
        book_up, book_down = latest CLOB books
        fill = simulator.apply_signal(signal, book_up, book_down)   # executes target diff
        equity = simulator.mark_to_market(book_up, book_down)
    simulator.finish_event(resolved_up_price, resolved_down_price)

The simulator owns position + cash state across events (they carry over),
but records per-event PnL attribution (intra-event vs resolution).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, replace

from polybench.model import EventResult, Side, Signal

log = logging.getLogger("polybench.pnl")


@dataclass(frozen=True, slots=True)
class BookTop:
    """Minimal view of a single-token book top, enough to fill paper trades.

    Buy-side tradability requires a sane ask; sell-side/liquidation
    tradability requires a sane bid. When both sides exist, the spread must
    not be crossed or implausibly wide. This prevents phantom PnL from bad
    synthetic books while still allowing exit marks on one-sided bid books.
    """

    best_bid: float
    best_ask: float
    mid: float

    @property
    def is_well_formed(self) -> bool:
        return (
            self.best_bid > 0.0
            and self.best_ask > 0.0
            and self.best_ask >= self.best_bid
            and (self.best_ask - self.best_bid) < 0.5    # sanity: max 50-cent spread on a [0,1] market
        )

    @property
    def is_tradable(self) -> bool:
        return self.is_bid_tradable and self.is_ask_tradable

    @property
    def _spread_is_sane(self) -> bool:
        if self.best_bid <= 0.0 or self.best_ask <= 0.0:
            return True
        return self.best_ask >= self.best_bid and (self.best_ask - self.best_bid) < 0.5

    @property
    def is_bid_tradable(self) -> bool:
        return self.best_bid > 0.0 and self._spread_is_sane

    @property
    def is_ask_tradable(self) -> bool:
        return self.best_ask > 0.0 and self._spread_is_sane


@dataclass(frozen=True, slots=True)
class Fill:
    """Record of a single leg executed on a tick."""

    side: Side            # which leg (UP or DOWN)
    shares_delta: float   # positive = buy, negative = sell
    fill_price: float
    notional: float       # signed: positive spent (buy), negative received (sell)
    fee: float = 0.0      # always positive, deducted from cash on top of notional


@dataclass
class Position:
    cash: float = 0.0
    up_shares: float = 0.0
    down_shares: float = 0.0

    def equity(self, up_exit: float, down_exit: float) -> float:
        return self.cash + self.up_shares * up_exit + self.down_shares * down_exit


@dataclass
class _EventAcc:
    event_id: str
    slug: str
    start_ts: float
    starting_equity: float
    last_mtm_equity: float
    last_up_exit: float = 0.0
    last_down_exit: float = 0.0
    n_trades: int = 0
    n_ticks: int = 0
    n_timeouts: int = 0


@dataclass
class _PendingSettlement:
    event_index: int
    slug: str
    provisional_up: float
    provisional_down: float
    up_shares: float
    down_shares: float
    still_held_mtm: float


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
        slippage_bps: float = 50.0,    # 0.5% = 50 basis points per order
        fee_rate: float = 0.072,        # Polymarket-style per-share fee coefficient
    ) -> None:
        self._starting_capital = float(starting_capital)
        self._slippage = float(slippage_bps) / 10_000.0
        self._fee_rate = float(fee_rate)
        self.position = Position(cash=self._starting_capital)
        self._current_event: _EventAcc | None = None
        self._completed_events: list[EventResult] = []
        self._pending_settlements: dict[str, _PendingSettlement] = {}

    # ---- event lifecycle ----

    @property
    def starting_capital(self) -> float:
        return self._starting_capital

    @property
    def slippage_bps(self) -> float:
        return self._slippage * 10_000.0

    @property
    def fee_rate(self) -> float:
        return self._fee_rate

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
            last_up_exit=up_mid,
            last_down_exit=down_mid,
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
        # keep the last observed liquidation marks (position remains open,
        # would eventually settle but we finalize PnL here).
        resolved_known = (
            resolved_up_price is not None and resolved_down_price is not None
        )
        if resolved_known:
            final_up = float(resolved_up_price)
            final_down = float(resolved_down_price)
        else:
            final_up = acc.last_up_exit
            final_down = acc.last_down_exit

        intra_pnl = acc.last_mtm_equity - acc.starting_equity

        # Snap any held shares to resolution price — this is the payoff
        # from holding through settlement.
        payout = (
            self.position.up_shares * final_up
            + self.position.down_shares * final_down
        )
        still_held_mtm = (
            self.position.up_shares * acc.last_up_exit
            + self.position.down_shares * acc.last_down_exit
        )
        resolution_pnl = payout - still_held_mtm

        held_up_shares = self.position.up_shares
        held_down_shares = self.position.down_shares

        # Always flatten at event end: tokens from event A are NOT fungible
        # with event B's tokens. If resolved, snap at $1/$0; otherwise cash
        # out at the last observed liquidation marks (best approximation
        # absent a resolution signal). Carrying positions across events produces
        # nonsensical MTM because the share count would be re-priced at
        # the next event's different tokens.
        self.position.cash += payout
        self.position.up_shares = 0.0
        self.position.down_shares = 0.0

        # Determine outcome label.
        outcome = _outcome_label(final_up, final_down) if resolved_known else "UNKNOWN"

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
            pending_resolution_up_shares=held_up_shares if not resolved_known else 0.0,
            pending_resolution_down_shares=held_down_shares if not resolved_known else 0.0,
            pending_resolution_up_mark=acc.last_up_exit if not resolved_known else 0.0,
            pending_resolution_down_mark=acc.last_down_exit if not resolved_known else 0.0,
        )
        self._completed_events.append(result)
        if not resolved_known:
            self._pending_settlements[acc.slug] = _PendingSettlement(
                event_index=len(self._completed_events) - 1,
                slug=acc.slug,
                provisional_up=final_up,
                provisional_down=final_down,
                up_shares=held_up_shares,
                down_shares=held_down_shares,
                still_held_mtm=still_held_mtm,
            )
        self._current_event = None
        return result

    def settle_pending_event(
        self,
        slug: str,
        ts: float,
        resolved_up_price: float,
        resolved_down_price: float,
    ) -> EventResult | None:
        pending = self._pending_settlements.pop(slug, None)
        if pending is None:
            return None
        if not (0 <= pending.event_index < len(self._completed_events)):
            return None

        final_up = float(resolved_up_price)
        final_down = float(resolved_down_price)
        old = self._completed_events[pending.event_index]

        provisional_payout = (
            pending.up_shares * pending.provisional_up
            + pending.down_shares * pending.provisional_down
        )
        final_payout = pending.up_shares * final_up + pending.down_shares * final_down
        cash_delta = final_payout - provisional_payout
        if abs(cash_delta) > 1e-12:
            self.position.cash += cash_delta
            if self._current_event is not None:
                self._current_event.starting_equity += cash_delta
                self._current_event.last_mtm_equity += cash_delta

        resolution_pnl = final_payout - pending.still_held_mtm
        outcome = _outcome_label(final_up, final_down)
        updated = replace(
            old,
            resolved_outcome=outcome,
            pnl_total=old.pnl_intra_event + resolution_pnl,
            pnl_resolution=resolution_pnl,
            pending_resolution_up_shares=0.0,
            pending_resolution_down_shares=0.0,
            pending_resolution_up_mark=0.0,
            pending_resolution_down_mark=0.0,
        )
        self._completed_events[pending.event_index] = updated
        return updated

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
        """Record liquidation-value MTM for the tick and return equity."""
        if self._current_event is None:
            raise RuntimeError("mark_to_market called with no active event")
        up_exit = self._liquidation_price(up_book)
        down_exit = self._liquidation_price(down_book)
        equity = self.position.equity(up_exit, down_exit)
        acc = self._current_event
        acc.last_up_exit = up_exit
        acc.last_down_exit = down_exit
        acc.last_mtm_equity = equity
        acc.n_ticks += 1
        return equity

    # ---- internal ----

    def _liquidation_price(self, book: BookTop) -> float:
        if not book.is_bid_tradable:
            return 0.0
        fill_price = book.best_bid * (1.0 - self._slippage)
        p = max(0.0, min(1.0, fill_price))
        per_share_fee = self._fee_rate * p * (1.0 - p)
        return max(0.0, fill_price - per_share_fee)

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
            self.position.cash -= fill_up.fee
        fill_down = self._execute_leg(
            Side.DOWN,
            delta=target_down - self.position.down_shares,
            book=down_book,
        )
        if fill_down is not None:
            fills.append(fill_down)
            self.position.down_shares += fill_down.shares_delta
            self.position.cash -= fill_down.notional
            self.position.cash -= fill_down.fee
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
        # Polymarket-style taker fee: fee = shares * fee_rate * p * (1-p)
        # where p = fill_price. Always positive; deducted from cash on top of
        # notional.
        p = max(0.0, min(1.0, fill_price))
        fee = abs(delta) * self._fee_rate * p * (1.0 - p)
        return Fill(
            side=side,
            shares_delta=delta,
            fill_price=fill_price,
            notional=notional,
            fee=fee,
        )


def _outcome_label(up_price: float, down_price: float) -> str:
    if up_price >= 0.99 and down_price <= 0.01:
        return "UP"
    if down_price >= 0.99 and up_price <= 0.01:
        return "DOWN"
    return "UNKNOWN"
