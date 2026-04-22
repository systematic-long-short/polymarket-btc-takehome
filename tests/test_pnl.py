"""Unit tests for the paper-trading simulator."""

from __future__ import annotations

import math

import pytest

from polybench.model import Side, Signal
from polybench.pnl import BookTop, PaperSimulator


def _book(bid: float, ask: float) -> BookTop:
    return BookTop(best_bid=bid, best_ask=ask, mid=(bid + ask) / 2)


def test_initial_state_is_cash_only() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    assert sim.position.cash == 1000.0
    assert sim.position.up_shares == 0.0
    assert sim.position.down_shares == 0.0


def test_full_size_up_buy_at_ask_with_slippage() -> None:
    # 2% slippage: buying 1000 USD of UP at ask=0.50 ⇒ fill at 0.51 ⇒ 1960.78 shares.
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=200.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    fills = sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),   # up
        _book(0.50, 0.51),   # down
    )
    assert len(fills) == 1
    assert fills[0].side == Side.UP
    assert fills[0].shares_delta > 0
    assert fills[0].fill_price == pytest.approx(0.51, rel=1e-9)
    # 1000 / 0.50 = 2000 target shares, all buys at 0.51 cost 2000 * 0.51 = 1020.
    assert sim.position.up_shares == pytest.approx(2000.0, rel=1e-9)
    assert sim.position.cash == pytest.approx(-20.0, rel=1e-9)


def test_flip_from_up_to_down_closes_at_bid() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)  # no slippage
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    # Buy UP full size at 0.50 → 2000 shares, cash=0
    sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),
        _book(0.50, 0.51),
    )
    assert sim.position.up_shares == pytest.approx(2000.0, rel=1e-9)
    assert sim.position.cash == pytest.approx(0.0, abs=1e-9)
    # Flip to DOWN full size. Sells 2000 UP at bid=0.55 (+$1100 cash),
    # then buys 1000 USD worth of DOWN at ask=0.45 → 2222.22 shares.
    sim.apply_signal(
        Signal(side=Side.DOWN, size=1.0),
        _book(0.55, 0.56),
        _book(0.44, 0.45),
    )
    assert sim.position.up_shares == pytest.approx(0.0, abs=1e-9)
    assert sim.position.down_shares == pytest.approx(1000.0 / 0.45, rel=1e-9)
    # Cash: started 0 → +2000*0.55 = +1100 → -1000 spent on DOWN = 100
    assert sim.position.cash == pytest.approx(100.0, rel=1e-9)


def test_resolution_settles_held_shares_to_one_dollar() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),
        _book(0.50, 0.51),
    )
    # Held 2000 UP shares through an event that resolves UP ([1, 0]).
    sim.mark_to_market(_book(0.49, 0.50), _book(0.50, 0.51))
    result = sim.finish_event(ts=300.0, resolved_up_price=1.0, resolved_down_price=0.0)
    # 2000 shares * $1 = $2000 - starting cap $1000 = +$1000 PnL.
    assert result.resolved_outcome == "UP"
    assert result.pnl_total == pytest.approx(1000.0, abs=1e-6)
    # All resolution PnL since we held through settlement.
    assert result.pnl_resolution > 500.0
    # After settlement, position is flat and cash reflects payout.
    assert sim.position.up_shares == 0.0
    assert sim.position.cash == pytest.approx(2000.0, rel=1e-9)


def test_closed_before_resolution_gives_intra_event_pnl_only() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),   # buy at 0.50
        _book(0.50, 0.51),
    )
    sim.mark_to_market(_book(0.54, 0.55), _book(0.45, 0.46))
    # Close at bid=0.54 with no slippage → 2000 * (0.54 - 0.50) = $80 intra-event.
    sim.apply_signal(
        Signal(side=Side.FLAT, size=0.0),
        _book(0.54, 0.55),
        _book(0.45, 0.46),
    )
    sim.mark_to_market(_book(0.54, 0.55), _book(0.45, 0.46))
    # Event resolves DOWN — but we're flat, so resolution PnL = 0.
    result = sim.finish_event(ts=300.0, resolved_up_price=0.0, resolved_down_price=1.0)
    assert result.pnl_resolution == pytest.approx(0.0, abs=1e-6)
    assert result.pnl_intra_event == pytest.approx(80.0, rel=1e-2)
    assert result.resolved_outcome == "DOWN"


def test_untradable_side_skips_buy() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    # No asks on UP side → fill should be skipped.
    fills = sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        BookTop(best_bid=0.49, best_ask=0.0, mid=0.49),
        _book(0.50, 0.51),
    )
    assert fills == []
    assert sim.position.up_shares == 0.0


def test_flat_signal_closes_existing_positions() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(Signal(side=Side.UP, size=1.0),
                     _book(0.49, 0.50), _book(0.50, 0.51))
    assert sim.position.up_shares > 0
    sim.apply_signal(Signal(side=Side.FLAT, size=0.0),
                     _book(0.49, 0.50), _book(0.50, 0.51))
    assert sim.position.up_shares == 0.0


def test_one_sided_book_rejects_fills() -> None:
    """A book with only a bid and no ask (or vice versa) must not fill.

    Regression: a transient empty-ask tick synthesized a 1.0/0.01 DOWN
    book via 1-p arbitrage which allowed buying 36,000 shares at a fake
    $0.01, locking in phantom profit when the real book returned.
    """
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    # UP has a bid but no ask — not tradable.
    degenerate_up = BookTop(best_bid=0.99, best_ask=0.0, mid=0.5)
    # DOWN has the inverted arbitrage crossed book: bid=1.0, ask=0.01.
    crossed_down = BookTop(best_bid=1.0, best_ask=0.01, mid=0.5)
    fills = sim.apply_signal(
        Signal(side=Side.DOWN, size=1.0),
        degenerate_up,
        crossed_down,
    )
    assert fills == []
    assert sim.position.up_shares == 0.0
    assert sim.position.down_shares == 0.0


def test_wildly_wide_spread_rejects_fills() -> None:
    """A >50-cent spread on a [0,1] market is treated as no real market."""
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    wide = BookTop(best_bid=0.01, best_ask=0.99, mid=0.5)
    fine = BookTop(best_bid=0.45, best_ask=0.46, mid=0.455)
    fills = sim.apply_signal(Signal(side=Side.UP, size=1.0), wide, fine)
    assert fills == []


def test_timeout_counted_per_tick() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.record_timeout()
    sim.record_timeout()
    result = sim.finish_event(ts=100.0, resolved_up_price=1.0, resolved_down_price=0.0)
    assert result.n_timeouts == 2
    assert result.n_ticks == 2
