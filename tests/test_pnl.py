"""Unit tests for the paper-trading simulator."""

from __future__ import annotations

import math

import pytest

from polybench.model import Side, Signal
from polybench.pnl import BookTop, PaperSimulator


def _book(bid: float, ask: float) -> BookTop:
    return BookTop(best_bid=bid, best_ask=ask, mid=(bid + ask) / 2)


def test_initial_state_is_cash_only() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    assert sim.position.cash == 1000.0
    assert sim.position.up_shares == 0.0
    assert sim.position.down_shares == 0.0


def test_default_slippage_is_half_percent_per_order() -> None:
    sim = PaperSimulator(starting_capital=1000.0, fee_rate=0.0)
    assert sim.slippage_bps == pytest.approx(50.0)


def test_full_size_up_buy_at_ask_with_slippage() -> None:
    # 2% slippage: buying 1000 USD of UP at ask=0.50 ⇒ fill at 0.51 ⇒ 1960.78 shares.
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=200.0, fee_rate=0.0)
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
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)  # no slippage
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
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
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
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
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


def test_mark_to_market_uses_liquidation_bid_not_mid() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),
        _book(0.50, 0.51),
    )

    equity = sim.mark_to_market(
        BookTop(best_bid=0.54, best_ask=0.56, mid=0.55),
        BookTop(best_bid=0.44, best_ask=0.46, mid=0.45),
    )

    assert equity == pytest.approx(1080.0, rel=1e-9)


def test_mark_to_market_uses_one_sided_bid_for_exit_value() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),
        _book(0.50, 0.51),
    )

    equity = sim.mark_to_market(
        BookTop(best_bid=0.90, best_ask=0.0, mid=0.90),
        BookTop(best_bid=0.10, best_ask=0.20, mid=0.15),
    )

    assert equity == pytest.approx(1800.0, rel=1e-9)


def test_mark_to_market_marks_zero_when_position_has_no_exit_bid() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        _book(0.49, 0.50),
        _book(0.50, 0.51),
    )

    equity = sim.mark_to_market(
        BookTop(best_bid=0.0, best_ask=0.90, mid=0.90),
        BookTop(best_bid=0.10, best_ask=0.20, mid=0.15),
    )

    assert equity == pytest.approx(0.0, abs=1e-9)


def test_untradable_side_skips_buy() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
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
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.apply_signal(Signal(side=Side.UP, size=1.0),
                     _book(0.49, 0.50), _book(0.50, 0.51))
    assert sim.position.up_shares > 0
    sim.apply_signal(Signal(side=Side.FLAT, size=0.0),
                     _book(0.49, 0.50), _book(0.50, 0.51))
    assert sim.position.up_shares == 0.0


def test_fee_is_max_at_p_0_5() -> None:
    """Per-fill fee = shares * fee_rate * p * (1-p). At p=0.5, p(1-p)=0.25."""
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.072)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    # Book with mid exactly 0.5: bid=0.50, ask=0.50 (no spread, no slippage).
    book = BookTop(best_bid=0.50, best_ask=0.50, mid=0.50)
    fills = sim.apply_signal(
        Signal(side=Side.UP, size=1.0),   # target $1000 notional
        book,
        BookTop(best_bid=0.49, best_ask=0.50, mid=0.495),
    )
    assert len(fills) == 1
    fill = fills[0]
    assert fill.side == Side.UP
    # Target notional = $1000, fill price = $0.50 → target 2000 shares.
    assert fill.shares_delta == pytest.approx(2000.0, rel=1e-9)
    # Notional = 2000 * 0.50 = $1000. Fee = 2000 * 0.072 * 0.5 * 0.5 = $36.
    assert fill.notional == pytest.approx(1000.0, rel=1e-9)
    assert fill.fee == pytest.approx(36.0, rel=1e-9)
    # Cash: started $1000, minus $1000 notional, minus $36 fee = -$36.
    assert sim.position.cash == pytest.approx(-36.0, rel=1e-9)


def test_fee_shrinks_at_extremes() -> None:
    """At p=0.1, p(1-p)=0.09, so per-share fees are smaller than at p=0.5."""
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.072)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.1, down_mid=0.9)
    book = BookTop(best_bid=0.10, best_ask=0.10, mid=0.10)
    fills = sim.apply_signal(
        Signal(side=Side.UP, size=1.0),
        book,
        BookTop(best_bid=0.89, best_ask=0.90, mid=0.895),
    )
    assert len(fills) == 1
    fill = fills[0]
    # Target notional = $1000, fill price = $0.10 → target 10000 shares.
    assert fill.shares_delta == pytest.approx(10000.0, rel=1e-9)
    # Notional = 10000 * 0.10 = $1000. Fee = 10000 * 0.072 * 0.1 * 0.9 = $64.80.
    assert fill.fee == pytest.approx(64.8, rel=1e-9)
    assert sim.position.cash == pytest.approx(-64.8, rel=1e-9)


def test_fee_charged_on_both_buy_and_sell() -> None:
    """Round-trip (open then close) pays fee both ways."""
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.072)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    book = BookTop(best_bid=0.50, best_ask=0.50, mid=0.50)
    # Open
    buy_fills = sim.apply_signal(Signal(side=Side.UP, size=1.0), book, book)
    open_fee = buy_fills[0].fee
    # Close (FLAT)
    sell_fills = sim.apply_signal(Signal(side=Side.FLAT, size=0.0), book, book)
    close_fee = sell_fills[0].fee
    # Selling 2000 shares at 0.50 charges the same per-share fee.
    assert open_fee == pytest.approx(36.0, rel=1e-9)
    assert close_fee == pytest.approx(36.0, rel=1e-9)


def test_mark_to_market_subtracts_share_based_exit_fee() -> None:
    sim = PaperSimulator(starting_capital=0.0, slippage_bps=0.0, fee_rate=0.072)
    sim.position.up_shares = 100.0
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)

    equity = sim.mark_to_market(
        BookTop(best_bid=0.50, best_ask=0.50, mid=0.50),
        BookTop(best_bid=0.49, best_ask=0.51, mid=0.50),
    )

    assert equity == pytest.approx(48.2, rel=1e-9)


def test_one_sided_book_rejects_fills() -> None:
    """A book with only a bid and no ask (or vice versa) must not fill.

    Regression: a transient empty-ask tick synthesized a 1.0/0.01 DOWN
    book via 1-p arbitrage which allowed buying 36,000 shares at a fake
    $0.01, locking in phantom profit when the real book returned.
    """
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
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
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    wide = BookTop(best_bid=0.01, best_ask=0.99, mid=0.5)
    fine = BookTop(best_bid=0.45, best_ask=0.46, mid=0.455)
    fills = sim.apply_signal(Signal(side=Side.UP, size=1.0), wide, fine)
    assert fills == []


def test_timeout_counted_per_tick() -> None:
    sim = PaperSimulator(starting_capital=1000.0, slippage_bps=0.0, fee_rate=0.0)
    sim.start_event("E1", "ev1", ts=0.0, up_mid=0.5, down_mid=0.5)
    sim.record_timeout()
    sim.record_timeout()
    result = sim.finish_event(ts=100.0, resolved_up_price=1.0, resolved_down_price=0.0)
    assert result.n_timeouts == 2
    assert result.n_ticks == 2
