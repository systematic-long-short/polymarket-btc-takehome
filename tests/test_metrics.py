"""Unit tests for metrics on known inputs."""

from __future__ import annotations

import math

import pytest

from polybench.metrics import (
    hit_rate,
    max_drawdown,
    outcome_accuracy,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    tick_returns,
)
from polybench.model import EventResult


def test_tick_returns_pct_changes() -> None:
    assert tick_returns([100, 101, 99, 100]) == pytest.approx(
        [1 / 100, -2 / 101, 1 / 99]
    )


def test_sharpe_zero_on_flat_returns() -> None:
    assert sharpe_ratio([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_sharpe_zero_on_zero_mean_symmetric_returns() -> None:
    # mean=0 → numerator is 0 regardless of std.
    assert sharpe_ratio([0.01, -0.01, 0.01, -0.01]) == 0.0


def test_sharpe_positive_on_positive_mean_returns() -> None:
    # Positive mean, nonzero std → Sharpe > 0.
    s = sharpe_ratio([0.01, 0.005, 0.01, 0.005, 0.01])
    assert s > 0.0
    # Should scale with sqrt of periods per year, so should be very large for 1Hz.
    assert s > 100.0


def test_sortino_zero_on_flat_returns() -> None:
    assert sortino_ratio([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_sortino_positive_when_no_downside() -> None:
    # Returns are never negative → downside std = 0 → we report 0 (undefined).
    assert sortino_ratio([0.01, 0.02, 0.005, 0.01]) == 0.0


def test_max_drawdown_peak_to_trough() -> None:
    assert max_drawdown([100, 120, 90, 110, 70]) == pytest.approx((120 - 70) / 120)


def test_max_drawdown_no_drawdown() -> None:
    assert max_drawdown([100, 110, 120, 130]) == 0.0


def test_max_drawdown_empty() -> None:
    assert max_drawdown([]) == 0.0


def _event(pnl: float, pnl_resolution: float = 0.0, outcome: str = "UP") -> EventResult:
    return EventResult(
        event_id="e",
        slug="s",
        start_ts=0.0,
        end_ts=300.0,
        resolved_outcome=outcome,
        pnl_total=pnl,
        pnl_intra_event=pnl - pnl_resolution,
        pnl_resolution=pnl_resolution,
        n_trades=0,
        n_ticks=100,
        n_timeouts=0,
    )


def test_hit_rate_counts_positive_pnl_events() -> None:
    events = [_event(100.0), _event(-50.0), _event(30.0)]
    assert hit_rate(events) == pytest.approx(2 / 3)


def test_hit_rate_empty_is_zero() -> None:
    assert hit_rate([]) == 0.0


def test_outcome_accuracy_uses_resolution_pnl() -> None:
    events = [
        _event(100.0, pnl_resolution=80.0, outcome="UP"),
        _event(-50.0, pnl_resolution=-40.0, outcome="DOWN"),
        _event(30.0, pnl_resolution=20.0, outcome="UP"),
    ]
    assert outcome_accuracy(events) == pytest.approx(2 / 3)


def test_outcome_accuracy_ignores_unknown() -> None:
    events = [
        _event(100.0, pnl_resolution=80.0, outcome="UNKNOWN"),
        _event(-50.0, pnl_resolution=0.0, outcome="UNKNOWN"),
    ]
    assert outcome_accuracy(events) == 0.0


def test_summarize_keys_and_types() -> None:
    equity = [1000.0, 1010.0, 1005.0, 1020.0]
    events = [_event(20.0, pnl_resolution=15.0, outcome="UP")]
    m = summarize(
        starting_capital=1000.0,
        final_equity=1020.0,
        equity_curve=equity,
        events=events,
    )
    assert m["pnl_total"] == 20.0
    assert m["pnl_pct"] == pytest.approx(0.02)
    assert math.isfinite(m["sharpe"])
    assert m["sharpe"] >= 0.0
    assert math.isfinite(m["sortino"])
    assert 0.0 <= m["max_drawdown"] <= 1.0
    assert m["n_events"] == 1
    assert m["hit_rate"] == 1.0
    # Primary score: PnL × max(Sharpe, 0) × (1 − max_drawdown).
    expected = m["pnl_total"] * m["sharpe"] * max(0.0, 1.0 - m["max_drawdown"])
    assert m["primary_score"] == pytest.approx(expected)


def test_primary_score_uses_full_run_pnl_not_event_ledger_sum() -> None:
    """The PnL term is final_equity - starting_capital for the full run.

    Event-level PnL rows are attribution/diagnostics. They should not become
    a separate scoring source if they ever diverge from the canonical final
    equity snapshot.
    """
    equity = [1000.0, 1005.0, 1015.0, 1020.0]
    events = [_event(999.0, pnl_resolution=0.0, outcome="UP")]
    m = summarize(
        starting_capital=1000.0,
        final_equity=1020.0,
        equity_curve=equity,
        events=events,
    )
    expected = 20.0 * m["sharpe"] * max(0.0, 1.0 - m["max_drawdown"])
    assert m["pnl_total"] == pytest.approx(20.0)
    assert m["primary_score"] == pytest.approx(expected)


def test_summarize_floors_negative_sharpe_at_zero() -> None:
    """A losing, consistently negative return stream should not earn a
    positive primary score from negative PnL times negative Sharpe."""
    equity = [1000.0, 1005.0, 990.0, 980.0]
    events = [_event(-20.0, pnl_resolution=-15.0, outcome="DOWN")]
    m = summarize(
        starting_capital=1000.0,
        final_equity=980.0,
        equity_curve=equity,
        events=events,
    )
    assert sharpe_ratio(tick_returns(equity)) < 0.0
    assert m["sharpe"] == 0.0
    assert m["primary_score"] == pytest.approx(0.0)


def test_primary_score_penalises_drawdown() -> None:
    """Identical PnL and Sharpe, heavier drawdown → lower primary_score."""
    clean = summarize(
        starting_capital=1000.0,
        final_equity=1100.0,
        equity_curve=[1000, 1020, 1050, 1080, 1100],
        events=[_event(100.0)],
    )
    drawdown = summarize(
        starting_capital=1000.0,
        final_equity=1100.0,
        equity_curve=[1000, 1200, 800, 1100, 1100],   # 33% peak-to-trough
        events=[_event(100.0)],
    )
    assert drawdown["max_drawdown"] > clean["max_drawdown"]
    # Drawdown penalty should produce a smaller primary_score for the volatile
    # track, even though PnL is identical.
    assert drawdown["primary_score"] < clean["primary_score"]
